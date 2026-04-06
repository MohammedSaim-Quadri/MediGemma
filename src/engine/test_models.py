# src/engine/vision_models.py
# Gemma 3 and MedGemma inference helpers

import base64
import logging
import os
import time
import gc
import threading
import requests
import torch
import yaml
from dataclasses import dataclass, field
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config name → Registry name mapping
# ---------------------------------------------------------------------------
MODEL_NAME_MAP = {
    "medgemma_27b": "medgemma",
    "medgemma_4b": "mg4b",
    "hulumed": "hulumed",
    "gemma3": None,  # Ollama backend, no Python registry
}

# ---------------------------------------------------------------------------
# Inference configuration and result dataclasses
# ---------------------------------------------------------------------------
@dataclass
class InferenceConfig:
    """Configuration for a single inference run."""
    model_name: str               # config name: "medgemma_27b" | "medgemma_4b" | "hulumed" | "gemma3"
    registry_name: str | None     # registry name: "medgemma" | "mg4b" | "hulumed" | None
    profile_name: str             # "default" | "creative" | ...
    prompt_template: str          # "clinician_v1" | "structured_output" | ...
    backend: str                  # "transformers" | "ollama"
    generate_params: dict = field(default_factory=dict)
    decode_params: dict = field(default_factory=dict)
    backend_options: dict = field(default_factory=dict)
    system_prompt: str = ""
    user_suffix: str = ""
    image_max_size: int = 896


@dataclass
class InferenceResult:
    """Result from a single inference run."""
    model_name: str
    profile_name: str
    prompt_template: str
    output: str
    inference_time_sec: float
    generate_params: dict = field(default_factory=dict)
    system_prompt: str = ""
    error: str | None = None


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------
def _get_config_dir() -> str:
    """Return absolute path to the config/ directory."""
    return os.path.join(os.path.dirname(__file__), "..", "..", "config")


def load_model_profiles() -> dict:
    """Load config/model_profiles.yaml and return the models dict."""
    path = os.path.join(_get_config_dir(), "model_profiles.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("models", {})


def load_prompt_templates() -> dict:
    """Load config/prompts.yaml and return the prompt_templates dict."""
    path = os.path.join(_get_config_dir(), "prompts.yaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("prompt_templates", {})


def build_inference_config(
    model_name: str,
    profile_name: str = "default",
    prompt_template: str = "clinician_v1",
) -> InferenceConfig:
    """
    Build an InferenceConfig by merging model profile + prompt template from YAML configs.

    Args:
        model_name: Config key like "medgemma_27b", "hulumed", "gemma3"
        profile_name: Profile within the model config ("default", "creative", etc.)
        prompt_template: Prompt template key ("clinician_v1", "structured_output", etc.)

    Returns:
        InferenceConfig ready to pass to run_inference()
    """
    profiles = load_model_profiles()
    prompts = load_prompt_templates()

    if model_name not in profiles:
        raise ValueError(f"Unknown model '{model_name}'. Available: {list(profiles.keys())}")

    model_cfg = profiles[model_name]
    available_profiles = model_cfg.get("profiles", {})

    if profile_name not in available_profiles:
        raise ValueError(
            f"Unknown profile '{profile_name}' for model '{model_name}'. "
            f"Available: {list(available_profiles.keys())}"
        )

    if prompt_template not in prompts:
        raise ValueError(
            f"Unknown prompt template '{prompt_template}'. Available: {list(prompts.keys())}"
        )

    profile = available_profiles[profile_name]
    prompt_cfg = prompts[prompt_template]

    return InferenceConfig(
        model_name=model_name,
        registry_name=model_cfg.get("registry_name"),
        profile_name=profile_name,
        prompt_template=prompt_template,
        backend=model_cfg.get("backend", "transformers"),
        generate_params=profile.get("generate_params", {}),
        decode_params=profile.get("decode_params", {}),
        backend_options=profile.get("backend_options", {}),
        system_prompt=prompt_cfg.get("system", ""),
        user_suffix=prompt_cfg.get("suffix", ""),
        image_max_size=model_cfg.get("image_max_size", 896),
    )

# ---------------------------------------------------------------------------
# Process-level global model registry (shared across all Streamlit sessions)
# ---------------------------------------------------------------------------
_model_registry = {
    "model": None,
    "processor": None,
    "model_name": None,  # "hulumed" / "medgemma" / "mg4b"
    "loading": None,     # model name currently being loaded (process-level)
}
_registry_lock = threading.Lock()


def set_model_loading(name: str | None):
    """Mark a model as currently loading (or None when done). Process-level, visible to all sessions."""
    with _registry_lock:
        _model_registry["loading"] = name


def get_model_loading() -> str | None:
    """Return the name of the model currently being loaded, or None."""
    with _registry_lock:
        return _model_registry["loading"]


def register_model(name: str, processor, model):
    """Register a loaded model in the global registry (replaces any previous)."""
    with _registry_lock:
        _model_registry["model"] = model
        _model_registry["processor"] = processor
        _model_registry["model_name"] = name
        _model_registry["loading"] = None  # loading finished
        logger.info(f"📋 Global registry: registered '{name}'")


def get_registered_model():
    """Return (name, processor, model) from the global registry, or (None, None, None)."""
    with _registry_lock:
        return (
            _model_registry["model_name"],
            _model_registry["processor"],
            _model_registry["model"],
        )


def clear_registry():
    """Clear the global model registry (does NOT delete the model objects themselves)."""
    with _registry_lock:
        _model_registry["model"] = None
        _model_registry["processor"] = None
        _model_registry["model_name"] = None
        logger.info("📋 Global registry: cleared")

OLLAMA_URL = "http://localhost:11434"  # Match your existing .env or config

# Max resolution per model (longest edge in pixels).
# MedGemma internally resizes to 896x896, so pre-resizing is lossless.
# Hulu-Med uses dynamic resolution; large images cause OOM on 24GB GPU.
IMAGE_MAX_SIZE = {
    "gemma3": 896,
    "medgemma": 896,
    "medgemma_4b": 896,
    "hulumed": 512,  # Conservative; 256 confirmed safe, test up to 768/1024 later
}


def preprocess_image(image_path: str, max_size: int) -> str:
    """Resize image so longest edge <= max_size. Maintains aspect ratio. Overwrites temp file."""
    try:
        img = Image.open(image_path)
        w, h = img.size

        if max(w, h) <= max_size:
            logger.info(f"Image {w}x{h} within {max_size}px limit, no resize needed.")
            return image_path

        if w >= h:
            new_w = max_size
            new_h = int(h * (max_size / w))
        else:
            new_h = max_size
            new_w = int(w * (max_size / h))

        logger.info(f"Resizing image from {w}x{h} to {new_w}x{new_h} (max_size={max_size})")
        img_resized = img.resize((new_w, new_h), Image.LANCZOS)
        img_resized.save(image_path, quality=95)
        return image_path

    except Exception as e:
        logger.warning(f"Image preprocessing failed: {e}. Using original.")
        return image_path

def force_vram_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.memory._dump_snapshot = False
    gc.collect()  # Double collect for stubborn references


def log_vram(stage=""):
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        free_gb = free / (1024**3)
        total_gb = total / (1024**3)
        logger.info(f"🧠 VRAM {stage}: {free_gb:.2f} GB free / {total_gb:.2f} GB total")


def evict_ollama_with_retry(required_free_gb=20.0, max_retries=10, retry_delay=3.0):
    logger.info(f"🧹 Starting aggressive Ollama eviction (need {required_free_gb:.1f} GB free)")
    
    for attempt in range(max_retries):
        try:
            # Step 1: Query running models
            resp = requests.get(f"{OLLAMA_URL}/api/ps", timeout=5)
            
            if resp.status_code == 200:
                running_models = resp.json().get('models', [])
                
                if running_models:
                    logger.info(f"Attempt {attempt + 1}/{max_retries}: Found {len(running_models)} models running")
                    
                    # Unload each model
                    for m in running_models:
                        model_name = m['name']
                        logger.info(f"  Unloading: {model_name}")
                        try:
                            requests.post(
                                f"{OLLAMA_URL}/api/generate",
                                json={"model": model_name, "keep_alive": 0},
                                timeout=10
                            )
                        except Exception as e:
                            logger.warning(f"  Failed to unload {model_name}: {e}")
                else:
                    logger.info(f"Attempt {attempt + 1}/{max_retries}: No Ollama models currently loaded")
            
            # Step 2: Aggressive Python cleanup
            force_vram_cleanup()
            
            # Step 3: Wait for VRAM to stabilize
            time.sleep(retry_delay)
            
            # Step 4: Check if we have enough free VRAM
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024 ** 3)
                total_gb = total_bytes / (1024 ** 3)
                
                logger.info(f"  VRAM Status: {free_gb:.1f} GB free / {total_gb:.1f} GB total")
                
                if free_gb >= required_free_gb:
                    logger.info(f"✅ SUCCESS: {free_gb:.1f} GB free (needed {required_free_gb:.1f} GB)")
                    return True
                else:
                    shortage = required_free_gb - free_gb
                    logger.warning(f"  ⚠️ Still need {shortage:.1f} GB more. Retrying...")
            else:
                logger.error("CUDA not available!")
                return False
                
        except Exception as e:
            logger.error(f"Attempt {attempt + 1} error: {e}")
        
        # Wait before next retry (with exponential backoff)
        if attempt < max_retries - 1:
            wait_time = retry_delay * (1.5 ** attempt)  # 3s, 4.5s, 6.75s, etc.
            logger.info(f"  Waiting {wait_time:.1f}s before retry {attempt + 2}/{max_retries}...")
            time.sleep(wait_time)
    
    # If we get here, all retries failed
    logger.error(f"❌ FAILED: Could not free {required_free_gb:.1f} GB after {max_retries} attempts")
    return False


def _destroy_cuda_model(model):
    """
    Aggressively release CUDA memory from a PyTorch model BEFORE dropping the reference.
    This works even if dangling references elsewhere prevent GC from collecting the model,
    because we replace every CUDA tensor with an empty CPU tensor in-place.
    """
    try:
        # Null out all parameter tensors (the bulk of VRAM usage)
        for param in model.parameters():
            param.data = torch.empty(0, device="cpu")
            if param.grad is not None:
                param.grad = None
        # Null out all buffer tensors (batch norm stats, etc.)
        for buf in model.buffers():
            buf.data = torch.empty(0, device="cpu")
        logger.info("  🗑️ All CUDA tensors replaced with empty CPU tensors")
    except Exception as e:
        logger.warning(f"  ⚠️ _destroy_cuda_model partial failure: {e}")


def cleanup_python_models():
    """
    Forcefully removes MedGemma, Hulu-Med, and LLaVA from Python memory and VRAM.
    Model objects live ONLY in the global registry — clearing it drops the sole
    reference, allowing GC to free GPU memory regardless of session state.
    """
    import streamlit as st
    logger.info("♻️ Cleaning up Python-resident models...")

    # 1. Extract model from registry, then clear registry entries
    # NOTE: Do NOT clear loading flag here — it's managed by set_model_loading()
    # and register_model(). Clearing it here would race with in-progress loads.
    old_model = None
    old_processor = None
    with _registry_lock:
        if _model_registry["model"] is not None:
            logger.info(f"  Deleting global registry model '{_model_registry['model_name']}'")
            old_model = _model_registry["model"]
            old_processor = _model_registry["processor"]
            _model_registry["model"] = None
            _model_registry["processor"] = None
            _model_registry["model_name"] = None

    # 2. Aggressively destroy CUDA tensors BEFORE dropping the reference
    # This ensures VRAM is freed even if circular refs keep the model alive
    if old_model is not None:
        _destroy_cuda_model(old_model)
        del old_model
    if old_processor is not None:
        del old_processor

    # 3. Clean up any legacy session_state model refs (from before registry-only migration)
    legacy_keys = [
        'mg4b_model', 'mg4b_processor',
        'medgemma_model', 'medgemma_processor',
        'hulumed_model', 'hulumed_processor',
        'vision_engine',
    ]
    for key in legacy_keys:
        if key in st.session_state:
            logger.info(f"  Deleting legacy {key} from session state...")
            try:
                del st.session_state[key]
            except Exception as e:
                logger.warning(f"  Could not delete {key}: {e}")

    # 4. Reset all loaded flags
    st.session_state['medgemma_loaded'] = False
    st.session_state['medgemma4b_loaded'] = False
    st.session_state['hulumed_loaded'] = False

    # 5. Aggressive VRAM cleanup
    force_vram_cleanup()

    logger.info("✅ Python VRAM cleared.")


def master_evict_with_retry(required_free_gb=18.0, max_retries=10):
    """
    The nuclear option: Clears EVERYTHING with retry logic.
    """
    logger.info(f"💣 MASTER EVICT: Clearing all models (need {required_free_gb:.1f} GB)")
    
    # Step 1: Clean up Python models first
    cleanup_python_models()
    
    # Step 2: Evict Ollama with retry
    success = evict_ollama_with_retry(
        required_free_gb=required_free_gb,
        max_retries=max_retries,
        retry_delay=3.0
    )
    
    if not success:
        logger.error("⚠️ WARNING: Could not free enough VRAM. Next load will likely fail.")
    
    return success


def unload_model_safely(model, processor, model_name="Model"):
    """
    Safely unloads a HuggingFace model with verification.
    """
    try:
        logger.info(f"♻️ Unloading {model_name}...")
        
        if model is not None:
            del model
        if processor is not None:
            del processor
        
        # Force cleanup
        force_vram_cleanup()
        log_vram(f"after unloading {model_name}")
        
        return None, None
        
    except Exception as e:
        logger.warning(f"⚠️ {model_name} unload error: {e}")
        return None, None


# ============================================================================
# GEMMA 3
# ============================================================================

def analyze_with_gemma3(image_path: str, user_question: str, conversation_history: list = [], config: InferenceConfig = None) -> str:
    """Sends image + question to Gemma 3 27B via Ollama. Multi-turn capable."""
    import ollama

    _default_system = (
        "You are a wound care specialist providing analysis to fellow clinicians "
        "in a hospital clinical decision support system. "
        "Give direct, specific clinical observations and assessments. "
        "Do not add disclaimers about consulting a doctor — the reader IS the doctor."
    )

    system_prompt = config.system_prompt if config else _default_system
    suffix = config.user_suffix if config else ""
    options = config.backend_options if config else {}
    max_size = config.image_max_size if config else IMAGE_MAX_SIZE["gemma3"]

    messages = [{"role": "system", "content": system_prompt}]

    messages.extend(conversation_history)

    processed_path = preprocess_image(image_path, max_size)

    messages.append({
        "role": "user",
        "content": user_question + suffix,
        "images": [processed_path]
    })

    try:
        response = ollama.chat(model="gemma3:27b", messages=messages, options=options)
        return response["message"]["content"]
    except Exception as e:
        logger.error(f"Gemma 3 error: {e}")
        return f"Error: {str(e)}\nMake sure 'ollama pull gemma3:27b' has been run."


# ============================================================================
# MEDGEMMA 27B
# ============================================================================

def load_medgemma_27b():
    """
    Loads google/medgemma-27b-it (multimodal, NOT the text-only variant).
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    logger.info("Loading MedGemma 27B...")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    logger.info(f"VRAM Status: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

    # Load MedGemma
    try:
        # Work around intermittent xFormers/SDPA alignment crashes on some CUDA stacks.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
        
        processor = AutoProcessor.from_pretrained("google/medgemma-27b-it", use_fast=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-27b-it",
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )
        logger.info("✅ MedGemma 27B loaded (4-bit quantized).")
        return processor, model
    except Exception as e:
        logger.error(f"❌ MedGemma load failed: {e}")
        return None, None


def analyze_with_medgemma(image_path: str, user_question: str, processor, model, config: InferenceConfig = None) -> str:
    """Runs inference on image + question using MedGemma 27B."""
    _default_system = (
        "You are a wound care specialist providing analysis to fellow clinicians "
        "in a hospital clinical decision support system. "
        "Give direct, specific clinical observations and assessments. "
        "Do not add disclaimers about consulting a doctor — the reader IS the doctor. "
        "Do not refuse to analyze. Provide your professional assessment directly."
    )
    _default_gen_params = {"max_new_tokens": 2048, "do_sample": False}

    system_prompt = config.system_prompt if config else _default_system
    suffix = config.user_suffix if config else ""
    gen_params = config.generate_params if config else _default_gen_params
    max_size = config.image_max_size if config else IMAGE_MAX_SIZE["medgemma"]

    try:
        processed_path = preprocess_image(image_path, max_size)
        image = Image.open(processed_path).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_question + suffix},
                    {"type": "image", "image": image}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, cache_implementation="hybrid", **gen_params)
            generation = generation[0][input_len:]

        return processor.decode(generation, skip_special_tokens=True)

    except Exception as e:
        logger.error(f"MedGemma error: {e}")
        return f"Error: {str(e)}"


def load_hulumed():
    """
    Loads Hulu-Med 32B
    """
    from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

    MODEL_ID = "ZJU-AI4H/Hulu-Med-32B"

    logger.info("Loading Hulu-Med 32B...")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 **3)
    total_gb = total_bytes / (1024 **3)
    logger.info(f"VRAM Status: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

    try:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        processor = AutoProcessor.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            use_fast=True
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            quantization_config=quantization_config,
            low_cpu_mem_usage=True,
            use_safetensors=True
        )

        logger.info("✅ Hulu-Med-32B loaded (4-bit quantized).")
        return processor, model

    except Exception as e:
        logger.error(f"❌ Failed to load Hulu-Med: {e}")
        return None, None

def analyze_with_hulumed(image_path: str, user_question: str, processor, model, config: InferenceConfig = None) -> str:
    """
    Runs inference on image + question using Hulu-Med-32B.
    Uses the HF-Version conversation format from their official docs.
    """
    _default_gen_params = {"max_new_tokens": 2048, "do_sample": True, "temperature": 0.6}
    _default_decode_params = {"use_think": False}

    suffix = config.user_suffix if config else "\nProvide a detailed clinical analysis."
    gen_params = config.generate_params if config else _default_gen_params
    decode_params = config.decode_params if config else _default_decode_params
    max_size = config.image_max_size if config else IMAGE_MAX_SIZE["hulumed"]

    try:
        processed_path = preprocess_image(image_path, max_size)

        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": {
                            "image_path": processed_path,
                        }
                    },
                    {
                        "type": "text",
                        "text": user_question + suffix
                    },
                ]
            }
        ]

        inputs = processor(
            conversation=conversation,
            add_system_prompt=True,
            add_generation_prompt=True,
            return_tensors="pt"
        )

        inputs = {k: v.to(model.device) if isinstance(v, torch.Tensor) else v
                  for k, v in inputs.items()}

        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.bfloat16)

        with torch.inference_mode():
            output_ids = model.generate(**inputs, **gen_params)

        # Hulu-Med's generate() returns only new tokens (not input+output),
        # so decode directly without stripping input tokens.
        output = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            **decode_params
        )[0].strip()

        return output

    except Exception as e:
        logger.error(f"Hulu-Med error: {e}")
        return f"Error: {str(e)}"

def load_medgemma_4b():
    """
    Loads google/medgemma-1.5-4b-it (multimodal, NOT the text-only variant).
    """
    from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

    logger.info("Loading MedGemma-1.5 4B...")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gb = free_bytes / (1024 ** 3)
    total_gb = total_bytes / (1024 ** 3)
    logger.info(f"VRAM Status: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

    # Load MedGemma 4B — small enough for full bf16 (~8GB), no quantization needed
    try:
        # Work around intermittent xFormers/SDPA alignment crashes on some CUDA stacks.
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

        processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it", use_fast=True)

        model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-1.5-4b-it",
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            use_safetensors=True,
            attn_implementation="eager",
        )
        logger.info("✅ MedGemma 4B loaded (bf16, eager attention).")
        return processor, model
    except Exception as e:
        logger.error(f"❌ MedGemma load failed: {e}")
        return None, None


def analyze_with_medgemma_4b(image_path: str, user_question: str, processor, model, config: InferenceConfig = None) -> str:
    """Inference for MedGemma 4B. Uses same logic as 27B but with its own image size config."""
    _default_system = (
        "You are a wound care specialist providing analysis to fellow clinicians "
        "in a hospital clinical decision support system. "
        "Give direct, specific clinical observations and assessments. "
        "Do not add disclaimers about consulting a doctor — the reader IS the doctor. "
        "Do not refuse to analyze. Provide your professional assessment directly."
    )
    _default_gen_params = {"max_new_tokens": 2048, "do_sample": False}

    system_prompt = config.system_prompt if config else _default_system
    suffix = config.user_suffix if config else ""
    gen_params = config.generate_params if config else _default_gen_params
    max_size = config.image_max_size if config else IMAGE_MAX_SIZE["medgemma_4b"]

    try:
        processed_path = preprocess_image(image_path, max_size)
        image = Image.open(processed_path).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": system_prompt}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_question + suffix},
                    {"type": "image", "image": image}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)

        input_len = inputs["input_ids"].shape[-1]

        with torch.inference_mode():
            generation = model.generate(**inputs, **gen_params)
            generation = generation[0][input_len:]

        decoded = processor.decode(generation, skip_special_tokens=True)

        # MedGemma 1.5 wraps its thinking chain in <unused94>thought...<unused95>.
        # These are normal vocab tokens so skip_special_tokens won't remove them.
        import re
        decoded = re.sub(r"<unused\d+>thought.*?<unused\d+>", "", decoded, flags=re.DOTALL).strip()

        return decoded

    except Exception as e:
        logger.error(f"MedGemma 4B error: {e}")
        return f"Error: {str(e)}"


# ---------------------------------------------------------------------------
# Unified inference entry point
# ---------------------------------------------------------------------------
def run_inference(
    image_path: str,
    question: str,
    config: InferenceConfig,
    processor=None,
    model=None,
) -> InferenceResult:
    """
    Unified inference entry point.

    Routes to the appropriate analyze_with_*() based on config.model_name.
    - transformers backend: requires processor + model (caller must provide)
    - ollama backend: processor/model not needed

    Args:
        image_path: Path to the input image
        question: User question about the image
        config: InferenceConfig built from build_inference_config()
        processor: HuggingFace processor (required for transformers backend)
        model: HuggingFace model (required for transformers backend)

    Returns:
        InferenceResult with output text, timing, and metadata
    """
    if config.backend == "transformers" and (processor is None or model is None):
        raise ValueError(
            f"transformers backend requires processor and model for {config.model_name}"
        )

    error = None
    output = ""

    start = time.time()
    try:
        if config.model_name == "medgemma_27b":
            output = analyze_with_medgemma(image_path, question, processor, model, config=config)
        elif config.model_name == "medgemma_4b":
            output = analyze_with_medgemma_4b(image_path, question, processor, model, config=config)
        elif config.model_name == "hulumed":
            output = analyze_with_hulumed(image_path, question, processor, model, config=config)
        elif config.model_name == "gemma3":
            output = analyze_with_gemma3(image_path, question, config=config)
        else:
            raise ValueError(f"Unknown model_name: {config.model_name}")

        # Check if the output itself is an error message from the analyze function
        if output.startswith("Error:"):
            error = output
    except Exception as e:
        error = str(e)
        output = f"Error: {e}"

    elapsed = time.time() - start

    return InferenceResult(
        model_name=config.model_name,
        profile_name=config.profile_name,
        prompt_template=config.prompt_template,
        output=output,
        inference_time_sec=round(elapsed, 2),
        generate_params=config.generate_params,
        system_prompt=config.system_prompt,
        error=error,
    )
