# src/engine/vision_models.py
# Gemma 3 and MedGemma inference helpers

import base64
import logging
import time
import gc
import requests
import torch
from PIL import Image

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434"  # Match your existing .env or config

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


def cleanup_python_models():
    """
    Forcefully removes MedGemma, Hulu-Med, and LLaVA from Python memory and VRAM.
    """
    import streamlit as st
    logger.info("♻️ Cleaning up Python-resident models...")
    
    # 1. Identify ALL potential model keys
    model_keys = [
        ('mg4b_model', 'mg4b_processor'),
        ('medgemma_model', 'medgemma_processor'),
        ('hulumed_model', 'hulumed_processor'),
        ('vision_engine',)
    ]
    
    # 2. Delete from session state
    for keys in model_keys:
        for key in keys:
            if key in st.session_state:
                logger.info(f"  Deleting {key} from session state...")
                try:
                    del st.session_state[key]
                except Exception as e:
                    logger.warning(f"  Could not delete {key}: {e}")
    
    # 3. Reset all loaded flags
    st.session_state['medgemma_loaded'] = False
    st.session_state['medgemma4b_loaded'] = False
    st.session_state['hulumed_loaded'] = False
    
    # 4. Aggressive VRAM cleanup
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

def analyze_with_gemma3(image_path: str, user_question: str, conversation_history: list = []) -> str:
    """Sends image + question to Gemma 3 27B via Ollama. Multi-turn capable."""
    import ollama

    messages = [
        {
            "role": "system",
            "content": (
                "You are Dr. Gemma, an Expert Wound Care Medical Director. "
                "Analyze wound images with medical grade specificity."
            )
        }
    ]

    messages.extend(conversation_history)

    messages.append({
        "role": "user",
        "content": user_question,
        "images": [image_path]
    })

    try:
        response = ollama.chat(model="gemma3:27b", messages=messages)
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
    logger.error(f"VRAM Status: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

    # Load MedGemma
    try:
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


def analyze_with_medgemma(image_path: str, user_question: str, processor, model) -> str:
    """Runs inference on image + question using MedGemma 27B."""
    try:
        image = Image.open(image_path).convert("RGB")

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are an expert medical imaging specialist."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_question},
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
            generation = model.generate(**inputs, max_new_tokens=512, do_sample=False)
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
    logger.error(f"VRAM Status: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

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

def analyze_with_hulumed(image_path: str, user_question: str, processor, model) -> str:
    """
    Runs inference on image + question using Hulu-Med-7B.
    Uses the HF-Version conversation format from their official docs.
    """
    try:
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": {
                            "image_path": image_path,
                        }
                    },
                    {
                        "type": "text",
                        "text": user_question
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
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False,
            )

        # use_think=False: final answer only, no reasoning chain
        output = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            use_think=False
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
    logger.error(f"VRAM Status: {free_gb:.1f} GB free / {total_gb:.1f} GB total")

    # Load MedGemma
    try:
        processor = AutoProcessor.from_pretrained("google/medgemma-1.5-4b-it", use_fast=True)

        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

        model = AutoModelForImageTextToText.from_pretrained(
            "google/medgemma-1.5-4b-it",
            torch_dtype=torch.bfloat16,
            device_map={"": 0},
            quantization_config=quantization_config,
            use_safetensors=True
        )
        logger.info("✅ MedGemma 4B loaded (4-bit quantized).")
        return processor, model
    except Exception as e:
        logger.error(f"❌ MedGemma load failed: {e}")
        return None, None


def analyze_with_medgemma_4b(image_path: str, user_question: str, processor, model) -> str:
    """Inference for MedGemma 4B."""
    return analyze_with_medgemma(image_path, user_question, processor, model)