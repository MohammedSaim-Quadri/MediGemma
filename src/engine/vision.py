import time
import torch
import gc
import logging
import os
import requests
from PIL import Image
from transformers import BitsAndBytesConfig
try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
except ImportError:
    pass

logger = logging.getLogger(__name__)

class VisionEngine:
    """
    Handles all visual perception tasks using the local LLaVA-Medical model.
    """
    def __init__(self, model_path=None):
        self.model_path = model_path or os.getenv("VISION_MODEL_PATH", "./LLaVA-Medical-Director")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.reasoning_model = os.getenv("REASONING_MODEL", "gemma2:27b")
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.loaded = False

    def _evict_ollama(self):
        """
        CRITICAL: Forces the Ollama server to unload the 18GB Gemma model.
        This clears the VRAM 'parking spot' for LLaVA.
        """
        try:
            logger.info("🧹 Orchestrator: Evicting Ollama model to free VRAM...")
            # Sending keep_alive: 0 forces an immediate unload
            response = requests.post(
                f"{self.ollama_url}/api/chat",
                json={
                    "model": self.reasoning_model, 
                    "messages": [], 
                    "keep_alive": 0
                }
            )
            if response.status_code == 200:
                logger.info("✅ Ollama Evicted. VRAM is clear.")

            # 2. VERIFICATION LOOP (The Fix)
            # We wait up to 10 seconds for VRAM to drop below a safe threshold.
            max_retries = 5
            required_free_gb = 8.0 # LLaVA needs ~6GB, plus overhead.
            
            logger.info("⏳ Waiting for VRAM release...")
            for i in range(max_retries):
                # Force Python Garbage Collection
                gc.collect()
                torch.cuda.empty_cache()
                
                # Check actual VRAM
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                free_gb = free_bytes / (1024 ** 3)
                
                if free_gb >= required_free_gb:
                    logger.info(f"✅ VRAM Clear: {free_gb:.1f} GB free. Ready for Vision.")
                    return # Success!
                
                logger.debug(f"   Retry {i+1}/{max_retries}: Only {free_gb:.1f} GB free. Waiting...")
                time.sleep(2) # Wait 2 seconds before checking again
            
            # If we get here, it didn't clear.
            logger.error("❌ TIMEOUT: Ollama did not release VRAM in time.")
            # We don't raise an error here to allow a 'Hail Mary' attempt, 
            # but LLaVA will likely crash if this log appears.

        except Exception as e:
            logger.warning(f"⚠️ Could not contact Ollama for eviction: {e}")

    def load_model(self):
        """Loads the Vision Model into VRAM."""
        if self.loaded:
            return
        
        self._evict_ollama()
        
        try:
            logger.info(f"🏥 Loading Vision Model from: {self.model_path}...")
            disable_torch_init()
            model_name = get_model_name_from_path(self.model_path)

            # We create the config explicitly to satisfy newer Transformers versions
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
            
            # Call loader with load_4bit=False to prevent LLaVA from setting the conflicting flag.
            # We pass our valid config in kwargs instead.
            self.tokenizer, self.model, self.image_processor, _ = load_pretrained_model(
                model_path=self.model_path, 
                model_base=None,
                model_name=model_name, 
                load_4bit=False,      # <--- CRITICAL: Turn off LLaVA's internal flag
                load_8bit=False,
                quantization_config=quantization_config, # <--- Pass our clean config
                device_map="cuda"
            )

            self.loaded = True
            logger.info("✅ Vision Model loaded successfully.")
            
        except Exception as e:
            logger.critical(f"❌ Failed to load Vision Model: {e}", exc_info=True)
            raise e

    def analyze(self, image_file, prompt=None):
        """
        Runs inference on a single image.
        """
        if prompt is None:
            prompt = (
                "ACT AS A MEDICAL IMAGING ANALYST. \n"
                "Analyze this wound image strictly. \n"
                "1. LOCATION: Identify the specific body part (e.g., heel, toe, sacrum).\n"
                "2. TISSUE TYPE: Identify visible tissue colors (Pink=Granulation, Yellow=Slough, Black=Eschar). "
                "Do NOT describe pink tissue as 'dead' or 'necrotic'.\n"
                "3. EDGES: Are edges macerated or defined?\n"
                "4. SIGNS OF INFECTION: Visible purulence or erythema only. Do not guess.\n"
                "Output a concise clinical description."
            )
        self.unload()

        try:
            self.load_model()
            # 1. Image Preprocessing
            image = Image.open(image_file).convert('RGB')
            image_tensor = process_images([image], self.image_processor, self.model.config)
            
            if type(image_tensor) is list:
                image_tensor = [img.to(self.model.device, dtype=torch.float16) for img in image_tensor]
            else:
                image_tensor = image_tensor.to(self.model.device, dtype=torch.float16)

            # 2. Prompt Formatting (Vicuna Style)
            if self.model.config.mm_use_im_start_end:
                qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + prompt
            else:
                qs = DEFAULT_IMAGE_TOKEN + '\n' + prompt

            conv = conv_templates["vicuna_v1"].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], None)
            prompt_str = conv.get_prompt()

            # 3. Tokenization
            input_ids = tokenizer_image_token(prompt_str, self.tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(self.model.device)

            # 4. Generation
            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=0.2,
                    max_new_tokens=512,
                )

            # 5. Decoding
            output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
            
            # 7. IMMEDIATE CLEANUP
            # Delete tensors first
            del image_tensor
            del input_ids
            # Then unload the model
            self.unload()
            
            return output_text

        except Exception as e:
            logger.error(f"Error during vision analysis: {e}")
            self.unload()
            return "Error during vision analysis."

    def unload(self):
        """Frees GPU memory manually."""
        if self.loaded:
            # Delete instance variables
            del self.model
            del self.tokenizer
            del self.image_processor
            
            # Reset flags
            self.model = None
            self.tokenizer = None
            self.image_processor = None
            self.loaded = False
            
            # Force Garbage Collection
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("♻️ Vision Model unloaded from VRAM.")
