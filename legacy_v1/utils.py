# utils.py
"""
🏥 MEDI-GEMMA CLINICAL DECISION SUPPORT ARCHITECTURE

STAGE 1: AI PERCEPTION (Visual Diagnosis)
- Model: Fine-Tuned LLaVA-v1.5-7B ("LLaVA-Medical-Director")
- Training Data: ~4,000 images (AZH Wound + DFU QUST datasets)
- Role: Identifies wound pathology from visual features (e.g., "Diabetic Foot Ulcer").

STAGE 2: CLINICAL ACTION (Protocol Mapping)
- Logic: Deterministic mapping of AI diagnosis to evidence-based protocols.
- Source: Standard Clinical Guidelines (ADA, WOCN, IWGDF).
- Role: Ensures management recommendations are safe, consistent, and hallucination-free.

This hybrid approach mimics FDA-cleared CDSS architectures where AI handles 
perception, but validated rule engines handle critical treatment recommendations.
"""
import os
import torch
import gc
import streamlit as st
from PIL import Image
import re
import traceback
import logging
from dotenv import load_dotenv

# --- LLAVA IMPORTS ---
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.conversation import conv_templates, SeparatorStyle
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images

# ---LLAVA INDEX IMPORTS ----
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

logger = logging.getLogger(__name__)

# --- CONFIG ---
load_dotenv()
VISION_MODEL_PATH = os.getenv("VISION_MODEL_PATH", "./LLaVA-Medical-Director")
REASONING_MODEL = os.getenv("REASONING_MODEL", "gemma2:27b")
BASE_MODEL_PATH = "liuhaotian/llava-v1.5-7b"

# --- CLINICAL PROTOCOLS (The "Safety Layer") ---
CLINICAL_PROTOCOLS = {
    "Diabetic Foot Ulcer": (
        "**⚠️ Clinical Severity:** High\n\n"
        "**📋 Evidence-Based Management Protocol:**\n\n"
        "• **Offloading:** Total Contact Casting (TCC) or specialized footwear to redistribute pressure (Gold Standard).\n\n"
        "• **Glycemic Control:** Target HbA1c < 7% to facilitate healing.\n\n"  # <--- Added \n
        "• **Debridement:** Removal of non-viable tissue (sharp or enzymatic) to promote granulation.\n\n"
        "• **Vascular Assessment:** Check ABI/TBI to rule out ischemia.\n\n"
        "• **Infection Control:** Monitor for osteomyelitis; obtain deep tissue culture if infection suspected."
    ),
    "Venous Leg Ulcer": (
        "**⚠️ Clinical Severity:** Moderate\n\n"
        "**📋 Evidence-Based Management Protocol:**\n\n"
        "• **Compression Therapy:** Multilayer bandaging (aim for 40mmHg at ankle) is the mainstay of treatment.\n\n"
        "• **Elevation:** Elevate legs above heart level for 30 min, 3-4 times daily.\n\n"
        "• **Exudate Management:** Use superabsorbent dressings (foams, alginates) to manage high drainage.\n\n"
        "• **Skin Care:** Apply barrier cream to peri-wound skin to prevent maceration."
    ),
    "Pressure Injury": (
        "**⚠️ Clinical Severity:** High (Stage dependent)\n\n"
        "**📋 Evidence-Based Management Protocol:**\n\n"
        "• **Pressure Redistribution:** Reposition patient every 2 hours; use air-fluidized or alternating pressure surfaces.\n\n"
        "• **Nutrition:** High-protein, high-calorie diet with vitamin supplementation (Zinc, Vit C).\n\n"
        "• **Moisture Management:** Manage incontinence; use barrier creams.\n\n"
        "• **Debridement:** Required for Stage 3/4 if slough/eschar is present."
    ),
    "Surgical Wound": (
        "**⚠️ Clinical Severity:** Moderate\n\n"
        "**📋 Evidence-Based Management Protocol:**\n\n"
        "• **Infection Surveillance:** Monitor for dehiscence, erythema, purulent drainage, or fever (SSI signs).\n\n"
        "• **Dressing:** Keep clean and dry; change dressings per aseptic technique.\n\n"
        "• **Nutrition:** Optimize glucose control and protein intake for collagen synthesis."
    ),
    "Normal Skin": (
        "**⚠️ Clinical Severity:** Low\n\n"
        "**📋 Preventive Care Protocol:**\n\n"
        "• **Routine Inspection:** Daily foot/skin checks for high-risk patients (diabetes, neuropathy).\n\n"
        "• **Moisturization:** Maintain skin hydration to prevent cracking/fissures.\n\n"
        "• **Education:** Instruct patient on proper footwear and offloading techniques."
    )
}

# --- AI MODEL SETUP ---
@st.cache_resource
def init_models():
    try:
        # 1. RAG Model (Text)
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5")
        Settings.llm = Ollama(model=REASONING_MODEL, request_timeout=300.0)
        
        # 2. VISION Model (Local LLaVA)
        logger.info("🏥 Loading Medical Director Vision Model: {VISION_MODEL_PATH}")
        disable_torch_init()
        model_name = get_model_name_from_path(VISION_MODEL_PATH)
        tokenizer, model, image_processor, context_len = load_pretrained_model(
            model_path = VISION_MODEL_PATH, # LoRA checkpoint
            model_base=None,
            model_name=model_name, 
            load_4bit=True, # QLoRA requirement
            device_map="cuda"
        )
        logger.info("Vision Model loaded successfully")
        return tokenizer, model, image_processor
    except Exception as e:
        logger.error(f"❌ MODEL LOAD ERROR: {e}", exc_info=True)
        return None, None, None

# --- HELPER FUNCTIONS ---
def extract_patient_ids(query_text):
    p_matches = re.findall(r'\b(P\d+)\b', query_text, re.IGNORECASE)
    num_matches = re.findall(r'\b(\d{3,6})\b', query_text)
    all_matches = list(set(p_matches + num_matches))
    logger.info(f"Extracted Patients IDs")
    return [m.upper() for m in all_matches]

def analyze_image(image_file):
    """
    Hybrid Logic: AI Vision + Clinical Protocol Mapping.
    Attempt: Matches CLI image processing pipeline EXACTLY.
    """
    if not image_file: return {"error": "No image uploaded"}
    
    # Garbage collection before loading model
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Garbage collection completed")

    tokenizer = model = image_processor = None
    
    try:
        # 1. Load model
        tokenizer, model, image_processor = init_models()
        logger.info("Model Loade successfully")
        if model is None:
            logger.error("Model failed to load.")
            return {"error": "Model failed to load."}
        
        # 2. Image Prep
        image = Image.open(image_file).convert('RGB')
        
        image_tensor = process_images([image], image_processor, model.config)

        if type(image_tensor) is list:
            image_tensor = [img.to(model.device, dtype=torch.float16) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(model.device, dtype=torch.float16)
        
        logger.info("Image processed successfully")
        # 3. Prompt (Vicuna format)
        qs = "Describe this wound in detail, focusing on tissue type and signs of infection."
        
        if model.config.mm_use_im_start_end:
            qs = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + qs
        else:
            qs = DEFAULT_IMAGE_TOKEN + '\n' + qs

        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        logger.info("Prompt formatted and generated successfully")
        
        # 4. tokenize
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
        logger.info("Input IDs tokenized successfully")
        
        # 5. Generate
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=image_tensor,
                do_sample=True,
                temperature=0.2, # low temp for factual medical data
                max_new_tokens=512,
                use_cache=True
            )

        logger.info("Model generated successfully")
        
        # 6. decode
        raw_diagnosis = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        logger.info(f"🧐 VISION OUTPUT: '{raw_diagnosis}'")

        # -- CLEANUP---
        # we must offload LLaVA to make room for Gemma 27B
        del model
        del image_tensor
        del input_ids
        torch.cuda.empty_cache()
        gc.collect()
        logger.info("Vision cleanup completed successfully")

        # --- PROTOCOL MAPPING (Keep your logic) ---
        management_plan = "Requires clinical assessment."
        clean_diagnosis = raw_diagnosis 
        diagnosis_lower = raw_diagnosis.lower()
        
        if any(x in diagnosis_lower for x in ["diabetic", "dfu", "foot", "toe", "ulcer", "neuropathic"]):
            clean_diagnosis = "Diabetic Foot Ulcer"
            management_plan = CLINICAL_PROTOCOLS["Diabetic Foot Ulcer"]
        elif any(x in diagnosis_lower for x in ["venous", "vlu", "stasis", "vein", "edema"]):
            clean_diagnosis = "Venous Leg Ulcer"
            management_plan = CLINICAL_PROTOCOLS["Venous Leg Ulcer"]
        elif any(x in diagnosis_lower for x in ["pressure", "decubitus", "sore", "sacral", "heel"]):
            clean_diagnosis = "Pressure Injury"
            management_plan = CLINICAL_PROTOCOLS["Pressure Injury"]
        elif any(x in diagnosis_lower for x in ["surgical", "incision", "suture", "dehiscence"]):
            clean_diagnosis = "Surgical Wound"
            management_plan = CLINICAL_PROTOCOLS["Surgical Wound"]
        elif any(x in diagnosis_lower for x in ["normal", "healthy", "intact"]):
            clean_diagnosis = "Normal Skin"
            management_plan = CLINICAL_PROTOCOLS["Normal Skin"]
        else:
            clean_diagnosis = "Unspecified Lesion"
            management_plan = "**⚠️ Uncertain Classification:** Clinical evaluation required."

        return {
            "diagnosis": raw_diagnosis,
            "protocol": management_plan,
            "full_report": f"**Diagnosis:** {clean_diagnosis}\n\n{management_plan}",
            "raw_output": raw_diagnosis
        }
        
    except torch.cuda.OutOfMemoryError:
        logger.warning("GPU OOM - Attempting to clear cache.")
        torch.cuda.empty_cache()
        return {
            "error": "GPU out of memory. Close other applications and retry.",
            "recovery": "automatic_cache_clear_attempted"
        }
    
    except RuntimeError as e:
        if "CUDA" in str(e) or "device-side assert" in str(e):
            logger.critical(f"FATAL GPU ERROR: {e}")
            return {
                "error": f"GPU driver error: {str(e)}",
                "recovery": "restart_required",
                "details": "GPU entered bad state. Kill all Python processes and restart."
            }
        logger.error(f"Runtime error: {e}", exc_info=True)
        raise  # Re-raise non-CUDA errors
    
    except Exception as e:
        logger.critical(f"❌ UNEXPECTED ERROR IN VISION MODULE:{e}", exc_info=True)
        return {
            "error": f"Internal error: {type(e).__name__}",
            "details": str(e),
            "trace": traceback.format_exc()
        }


def get_patient_current_state(patient_id, df_preview):
    """
    Retrieves the absolute latest clinical state from the structured dataframe.
    This acts as the 'Ground Truth' to prevent RAG hallucinations.
    """
    try:
        # ensure ID is string format
        pid = str(patient_id).strip()

        #Filter and sort by date (newest first)
        p_data = df_preview[df_preview['Patient_ID'].astype(str) == pid]

        if p_data.empty:
            return None
        
        # sort to get the latest row
        p_data = p_data.sort_values(by='Encounter_Date', ascending=False)
        latest = p_data.iloc[0]

        # Determine simple severity flag based on narrative keywords
        severity = "Stable"
        narrative = str(latest.get('Narrative', '')).lower()
        if "decline" in narrative or "deteriorating" in narrative or "infection" in narrative:
            severity = "Critical/Watch"

        logger.info(f"Patient {pid} current state: {severity}")
            
        return {
            "last_visit": latest.get('Encounter_Date', 'Unknown'),
            "wound_dims": f"{latest.get('Wound_Size_Length_cm', '?')} x {latest.get('Wound_Size_Width_cm', '?')} cm",
            "narrative": latest.get('Narrative', 'No notes available'),
            "severity": severity
        }
    except Exception as e:
        logger.error(f"Error fetching state: {e}")
        return None