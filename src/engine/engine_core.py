import logging
import re
import os
import json
import time
import torch
import gc
import requests
import pandas as pd
from src.data_manager import DataManager
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
from llama_index.core import Settings, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.llms.ollama import Ollama
from llama_index.experimental.query_engine import PandasQueryEngine


logger = logging.getLogger(__name__)

class LLMEngine:
    """
    Handles clinical reasoning and text generation via Ollama (Gemma 2).
    """
    def __init__(self, model_name="gemma2:27b", timeout=300.0):
        self.model_name = model_name
        self.timeout = timeout
        self.llm = None

    def initialize(self):
        """Connects to the local Ollama instance."""
        try:
            logger.info(f"🧠 Connecting to Inference Engine: {self.model_name}...")
            self.llm = Ollama(model=self.model_name, request_timeout=self.timeout)
            Settings.llm = self.llm # Set global LlamaIndex setting
            logger.info("✅ Inference Engine connected.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
            raise e

    def generate(self, prompt):
        """Direct text completion."""
        if not self.llm:
            self.initialize()
        return self.llm.complete(prompt).text

    def chat(self, messages):
        """Chat completion (list of messages)."""
        if not self.llm:
            self.initialize()
        return self.llm.chat(messages)


class ClinicalRAGEngine:
    """
    Handles Clinical Assessments using RAG (Retrieval Augmented Generation).
    Uses the Vector Index to find patient history.
    """
    def __init__(self, index):
        self.index = index
        self.chat_engine = None

        # Load constrastive style guide
        contrastive_block = self._load_style_guide()

        # --- 1. THE "SELF-CHECK" PROMPT ---
        self.system_prompt_str = (
            "You are Dr. Gemma, the Expert Medical Director of Wound Care Services.\n"
            "You are communicating directly with a bedside physician. Your tone must be:\n"
            "1. **Authoritative:** Do not be passive. Make clear recommendations.\n"
            "2. **Concise:** Physicians are busy. Get to the point.\n"
            "3. **Clinical:** Use professional terminology (e.g., 'Contraindicated', 'Indicated', 'Etiology').\n"
            f"{contrastive_block}\n"
            "--------------------------------------------------------\n"
            "### CORE INSTRUCTION: DETERMINE THE RESPONSE FORMAT ###\n"
            "Classify the user's query and strictly follow the formatting rules:\n\n"

            "**SCENARIO A: SPECIFIC CLINICAL QUESTION (e.g., 'Should we use Santyl?', 'Is it infected?')**\n"
            "FORMAT:\n"
            "1. **Clinical Verdict:** Start with a decisive judgment (e.g., 'Recommendation: Contraindicated', 'Yes, Indicated').\n"
            "2. **Evidence:** Cite the specific data (Date, Size, Tissue %) supporting your verdict.\n"
            "3. **Reasoning:** Briefly explain the pathophysiology or guideline basis.\n"
            "🛑 **CONSTRAINT:** Keep it under 100 words. NO 'Clinical Assessment' headers.\n\n"

            "**SCENARIO B: GENERAL PATIENT REVIEW (e.g., 'Status update?', 'Full assessment')**\n"
            "FORMAT:\n"
            "**🏥 Clinical Assessment:** [Synthesize wound state + key comorbidities]\n"
            "**⚠️ Risk Stratification:** [High/Medium/Low] - [Primary Driver]\n"
            "**📋 Recommended Plan:**\n"
            "  1. [Actionable Step]\n"
            "  2. [Diagnostic Step]\n"
            "**👨‍⚕️ Confidence:** [High/Medium] - [Reasoning]\n"
            "--------------------------------------------------------"
        )
        
        self.clinical_template = PromptTemplate(
            self.system_prompt_str + 
            "\n\nCLINICAL CONTEXT FROM DATABASE:\n{context_str}\n\n"
            "--------------------------------------------------------\n"
            "### IMMEDIATE INSTRUCTION ###\n"
            "User Query: {query_str}\n\n"
            "STOP. Before answering, determine the scenario:\n"
            "1. IF asking a specific question -> Answer directly (Yes/No) + Evidence. NO lengthy assessment.\n"
            "2. IF asking for a summary -> Provide full Clinical Assessment.\n"
            "RESPONSE:"
        )

    def _load_style_guide(self):
        """
        Loads contrastive examples from JSON config.
        Returns formatted string for prompt injection.
        """
        style_path = os.path.join(os.path.dirname(__file__), '../../config/style_guide.json')
        
        try:
            if not os.path.exists(style_path):
                logger.warning(f"Style guide not found at {style_path}")
                return ""
            
            with open(style_path, 'r') as f:
                guides = json.load(f)
            
            # Build multi-example contrastive block
            examples = []
            for category, content in guides.items():
                bad = content.get('bad_example', '')
                good = content.get('good_example', '')
                if bad and good:
                    examples.append(
                        f"**{category.replace('_', ' ').title()}:**\n"
                        f"❌ AVOID: {bad}\n"
                        f"✅ ADOPT: {good}"
                    )
            
            if examples:
                return (
                    "\n--------------------------------------------------------\n"
                    "CONTRASTIVE STYLE GUIDE (Learn from these examples):\n" +
                    "\n\n".join(examples) +
                    "\n--------------------------------------------------------"
                )
            else:
                return ""
                
        except Exception as e:
            logger.error(f"Error loading style guide: {e}")
            return ""

    def _extract_patient_id(self, text):
        match = re.search(r'\b(patient|p|id|encounter)\s*[:#-]?\s*(\d{1,10})', text.lower())
        if match:
            return match.group(2)
        return None
        
    def initialize(self):
        """Builds the Chat Engine with Context."""
        if not self.index:
            logger.warning("⚠️ No Index provided to RAG Engine.")
            return

        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        self.chat_engine = self.index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=self.system_prompt_str,
            llm=Settings.llm,
            verbose=True
        )
        logger.info("✅ Clinical RAG Engine initialized.")

    def chat(self, user_query, history=[]):
        """
        Queries the patient history or Visual Context from history.
        """
        if not self.chat_engine:
            self.initialize()
            
        if not self.chat_engine:
            return "System Error: Clinical Data Index is not available."

        try:
            # --- DYNAMIC METADATA FILTERING ---
            patient_id = self._extract_patient_id(user_query)

            # Look at last 3 messages for the [SYSTEM UPDATE] tag
            visual_context = None
            
            # --- FIX 2: Correct variable name (history vs conversation_history) ---
            if history:
                last_3 = history[-3:] if len(history) >= 3 else history
                for msg in last_3:
                    content = msg.get('content', '')
                    if '[SYSTEM UPDATE]' in str(content):
                        visual_context = content
                        break
            
            if patient_id:
                logger.info(f"🎯 Detected Patient ID {patient_id}. Applying Strict Filter.")
                # CRITICAL FIX: Inject Visual Context into the Query if it exists
                # This ensures the LLM 'sees' the image analysis alongside the database records
                final_query = user_query
                if visual_context:
                    final_query = (
                        f"{user_query}\n\n"
                        f"VISUAL OBSERVATIONS:\n{visual_context}\n\n"
                        "INSTRUCTION: Combine the visual observations above with the patient records below."
                    )
                filters = MetadataFilters(
                    filters=[MetadataFilter(key="patient_id", value=str(patient_id), operator="==")]
                )
                # This returns a standard Response Object
                return self.index.as_query_engine(
                    filters=filters,
                    llm=Settings.llm,
                    similarity_top_k=5,
                    text_qa_template=self.clinical_template
                ).query(final_query)
            
            # --- CASE B: Visual Follow-Up (Restoring Old Version Logic) ---
            elif visual_context and not patient_id:
                logger.info("👁️ Visual Answering: Answering from image only")
                
                # Use LLM directly with visual context (skip database)
                prompt = (
                    f"{self.system_prompt_str}\n\n"
                    f"VISUAL CONTEXT:\n{visual_context}\n\n"
                    f"USER QUERY: {user_query}\n\n"
                    "INSTRUCTION: Answer based ONLY on the visual context above. "
                    "Do NOT invent patient history."
                )
                
                response_text = Settings.llm.complete(prompt).text
                
                # Create mock response object for explainability UI
                class MockResponse:
                    def __init__(self, text, source):
                        self.response = text
                        self.source_nodes = [
                            NodeWithScore(
                                node=TextNode(text=source),
                                score=1.0
                            )
                        ]
                    def __str__(self):
                        return self.response
                
                return MockResponse(
                    response_text, 
                    f"Source: Visual Analysis (Current Session)\n{visual_context}"
                )

            else:
                logger.info("🛡️ No ID detected. Bypassing Database to prevent Context Leak.")
                
                # 1. Use the Class's System Prompt (Maintains Persona)
                prompt = (
                    f"{self.system_prompt_str}\n\n"
                    f"USER QUERY: {user_query}\n\n"
                    "INSTRUCTION: The user is asking a general medical question. "
                    "Do NOT retrieve or invent specific patient records. "
                    "Answer using only your general medical knowledge and guidelines."
                )
                
                response_text = Settings.llm.complete(prompt).text
                
                # 2. Return an Object, NOT a String (Prevents UI Crash)
                class GeneralResponse:
                    def __init__(self, text):
                        self.response = text
                        self.source_nodes = [] # Empty list = "No specific source"
                    def __str__(self):
                        return self.response

                return GeneralResponse(response_text)

        except Exception as e:
            logger.error(f"RAG Error: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"


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


class AnalyticsEngine:
    """
    Deterministic data queries using LlamaIndex PandasQueryEngine.
    Converts natural language -> Pandas code -> Exact answers.
    """
    def __init__(self, data_manager):
        self.dm = data_manager
        self.engine = None
        
    def initialize(self):
        """Setup the query engine with dataframe schema."""
        # Get the clean dataframe from your DataManager
        df = self.dm.get_preview()

        # 1. Force Local LLM (Gemma 2)
        # This prevents the "No OpenAI API Key" error
        local_llm = Ollama(model="gemma2:27b", request_timeout=300.0)
        Settings.llm = local_llm  # Set global setting just in case
        
        # Instruction string tells LLM how to use the dataframe
        instruction = (
            "You are a python data analyst. "
            "IMPORTANT RULES:\n"
            "1. When asked 'how many patients', ALWAYS count unique Patient_IDs using: df['Patient_ID'].nunique()\n"
            "2. NEVER use len(df) for patient counts, as the data has multiple rows per patient.\n"
            "3. For filtering (e.g. 'diabetic'), use str.contains(..., case=False, na=False).\n"
            "4. CRITICAL/URGENT CHECKS: The 'Status' column is the Source of Truth.\n"
            "   - If asked 'How many are critical?', use: df[df['Status'] == 'Critical']['Patient_ID'].nunique()\n"
            "   - If asked 'How many are stable?', use: df[df['Status'] == 'Stable']['Patient_ID'].nunique()\n"
            "5. Return ONLY the executable Python code using 'df'. Do not explain."
            "Available columns: Patient_ID, Age, Sex, Wound_Size_Length_cm, "
            "Wound_Size_Width_cm, Comorbidities, Encounter_Date, Narrative.\n"
        )
        
        self.engine = PandasQueryEngine(
            df=df,
            instruction_str=instruction,
            llm=local_llm,
            verbose=True,  # Shows the generated code in logs
            synthesize_response=True # Turns the result back into a sentence
        )
        logger.info("✅ AnalyticsEngine (Pandas) initialized.")
        
    def execute_query(self, query):
        """
        Executes natural language query deterministically.
        """
        if not self.engine:
            self.initialize()
            
        try:
            # This generates Python code, runs it on the DF, and returns result
            response = self.engine.query(query)
            return {
                "type": "text",
                "content": str(response),
                "text": str(response)
            }
        except Exception as e:
            logger.error(f"Analytics error: {e}")
            return {
                "type": "error",
                "text": f"Could not process query: {str(e)}"
            }