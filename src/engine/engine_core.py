import logging
import re
import os
import json
import time
import torch
import gc
import requests
import pandas as pd
from PIL import Image
from transformers import BitsAndBytesConfig, AutoTokenizer
try:
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, process_images
    from llava.model import LlavaLlamaForCausalLM
except ImportError:
    pass

from llama_index.core import Settings, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.schema import NodeWithScore, TextNode
# from llama_index.llms.ollama import Ollama
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

logger = logging.getLogger(__name__)

class DataManager:
    """
    Centralized data handler.
    Features:
    - Auto-cleaning of dirty data (Fixes 'N/A' crashes).
    - LOCAL Embeddings (No OpenAI).
    - Rich Vector Index for Clinical RAG.
    """
    def __init__(self):
        self.df = None
        self.index = None
        
        # --- 1. SETUP LOCAL EMBEDDINGS (Crucial for RAG) ---
        try:
            logger.info("⚙️ Loading Local Embedding Model (BAAI/bge-small-en-v1.5)...")
            self.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5", device="cpu")
            
            # CRITICAL: Apply to Global Settings
            Settings.embed_model = self.embed_model
            
            # Explicitly set LLM to None here to prevent OpenAI default
            Settings.llm = None
            Settings.chunk_size = 512 
            
        except Exception as e:
            logger.error(f"❌ Failed to load Embedding Model: {e}")

    def load_data(self, enc_file, pat_file=None):
        try:
            logger.info("📂 Processing Uploaded Data...")
            
            # --- Load CSVs ---
            df_enc = pd.read_csv(enc_file)

            # Clean column names (strip whitespace)
            # 2. Normalize Patient_ID (Force to String)
            # This fixes the merge failure (Int vs String mismatch)
            if 'Patient_ID' in df_enc.columns:
                df_enc['Patient_ID'] = df_enc['Patient_ID'].astype(str).str.strip()
            elif 'patient_id' in df_enc.columns:
                df_enc['Patient_ID'] = df_enc['patient_id'].astype(str).str.strip()
            
            # 3. Merge Demographics (If provided)
            if pat_file:
                df_pat = pd.read_csv(pat_file)
                
                # Normalize Patient_ID in demographics too
                if 'Patient_ID' in df_pat.columns:
                    df_pat['Patient_ID'] = df_pat['Patient_ID'].astype(str).str.strip()
                elif 'patient_id' in df_pat.columns:
                    df_pat['Patient_ID'] = df_pat['patient_id'].astype(str).str.strip()
                
                # Perform Left Merge
                self.df = pd.merge(df_enc, df_pat, on='Patient_ID', how='outer', suffixes=('', '_demo'))
                logger.info(f"✅ Merged demographics. Shape: {self.df.shape}")
            else:
                self.df = df_enc
                logger.warning("⚠️ No demographics file provided. Using encounters only.")

            # --- 🛡️ CRITICAL FIX: AGGRESSIVE TYPE CLEANING ---
            # 1. Force Numeric Columns (Fixes ArrowInvalid Crash)
            numeric_cols = [
                'Wound_Size_Length_cm', 
                'Wound_Size_Width_cm', 
                'Wound_Size_Area_cm2',
                'Patient_Age',
                'Age',
                'Necrosis_Percent', 
                'Slough_Percent', 
                'Granulation_Percent',
                'Pain_Level'
            ]
            
            for col in numeric_cols:
                if col in self.df.columns:
                    # Coerce errors (turn "N/A" -> NaN)
                    self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                    # Fill NaN with 0.0 (Safe for Dashboard)
                    self.df[col] = self.df[col].fillna(0.0)

            # Ensure Comorbidities is never null (Fixes 'History: None')
            if 'Comorbidities' in self.df.columns:
                self.df['Comorbidities'] = self.df['Comorbidities'].fillna("No medical history documented").astype(str)
            else:
                self.df['Comorbidities'] = "No medical history documented"
            
            text_cols = ['Narrative', 'Notes', 'Treatment_Plan', 'Exudate_Type', 'Tissue_Exposed']
            for col in text_cols:
                if col in self.df.columns:
                    self.df[col] = self.df[col].fillna('N/A').astype(str)

            logger.info(f"✅ Data Loaded & Cleaned. Rows: {len(self.df)}")
            
            # --- Build RAG Index (Rich Context Strategy) ---
            self._build_index()
            
            return True

        except Exception as e:
            logger.error(f"❌ Data Manager Error: {e}")
            return False

    def _build_index(self):
        logger.info("🧠 Building Clinical Vector Index (Per-Visit Strategy)...")
        
        # Ensure Settings are persistent
        if Settings.embed_model is None:
             Settings.embed_model = self.embed_model

        documents = []
        
        for _, row in self.df.iterrows():
            
            # 1. Context Header (Rich Patient Context)
            context_header = (
                f"PATIENT CONTEXT: ID {row.get('Patient_ID', 'Unknown')}\n"
                f"Demographics: Age {row.get('Age', 'N/A')}, Sex {row.get('Sex', 'N/A')}\n"
                f"Medical History: {row.get('Comorbidities', 'None')}\n"
            )
            
            # 2. Specific Visit Details
            # Note: We use .get with defaults to handle missing cols safely
            visit_details = (
                f"ENCOUNTER DATE: {row.get('Encounter_Date', 'Unknown')}\n"
                f"Wound Dims: {row.get('Wound_Size_Length_cm', '?')} x {row.get('Wound_Size_Width_cm', '?')} cm\n"
                f"TISSUE: Necrosis {row.get('Necrosis_Percent', 0)}%, Slough {row.get('Slough_Percent', 0)}%, Granulation {row.get('Granulation_Percent', 0)}%\n"
                f"EXUDATE: {row.get('Exudate_Type', 'N/A')}\n"
                f"PAIN: {row.get('Pain_Level', 'N/A')}/10\n"
                f"Narrative: {row.get('Narrative', 'No notes')}\n"
                f"Plan: {row.get('Treatment_Plan', 'None')}\n"
            )
            
            # 3. Combine
            full_text = context_header + "--- VISIT DETAILS ---\n" + visit_details
            
            # 4. Create Document with Metadata
            doc = Document(
                text=full_text, 
                metadata={
                    "patient_id": str(row.get('Patient_ID', 'Unknown')),
                    "date": str(row.get('Encounter_Date', 'Unknown'))
                }
            )
            documents.append(doc)
        
        if documents:
            self.index = VectorStoreIndex.from_documents(documents)
            logger.info(f"✅ Index Built Successfully ({len(documents)} clinical encounters).")

    def get_preview(self):
        return self.df

    # In src/data_manager.py - Add this method to the DataManager class

    def get_patient_current_state(self, patient_id):
        """
        Retrieves the absolute latest clinical state (Ground Truth).
        """
        try:
            # Ensure ID is string
            pid = str(patient_id).strip()
            
            # Filter for patient
            p_data = self.df[self.df['Patient_ID'] == pid].copy()
            
            if p_data.empty:
                return None

            p_data['Encounter_Date'] = pd.to_datetime(p_data['Encounter_Date'])
            
            # Sort by Date AND Visit Number (Latest date, highest visit number)
            # This handles cases where a patient has 2 visits on the same day
            if 'Visit_Number' in p_data.columns:
                p_data = p_data.sort_values(by=['Encounter_Date', 'Visit_Number'], ascending=[False, False])
            else:
                p_data = p_data.sort_values(by='Encounter_Date', ascending=False)
                
            latest = p_data.iloc[0]
            
            # Convert back to string for display
            date_str = latest['Encounter_Date'].strftime('%Y-%m-%d')
            
            return {
                "last_visit": date_str,
                "wound_dims": f"{latest.get('Wound_Size_Length_cm', '?')} x {latest.get('Wound_Size_Width_cm', '?')} cm",
                "narrative": str(latest.get('Narrative', 'No notes'))[:200] + "...", # Truncate for token limits
                "severity": "Critical" if "deteriorating" in str(latest.get('Narrative', '')).lower() else "Stable"
            }
        except Exception as e:
            logger.error(f"Error fetching state: {e}")
            return None


class LLMEngine:
    """
    Handles clinical reasoning and text generation via Ollama (Gemma 2).
    """
    def __init__(self, model_name="gemma2:27b", timeout=300.0):
        logger.warning("⚠️ LLMEngine (Gemma2) is DISABLED. Use Gemma3 for all tasks.")
        self.model_name = model_name
        self.timeout = timeout
        self.llm = None

    def initialize(self):
        """Connects to the local Ollama instance."""
        """Does nothing - Gemma2 is disabled."""
        logger.info("ℹ️ Gemma2 is disabled. Skipping initialization.")
        pass
        # try:
        #     logger.info(f"🧠 Connecting to Inference Engine: {self.model_name}...")
        #     self.llm = Ollama(model=self.model_name, request_timeout=self.timeout)
        #     Settings.llm = self.llm # Set global LlamaIndex setting
        #     logger.info("✅ Inference Engine connected.")
        # except Exception as e:
        #     logger.error(f"❌ Failed to connect to Ollama: {e}")
        #     raise e

    def generate(self, prompt):
        """Direct text completion."""
        """Does nothing - Gemma2 is disabled."""
        pass
        # if not self.llm:
        #     self.initialize()
        # return self.llm.complete(prompt).text

    def chat(self, messages):
        """Chat completion (list of messages)."""
        """Does nothing - Gemma2 is disabled."""
        pass
        # if not self.llm:
        #     self.initialize()
        # return self.llm.chat(messages)


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

            "**SCENARIO A: SPECIFIC QUESTION OR FACTUAL QUERY**\n"
            "FORMAT:\n"
            "1. **Direct Answer/Verdict:** If asking a fact (e.g. 'What composition?'), state the data directly. If asking a decision, state the verdict (Indicated/Contraindicated).\n"
            "2. **Evidence:** Cite the specific data (Date, Size, Tissue %) supporting your answer.\n"
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
        if not self.chat_engine and self.index: 
            self.initialize()

        try:
            # --- DYNAMIC METADATA FILTERING ---
            patient_id = self._extract_patient_id(user_query)

            # Look at last 3 messages for the [SYSTEM UPDATE] tag
            visual_context = None
            
            # --- FIX 2: Correct variable name (history vs conversation_history) ---
            if history:
                #last_3 = history[-3:] if len(history) >= 3 else history
                for msg in history:
                    content = msg.get('content', '')
                    if '[SYSTEM UPDATE]' in str(content):
                        visual_context = content
                        break
            
            if patient_id and self.index:
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
            elif visual_context:
                logger.info("👁️ Visual Answering: Answering from image only")
                
                # Use LLM directly with visual context (skip database)
                prompt = (
                    f"{self.system_prompt_str}\n\n"
                    f"VISUAL CONTEXT (raw findings from imaging analysis):\n{visual_context}\n\n"
                    f"USER QUERY: {user_query}\n\n"
                    "INSTRUCTION:\n"
                    "1. The VISUAL CONTEXT above is raw data from an imaging pipeline. It is NOT your answer — it is your source material.\n"
                    "2. You must INTERPRET and SYNTHESIZE the findings into a clinical response. Do not quote or repeat the visual context back.\n"
                    "3. Answer the USER QUERY directly. If the query is a specific clinical question, give a decisive verdict + your reasoning from the data.\n"
                    "4. If the visual context does not contain enough information to answer the query, say so explicitly.\n"
                    "5. Do NOT invent any patient history or data that is not in the visual context.\n"
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
        self.tokenizer = None
        self.model = None
        self.image_processor = None
        self.loaded = False

    def load_model(self):
        """Loads the Vision Model into VRAM."""
        if self.loaded:
            logger.info("✅ LLaVA already loaded")
            return

        from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM
        
        # FIX 1: cache_position
        original_forward = LlavaLlamaForCausalLM.forward
        def patched_forward(self, *args, **kwargs):
            kwargs.pop("cache_position", None)
            return original_forward(self, *args, **kwargs)
        LlavaLlamaForCausalLM.forward = patched_forward

        # FIX 2: input validation (The 'inputs_embeds' fix)
        # We force the model to accept inputs_embeds without raising a ValueError
        def patched_validate_kwargs(self, model_kwargs):
            # We just bypass the validation entirely for LLaVA
            return 
        
        LlavaLlamaForCausalLM._validate_model_kwargs = patched_validate_kwargs
        
        # FIX 3: Cache handling
        LlavaLlamaForCausalLM.prepare_inputs_for_generation = LlavaLlamaForCausalLM.prepare_inputs_for_generation

        try:
            from src.engine.test_models import master_evict_with_retry
            logger.info("🧹 Evicting all models before loading LLaVA...")
            success = master_evict_with_retry(required_free_gb=8.0, max_retries=10)
            
            if not success:
                raise RuntimeError("Could not free enough VRAM for LLaVA")
        except Exception as e:
            logger.error(f"Eviction failed: {e}")
            raise e
        
        try:
            logger.info(f"🏥 Loading Vision Model from: {self.model_path}...")
            disable_torch_init()

            # We create the config explicitly to satisfy newer Transformers versions
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                use_fast=False
            )

            # Load Model directly (Bypassing the buggy builder)
            self.model = LlavaLlamaForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
            )

            # Initialize the Vision Tower
            vision_tower = self.model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            
            self.image_processor = vision_tower.image_processor

            self.loaded = True
            logger.info("✅ Vision Model loaded successfully.")
            
        except Exception as e:
            logger.critical(f"❌ Failed to load Vision Model: {e}", exc_info=True)
            raise e

    def analyze(self, image_file, prompt=None):
        """
        Runs inference on a single image.
        """
        if not self.loaded:
            logger.info("LLaVA not loaded. Loading now...")
            self.load_model()

        if prompt is None:
            prompt = (
                "Describe the wound in this image. "
                "Include details about location, tissue type (necrotic, slough, granulation), "
                "and any signs of infection."
            )

        try:
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


# 5. TRIAGE LOGIC (Moved here for Binary Compatibility)
def generate_priority_report(df):
    report_data = []
    working_df = df.copy() 
    
    if 'Patient_ID' in working_df.columns:
        working_df['Patient_ID'] = working_df['Patient_ID'].astype(str)
    
    for pid, group in working_df.groupby('Patient_ID'):
        if group['Encounter_Date'].isna().all(): continue
        group = group.sort_values(by='Encounter_Date', ascending=False)
        latest_visit = group.iloc[0]
        
        flags = []
        severity = "Normal"
        
        curr_area = 0
        prev_area = 0
        if len(group) > 1:
            prev_visit = group.iloc[1]
            try:
                curr_area = float(latest_visit.get('Wound_Size_Length_cm', 0)) * float(latest_visit.get('Wound_Size_Width_cm', 0))
                prev_area = float(prev_visit.get('Wound_Size_Length_cm', 0)) * float(prev_visit.get('Wound_Size_Width_cm', 0))
            except: pass

        if len(group) > 1 and curr_area > prev_area * 1.1:
            flags.append(f"⚠️ **Wound Deteriorating**")
            severity = "Critical"
        
        try:
            pain = float(latest_visit.get('Pain_Level', 0))
            if pain >= 7:
                flags.append(f"🔴 **Severe Pain**")
                severity = "Critical"
        except: pass
            
        narrative = str(latest_visit.get('Narrative', '')).lower()
        if any(x in narrative for x in ['odor', 'pus', 'infection']):
            flags.append("☣️ **Infection Risk**")
            if severity != "Critical": severity = "Urgent"

        if len(group) > 1 and curr_area > 0 and (prev_area * 0.95 <= curr_area <= prev_area * 1.05):
            flags.append(f"🐢 **Stalled Healing**")
            if severity == "Normal": severity = "Urgent" 
        
        if flags:
            report_data.append({
                "Patient ID": pid,
                "Severity": severity,
                "Date": latest_visit.get('Encounter_Date', 'Unknown'),
                "Age": latest_visit.get('Age', 'Unknown'),
                "Sex": latest_visit.get('Sex', 'Unknown'),
                "Comorbidities": str(latest_visit.get('Comorbidities', 'None')),
                "Wound Size (cm)": f"{latest_visit.get('Wound_Size_Length_cm','?')} x {latest_visit.get('Wound_Size_Width_cm','?')}",
                "Latest Note": str(latest_visit.get('Narrative', ''))[:50],
                "Alerts": flags
            })
    
    report_data.sort(key=lambda x: (x['Severity'] != 'Critical', x['Severity'] != 'Urgent'))
    return report_data