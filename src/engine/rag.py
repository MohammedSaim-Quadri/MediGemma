import re
import logging
import os
import json
from llama_index.core import Settings, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
from llama_index.core.schema import NodeWithScore, TextNode

logger = logging.getLogger(__name__)

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
        style_path = os.path.join(os.path.dirname(__file__), '../config/style_guide.json')
        
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