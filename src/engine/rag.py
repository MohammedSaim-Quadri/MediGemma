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
            "You are Dr. Gemma, Medical Director of Wound Care Services.\n"
            "You provide authoritative clinical assessments by synthesizing visual findings with patient history.\n"
            f"{contrastive_block}\n"
            "--------------------------------------------------------\n"
            "CORE PRINCIPLES:\n"
            "1. **Synthesis Over Repetition:** Don't just quote records. Analyze patterns.\n"
            "2. **Structured Output:** Use sections: Assessment → Risk → Plan.\n"
            "3. **Confidence Calibration:** Always end with [High/Medium/Low Confidence].\n"
            "4. **Evidence-Based:** Reference specific dates, measurements, or visual findings.\n"
            "5. **Actionable:** Every recommendation must be concrete and implementable.\n"
            "--------------------------------------------------------\n"
            "RESPONSE FRAMEWORK:\n"
            "**Clinical Assessment:** [Synthesize wound + comorbidities + visual findings]\n"
            "**Risk Stratification:** [High/Medium/Low] + specific risk factors\n"
            "**Recommended Plan:**\n"
            "  1. [Immediate action]\n"
            "  2. [Diagnostic workup]\n"
            "  3. [Ongoing management]\n"
            "**Confidence:** [High/Medium/Low Confidence] + reasoning\n"
            "--------------------------------------------------------"
        )
        
        self.clinical_template = PromptTemplate(
            self.system_prompt_str + 
            "\n\nCLINICAL CONTEXT:\n{context_str}\n\nDIRECTOR QUERY: {query_str}\n\nASSESSMENT:"
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
                        f"CRITICAL VISUAL FINDINGS FROM IMAGE:\n{visual_context}\n\n"
                        "INSTRUCTION: Synthesize the patient's historical records (retrieved below) "
                        "with these visual findings to form a diagnosis."
                    )
                filters = MetadataFilters(
                    filters=[MetadataFilter(key="patient_id", value=str(patient_id), operator="==")]
                )
                specialized_engine = self.index.as_query_engine(
                    filters=filters,
                    llm=Settings.llm,
                    similarity_top_k=5,
                    text_qa_template=self.clinical_template,
                )
                return specialized_engine.query(final_query)
            
            # --- CASE B: Visual Follow-Up (Restoring Old Version Logic) ---
            elif visual_context and not patient_id:
                logger.info("👁️ Visual Answering: Answering from image only")
                
                # Use LLM directly with visual context (skip database)
                prompt = (
                    f"{self.system_prompt_str}\n\n"
                    f"VISUAL CONTEXT FROM HISTORY:\n{visual_context}\n\n"
                    f"USER QUERY: {user_query}\n\n"
                    "Answer based ONLY on the visual context above."
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
                # Standard Chat Mode
                return self.chat_engine.chat(user_query)

        except Exception as e:
            logger.error(f"RAG Error: {e}")
            return f"I apologize, but I encountered an error: {str(e)}"