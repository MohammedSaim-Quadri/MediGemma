import re
from llama_index.core import Settings, PromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters
import logging

logger = logging.getLogger(__name__)

class ClinicalRAGEngine:
    """
    Handles Clinical Assessments using RAG (Retrieval Augmented Generation).
    Uses the Vector Index to find patient history.
    """
    def __init__(self, index):
        self.index = index
        self.chat_engine = None

        # --- 1. THE "SELF-CHECK" PROMPT (Requirement #3) ---
        # This prompt forces the LLM to critique itself and cite sources.
        self.system_prompt_str = (
            "You are the Medi-Gemma Clinical Intelligence Engine.\n"
            "--------------------------------------------------------\n"
            "RESPONSE PROTOCOL:\n"
            "1. PRIORITY: If a '[SYSTEM UPDATE] Visual Analysis' is in the history, use THAT for visual questions.\n"
            "2. CITE SOURCES: Reference 'Visual Analysis' or 'Clinical Record [Date]'.\n"
            "3. UNCERTAINTY: Do not guess. If no record matches the patient, say so.\n"
            "4. PROTOCOL: Use T.I.M.E. framework for wounds.\n"
            "5. CONFIDENCE: End your response with a confidence assessment: [High/Medium/Low Confidence].\n"
            "--------------------------------------------------------"
        )
        
        # Template for specific Query Engine calls (when filtering by Patient ID)
        self.clinical_template = PromptTemplate(
            self.system_prompt_str + "\n\nCONTEXT:\n{context_str}\n\nQUERY: {query_str}\nANSWER:"
        )

    def _extract_patient_id(self, text):
        """
        Extracts patient ID from query text.
        Returns the ID as a string if found, else None.
        """
        match = re.search(r'\b(patient|p|id|encounter)\s*[:#-]?\s*(\d{4,6})', text.lower())
        if match:
            return match.group(2)
        return None
        
    def initialize(self):
        """Builds the Chat Engine with Context."""
        if not self.index:
            logger.warning("⚠️ No Index provided to RAG Engine.")
            return

        # Memory buffer for context (remembers last 3 turns)
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
        Queries the patient history.
        conversation_history: Optional list of message dicts from Streamlit session state
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
            if conversation_history:
                last_3 = conversation_history[-3:] if len(conversation_history) >= 3 else conversation_history
                for msg in last_3:
                    content = msg.get('content', '')
                    if '[SYSTEM UPDATE]' in str(content):
                        visual_context = content
                        break
            
            if patient_id:
                logger.info(f"🎯 Detected Patient ID {patient_id}. Applying Strict Filter.")
                
                # Create a specialized Query Engine just for this patient
                filters = MetadataFilters(
                    filters=[MetadataFilter(key="patient_id", value=str(patient_id), operator="==")]
                )
                
                # Use a query engine for precision
                specialized_engine = self.index.as_query_engine(
                    filters=filters,
                    llm=Settings.llm,
                    similarity_top_k=5,
                    text_qa_template=self.clinical_template,
                )
                
                return specialized_engine.query(user_query)
            
            # --- CASE B: Visual Follow-Up (THE FIX) ---
            elif visual_context and not patient_id:
                logger.info("👁️ Visual short-circuit: Answering from image only")
                
                # Use LLM directly with visual context (skip database)
                prompt = (
                    f"{self.system_prompt_str}\n\n"
                    f"VISUAL CONTEXT:\n{visual_context}\n\n"
                    f"USER QUERY: {user_query}\n\n"
                    "Answer based ONLY on the visual context above. Do NOT search patient records."
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
                # Standard Chat Mode for general questions
                # 2. General Query -> Use Chat Engine (Conversation)
                # Note: The Chat Engine automatically handles history context via LlamaIndex memory
                return self.chat_engine.chat(user_query)

        except Exception as e:
            logger.error(f"RAG Error: {e}")
            return "I apologize, but I encountered an error searching the clinical notes."