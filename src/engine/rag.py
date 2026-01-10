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

        # --- THE "BRAIN": CENTRAL CLINICAL PROMPT ---
        self.clinical_prompt_str = (
            "You are the Medi-Gemma Clinical Intelligence Engine. "
            "Analyze the retrieved clinical records to answer the query.\n"
            "--------------------------------------------------------\n"
            "DATA INTERPRETATION RULES:\n"
            "1. TISSUE ANALYSIS:\n"
            "   - If 'Necrosis' > 0% OR 'Slough' > 0% -> IMPLIES 'Enzymatic Debridement (Santyl)' is indicated.\n"
            "   - If 'Granulation' > 75% -> IMPLIES 'Moisture Balance' is indicated.\n"
            "2. STATUS CHECKS:\n"
            "   - Compare dimensions (LxW) over time to determine 'Improving' or 'Declining'.\n"
            "   - Look for keywords: 'Odor', 'Pus', 'Erythema' -> Signs of Infection.\n"
            "--------------------------------------------------------\n"
            "RESPONSE FORMAT:\n"
            "1. DATA SUMMARY: Start with 'Patient [ID] Records show...'\n"
            "2. PROTOCOL MATCH: 'Based on Necrosis of [X]%, the protocol indicates...'\n"
            "3. RECOMMENDATION: Explicitly state if Santyl is suitable based on the rules above.\n"
            "4. DISCLAIMER: End with 'Generated for physician review.'\n"
            "--------------------------------------------------------\n"
            "CONTEXT:\n"
            "{context_str}\n"
            "--------------------------------------------------------\n"
            "QUERY: {query_str}\n"
            "CLINICAL ANALYSIS:"
        )
        self.clinical_template = PromptTemplate(self.clinical_prompt_str)

    def _extract_patient_id(self, text):
        """
        Extracts patient ID from query text.
        Returns the ID as a string if found, else None.
        """
        match = re.search(r'\b\d{4,6}\b', text)
        if match:
            patient_id = match.group(0)
            logger.info(f"🎯 Extracted Patient ID: {patient_id}")
            return patient_id
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
            system_prompt="You are a Clinical Intelligence Engine. Use T.I.M.E. protocols to analyze wounds.",
            llm=Settings.llm,
            verbose=True
        )
        logger.info("✅ Clinical RAG Engine initialized.")

    def chat(self, user_query, conversation_history=None):  # ← FIXED: Added parameter
        """
        Queries the patient history.
        conversation_history: Optional list of message dicts from Streamlit session state
        """
        if not self.chat_engine:
            self.initialize()
            
        if not self.chat_engine:
            return "System Error: Clinical Data Index is not available."

        try:
            # --- NEW: INJECT RECENT VISUAL CONTEXT ---
            # If we have conversation history with a recent image analysis,
            # prepend it to the query so the RAG engine sees it
            if conversation_history:  # ← FIXED: Now this variable exists
                recent_updates = [
                    msg['content'] for msg in conversation_history[-3:]
                    if '[SYSTEM UPDATE]' in msg.get('content', '')
                ]
                if recent_updates:
                    # Combine the most recent visual context with the user's question
                    enhanced_query = f"{recent_updates[-1]}\n\nUser Question: {user_query}"
                    user_query = enhanced_query
                    logger.info("✅ Enhanced query with visual context")
                
            # --- DYNAMIC METADATA FILTERING ---
            patient_id = self._extract_patient_id(user_query)
            
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
                    similarity_top_k=20,
                    text_qa_template=self.clinical_template,
                    verbose=True
                )
                
                response = specialized_engine.query(user_query)
                return str(response)

            else:
                # Standard Chat Mode for general questions
                response = self.chat_engine.chat(user_query)
                return str(response)

        except Exception as e:
            logger.error(f"RAG Error: {e}")
            return "I apologize, but I encountered an error searching the clinical notes."