import re
from llama_index.core import Settings
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

        # System Prompt (The "Medical Director" Persona)
        system_prompt = (
            "You are the Medi-Gemma Clinical Intelligence Engine. "
            "Your job is to COMPARE patient data against the T.I.M.E. Clinical Framework.\n"
            "--------------------------------------------------------\n"
            "RULES OF ENGAGEMENT:\n"
            "1. NO GENERIC REFUSALS: Do not start with 'I cannot provide medical advice'. "
            "Instead, state 'Based on the available records...'.\n"
            "2. DATA-DRIVEN: You are analyzing DATA, not treating a human. "
            "If the data shows Necrosis, report it. If the data is empty, report 'Insufficient Data'.\n"
            "3. FRAMEWORK LOGIC (T.I.M.E.):\n"
            "   - Tissue: Necrosis/Slough > 0% -> Matches protocol for Enzymatic Debridement (Santyl).\n"
            "   - Tissue: Granulation 100% -> Matches protocol for Moisture Balance (No Santyl).\n"
            "   - Infection: Signs present -> Matches protocol for Antimicrobials.\n"
            "--------------------------------------------------------\n"
            "INSTRUCTIONS:\n"
            "1. RETRIEVE: Look at the latest encounter dates.\n"
            "2. ANALYZE: Check the 'Necrosis_Percent' and 'Slough_Percent' values.\n"
            "3. CONCLUDE:\n"
            "   - If Necrosis > 0%: 'Data shows [X]% Necrosis. Protocol indicates Santyl.'\n"
            "   - If Necrosis == 0%: 'Data shows 0% Necrosis. Santyl is NOT indicated.'\n"
            "   - If Data Missing: 'Clinical records do not contain tissue composition data. Cannot determine Santyl suitability.'\n"
            "4. REQUIRED CLOSING: 'Clinical recommendation generated for physician review and approval.'"
        )

        self.chat_engine = self.index.as_chat_engine(
            chat_mode="context",
            memory=memory,
            system_prompt=system_prompt,
            llm=Settings.llm,
            verbose=True
        )
        logger.info("✅ Clinical RAG Engine initialized.")

    def chat(self, user_query):
        """
        Queries the patient history.
        """
        if not self.chat_engine:
            self.initialize()
            
        if not self.chat_engine:
            return "System Error: Clinical Data Index is not available."

        try:
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
                    similarity_top_k=15,
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
