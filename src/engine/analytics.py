from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings
import pandas as pd
import logging
from src.data_manager import DataManager

logger = logging.getLogger(__name__)

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
