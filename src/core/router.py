from enum import Enum
import re
import logging
from llama_index.core import Settings

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    SIMPLE_FACT = "simple_fact"          # "What is Santyl?"
    PATIENT_ASSESSMENT = "patient_assessment"    # "How is patient 20152?"
    DATA_ANALYTICS = "data_analytics"    # "Show me the table", "How many patients?"
    UNKNOWN = "unknown"

class IntentRouter:
    def __init__(self, llm_engine):
        self.llm = Settings.llm

    def classify(self, user_query):
        """
        DETERMINISTIC FIRST, LLM fallback only.
        """
        q = user_query.lower()
        
        # === 1. DETERMINISTIC RULES (FUNDING-CRITICAL) ===
        
        # Data/Analytics triggers (Aggressive matching)
        data_keywords = [
            "table", "list", "show me", "how many", "count", 
            "statistics", "average", "mean", "total", "sum",
            "patients with", "age", "over", "under", "more than",
            "diabetic", "stage", "size", "filter", "where"
        ]
        if any(kw in q for kw in data_keywords):
            return QueryIntent.DATA_ANALYTICS
        
        # A. Patient ID Regex (The "Specific Lookup")
        # Matches: "patient 10770", "id: 5050", "p-12345", "encounter 9999"
        if re.search(r'\b(patient|p|id|encounter)\s*[:#-]?\s*\d{4,}', q):
             return QueryIntent.PATIENT_ASSESSMENT

        # B. Clinical/Visual Keywords (The "Context Override")
        # Forces the RAG engine for visual descriptions, even if the LLM thinks it's "data"
        clinical_triggers = [
            "image", "visual", "tissue", "wound", "edge", "appearance", 
            "color", "drainage", "pus", "protocol", "recommend", "treatment",
            "infection", "necrotic", "granulation"
        ]
        if any(trigger in q for trigger in clinical_triggers):
            return QueryIntent.PATIENT_ASSESSMENT
        
        # Simple fact triggers (drugs, definitions)
        fact_keywords = ["what is", "define", "explain", "santyl", "collagenase", "treatment for"]
        if any(kw in q for kw in fact_keywords):
            return QueryIntent.SIMPLE_FACT
        
        # === 2. LLM FALLBACK (Only if rules fail) ===
        prompt = (
            "Classify the following medical query into exactly one category:\n"
            "1. 'data_analytics': ONLY for questions asking to count patients, calculate averages, or aggregate statistics from a database.\n"
            "2. 'patient_assessment': For questions about specific patient history, clinical notes, treatment plans, OR VISUAL/IMAGE ANALYSIS.\n"
            "3. 'simple_fact': For general medical definitions or questions not related to specific data.\n\n"
            "CRITICAL RULES:\n"
            "- If the query asks to 'describe the image', 'what do you see', or refers to 'visual evidence', choose 'patient_assessment'.\n"
            "- If the query asks about 'tissue types' or 'wound edges' (visual features), choose 'patient_assessment'.\n"
            "- Only choose 'data_analytics' if the user asks for numbers, counts, or statistics across the population.\n\n"
            f"Query: \"{query}\"\n\n"
            "Category:"
        )
        
        try:
            response = self.llm.generate(prompt).strip().lower()
            if "data" in response or "analytics" in response:
                return QueryIntent.DATA_ANALYTICS
            elif "patient" in response or "assessment" in response or "visual" in response:
                return QueryIntent.PATIENT_ASSESSMENT
            else:
                return QueryIntent.SIMPLE_FACT
        except Exception as e:
            logger.error(f"Router Error: {e}")
            return QueryIntent.SIMPLE_FACT
