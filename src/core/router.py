from enum import Enum
import re
import logging

logger = logging.getLogger(__name__)

class QueryIntent(Enum):
    SIMPLE_FACT = "simple_fact"          # "What is Santyl?"
    PATIENT_ASSESSMENT = "assessment"    # "How is patient 20152?"
    DATA_ANALYTICS = "data_analytics"    # "Show me the table", "How many patients?"
    UNKNOWN = "unknown"

class IntentRouter:
    def __init__(self, llm_engine):
        self.llm = llm_engine

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
        
        # Patient-specific triggers (Regex for IDs like 10770 or P-100)
        if re.search(r'\b(patient|p|id)\s*[:#-]?\s*\d{4,}', q):
            return QueryIntent.PATIENT_ASSESSMENT
        
        # Simple fact triggers (drugs, definitions)
        fact_keywords = ["what is", "define", "explain", "santyl", "collagenase", "treatment for"]
        if any(kw in q for kw in fact_keywords):
            return QueryIntent.SIMPLE_FACT
        
        # === 2. LLM FALLBACK (Only if rules fail) ===
        prompt = (
            f"Classify this query:\n"
            f"- data_analytics: Requests counts, lists, statistics\n"
            f"- simple_fact: Definitions, drug info\n"
            f"- assessment: Analysis of specific patient\n\n"
            f"Query: '{user_query}'\n\n"
            f"Respond with ONLY the category."
        )
        
        try:
            response = self.llm.generate(prompt).strip().lower()
            if "data" in response: return QueryIntent.DATA_ANALYTICS
            if "fact" in response: return QueryIntent.SIMPLE_FACT
            if "assess" in response: return QueryIntent.PATIENT_ASSESSMENT
        except Exception as e:
            logger.error(f"Classification failed: {e}")
        
        # Safe default
        return QueryIntent.SIMPLE_FACT
