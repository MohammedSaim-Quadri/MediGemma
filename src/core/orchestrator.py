import logging
import re
from src.core.router import classify_query, QueryIntent

logger = logging.getLogger(__name__)

class ClinicalOrchestrator:
    """
    The 'Brain' of the application. 
    Coordinates the Router, Data Manager, and Engines (Analytics/RAG/LLM).
    """
    def __init__(self, analytics_engine, rag_engine, llm_engine, data_manager):
        self.analytics = analytics_engine
        self.rag = rag_engine
        self.llm = llm_engine
        self.dm = data_manager

    def process_query(self, user_query, chat_history):
        """
        Main Logic Loop:
        1. Detects Intent (Data vs Clinical)
        2. Injects Context (Patient IDs)
        3. Executes the appropriate Engine
        
        Returns: (response_text, source_label, response_object, intent)
        """
        # 1. Routing
        intent = classify_query(user_query, chat_history)
        logger.info(f"🔀 Orchestrator Route: {intent}")

        # 2. Execution
        if intent == QueryIntent.DATA:
            # --- DATA PATH ---
            if self.dm.df is not None:
                result = self.analytics.execute_query(user_query)
                return result['text'], "Analytics Engine", None, intent
            else:
                return "⚠️ Analytics Unavailable: No CSV loaded.", "System", None, intent

        else: 
            # --- CLINICAL PATH (RAG/LLM) ---
            
            # A. Ground Truth Injection
            # Check for "Patient 10770" mentions using Regex
            patient_id_match = re.search(r'\b(10\d{3}|20\d{3}|3\d{4})\b', user_query)
            final_query = user_query
            
            if patient_id_match:
                pid = patient_id_match.group(0)
                # Fetch the incontestable truth from Pandas
                state = self.dm.get_patient_current_state(pid)
                
                if state:
                    logger.info(f"🔎 Injecting Ground Truth for {pid}")
                    ground_truth = (
                        f"\n\n[SYSTEM INJECTED GROUND TRUTH - PRIORITY OVER RETRIEVAL]\n"
                        f"Latest Encounter Date: {state['last_visit']}\n"
                        f"Current Wound Size: {state['wound_dims']}\n"
                        f"Current Status: {state['severity']}\n"
                        f"Latest Note Snippet: {state['narrative']}\n"
                        f"INSTRUCTION: You MUST use this date ({state['last_visit']}) as the current status.\n"
                    )
                    final_query = user_query + ground_truth

            # B. Engine Selection
            if self.rag.index:
                # Use RAG if index exists
                # This matches your app_main.py logic: rag_engine.chat(enhanced_prompt, messages)
                response_obj = self.rag.chat(final_query, chat_history)
                
                # Handle response types safely (String vs Object)
                if hasattr(response_obj, 'response'):
                    return str(response_obj.response), "Clinical Engine", response_obj, intent
                else:
                    return str(response_obj), "Clinical Engine", response_obj, intent
            else:
                # Fallback to pure LLM if no data loaded (General Knowledge)
                # Construct simple history for LLM
                messages = [{"role": m["role"], "content": m["content"]} for m in chat_history]
                messages.append({"role": "user", "content": final_query})
                
                resp = self.llm.chat(messages)
                return resp.message.content, "LLM (General Knowledge)", None, intent