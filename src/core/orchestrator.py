import logging
import re
from src.core.router import classify_query, QueryIntent
from llama_index.core.llms import ChatMessage, MessageRole

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

            # CHECK: Do we have a recent image in history?
            has_visual = any('[SYSTEM UPDATE]' in str(m.get('content','')) for m in chat_history)
            
            # A. Ground Truth Injection (Robust Lookup)
            final_query = user_query
            pid = None

            # Only attempt ID lookup if we have a Database loaded
            if self.dm.df is not None:
                # FIX: Pull every numeric token (3-6 digits) and validate against actual Data
                candidates = re.findall(r'\b(\d{3,6})\b', user_query)
                valid_ids = set(self.dm.df['Patient_ID'].astype(str).unique())
                
                for c in candidates:
                    if c in valid_ids:
                        pid = c
                        break
            
            if pid:
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
            if self.rag.index or has_visual:
                # Use RAG if index exists
                response_obj = self.rag.chat(final_query, chat_history)
                
                # Handle response types safely (String vs Object)
                if hasattr(response_obj, 'response'):
                    return str(response_obj.response), "Clinical Engine", response_obj, intent
                else:
                    return str(response_obj), "Clinical Engine", response_obj, intent
            else:
                # Fallback to pure LLM if no data loaded (General Knowledge)
                messages = []
                
                # Convert History
                for m in chat_history:
                    role_str = m.get("role", "user")
                    role_enum = MessageRole.USER if role_str == "user" else MessageRole.ASSISTANT
                    # Handle System role if present
                    if role_str == "system": role_enum = MessageRole.SYSTEM
                        
                    messages.append(ChatMessage(role=role_enum, content=str(m.get("content", ""))))

                # Append Current Query
                messages.append(ChatMessage(role=MessageRole.USER, content=final_query))
                
                # Now this will work because 'messages' contains Objects, not Dicts
                resp = self.llm.chat(messages)
                return resp.message.content, "LLM (General Knowledge)", None, intent