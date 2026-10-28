import logging
import re
from src.core.router import classify_query, QueryIntent
from llama_index.core.llms import ChatMessage, MessageRole
import ollama
from llama_index.core.vector_stores import (
    MetadataFilters,
    ExactMatchFilter,
)

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

    
    def process_query(self, user_query, chat_history, image_path=None):
        """
        Unified Gemma 3 clinical inference:
        - Analytics stays deterministic
        - Optional image is sent directly to Gemma 3
        - Patient history is retrieved before inference
        - One Ollama inference call per clinical question
        """
        intent = classify_query(user_query, chat_history)
        logger.info("Orchestrator route: %s", intent)

        # Preserve deterministic analytics for text-only questions.
        if image_path is None and intent == QueryIntent.DATA:
            if self.dm.df is None:
                return (
                    "Analytics unavailable: No CSV loaded.",
                    "System",
                    None,
                    intent,
                )
            result = self.analytics.execute_query(user_query)
            return result["text"], "Analytics Engine", None, intent

        context_parts = []
        pid = None

        # Identify a patient only if the ID exists in the dataset.
        if self.dm.df is not None:
            candidates = re.findall(r"\b(\d{3,6})\b", user_query)
            valid_ids = set(
                self.dm.df["Patient_ID"].astype(str).unique()
            )
            pid = next(
                (candidate for candidate in candidates
                 if candidate in valid_ids),
                None,
            )

        # Inject the latest patient record.
        if pid:
            state = self.dm.get_patient_current_state(pid)

            if state:
                context_parts.append(
                    f"PATIENT {pid} - LATEST CLINICAL RECORD\n"
                    f"Date: {state['last_visit']}\n"
                    f"Wound size: {state['wound_dims']}\n"
                    f"Status: {state['severity']}\n"
                    f"Latest note: {state['narrative']}"
                )

        # Retrieve supporting RAG context.
        if self.rag.index is not None:
            try:
                retriever_kwargs = {"similarity_top_k": 3}

                # Never retrieve other patients' records
                # when a valid patient ID is specified.
                if pid:
                    retriever_kwargs["filters"] = MetadataFilters(
                        filters=[
                            ExactMatchFilter(
                                key="patient_id",
                                value=str(pid),
                            )
                        ]
                    )
                elif self.dm.df is not None:
                    # Avoid mixing unidentified patient records.
                    retriever_kwargs = None

                if retriever_kwargs is not None:
                    retriever = self.rag.index.as_retriever(
                        **retriever_kwargs
                    )
                    nodes = retriever.retrieve(user_query)

                    if nodes:
                        retrieved = "\n\n".join(
                            node.node.get_content()
                            for node in nodes
                        )
                        context_parts.append(
                            "RETRIEVED CLINICAL RECORDS:\n"
                            + retrieved
                        )
            except Exception:
                logger.exception("RAG retrieval failed")

        system_prompt = (
            "You are a clinical decision-support assistant "
            "supporting wound-care clinicians. "
            "Use the provided image, patient history, and "
            "clinical records to answer the user's question. "
            "Distinguish visual observations from documented "
            "patient facts. Do not invent clinical measurements, "
            "laboratory results, or unseen findings. "
            "If an image is provided, analyze the image directly. "
            "Provide a clear clinical assessment and relevant "
            "next steps. State important uncertainty."
        )

        # Build a concise conversation history.
        messages = [
            {"role": "system", "content": system_prompt}
        ]

        for entry in chat_history[-6:]:
            role = entry.get("role")
            content = str(entry.get("content", ""))

            if role not in ("user", "assistant"):
                continue

            # Exclude old machine-generated visual log messages.
            if "[SYSTEM UPDATE]" in content:
                continue

            messages.append({
                "role": role,
                "content": content,
            })

        question = user_query

        if context_parts:
            question += (
                "\n\nCLINICAL CONTEXT:\n"
                + "\n\n".join(context_parts)
            )

        user_message = {
            "role": "user",
            "content": question,
        }

        if image_path is not None:
            # Image bytes are passed directly to Ollama.
            with open(image_path, "rb") as image_file:
                user_message["images"] = [image_file.read()]

        messages.append(user_message)

        logger.info(
            "Calling Gemma 3 multimodal: image=%s, patient=%s",
            bool(image_path),
            pid,
        )

        response = ollama.chat(
            model="gemma3:27b",
            messages=messages,
            options={"temperature": 0.2},
        )

        answer = response["message"]["content"]

        return (
            answer,
            "Gemma 3 Multimodal" if image_path else "Gemma 3",
            None,
            intent,
        )