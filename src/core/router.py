import re

class QueryIntent:
    DATA = "data"
    CLINICAL = "clinical"

def classify_query(query, session_messages):
    """
    100% Deterministic Router.
    """
    q = query.lower()
    
    # RULE 1: Explicit Data Analytics Keywords
    # If the user asks for counts, lists, or tables, it is ALWAYS a data request.
    data_keywords = [
        "how many", "count", "total", "list", "show me", "table", 
        "average", "percentage of patients", "statistics", "graph", "plot"
    ]
    if any(kw in q for kw in data_keywords):
        return QueryIntent.DATA
    
    # RULE 2: Visual Context Override
    # If the system just analyzed an image (visible in history), 
    # follow-up questions ("Describe tissue") must go to Clinical/RAG.
    # We check the last 3 messages for the [SYSTEM UPDATE] log.
    recent_history = session_messages[-3:] if len(session_messages) > 3 else session_messages
    has_recent_image = any("[SYSTEM UPDATE]" in str(msg.get("content", "")) for msg in recent_history)
    
    if has_recent_image:
        return QueryIntent.CLINICAL
    
    # RULE 3: Patient ID Check
    # "Patient 10770" is always a clinical record lookup.
    if re.search(r'\b(patient|p|id|encounter)\s*[:#-]?\s*\d{4,}', q):
        return QueryIntent.CLINICAL

    # DEFAULT: This is a medical app. If unsure, treat as Clinical Chat.
    return QueryIntent.CLINICAL