import re

class QueryIntent:
    DATA = "data"
    CLINICAL = "clinical"

def classify_query(query, session_messages):
    """
    Refined Router: Prioritizes Specific IDs > Hard Stats > Visual Context > General Lists.
    """
    q = query.lower()
    
    # --- 1. Strong "Specific Patient" Override ---
    # If a specific ID is mentioned, it is almost always a clinical lookup.
    # "List meds for Patient 123" -> CLINICAL (Specific) vs "List all patients" -> DATA (Aggregate)
    if re.search(r'\b(patient|p|id|encounter)\s*[:#-]?\s*\d{4,}', q):
        return QueryIntent.CLINICAL

    # --- 2. Hard Statistics (The "Count" Fix) ---
    # These words imply aggregation across the dataset.
    strong_data_keywords = [
        "how many", "count", "total", "average", "mean", "percentage", 
        "statistics", "graph", "plot", "trend", "distribution", "breakdown",
        "census", "number of"
    ]
    if any(kw in q for kw in strong_data_keywords):
        return QueryIntent.DATA

    # --- 3. Visual Context Logic (Context Aware) ---
    # Only override if the user is likely referring to the image content ("this", "it")
    # AND they didn't ask for a count.
    recent_history = session_messages[-3:] if len(session_messages) > 3 else session_messages
    has_recent_image = any("[SYSTEM UPDATE]" in str(msg.get("content", "")) for msg in recent_history)
    
    if has_recent_image:
        # User is likely asking follow-up questions about the image
        return QueryIntent.CLINICAL

    # --- 4. The "List/Table" Fallback ---
    # "List patients", "Show me a table of X"
    # We check this LAST so that "List Patient 123" is caught by Rule 1.
    if any(kw in q for kw in ["list", "table", "show me", "dataset"]):
        return QueryIntent.DATA

    # Default: Medical App = Clinical Conversation
    return QueryIntent.CLINICAL