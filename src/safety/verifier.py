import logging

logger = logging.getLogger(__name__)

class SafetyVerifier:
    """
    The 'Critic' layer that checks LLM outputs before showing them to the user.
    """
    def __init__(self):
        self.banned_phrases = [
            "ignore previous instructions",
            "take 500mg", 
            "stop taking your medication"
        ]

    def verify(self, response_text):
        """
        Simple keyword-based safety check.
        Returns: (is_safe: bool, reason: str)
        """
        text = response_text.lower()
        
        for phrase in self.banned_phrases:
            if phrase in text:
                logger.warning(f"⚠️ Safety Violation: Found banned phrase '{phrase}'")
                return False, f"Response blocked due to safety keyword: {phrase}"
                
        return True, "Safe"
