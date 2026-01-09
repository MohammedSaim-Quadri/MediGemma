import logging
from llama_index.llms.ollama import Ollama
from llama_index.core import Settings

logger = logging.getLogger(__name__)

class LLMEngine:
    """
    Handles clinical reasoning and text generation via Ollama (Gemma 2).
    """
    def __init__(self, model_name="gemma2:27b", timeout=300.0):
        self.model_name = model_name
        self.timeout = timeout
        self.llm = None

    def initialize(self):
        """Connects to the local Ollama instance."""
        try:
            logger.info(f"🧠 Connecting to Inference Engine: {self.model_name}...")
            self.llm = Ollama(model=self.model_name, request_timeout=self.timeout)
            Settings.llm = self.llm # Set global LlamaIndex setting
            logger.info("✅ Inference Engine connected.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Ollama: {e}")
            raise e

    def generate(self, prompt):
        """Direct text completion."""
        if not self.llm:
            self.initialize()
        return self.llm.complete(prompt).text

    def chat(self, messages):
        """Chat completion (list of messages)."""
        if not self.llm:
            self.initialize()
        return self.llm.chat(messages)
