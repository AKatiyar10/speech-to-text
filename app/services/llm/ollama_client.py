import ollama
from app.services.llm.base import LLMProvider
from app.core.config import settings
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger()

class OllamaClient(LLMProvider):
    def __init__(self):
        self.model = settings.OLLAMA_MODEL
        self.base_url = settings.OLLAMA_BASE_URL
        # Ensure ollama client is configured if needed, though standard client uses env vars or defaults
        
    async def refine_text(self, text: str, context: List[Dict[str, Any]] = None) -> str:
        prompt = f"""You are a helpful assistant that corrects grammar and punctuation.
        Output ONLY the corrected text. Do not add any conversational filler.
        
        Raw text: "{text}"
        """
        
        if context:
            # Format context for the LLM if provided
            context_str = "\n".join([f"{c['speaker']}: {c['text']}" for c in context[-10:]])
            prompt = f"Context:\n{context_str}\n\n{prompt}"

        try:
            response = ollama.chat(model=self.model, messages=[
                {'role': 'user', 'content': prompt},
            ])
            return response['message']['content'].strip()
        except Exception as e:
            logger.error("ollama_refine_failed", error=str(e))
            return text  # Fallback to original text
