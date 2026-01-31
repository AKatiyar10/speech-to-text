from openai import AsyncOpenAI
from app.services.llm.base import LLMProvider
from app.core.config import settings
from typing import List, Dict, Any
import structlog

logger = structlog.get_logger()

class OpenAIClient(LLMProvider):
    def __init__(self):
        self.client = AsyncOpenAI(api_key=settings.OPENAI_API_KEY)
        self.model = settings.OPENAI_MODEL
        
    async def refine_text(self, text: str, context: List[Dict[str, Any]] = None) -> str:
        if not settings.OPENAI_API_KEY:
             logger.error("openai_api_key_missing")
             return text

        system_prompt = "You are a helpful assistant that corrects grammar and punctuation. Output ONLY the corrected text."
        
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        if context:
             for turn in context[-10:]:
                 messages.append({"role": "user", "name": turn.get('speaker', 'User'), "content": turn.get('text', '')})
        
        messages.append({"role": "user", "content": f"Correct this: {text}"})

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error("openai_refine_failed", error=str(e))
            return text
