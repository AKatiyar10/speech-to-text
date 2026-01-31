from app.core.config import settings, LLMProvider
from app.services.llm.base import LLMProvider as BaseLLMProvider
from app.services.llm.ollama_client import OllamaClient
from app.services.llm.openai_client import OpenAIClient
import structlog

logger = structlog.get_logger()

def get_llm_client() -> BaseLLMProvider:
    if settings.LLM_PROVIDER == LLMProvider.OPENAI:
        logger.info("using_openai_provider")
        return OpenAIClient()
    else:
        logger.info("using_ollama_provider")
        return OllamaClient()
