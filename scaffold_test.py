import asyncio
from app.core.config import settings, LLMProvider
from app.services.llm.factory import get_llm_client
from app.core.logging import setup_logging
import structlog

setup_logging()
logger = structlog.get_logger()

async def main():
    logger.info("testing_scaffolding")
    logger.info("current_configuration", 
                llm_provider=settings.LLM_PROVIDER, 
                debug=settings.DEBUG,
                ollama_model=settings.OLLAMA_MODEL)
    
    # Test Factory
    client = get_llm_client()
    logger.info("llm_client_instantiated", type=type(client).__name__)
    
    # Check simple refinement (mocking actual call if needed, but here we just check structure)
    # in a real run, this would try to hit localhost:11434
    
    print("\n✅ Scaffolding verification successful!")

if __name__ == "__main__":
    asyncio.run(main())
