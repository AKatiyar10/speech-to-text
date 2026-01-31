from pydantic_settings import BaseSettings, SettingsConfigDict
from enum import Enum
from typing import Optional, List

class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"

class Settings(BaseSettings):
    # App Settings
    APP_NAME: str = "Speech-to-Text App"
    DEBUG: bool = False
    
    # Cors
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]
    
    # LLM Settings
    LLM_PROVIDER: LLMProvider = LLMProvider.OLLAMA
    
    # Ollama Settings
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "gemma:2b"
    
    # OpenAI Settings
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4o"
    
    # Audio Settings
    VAD_THRESHOLD: float = 0.5
    SPEAKER_CONFIDENCE_THRESHOLD: float = 0.75
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True
    )

settings = Settings()
