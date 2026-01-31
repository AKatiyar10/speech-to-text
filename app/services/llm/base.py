from abc import ABC, abstractmethod
from typing import List, Dict, Any

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    """
    
    @abstractmethod
    async def refine_text(self, text: str, context: List[Dict[str, Any]] = None) -> str:
        """
        Refine the given text (grammar, punctuation) using the LLM.
        
        Args:
            text: The raw text to refine.
            context: Optional list of previous conversation turns for context.
            
        Returns:
            Refined text.
        """
        pass
