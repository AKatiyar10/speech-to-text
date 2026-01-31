"""
Enhanced refinement engine with async lock for thread safety.
"""
import asyncio
import logging
import ollama
import time
from enum import Enum

logger = logging.getLogger(__name__)

# ============ VERBOSE LOGGING ============
VERBOSE_LOGGING = True
# ===========================================


class OutputMode(str, Enum):
    """Feature flags for output control"""
    RAW_ONLY = "raw"
    REFINED_ONLY = "refined"
    REFINED_WITH_FEEDBACK = "full"
    ALL = "all"


class EnhancedRefinementEngine:
    """Advanced refinement with on-demand feedback and async lock for thread safety"""
    def __init__(self, model_name="phi3:mini"):
        self.model_name = model_name
        self.client = None
        self.enabled = False
        self._lock = None  # Will be initialized as asyncio.Lock() when event loop is available
        logger.info(f"Loading Refinement Engine (model={model_name})...")
        
        try:
            self.client = ollama.Client(host='http://localhost:11434')
            # Test connection
            self.client.generate(model=self.model_name, prompt="test", options={'num_predict': 1})
            self.enabled = True
            logger.info(f"✓ Refinement Engine enabled with {model_name}")
        except Exception as e:
            logger.warning(f"Refinement disabled: {e}")
            self.enabled = False
    
    def _get_lock(self):
        """Lazy initialization of asyncio.Lock (must be called from async context)"""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
    
    async def refine_text(self, raw_text: str, context: str = "") -> str:
        """Refine text ONLY (no feedback generation) - Thread-safe"""
        if not self.enabled or len(raw_text.strip()) < 5:
            return raw_text

        lock = self._get_lock()
        async with lock:  # Prevent concurrent access to ollama client
            try:
                start = time.time()
                logger.info(f"🔧 Refining text: '{raw_text[:50]}...'")

                # Add context to prompt if provided
                context_prompt = f"\n\nPast conversation context:\n{context}" if context else ""
                prompt = f"""You are an expert in speech transcription and text refinement.
Refine this transcription to improve clarity, remove filler words, fix grammar, and enhance readability.

Original transcription: {raw_text}{context_prompt}

Guidelines:
1. Keep the main meaning intact
2. Remove unnecessary filler words (um, uh, like, etc.)
3. Fix any grammatical errors
4. Improve sentence structure for better readability
5. Maintain the original speaker's tone and style

Refined text:"""

                response = await asyncio.to_thread(
                    self.client.generate,
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': 0.4,
                        'top_p': 0.9,
                        'top_k': 40,
                        'num_predict': 250,
                        'repeat_penalty': 1.1
                    }
                )

                refined = response['response'].strip()

                elapsed = time.time() - start
                logger.info(f"✓ Refined in {elapsed:.2f}s")

                return refined if refined else raw_text

            except Exception as e:
                logger.error(f"Refinement error: {e}")
                return raw_text
    
    async def generate_feedback(self, raw_text: str) -> str:
        """Generate speaking feedback - Thread-safe"""
        if not self.enabled or len(raw_text.strip()) < 5:
            return "Refinement engine disabled."
        
        lock = self._get_lock()
        async with lock:
            try:
                start = time.time()
                logger.info(f"💡 Generating feedback for: '{raw_text[:50]}...'")
                
                prompt = f"""Analyze this speech and give 3-5 brief feedback points on filler words, clarity, effectiveness. Be constructive.

TEXT: {raw_text}"""
                
                response = await asyncio.to_thread(
                    self.client.generate,
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': 0.4,
                        'top_p': 0.9,
                        'top_k': 40,
                        'num_predict': 250,
                        'repeat_penalty': 1.1
                    }
                )
                
                feedback = response['response'].strip()
                
                if not feedback or len(feedback) < 10:
                    feedback = "Great job! Your speech was clear and well-structured."
                
                elapsed = time.time() - start
                logger.info(f"✓ Feedback generated in {elapsed:.2f}s")
                
                return feedback
                
            except Exception as e:
                logger.error(f"Feedback generation error: {e}")
                return f"Error generating feedback: {str(e)}"
