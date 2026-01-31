"""
Enhanced refinement engine with async lock for thread safety.
"""
import asyncio
import logging
import time
from enum import Enum
import openai

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
    def __init__(self, model_name="glm-4.7:cloud", base_url="http://localhost:11434/v1", api_key="ollama"):
        self.model_name = model_name
        self.client = None
        self.enabled = True # Assume enabled initially, handle failures at runtime
        self._lock = None  # Will be initialized as asyncio.Lock() when event loop is available
        logger.info(f"Loading Refinement Engine (model={model_name})...")
        
        try:
            self.client = openai.AsyncOpenAI(
                base_url=base_url,
                api_key=api_key
            )
            # Connectivity check removed from __init__ to avoid blocking async event loop.
            # Connection will be verified on first request.
            logger.info(f"✓ Refinement Engine initialized with {model_name} via OpenAI Client")
        except Exception as e:
            logger.warning(f"Refinement initialization failed: {e}")
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
        async with lock:  # Prevent concurrent access to local LLM
            try:
                start = time.time()
                logger.info(f"🔧 Refining text: '{raw_text[:50]}...'")

                # Add context to prompt if provided
                context_prompt = f"\n\nPast conversation context:\n{context}" if context else ""
                
                system_prompt = "You are an expert in speech transcription and text refinement."
                user_prompt = f"""Refine this transcription to improve clarity, remove filler words, fix grammar, and enhance readability.

Original transcription: {raw_text}{context_prompt}

Guidelines:
1. Keep the main meaning intact
2. Remove unnecessary filler words (um, uh, like, etc.)
3. Fix any grammatical errors
4. Improve sentence structure for better readability
5. Maintain the original speaker's tone and style

Refined text:"""

                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.4,
                    top_p=0.9,
                    max_tokens=250, # Replaces num_predict
                    # repeat_penalty is not standard OpenAI API, omitted for compatibility
                )

                refined = response.choices[0].message.content.strip()

                elapsed = time.time() - start
                logger.info(f"✓ Refined in {elapsed:.2f}s")

                return refined if refined else raw_text

            except Exception as e:
                logger.error(f"Refinement error: {e}")
                # Optional: Disable on critical connection error?
                # self.enabled = False 
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
                
                response = await self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.4,
                    top_p=0.9,
                    max_tokens=250,
                )
                
                feedback = response.choices[0].message.content.strip()
                
                if not feedback or len(feedback) < 10:
                    feedback = "Great job! Your speech was clear and well-structured."
                
                elapsed = time.time() - start
                logger.info(f"✓ Feedback generated in {elapsed:.2f}s")
                
                return feedback
                
            except Exception as e:
                logger.error(f"Feedback generation error: {e}")
                return f"Error generating feedback: {str(e)}"
