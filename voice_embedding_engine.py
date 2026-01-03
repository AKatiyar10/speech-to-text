"""
Voice embedding engine using Resemblyzer for 100% local speaker identification.
"""
import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional

from speaker_match import SpeakerMatch

logger = logging.getLogger(__name__)

# ============ VERBOSE LOGGING ============
VERBOSE_LOGGING = True
# ===========================================


class VoiceEmbeddingEngine:
    """Extracts and compares voice embeddings using Resemblyzer (100% local)"""
    
    def __init__(self):
        self.enabled = False
        self.encoder = None
        self.enrolled_speakers: Dict[str, List[np.ndarray]] = {}
        self._lock = asyncio.Lock()
        self._load_enrolled_speakers()
        
        try:
            from resemblyzer import VoiceEncoder
            self.encoder = VoiceEncoder("cpu")
            self.enabled = True
            logger.info("✓ Resemblyzer VoiceEncoder loaded (local, offline)")
        except ImportError:
            logger.warning("✗ Resemblyzer not installed: pip install resemblyzer")
            logger.warning("Speaker identification disabled (core features still work)")
        except Exception as e:
            logger.error(f"Failed to load Resemblyzer: {e}")
    
    def _load_enrolled_speakers(self):
        """Load enrolled speakers from disk"""
        emb_dir = Path("speakers/embeddings")
        if emb_dir.exists():
            for emb_file in emb_dir.glob("*.npy"):
                name = emb_file.stem
                try:
                    embedding = np.load(emb_file)
                    if name not in self.enrolled_speakers:
                        self.enrolled_speakers[name] = []
                    self.enrolled_speakers[name].append(embedding)
                    if VERBOSE_LOGGING:
                        logger.info(f"[VERBOSE] Loaded embedding for speaker: {name}")
                except Exception as e:
                    logger.error(f"Failed to load embedding {emb_file}: {e}")
        
        if VERBOSE_LOGGING:
            logger.info(f"[VERBOSE] Loaded {len(self.enrolled_speakers)} enrolled speakers")
    
    async def extract_embedding(self, wav_path: str) -> np.ndarray:
        """Extract voice embedding from WAV file"""
        if not self.enabled or not self.encoder:
            if VERBOSE_LOGGING:
                logger.warning("[VERBOSE] Voice encoder not available, returning zero embedding")
            return np.zeros(256)
        
        return await asyncio.to_thread(self.encoder.embed_utterance, wav_path)
    
    async def enroll_speaker(self, name: str, wav_path: str) -> Dict[str, Any]:
        """Enroll a new speaker with voice sample"""
        if not self.enabled or not self.encoder:
            return {"success": False, "error": "Voice encoder not available"}
        
        embedding = await self.extract_embedding(wav_path)
        
        # Save to disk
        emb_dir = Path("speakers/embeddings")
        emb_dir.mkdir(parents=True, exist_ok=True)
        emb_file = emb_dir / f"{name}.npy"
        np.save(emb_file, embedding)
        
        # Add to memory
        async with self._lock:
            if name not in self.enrolled_speakers:
                self.enrolled_speakers[name] = []
            self.enrolled_speakers[name].append(embedding)
        
        logger.info(f"✓ Enrolled speaker: {name}")
        return {"success": True, "speaker_name": name, "embedding": embedding}
    
    async def identify_speaker(self, wav_path: str, confidence_threshold: float = 0.75):
        """Compare and identify speaker using cosine similarity"""
        if VERBOSE_LOGGING:
            logger.info(f"[VERBOSE] identify_speaker called: wav_path={wav_path}, threshold={confidence_threshold}")
        
        embedding = await self.extract_embedding(wav_path)
        
        if VERBOSE_LOGGING:
            logger.info(f"[VERBOSE] Extracted embedding: shape={embedding.shape}")
        
        best_match = None
        best_score = -1.0
        
        async with self._lock:
            if not self.enrolled_speakers:
                if VERBOSE_LOGGING:
                    logger.info("[VERBOSE] No enrolled speakers, returning UNKNOWN")
                return SpeakerMatch(
                    name="UNKNOWN",
                    confidence=0.0,
                    is_match=False,
                    embedding=embedding
                )
            
            if VERBOSE_LOGGING:
                logger.info(f"[VERBOSE] Comparing against {len(self.enrolled_speakers)} enrolled speakers")
            
            for name, embeddings in self.enrolled_speakers.items():
                for enrolled_emb in embeddings:
                    score = np.dot(embedding, enrolled_emb) / (
                        np.linalg.norm(embedding) * np.linalg.norm(enrolled_emb)
                    )
                    
                    if VERBOSE_LOGGING and score > 0.5:
                        logger.info(f"[VERBOSE] Similarity with {name}: {score:.4f}")
                    
                    if score > best_score:
                        best_score = score
                        best_match = name
            
            if VERBOSE_LOGGING:
                logger.info(f"[VERBOSE] Best match: {best_match}, Score: {best_score:.4f}, Threshold: {confidence_threshold}")
            
            if best_score >= confidence_threshold and best_match:
                if VERBOSE_LOGGING:
                    logger.info(f"[VERBOSE] ✓ Speaker identified: {best_match} (confidence: {best_score:.4f})")
                return SpeakerMatch(
                    name=best_match,
                    confidence=float(best_score),
                    is_match=True,
                    embedding=embedding
                )
            else:
                if VERBOSE_LOGGING:
                    logger.info(f"[VERBOSE] ✗ Unknown speaker (score {best_score:.4f} < threshold {confidence_threshold})")
                return SpeakerMatch(
                    name="UNKNOWN",
                    confidence=float(best_score),
                    is_match=False,
                    embedding=embedding
                )
