"""
Simple Voice Activity Detection using energy-based approach.
"""
import logging
import numpy as np
import wave
import tempfile
import os

logger = logging.getLogger(__name__)

# ============ VERBOSE LOGGING ============
VERBOSE_LOGGING = True
# ===========================================


class SimpleVAD:
    """Energy-based VAD optimized for 16kHz single-channel audio"""
    def __init__(self, energy_threshold: float = 0.01):
        self.energy_threshold = energy_threshold
    
    def has_speech(self, wav_path: str) -> bool:
        """Check if WAV file contains speech using RMS energy"""
        try:
            with wave.open(wav_path, 'rb') as wf:
                # Verify format: 16kHz, mono, 16-bit
                if wf.getframerate() != 16000 or wf.getnchannels() != 1 or wf.getsampwidth() != 2:
                    logger.warning(f"Unexpected WAV format: {wf.getframerate()}Hz, {wf.getnchannels()}ch, {wf.getsampwidth()}bytes")
                    return True
                
                audio_data = wf.readframes(wf.getnframes())
            
            # Optimized: Use numpy's efficient operations
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_normalized = audio_array.astype(np.float32) * (1.0 / 32768.0)
            rms_energy = np.sqrt(np.mean(np.square(audio_normalized)))
            has_speech = rms_energy > self.energy_threshold
            
            logger.info(f"VAD: Energy={rms_energy:.4f} {'✓ Speech' if has_speech else '✗ Silence'}")
            return has_speech
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True
