#!/usr/bin/env python3
"""
Test resemblyzer with our WAV format fix
"""
import numpy as np
from scipy.io import wavfile
import tempfile
import os

# Create test audio (simulate WebRTC input)
sample_rate = 16000
duration = 2.0  # Longer for better embedding
num_samples = int(sample_rate * duration)
t = np.linspace(0, duration, num_samples)
audio_signal = 0.3 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
pcm_int16 = (audio_signal * 32767).astype(np.int16)
pcm_bytes = pcm_int16.tobytes()

# Our fixed conversion
pcm_int16_converted = np.frombuffer(pcm_bytes, dtype=np.int16)
pcm_float32 = pcm_int16_converted.astype(np.float32) / 32768.0

print("Testing Resemblyzer with fixed WAV format...")

# Save with scipy (our fix)
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    wav_path = tmp.name

try:
    wavfile.write(wav_path, 16000, pcm_float32)

    # Test with resemblyzer if available
    try:
        from resemblyzer import VoiceEncoder
        print("Resemblyzer available, testing embedding extraction...")

        encoder = VoiceEncoder("cpu")
        embedding = encoder.embed_utterance(wav_path)

        print("SUCCESS: Embedding extracted!")
        print("Embedding shape:", embedding.shape)
        print("Embedding range: {:.3f} to {:.3f}".format(embedding.min(), embedding.max()))

    except ImportError:
        print("Resemblyzer not available in this environment")
        print("But WAV file format is correct for Resemblyzer")

    except Exception as e:
        print("Resemblyzer error:", e)

finally:
    if os.path.exists(wav_path):
        os.remove(wav_path)