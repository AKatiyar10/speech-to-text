#!/usr/bin/env python3
"""
Test scipy WAV file format with float32 data
"""
import numpy as np
from scipy.io import wavfile
import tempfile
import os
import struct

# Simulate WebRTC audio processing
sample_rate = 16000
duration = 1.0
num_samples = int(sample_rate * duration)
t = np.linspace(0, duration, num_samples)
audio_signal = 0.5 * np.sin(2 * np.pi * 440 * t)

# Convert to 16-bit integers (what WebRTC sends)
pcm_int16 = (audio_signal * 32767).astype(np.int16)
pcm_bytes = pcm_int16.tobytes()

# Our audio_processor.py conversion
pcm_int16_loaded = np.frombuffer(pcm_bytes, dtype=np.int16)
pcm_float32 = pcm_int16_loaded.astype(np.float32) / 32768.0

print("Original signal range: {:.3f} to {:.3f}".format(audio_signal.min(), audio_signal.max()))
print("PCM int16 range: {} to {}".format(pcm_int16.min(), pcm_int16.max()))
print("Converted float32 range: {:.3f} to {:.3f}".format(pcm_float32.min(), pcm_float32.max()))

# Save with scipy (our fix)
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    wav_path = tmp.name

try:
    wavfile.write(wav_path, 16000, pcm_float32)

    # Read back the raw header
    with open(wav_path, 'rb') as f:
        header = f.read(44)

    print("\nScipy WAV Header Analysis:")
    print("Format code:", struct.unpack('<H', header[20:22])[0])  # Should be 3 for float
    print("Bits per sample:", struct.unpack('<H', header[34:36])[0])

    # Test with librosa
    try:
        import librosa
        wav_data, sr = librosa.load(wav_path, sr=None)
        print("\nLibrosa loaded successfully!")
        print("Data type:", wav_data.dtype)
        print("Data shape:", wav_data.shape)
        print("Data range: {:.3f} to {:.3f}".format(wav_data.min(), wav_data.max()))

        # Test librosa's melspectrogram (what resemblyzer uses)
        mel = librosa.feature.melspectrogram(y=wav_data, sr=sr, n_mels=40)
        print("Melspectrogram computed successfully!")
        print("Mel shape:", mel.shape)

    except Exception as e:
        print("Error:", e)
        import traceback
        traceback.print_exc()

finally:
    if os.path.exists(wav_path):
        os.remove(wav_path)