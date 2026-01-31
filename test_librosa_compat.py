#!/usr/bin/env python3
"""
Test different WAV saving methods for librosa compatibility
"""
import numpy as np
import tempfile
import os

try:
    from scipy.io import wavfile
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    import librosa
    HAS_LIBROSA = True
except ImportError:
    HAS_LIBROSA = False

# Create test audio (simulate what we get from WebRTC)
sample_rate = 16000
duration = 1.0
num_samples = int(sample_rate * duration)

# Create a sine wave
t = np.linspace(0, duration, num_samples)
audio_signal = 0.5 * np.sin(2 * np.pi * 440 * t)

# Convert to 16-bit integers (what WebRTC sends)
pcm_int16 = (audio_signal * 32767).astype(np.int16)
pcm_bytes = pcm_int16.tobytes()

print("Original signal range: {:.3f} to {:.3f}".format(audio_signal.min(), audio_signal.max()))
print("PCM int16 range: {} to {}".format(pcm_int16.min(), pcm_int16.max()))

# Our conversion (from audio_processor.py)
pcm_int16_converted = np.frombuffer(pcm_bytes, dtype=np.int16)
pcm_float32 = pcm_int16_converted.astype(np.float32) / 32768.0

print("Converted float32 range: {:.3f} to {:.3f}".format(pcm_float32.min(), pcm_float32.max()))

def test_wav_method(method_name, save_func):
    """Test a WAV saving method"""
    print("\n=== Testing {} ===".format(method_name))

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav_path = tmp.name

    try:
        save_func(wav_path, pcm_float32)

        if HAS_LIBROSA:
            print("Testing librosa loading...")
            try:
                wav_data, sr = librosa.load(wav_path, sr=None)
                print("[PASS] Librosa loaded successfully!")
                print("  Sample rate: {}".format(sr))
                print("  Data shape: {}".format(wav_data.shape))
                print("  Data range: {:.3f} to {:.3f}".format(wav_data.min(), wav_data.max()))
                print("  Data type: {}".format(wav_data.dtype))
                return True
            except Exception as e:
                print("[FAIL] Librosa failed: {}".format(e))
                return False
        else:
            print("Librosa not available for testing")
            return True
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

# Test different methods
results = []

# Method 1: Python wave module (current implementation)
def save_wave_module(path, data):
    import wave
    with wave.open(path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)  # 4 bytes for 32-bit float
        wf.setframerate(16000)
        wf.writeframes(data.tobytes())

results.append(("Python wave module", test_wav_method("Python wave module", save_wave_module)))

# Method 2: scipy.io.wavfile (if available)
if HAS_SCIPY:
    def save_scipy(path, data):
        wavfile.write(path, 16000, data)

    results.append(("scipy.io.wavfile", test_wav_method("scipy.io.wavfile", save_scipy)))
else:
    print("\nscipy not available, skipping scipy test")

print("\n=== SUMMARY ===")
for method, success in results:
    status = "[WORKS]" if success else "[FAILS]"
    print("{}: {}".format(method, status))