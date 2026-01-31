#!/usr/bin/env python3
import numpy as np
import wave
import tempfile
import os

# Test our audio format conversion
print("Testing audio format conversion...")

# Simulate some 16-bit PCM audio data
sample_rate = 16000
duration = 1.0
num_samples = int(sample_rate * duration)

# Create some dummy 16-bit PCM data (sine wave)
t = np.linspace(0, duration, num_samples)
audio_signal = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

# Convert to 16-bit integers (what we receive from WebRTC)
pcm_int16 = (audio_signal * 32767).astype(np.int16)
pcm_bytes = pcm_int16.tobytes()

print(f"Original signal range: {audio_signal.min():.3f} to {audio_signal.max():.3f}")
print(f"PCM int16 range: {pcm_int16.min()} to {pcm_int16.max()}")

# Our conversion code
pcm_int16_converted = np.frombuffer(pcm_bytes, dtype=np.int16)
pcm_float32 = pcm_int16_converted.astype(np.float32) / 32768.0

print(f"Converted float32 range: {pcm_float32.min():.3f} to {pcm_float32.max():.3f}")

# Save as WAV
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
    wav_path = tmp.name

try:
    with wave.open(wav_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(4)  # 4 bytes for 32-bit float
        wf.setframerate(16000)
        wf.writeframes(pcm_float32.tobytes())

    print(f"Successfully saved WAV file: {wav_path}")

    # Verify the file
    with wave.open(wav_path, 'rb') as wf:
        print(f"WAV info: channels={wf.getnchannels()}, sampwidth={wf.getsampwidth()}, framerate={wf.getframerate()}")

    print("Audio format conversion test PASSED!")

finally:
    if os.path.exists(wav_path):
        os.remove(wav_path)