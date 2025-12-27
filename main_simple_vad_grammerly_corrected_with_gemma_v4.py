from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import tempfile
import whisper
import logging
import time
import wave
import numpy as np
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Speech-to-Text WITH Simple VAD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleVAD:
    """
    Simple energy-based VAD
    No external dependencies, pure Python
    Works by measuring audio energy levels
    """
    def __init__(self, energy_threshold=0.01):
        self.energy_threshold = energy_threshold
        self.enabled = True
        logger.info(f"✓ Simple VAD loaded (threshold={energy_threshold})")
    
    def has_speech(self, wav_path: str) -> bool:
        """Check if audio contains speech based on energy"""
        if not self.enabled:
            return True
        
        try:
            # Read WAV file
            with wave.open(wav_path, 'rb') as wf:
                audio_data = wf.readframes(wf.getnframes())
                sample_width = wf.getsampwidth()
            
            # Convert to numpy array
            if sample_width == 2:  # 16-bit
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                logger.warning(f"Unexpected sample width: {sample_width}")
                return True
            
            # Normalize to [-1, 1]
            audio_normalized = audio_array.astype(np.float32) / 32768.0
            
            # Calculate RMS energy
            rms_energy = np.sqrt(np.mean(audio_normalized ** 2))
            
            # Check if energy exceeds threshold
            has_speech = rms_energy > self.energy_threshold
            
            logger.info(f"VAD: Energy={rms_energy:.4f} {'✓ Speech' if has_speech else '✗ Silence'}")
            
            return has_speech
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True  # Assume speech on error

class EnhancedGrammarCorrector:
    """
    Enhanced sentence refinement using Gemma 2B
    Focus: Natural flow, professional polish, sentence structure
    NOT just grammar/punctuation fixes
    """
    def __init__(self):
        logger.info("Loading Enhanced Grammar Corrector...")
        try:
            self.client = ollama.Client(host='http://localhost:11434')
            self.client.generate(model="gemma2:2b", prompt="test", options={'num_predict': 1})
            self.enabled = True
            logger.info("✓ Enhanced Grammar enabled")
        except Exception as e:
            logger.warning(f"Grammar disabled: {e}")
            self.enabled = False
    
    async def correct(self, text: str) -> str:
        """
        Refine text to be more natural and professional
        Improves sentence flow, word choice, and structure
        """
        if not self.enabled or len(text.strip()) < 5:
            return text
        
        try:
            start = time.time()
            
            # Enhanced prompt for sentence-level refinement
            prompt = f"""Refine this transcribed speech into polished, professional text.

REQUIREMENTS:
1. Improve sentence flow and natural rhythm
2. Use better word choices where appropriate
3. Maintain the exact same meaning and intent
4. Keep approximately the same length (±10%)
5. Fix grammar, punctuation, and capitalization
6. Make it sound like professionally written text
7. Remove filler words (um, uh, like) if present
8. Ensure smooth transitions between ideas

ORIGINAL TEXT:
{text}

REFINED TEXT:"""
            
            response = await asyncio.to_thread(
                self.client.generate,
                model="gemma2:2b",
                prompt=prompt,
                options={
                    'temperature': 0.3,      # Slightly higher for creativity
                    'top_p': 0.9,
                    'top_k': 40,
                    'num_predict': 150,      # Allow more tokens for refinement
                    'repeat_penalty': 1.1,
                    'stop': ['\n\nORIGINAL', '\n\nREQUIREMENTS']
                }
            )
            
            refined = response['response'].strip()
            
            # Clean up common artifacts
            refined = refined.replace('REFINED TEXT:', '').strip()
            refined = refined.replace('Refined text:', '').strip()
            refined = refined.strip('"\'')
            refined = refined.strip()
            
            # Validate refinement
            if not refined or len(refined) < 3:
                logger.warning("Refinement produced empty result, using original")
                return text
            
            # Check length difference (shouldn't change too much)
            length_ratio = len(refined) / len(text)
            if length_ratio < 0.5 or length_ratio > 2.0:
                logger.warning(f"Refinement changed length too much ({length_ratio:.1f}x), using original")
                return text
            
            elapsed = time.time() - start
            
            if refined != text:
                logger.info(f"Refined ({elapsed:.2f}s):")
                logger.info(f"  Before: '{text}'")
                logger.info(f"  After:  '{refined}'")
            else:
                logger.info(f"No refinement needed ({elapsed:.2f}s)")
            
            return refined
            
        except Exception as e:
            logger.error(f"Refinement error: {e}")
            return text


class GrammarCorrector:
    """Gemma grammar correction"""
    def __init__(self):
        logger.info("Loading Grammar Corrector...")
        try:
            self.client = ollama.Client(host='http://localhost:11434')
            self.client.generate(model="gemma2:2b", prompt="test", options={'num_predict': 1})
            self.enabled = True
            logger.info("✓ Grammar enabled")
        except Exception as e:
            logger.warning(f"Grammar disabled: {e}")
            self.enabled = False
    
    async def correct(self, text: str) -> str:
        if not self.enabled or len(text.strip()) < 5:
            return text
        
        try:
            prompt = f"Fix grammar and punctuation. Return only corrected text:\n{text}"
            response = await asyncio.to_thread(
                self.client.generate,
                model="gemma2:2b",
                prompt=prompt,
                options={'temperature': 0.0, 'num_predict': 100}
            )
            corrected = response['response'].strip().strip('"\'')
            if corrected and corrected != text:
                logger.info(f"Grammar: '{text}' → '{corrected}'")
            return corrected if corrected else text
        except Exception as e:
            logger.error(f"Grammar error: {e}")
            return text

class AudioProcessor:
    """Processes audio with Simple VAD filtering"""
    def __init__(self, whisper_model, vad, grammar):
        self.model = whisper_model
        self.vad = vad
        self.grammar = grammar
        
        self.audio_buffer = []
        self.buffer_duration = 3.0  # Process every 3 seconds
        
        self.stats = {"processed": 0, "skipped_vad": 0, "skipped_empty": 0}
        logger.info("AudioProcessor WITH Simple VAD ready")
    
    def add_audio(self, pcm_data: bytes):
        """Add PCM audio chunk"""
        self.audio_buffer.append(pcm_data)
        
        # Calculate duration (16-bit PCM, 16kHz, mono)
        total_bytes = sum(len(chunk) for chunk in self.audio_buffer)
        duration = (total_bytes / 2) / 16000
        
        return duration >= self.buffer_duration
    
    async def process(self) -> dict:
        """Process audio buffer"""
        if not self.audio_buffer:
            return {"text": "", "skipped": True}
        
        try:
            logger.info("="*60)
            
            # Combine audio
            combined = b''.join(self.audio_buffer)
            duration = (len(combined) / 2) / 16000
            logger.info(f"Processing {duration:.1f}s ({len(combined)/1024:.1f}KB)")
            
            # Minimum size check
            if len(combined) < 16000:  # Less than 0.5s
                logger.warning("Audio too short")
                self.audio_buffer.clear()
                return {"text": "", "skipped": True}
            
            # Save as WAV
            wav_path = tempfile.mktemp(suffix=".wav")
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)       # Mono
                wf.setsampwidth(2)       # 16-bit
                wf.setframerate(16000)   # 16kHz
                wf.writeframes(combined)
            
            wav_size = os.path.getsize(wav_path)
            logger.info(f"WAV created: {wav_size/1024:.1f}KB")
            
            # VAD check
            if not self.vad.has_speech(wav_path):
                logger.info("⚠️ Skipped by VAD (silence/noise)")
                self.audio_buffer.clear()
                os.remove(wav_path)
                self.stats["skipped_vad"] += 1
                return {"text": "", "skipped": True, "reason": "no_speech"}
            
            # Transcribe
            start = time.time()
            transcription = await asyncio.to_thread(self._transcribe, wav_path)
            whisper_time = time.time() - start
            
            os.remove(wav_path)
            
            if not transcription or len(transcription.strip()) < 3:
                logger.info(f"Empty transcription ({whisper_time:.2f}s)")
                self.audio_buffer.clear()
                self.stats["skipped_empty"] += 1
                return {"text": "", "skipped": True}
            
            logger.info(f"Whisper ({whisper_time:.2f}s): '{transcription}'")
            
            # Grammar
            if self.grammar.enabled:
                transcription = await self.grammar.correct(transcription)
            
            self.audio_buffer.clear()
            self.stats["processed"] += 1
            
            logger.info(f"✓ Result: '{transcription}'")
            logger.info("="*60)
            
            return {"text": transcription, "skipped": False, "stats": self.stats}
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.audio_buffer.clear()
            return {"text": "", "skipped": True}
    
    def _transcribe(self, wav_path: str) -> str:
        """Transcribe with Whisper"""
        try:
            result = self.model.transcribe(
                wav_path,
                language="en",
                fp16=False,
                temperature=0.0,
                beam_size=5,
                verbose=False
            )
            return result["text"].strip()
        except Exception as e:
            logger.error(f"Whisper error: {e}")
            return ""

class ConnectionManager:
    """Manages connections"""
    def __init__(self):
        self.connections = {}
        self.processors = {}
        
        logger.info("="*60)
        logger.info(" LOADING MODELS")
        logger.info("="*60)
        
        self.whisper = whisper.load_model("small")
        logger.info("✓ Whisper loaded")
        
        self.vad = SimpleVAD(energy_threshold=0.02)
        self.grammar = GrammarCorrector()
        
        logger.info("="*60)
        logger.info(" SYSTEM READY WITH Simple VAD")
        logger.info("="*60)
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.connections[client_id] = websocket
        
        processor = AudioProcessor(self.whisper, self.vad, self.grammar)
        self.processors[client_id] = processor
        
        logger.info(f"✓ Client connected: {client_id}")
    
    async def disconnect(self, client_id: str):
        if client_id in self.processors:
            logger.info(f"Stats: {self.processors[client_id].stats}")
            del self.processors[client_id]
        
        if client_id in self.connections:
            del self.connections[client_id]
        
        logger.info(f"✗ Client disconnected: {client_id}")
    
    async def handle_audio(self, client_id: str, audio_data: bytes):
        """Handle incoming audio"""
        processor = self.processors.get(client_id)
        websocket = self.connections.get(client_id)
        
        if not processor or not websocket:
            return
        
        try:
            should_process = processor.add_audio(audio_data)
            
            if should_process:
                result = await processor.process()
                
                if result.get('text'):
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result['text'],
                        "stats": result.get('stats', {})
                    })
                    logger.info(f"✓ Sent: '{result['text']}'")
                # elif result.get('reason') == 'no_speech':
                #     await websocket.send_json({
                #         "type": "info",
                #         "message": "No speech detected (silence/noise filtered)"
                #     })
        except Exception as e:
            logger.error(f"Error: {e}")

manager = ConnectionManager()

@app.get("/")
async def root():
    return {
        "service": "Speech-to-Text WITH Simple VAD",
        "status": "running",
        "vad": "simple_energy" if manager.vad.enabled else "disabled",
        "grammar": manager.grammar.enabled
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "connections": len(manager.connections)
    }

@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Ready WITH Simple VAD"
        })
        
        while True:
            data = await websocket.receive_bytes()
            await manager.handle_audio(client_id, data)
            
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print(" SPEECH-TO-TEXT WITH Simple VAD")
    print(" - Energy-based VAD (pure Python)")
    print(" - Filters silence/low energy audio")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
