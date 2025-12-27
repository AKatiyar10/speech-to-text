from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import tempfile
import whisper
import logging
import time
import wave
import torch
import ollama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Speech-to-Text WITH VAD")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SileroVAD:
    """Voice Activity Detection - filters non-speech audio"""
    def __init__(self):
        logger.info("Loading Silero VAD...")
        try:
            torch.set_num_threads(1)
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            self.get_speech_timestamps = utils[0]
            self.read_audio = utils[2]
            self.sample_rate = 16000
            self.enabled = True
            logger.info("✓ Silero VAD loaded!")
        except Exception as e:
            logger.error(f"VAD failed: {e}")
            self.enabled = False
    
    def has_speech(self, wav_path: str) -> bool:
        """Check if audio contains speech"""
        if not self.enabled:
            return True
        
        try:
            wav = self.read_audio(wav_path, sampling_rate=self.sample_rate)
            speech_timestamps = self.get_speech_timestamps(
                wav, self.model,
                threshold=0.5,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=250
            )
            
            has_speech = len(speech_timestamps) > 0
            logger.info(f"VAD: {'✓ Speech detected' if has_speech else '✗ No speech'}")
            return has_speech
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True

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
    """Processes audio with VAD filtering"""
    def __init__(self, whisper_model, vad, grammar):
        self.model = whisper_model
        self.vad = vad
        self.grammar = grammar
        
        self.audio_buffer = []
        self.buffer_duration = 3.0  # Process every 3 seconds
        
        self.stats = {"processed": 0, "skipped_vad": 0, "skipped_empty": 0}
        logger.info("AudioProcessor WITH VAD ready")
    
    def add_audio(self, pcm_data: bytes):
        """Add PCM audio chunk"""
        self.audio_buffer.append(pcm_data)
        
        # Calculate duration (16-bit PCM, 16kHz, mono)
        total_bytes = sum(len(chunk) for chunk in self.audio_buffer)
        duration = (total_bytes / 2) / 16000  # bytes / sample_width / sample_rate
        
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
            
            # Save as WAV
            wav_path = tempfile.mktemp(suffix=".wav")
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(combined)
            
            # VAD check
            if not self.vad.has_speech(wav_path):
                logger.info("⚠️ Skipped by VAD (no speech)")
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
        
        self.whisper = whisper.load_model("base")
        logger.info("✓ Whisper loaded")
        
        self.vad = SileroVAD()
        self.grammar = GrammarCorrector()
        
        logger.info("="*60)
        logger.info(" SYSTEM READY WITH VAD")
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
            # Add audio
            should_process = processor.add_audio(audio_data)
            
            # Process when buffer full
            if should_process:
                result = await processor.process()
                
                if result.get('text'):
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result['text'],
                        "stats": result.get('stats', {})
                    })
                    logger.info(f"✓ Sent: '{result['text']}'")
                elif result.get('reason') == 'no_speech':
                    await websocket.send_json({
                        "type": "info",
                        "message": "No speech detected"
                    })
        except Exception as e:
            logger.error(f"Error: {e}")

manager = ConnectionManager()

@app.get("/")
async def root():
    return {
        "service": "Speech-to-Text WITH VAD",
        "status": "running",
        "vad": manager.vad.enabled,
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
            "message": "Ready WITH VAD"
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
    print(" SPEECH-TO-TEXT WITH VAD")
    print(" - Filters noise/silence")
    print(" - Processes only speech")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
