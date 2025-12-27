from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import io
import os
import json
from typing import Optional
from pydub import AudioSegment
from vosk import Model, KaldiRecognizer, SetLogLevel
import logging

# Disable Vosk logs (they're very verbose)
SetLogLevel(-1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local Real-Time Speech-to-Text API (Vosk)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoskProcessor:
    """
    Ultra-fast local speech recognition using Vosk
    Optimized for real-time streaming with low latency (50-200ms)
    Perfect for low-resource environments (Raspberry Pi, old laptops)
    """
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
        """
        Initialize Vosk model
        
        Download models from: https://alphacephei.com/vosk/models
        
        Recommended models:
        - vosk-model-small-en-us-0.15 (40MB) - Fast, good for real-time
        - vosk-model-en-us-0.22 (1.8GB) - High accuracy
        - vosk-model-en-us-0.22-lgraph (128MB) - Balanced
        """
        model_dir = os.path.join("models", model_path)
        
        if not os.path.exists(model_dir):
            logger.error(f"Model not found at {model_dir}")
            logger.info("Download models from: https://alphacephei.com/vosk/models")
            logger.info("Extract to ./models/ directory")
            raise FileNotFoundError(f"Vosk model not found: {model_dir}")
        
        logger.info(f"Loading Vosk model from {model_dir}")
        self.model = Model(model_dir)
        self.sample_rate = 16000
        logger.info("Vosk model loaded successfully!")
        
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """
        Process audio chunk with Vosk
        Vosk is designed for streaming, so it's very fast
        """
        try:
            # Convert WebM to WAV
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_data),
                format="webm"
            )
            
            # Vosk requires 16kHz mono
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            # Get raw audio data
            audio_bytes = audio_segment.raw_data
            
            # Transcribe with Vosk
            transcription = await asyncio.to_thread(
                self._transcribe_audio,
                audio_bytes
            )
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error processing audio: {type(e).__name__}: {str(e)}")
            return None
    
    def _transcribe_audio(self, audio_bytes: bytes) -> str:
        """
        Synchronous transcription with Vosk
        Vosk is extremely fast - typically 50-200ms latency
        """
        try:
            # Create recognizer for this audio chunk
            recognizer = KaldiRecognizer(self.model, self.sample_rate)
            recognizer.SetWords(True)  # Get word-level timestamps if needed
            
            # Process audio
            if recognizer.AcceptWaveform(audio_bytes):
                result = json.loads(recognizer.Result())
                text = result.get("text", "").strip()
            else:
                # Get partial result if not complete
                result = json.loads(recognizer.PartialResult())
                text = result.get("partial", "").strip()
            
            if text:
                logger.info(f"Transcription: {text}")
                
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return ""

class ConnectionManager:
    """
    Manages WebSocket connections
    Vosk is so lightweight we can create separate instances per user
    """
    def __init__(self, model_path="vosk-model-small-en-us-0.15"):
        self.active_connections: dict[str, WebSocket] = {}
        self.processors: dict[str, VoskProcessor] = {}
        self.model_path = model_path
        self.connection_count = 0
        # Pre-load model for faster connection
        self.shared_processor = VoskProcessor(model_path)
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        self.processors[client_id] = self.shared_processor
        self.connection_count += 1
        logger.info(f"✓ Client {client_id} connected | Total: {self.connection_count}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            self.connection_count -= 1
        if client_id in self.processors:
            del self.processors[client_id]
        logger.info(f"✗ Client {client_id} disconnected | Total: {self.connection_count}")
    
    async def process_and_send(self, client_id: str, audio_data: bytes):
        processor = self.processors.get(client_id)
        websocket = self.active_connections.get(client_id)
        
        if not processor or not websocket:
            return
        
        try:
            transcription = await processor.process_audio_chunk(audio_data)
            if transcription:
                await websocket.send_json({
                    "type": "transcription",
                    "text": transcription,
                    "is_final": True
                })
        except Exception as e:
            logger.error(f"Error in process_and_send: {str(e)}")

manager = ConnectionManager(model_path="vosk-model-small-en-us-0.15")

@app.get("/")
async def root():
    return {
        "service": "Local Real-Time Speech-to-Text API",
        "engine": "Vosk",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": manager.connection_count,
        "engine": "vosk",
        "model": "local"
    }

@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to local speech-to-text service",
            "engine": "vosk",
            "client_id": client_id
        })
        
        while True:
            data = await websocket.receive_bytes()
            await manager.process_and_send(client_id, data)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
        manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Starting Local Real-Time Speech-to-Text Server")
    print("Engine: Vosk (Local, Ultra-Fast, Low Resource)")
    print("=" * 60)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
