from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import io
import os
import tempfile
from typing import Optional
from pydub import AudioSegment
import whisper
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local Real-Time Speech-to-Text API (Whisper)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WhisperProcessor:
    """
    Local speech recognition using OpenAI Whisper
    100% offline, no API calls, completely free
    Works perfectly on Windows without compilation issues
    """
    def __init__(self, model_size="base"):
        """
        Initialize Whisper model
        
        Model sizes:
        - tiny: ~1GB RAM, fastest, 32M params
        - base: ~1GB RAM, good balance, 74M params (RECOMMENDED)
        - small: ~2GB RAM, better accuracy, 244M params
        - medium: ~5GB RAM, high accuracy, 769M params
        - large: ~10GB RAM, best accuracy, 1550M params
        """
        logger.info(f"Loading Whisper model: {model_size}")
        logger.info("First run will download the model (may take 2-5 minutes)")
        
        # Load model - downloads automatically on first run
        self.model = whisper.load_model(model_size)
        
        logger.info(f"✓ Whisper {model_size} model loaded successfully!")
        self.min_audio_length = 0.5  # Minimum 0.5 seconds
        
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """
        Process audio chunk and return transcription
        """
        try:
            # Convert WebM to WAV format
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_data),
                format="webm"
            )
            
            # Check minimum duration
            duration_seconds = len(audio_segment) / 1000.0
            if duration_seconds < self.min_audio_length:
                logger.debug(f"Audio too short: {duration_seconds:.2f}s")
                return None
            
            # Optimize for speech recognition (16kHz, mono)
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                temp_path = temp_file.name
                audio_segment.export(temp_path, format="wav")
            
            # Transcribe
            transcription = await asyncio.to_thread(
                self._transcribe_audio,
                temp_path
            )
            
            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error processing audio: {type(e).__name__}: {str(e)}")
            return None
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """
        Synchronous transcription with Whisper
        """
        try:
            # Transcribe with optimized parameters
            result = self.model.transcribe(
                audio_path,
                language="en",  # Set to None for auto-detection (99 languages)
                fp16=False,  # Use FP32 for CPU compatibility
                temperature=0.0,  # Deterministic results
                condition_on_previous_text=False,  # Better for real-time
                verbose=False,  # Reduce console output
                initial_prompt=None  # Can add context here
            )
            
            text = result["text"].strip()
            
            if text:
                logger.info(f"✓ Transcription: {text}")
            else:
                logger.debug("No speech detected in audio chunk")
                
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return ""

class ConnectionManager:
    """
    Manages WebSocket connections for multiple concurrent users
    """
    def __init__(self, model_size="base"):
        self.active_connections: dict[str, WebSocket] = {}
        self.processors: dict[str, WhisperProcessor] = {}
        
        # Initialize shared model instance (saves RAM)
        logger.info("Initializing shared Whisper model...")
        self.shared_processor = WhisperProcessor(model_size=model_size)
        
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        # Share the processor across connections
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
        """
        Process audio and send transcription back to client
        """
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
                    "is_final": True,
                    "timestamp": asyncio.get_event_loop().time()
                })
        except Exception as e:
            logger.error(f"Error in process_and_send: {str(e)}")
            try:
                await websocket.send_json({
                    "type": "error",
                    "message": "Transcription failed"
                })
            except:
                pass

# Initialize with base model
# Change to "tiny" for faster processing or "small" for better accuracy
manager = ConnectionManager(model_size="base")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Local Real-Time Speech-to-Text API",
        "engine": "OpenAI Whisper (Local)",
        "status": "running",
        "version": "1.0.0",
        "note": "100% offline, no API calls required"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_connections": manager.connection_count,
        "engine": "whisper-local",
        "model": "base"
    }

@app.get("/info")
async def info():
    """System information endpoint"""
    return {
        "supported_languages": "99 languages (auto-detection available)",
        "model_options": ["tiny", "base", "small", "medium", "large"],
        "current_model": "base",
        "features": [
            "Multi-language support (99 languages)",
            "Real-time streaming",
            "100% offline operation",
            "No API costs",
            "High accuracy"
        ]
    }

@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    Main WebSocket endpoint for real-time audio streaming
    """
    await manager.connect(websocket, client_id)
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to local speech-to-text service",
            "engine": "whisper-local",
            "client_id": client_id
        })
        
        while True:
            # Receive binary audio data from client
            data = await websocket.receive_bytes()
            
            # Process audio asynchronously
            await manager.process_and_send(client_id, data)
            
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"Unexpected error: {type(e).__name__}: {str(e)}")
        manager.disconnect(client_id)
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print(" Local Real-Time Speech-to-Text Server")
    print(" Engine: OpenAI Whisper (100% Local, No API Required)")
    print(" Supports 99 Languages | No Compilation Issues")
    print("=" * 70)
    print()
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        access_log=True
    )
