from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import io
import os
from typing import Optional
from pydub import AudioSegment
from faster_whisper import WhisperModel
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local Real-Time Speech-to-Text API (Faster-Whisper)")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class FasterWhisperProcessor:
    """
    Local speech recognition using Faster-Whisper
    Optimized for production with GPU/CPU auto-detection
    4x faster than original Whisper with same accuracy
    """
    def __init__(self, model_size="base", device="cpu", compute_type="int8"):
        """
        Initialize Faster-Whisper model
        
        Args:
            model_size: "tiny", "base", "small", "medium", "large-v3"
                       tiny: 39M params, 1GB RAM, fastest
                       base: 74M params, 1GB RAM, good balance
                       small: 244M params, 2GB RAM, better accuracy
                       medium: 769M params, 5GB RAM, high accuracy
                       large-v3: 1550M params, 10GB RAM, best accuracy
            device: "cpu" or "cuda" (auto-detect GPU)
            compute_type: "int8" (CPU), "float16" (GPU), "float32" (highest quality)
        """
        logger.info(f"Loading Faster-Whisper model: {model_size} on {device}")
        
        # Auto-detect GPU if available
        try:
            import torch
            if torch.cuda.is_available() and device == "auto":
                device = "cuda"
                compute_type = "float16"
                logger.info("GPU detected! Using CUDA acceleration")
        except ImportError:
            device = "cpu"
            compute_type = "int8"
        
        # Initialize model - downloads automatically on first run
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
            download_root="./models",  # Store models locally
            num_workers=4  # Parallel processing
        )
        
        self.min_audio_length = 0.5  # Minimum 0.5 seconds
        logger.info(f"Model loaded successfully! Using {device} with {compute_type}")
        
    async def process_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """
        Process audio chunk and return transcription
        Uses streaming approach with VAD (Voice Activity Detection)
        """
        try:
            # Convert WebM to WAV
            audio_segment = AudioSegment.from_file(
                io.BytesIO(audio_data),
                format="webm"
            )
            
            # Check minimum duration
            duration_seconds = len(audio_segment) / 1000.0
            if duration_seconds < self.min_audio_length:
                return None
            
            # Optimize for speech recognition
            audio_segment = audio_segment.set_frame_rate(16000).set_channels(1)
            
            # Export to temporary WAV in memory
            wav_io = io.BytesIO()
            audio_segment.export(wav_io, format="wav")
            wav_io.seek(0)
            
            # Save temporarily for faster-whisper processing
            temp_path = "temp_audio.wav"
            with open(temp_path, "wb") as f:
                f.write(wav_io.getvalue())
            
            # Transcribe with faster-whisper
            transcription = await asyncio.to_thread(
                self._transcribe_audio,
                temp_path
            )
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error processing audio: {type(e).__name__}: {str(e)}")
            return None
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """
        Synchronous transcription with Faster-Whisper
        Includes VAD filtering and beam search optimization
        """
        try:
            # Transcribe with optimizations
            segments, info = self.model.transcribe(
                audio_path,
                language="en",  # Set to None for auto-detection
                beam_size=5,  # Higher = more accurate but slower (1-10)
                vad_filter=True,  # Voice Activity Detection - filters silence
                vad_parameters=dict(
                    min_speech_duration_ms=250,  # Minimum speech duration
                    min_silence_duration_ms=500  # Minimum silence to split
                ),
                temperature=0.0,  # Deterministic results
                compression_ratio_threshold=2.4,  # Reject bad audio
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=False  # Better for real-time
            )
            
            # Collect all segments
            transcription_parts = []
            for segment in segments:
                transcription_parts.append(segment.text.strip())
            
            result = " ".join(transcription_parts).strip()
            
            if result:
                logger.info(f"Transcription: {result}")
                
            return result
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            raise

class ConnectionManager:
    """
    Manages WebSocket connections for multiple concurrent users
    """
    def __init__(self, model_size="base"):
        self.active_connections: dict[str, WebSocket] = {}
        self.processors: dict[str, FasterWhisperProcessor] = {}
        self.shared_processor = FasterWhisperProcessor(
            model_size=model_size,
            device="cpu",  # Change to "auto" for GPU auto-detection
            compute_type="int8"
        )
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        # Share single processor instance across connections for efficiency
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
            await websocket.send_json({
                "type": "error",
                "message": "Transcription failed"
            })

# Initialize with base model (good balance of speed/accuracy)
# Options: "tiny", "base", "small", "medium", "large-v3"
manager = ConnectionManager(model_size="base")

@app.get("/")
async def root():
    return {
        "service": "Local Real-Time Speech-to-Text API",
        "engine": "Faster-Whisper",
        "status": "running",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": manager.connection_count,
        "engine": "faster-whisper",
        "model": "local"
    }

@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time audio streaming
    """
    await manager.connect(websocket, client_id)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Connected to local speech-to-text service",
            "engine": "faster-whisper",
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
        try:
            await websocket.close()
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("Starting Local Real-Time Speech-to-Text Server")
    print("Engine: Faster-Whisper (Local, No API Required)")
    print("=" * 60)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
