from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import tempfile
from typing import Optional
import whisper
import logging
from collections import deque
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Local Speech-to-Text API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class WhisperProcessor:
    """
    Whisper processor with audio buffering
    Accumulates chunks before processing to ensure valid audio
    """
    def __init__(self, model_size="base"):
        logger.info(f"Loading Whisper model: {model_size}")
        self.model = whisper.load_model(model_size)
        logger.info(f"✓ Whisper model loaded successfully!")
        self.audio_buffer = bytearray()
        self.buffer_size_threshold = 50000  # ~50KB minimum
        self.last_process_time = time.time()
        self.min_time_between_transcriptions = 2.0  # Process every 2 seconds minimum
        
    async def add_audio_chunk(self, audio_data: bytes) -> Optional[str]:
        """
        Add audio chunk to buffer and process when threshold reached
        """
        self.audio_buffer.extend(audio_data)
        logger.info(f"Buffer size: {len(self.audio_buffer)} bytes")
        
        current_time = time.time()
        time_since_last = current_time - self.last_process_time
        
        # Process if buffer is large enough AND enough time has passed
        if (len(self.audio_buffer) >= self.buffer_size_threshold and 
            time_since_last >= self.min_time_between_transcriptions):
            
            result = await self._process_buffer()
            return result
        
        return None
    
    async def _process_buffer(self) -> Optional[str]:
        """
        Process accumulated audio buffer
        """
        if len(self.audio_buffer) == 0:
            return None
        
        try:
            logger.info(f"Processing buffer of {len(self.audio_buffer)} bytes")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(bytes(self.audio_buffer))
            
            # Transcribe
            transcription = await asyncio.to_thread(
                self._transcribe_audio,
                temp_path
            )
            
            # Clear buffer after successful processing
            self.audio_buffer.clear()
            self.last_process_time = time.time()
            
            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass
            
            return transcription
            
        except Exception as e:
            logger.error(f"Error processing buffer: {type(e).__name__}: {str(e)}")
            # Clear buffer on error to prevent bad data accumulation
            self.audio_buffer.clear()
            return None
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio file with Whisper
        """
        try:
            result = self.model.transcribe(
                audio_path,
                language="en",
                fp16=False,
                temperature=0.0,
                condition_on_previous_text=False,
                verbose=False
            )
            
            text = result["text"].strip()
            
            if text:
                logger.info(f"✓ Transcription: {text}")
            else:
                logger.info("No speech detected")
                
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return ""
    
    async def flush_buffer(self) -> Optional[str]:
        """
        Force process whatever is in the buffer (called on disconnect)
        """
        if len(self.audio_buffer) > 10000:  # Only if we have some data
            return await self._process_buffer()
        return None

class ConnectionManager:
    def __init__(self, model_size="base"):
        self.active_connections = {}
        self.processors = {}
        logger.info("Initializing Whisper model...")
        # Each client gets their own processor for separate buffering
        self.model = whisper.load_model(model_size)
        logger.info("✓ Model loaded and ready!")
        self.connection_count = 0
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        # Create new processor instance for this client
        processor = WhisperProcessor("base")
        processor.model = self.model  # Share the loaded model
        self.processors[client_id] = processor
        self.connection_count += 1
        logger.info(f"✓ Client {client_id} connected | Total: {self.connection_count}")
    
    async def disconnect(self, client_id: str):
        # Try to process any remaining audio in buffer
        if client_id in self.processors:
            processor = self.processors[client_id]
            websocket = self.active_connections.get(client_id)
            
            final_text = await processor.flush_buffer()
            if final_text and websocket:
                try:
                    await websocket.send_json({
                        "type": "transcription",
                        "text": final_text,
                        "is_final": True
                    })
                except:
                    pass
        
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
            # Add chunk to buffer, may return transcription
            transcription = await processor.add_audio_chunk(audio_data)
            
            if transcription:
                await websocket.send_json({
                    "type": "transcription",
                    "text": transcription,
                    "is_final": True
                })
                logger.info(f"✓ Sent transcription to {client_id}")
        except Exception as e:
            logger.error(f"Error in process_and_send: {str(e)}")

manager = ConnectionManager(model_size="base")

@app.get("/")
async def root():
    return {
        "service": "Local Speech-to-Text API",
        "engine": "Whisper (Buffered)",
        "status": "running",
        "note": "Processes audio in 2-second intervals"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": manager.connection_count,
        "engine": "whisper-buffered"
    }

@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Connected - speak for 2-3 seconds for transcription",
            "client_id": client_id
        })
        
        while True:
            data = await websocket.receive_bytes()
            logger.info(f"Received {len(data)} bytes from {client_id}")
            await manager.process_and_send(client_id, data)
            
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"Error: {type(e).__name__}: {str(e)}")
        await manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    print("=" * 70)
    print(" Local Speech-to-Text Server (Buffered Mode)")
    print(" Speak for 2-3 seconds, then pause for transcription")
    print("=" * 70)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
