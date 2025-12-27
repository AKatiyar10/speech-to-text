from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import os
import tempfile
from typing import Optional
import whisper
import logging
import time
import torch
import torchaudio
from collections import deque
import ollama

# Silero VAD for voice activity detection
torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Speech-to-Text API with VAD & Grammar Correction")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class VoiceActivityDetector:
    """
    Silero VAD - Enterprise-grade voice activity detection
    Filters out non-speech audio to save CPU resources
    """
    def __init__(self):
        logger.info("Loading Silero VAD model...")
        
        # Load Silero VAD model (lightweight, <1MB)
        self.model, self.utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=False
        )
        
        (self.get_speech_timestamps,
         self.save_audio,
         self.read_audio,
         self.VADIterator,
         self.collect_chunks) = self.utils
        
        self.sample_rate = 16000
        logger.info("✓ Silero VAD loaded successfully!")
    
    def contains_speech(self, audio_path: str, threshold: float = 0.5) -> bool:
        """
        Detect if audio contains human speech
        Returns True if speech detected, False otherwise
        """
        try:
            # Read audio file
            wav = self.read_audio(audio_path, sampling_rate=self.sample_rate)
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                wav, 
                self.model,
                threshold=threshold,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=250,  # Minimum speech duration
                min_silence_duration_ms=100  # Minimum silence between speeches
            )
            
            if len(speech_timestamps) > 0:
                # Calculate total speech duration
                total_speech_duration = sum(
                    (ts['end'] - ts['start']) / self.sample_rate 
                    for ts in speech_timestamps
                )
                logger.info(f"✓ Speech detected: {total_speech_duration:.2f}s")
                return True
            else:
                logger.info("✗ No speech detected (background noise/silence)")
                return False
                
        except Exception as e:
            logger.error(f"VAD error: {str(e)}")
            # On error, assume speech to avoid false negatives
            return True

class GrammarCorrector:
    """
    Local LLM-based grammar correction using Ollama
    Refines transcriptions for better readability
    """
    def __init__(self, model_name="gemma2:2b"):
        self.model_name = model_name
        self.client = ollama.Client(host='http://localhost:11434')  # Explicit host
        logger.info(f"Initializing Grammar Corrector with {model_name}")
        
        # Test Ollama connection
        try:
            self.client.list()
            logger.info("✓ Ollama connected successfully!")
        except Exception as e:
            logger.error(f"Ollama connection failed: {e}")
            logger.error("Make sure Ollama is running: 'ollama serve'")
    
    async def correct_text(self, text: str) -> str:
        """
        Correct grammar and improve text readability
        Using local LLM for privacy and no API costs
        """
        if not text or len(text.strip()) < 3:
            return text
        
        try:
            logger.info(f"Correcting grammar for: '{text[:50]}...'")
            
            # Optimized prompt for grammar correction
            prompt = f"""Fix only grammar, spelling, and punctuation errors in this text. 
Keep the original meaning and words as much as possible. 
Return ONLY the corrected text without explanations.

Text: {text}

Corrected:"""
            
            # Call local LLM
            response = await asyncio.to_thread(
                self.client.generate,
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.1,  # Low temperature for consistency
                    'top_p': 0.9,
                    'top_k': 40,
                    'num_predict': 256,  # Max tokens
                }
            )
            
            corrected_text = response['response'].strip()
            
            # Remove common LLM artifacts
            corrected_text = corrected_text.replace('Corrected:', '').strip()
            corrected_text = corrected_text.strip('"\'')
            
            logger.info(f"✓ Corrected: '{corrected_text[:50]}...'")
            return corrected_text
            
        except Exception as e:
            logger.error(f"Grammar correction error: {str(e)}")
            # Return original text if correction fails
            return text

class EnhancedWhisperProcessor:
    """
    Enhanced Whisper processor with VAD and grammar correction
    Optimized for CPU efficiency and text quality
    """
    def __init__(self, model_size="base", enable_vad=True, enable_grammar=True):
        logger.info(f"Loading Enhanced Whisper Processor...")
        
        # Load Whisper model
        self.model = whisper.load_model(model_size)
        logger.info(f"✓ Whisper {model_size} model loaded!")
        
        # Initialize VAD
        self.vad = VoiceActivityDetector() if enable_vad else None
        
        # Initialize Grammar Corrector
        self.grammar_corrector = GrammarCorrector() if enable_grammar else None
        
        # Buffer management
        self.audio_buffer = bytearray()
        self.buffer_size_threshold = 50000  # ~50KB
        self.last_process_time = time.time()
        self.min_time_between_transcriptions = 2.0
        
    async def add_audio_chunk(self, audio_data: bytes) -> Optional[dict]:
        """
        Add audio chunk and process when ready
        Returns dict with transcription and metadata
        """
        self.audio_buffer.extend(audio_data)
        
        current_time = time.time()
        time_since_last = current_time - self.last_process_time
        
        # Process if buffer threshold reached
        if (len(self.audio_buffer) >= self.buffer_size_threshold and 
            time_since_last >= self.min_time_between_transcriptions):
            
            result = await self._process_buffer()
            return result
        
        return None
    
    async def _process_buffer(self) -> Optional[dict]:
        """
        Process buffer with VAD filtering and grammar correction
        """
        if len(self.audio_buffer) == 0:
            return None
        
        try:
            logger.info(f"Processing buffer: {len(self.audio_buffer)} bytes")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(bytes(self.audio_buffer))
            
            # Step 1: Voice Activity Detection
            if self.vad:
                contains_speech = self.vad.contains_speech(temp_path, threshold=0.5)
                
                if not contains_speech:
                    logger.info("⚠️ Skipping transcription - no speech detected")
                    self.audio_buffer.clear()
                    self.last_process_time = time.time()
                    
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    
                    return {
                        "text": "",
                        "skipped": True,
                        "reason": "no_speech_detected",
                        "processing_time": 0
                    }
            
            # Step 2: Transcribe with Whisper
            start_time = time.time()
            transcription = await asyncio.to_thread(
                self._transcribe_audio,
                temp_path
            )
            
            # Step 3: Grammar Correction
            if transcription and self.grammar_corrector:
                transcription = await self.grammar_corrector.correct_text(transcription)
            
            processing_time = time.time() - start_time
            
            # Clear buffer
            self.audio_buffer.clear()
            self.last_process_time = time.time()
            
            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass
            
            return {
                "text": transcription,
                "skipped": False,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            self.audio_buffer.clear()
            return None
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe with Whisper
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
                logger.info(f"Raw transcription: '{text}'")
                
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return ""
    
    async def flush_buffer(self) -> Optional[dict]:
        """
        Force process remaining buffer
        """
        if len(self.audio_buffer) > 10000:
            return await self._process_buffer()
        return None

class ConnectionManager:
    def __init__(self, model_size="base", enable_vad=True, enable_grammar=True):
        self.active_connections = {}
        self.processors = {}
        self.connection_count = 0
        
        # Shared Whisper model
        logger.info("Initializing Enhanced System...")
        self.shared_model = whisper.load_model(model_size)
        
        # Shared VAD and Grammar Corrector
        self.vad = VoiceActivityDetector() if enable_vad else None
        self.grammar_corrector = GrammarCorrector() if enable_grammar else None
        
        logger.info("✓ Enhanced System Ready!")
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        # Create processor for this client
        processor = EnhancedWhisperProcessor("base", enable_vad=True, enable_grammar=True)
        processor.model = self.shared_model
        processor.vad = self.vad
        processor.grammar_corrector = self.grammar_corrector
        
        self.processors[client_id] = processor
        self.connection_count += 1
        logger.info(f"✓ Client {client_id} connected | Total: {self.connection_count}")
    
    async def disconnect(self, client_id: str):
        if client_id in self.processors:
            processor = self.processors[client_id]
            websocket = self.active_connections.get(client_id)
            
            final_result = await processor.flush_buffer()
            if final_result and final_result.get('text') and websocket:
                try:
                    await websocket.send_json({
                        "type": "transcription",
                        "text": final_result['text'],
                        "is_final": True,
                        "processing_time": final_result.get('processing_time', 0)
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
            result = await processor.add_audio_chunk(audio_data)
            
            if result:
                if result.get('skipped'):
                    # Send feedback that audio was skipped (optional)
                    await websocket.send_json({
                        "type": "info",
                        "message": "No speech detected - audio skipped",
                        "reason": result.get('reason')
                    })
                elif result.get('text'):
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result['text'],
                        "is_final": True,
                        "processing_time": result.get('processing_time', 0)
                    })
        except Exception as e:
            logger.error(f"Error: {str(e)}")

# Initialize with VAD and Grammar Correction enabled
manager = ConnectionManager(model_size="base", enable_vad=True, enable_grammar=True)

@app.get("/")
async def root():
    return {
        "service": "Enhanced Speech-to-Text API",
        "engine": "Whisper + Silero VAD + Local LLM",
        "features": [
            "Voice Activity Detection (filters non-speech)",
            "Grammar correction with local LLM",
            "100% offline & private",
            "CPU optimized"
        ],
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": manager.connection_count,
        "vad_enabled": manager.vad is not None,
        "grammar_correction_enabled": manager.grammar_corrector is not None
    }

@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Enhanced system ready - VAD + Grammar correction enabled",
            "client_id": client_id
        })
        
        while True:
            data = await websocket.receive_bytes()
            await manager.process_and_send(client_id, data)
            
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        await manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    print("=" * 80)
    print(" Enhanced Real-Time Speech-to-Text Server")
    print(" ✓ Voice Activity Detection (Silero VAD)")
    print(" ✓ Grammar Correction (Local LLM)")
    print(" ✓ 100% Offline & Private")
    print("=" * 80)
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
