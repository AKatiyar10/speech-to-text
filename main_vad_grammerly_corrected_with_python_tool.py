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
import language_tool_python

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
    Only processes audio that contains human speech
    """
    def __init__(self):
        logger.info("Loading Silero VAD model...")
        
        try:
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
            self.enabled = True
            logger.info("✓ Silero VAD loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load VAD: {e}")
            logger.warning("VAD disabled - will process all audio")
            self.enabled = False
    
    def contains_speech(self, audio_path: str, threshold: float = 0.5) -> tuple[bool, dict]:
        """
        Detect if audio contains human speech
        Returns (has_speech, metadata)
        """
        if not self.enabled:
            return True, {"reason": "vad_disabled"}
        
        try:
            # Read audio file
            wav = self.read_audio(audio_path, sampling_rate=self.sample_rate)
            
            # Get speech timestamps
            speech_timestamps = self.get_speech_timestamps(
                wav, 
                self.model,
                threshold=threshold,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100
            )
            
            if len(speech_timestamps) > 0:
                # Calculate speech metrics
                total_speech_duration = sum(
                    (ts['end'] - ts['start']) / self.sample_rate 
                    for ts in speech_timestamps
                )
                
                audio_duration = len(wav) / self.sample_rate
                speech_ratio = total_speech_duration / audio_duration if audio_duration > 0 else 0
                
                logger.info(f"✓ Speech detected: {total_speech_duration:.2f}s ({speech_ratio:.1%} of audio)")
                
                return True, {
                    "speech_duration": total_speech_duration,
                    "speech_ratio": speech_ratio,
                    "segments": len(speech_timestamps)
                }
            else:
                logger.info("✗ No speech detected (background noise/silence/music)")
                return False, {
                    "reason": "no_speech",
                    "audio_duration": len(wav) / self.sample_rate
                }
                
        except Exception as e:
            logger.error(f"VAD error: {str(e)}")
            # On error, assume speech to avoid false negatives
            return True, {"reason": "vad_error"}

class FastGrammarCorrector:
    """
    Fast rule-based grammar correction using LanguageTool
    Advantages over LLM:
    - 10-100x faster (50-200ms vs 500-2000ms)
    - No additional memory overhead
    - Deterministic results
    - Handles punctuation, spelling, grammar
    """
    def __init__(self, language='en-US'):
        logger.info("Loading LanguageTool grammar checker...")
        
        try:
            # Initialize LanguageTool
            self.tool = language_tool_python.LanguageTool(language)
            self.enabled = True
            logger.info("✓ LanguageTool loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load LanguageTool: {e}")
            logger.warning("Grammar correction disabled")
            self.enabled = False
    
    async def correct_text(self, text: str) -> tuple[str, dict]:
        """
        Correct grammar, spelling, and punctuation
        Returns (corrected_text, correction_metadata)
        """
        if not self.enabled or not text or len(text.strip()) < 3:
            return text, {"corrections": 0}
        
        try:
            start_time = time.time()
            logger.info(f"Correcting: '{text[:60]}...'")
            
            # Check for errors
            matches = await asyncio.to_thread(self.tool.check, text)
            
            if len(matches) == 0:
                logger.info("✓ No corrections needed")
                return text, {
                    "corrections": 0,
                    "processing_time": time.time() - start_time
                }
            
            # Apply corrections
            corrected_text = await asyncio.to_thread(
                language_tool_python.utils.correct,
                text,
                matches
            )
            
            processing_time = time.time() - start_time
            
            # Collect correction details
            correction_types = {}
            for match in matches:
                rule_id = match.ruleId
                correction_types[rule_id] = correction_types.get(rule_id, 0) + 1
            
            logger.info(f"✓ Applied {len(matches)} corrections in {processing_time:.3f}s")
            logger.info(f"  Original:  '{text}'")
            logger.info(f"  Corrected: '{corrected_text}'")
            
            return corrected_text, {
                "corrections": len(matches),
                "processing_time": processing_time,
                "correction_types": correction_types
            }
            
        except Exception as e:
            logger.error(f"Grammar correction error: {str(e)}")
            return text, {"corrections": 0, "error": str(e)}
    
    def close(self):
        """Clean up resources"""
        if self.enabled:
            self.tool.close()

class EnhancedWhisperProcessor:
    """
    Enhanced Whisper processor with:
    1. Voice Activity Detection (VAD) - filters non-speech
    2. Whisper Transcription - speech-to-text
    3. Grammar Correction - refines output
    
    Architecture optimized for 10+ years production use
    """
    def __init__(self, model_size="base", enable_vad=True, enable_grammar=True):
        logger.info(f"Initializing Enhanced Whisper Processor...")
        
        # Load Whisper model
        self.model = whisper.load_model(model_size)
        logger.info(f"✓ Whisper {model_size} model loaded!")
        
        # Initialize VAD
        self.vad = VoiceActivityDetector() if enable_vad else None
        self.vad_enabled = enable_vad and (self.vad.enabled if self.vad else False)
        
        # Initialize Grammar Corrector
        self.grammar_corrector = FastGrammarCorrector() if enable_grammar else None
        self.grammar_enabled = enable_grammar and (self.grammar_corrector.enabled if self.grammar_corrector else False)
        
        # Buffer management
        self.audio_buffer = bytearray()
        self.buffer_size_threshold = 50000  # ~50KB minimum
        self.last_process_time = time.time()
        self.min_time_between_transcriptions = 2.0  # Process every 2 seconds minimum
        
        # Statistics
        self.stats = {
            "total_chunks": 0,
            "processed_chunks": 0,
            "skipped_chunks": 0,
            "total_corrections": 0
        }
        
        logger.info(f"Configuration:")
        logger.info(f"  VAD: {'✓ Enabled' if self.vad_enabled else '✗ Disabled'}")
        logger.info(f"  Grammar: {'✓ Enabled' if self.grammar_enabled else '✗ Disabled'}")
        
    async def add_audio_chunk(self, audio_data: bytes) -> Optional[dict]:
        """
        Add audio chunk to buffer and process when threshold reached
        """
        self.audio_buffer.extend(audio_data)
        self.stats["total_chunks"] += 1
        
        logger.debug(f"Buffer size: {len(self.audio_buffer)} bytes")
        
        current_time = time.time()
        time_since_last = current_time - self.last_process_time
        
        # Process if buffer is large enough AND enough time has passed
        if (len(self.audio_buffer) >= self.buffer_size_threshold and 
            time_since_last >= self.min_time_between_transcriptions):
            
            result = await self._process_buffer()
            return result
        
        return None
    
    async def _process_buffer(self) -> Optional[dict]:
        """
        Process accumulated audio buffer through the pipeline:
        Buffer -> VAD -> Whisper -> Grammar -> Result
        """
        if len(self.audio_buffer) == 0:
            return None
        
        pipeline_start = time.time()
        
        try:
            logger.info(f"{'='*60}")
            logger.info(f"Processing buffer: {len(self.audio_buffer)} bytes")
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(bytes(self.audio_buffer))
            
            # Stage 1: Voice Activity Detection
            vad_start = time.time()
            has_speech = True
            vad_metadata = {}
            
            if self.vad_enabled:
                has_speech, vad_metadata = self.vad.contains_speech(temp_path, threshold=0.5)
                vad_time = time.time() - vad_start
                logger.info(f"VAD: {vad_time:.3f}s - Speech: {has_speech}")
                
                if not has_speech:
                    self.stats["skipped_chunks"] += 1
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
                        "vad_metadata": vad_metadata,
                        "processing_time": time.time() - pipeline_start
                    }
            
            # Stage 2: Whisper Transcription
            whisper_start = time.time()
            transcription = await asyncio.to_thread(
                self._transcribe_audio,
                temp_path
            )
            whisper_time = time.time() - whisper_start
            logger.info(f"Whisper: {whisper_time:.3f}s")
            
            if not transcription or len(transcription.strip()) < 2:
                logger.info("Empty transcription - skipping")
                self.audio_buffer.clear()
                self.last_process_time = time.time()
                
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                return {
                    "text": "",
                    "skipped": True,
                    "reason": "empty_transcription",
                    "processing_time": time.time() - pipeline_start
                }
            
            # Stage 3: Grammar Correction
            grammar_metadata = {}
            if self.grammar_enabled:
                grammar_start = time.time()
                corrected_text, grammar_metadata = await self.grammar_corrector.correct_text(transcription)
                grammar_time = time.time() - grammar_start
                logger.info(f"Grammar: {grammar_time:.3f}s - {grammar_metadata.get('corrections', 0)} corrections")
                
                transcription = corrected_text
                self.stats["total_corrections"] += grammar_metadata.get('corrections', 0)
            
            # Update statistics
            self.stats["processed_chunks"] += 1
            processing_time = time.time() - pipeline_start
            
            # Clear buffer
            self.audio_buffer.clear()
            self.last_process_time = time.time()
            
            # Cleanup
            try:
                os.remove(temp_path)
            except:
                pass
            
            logger.info(f"Total pipeline: {processing_time:.3f}s")
            logger.info(f"{'='*60}")
            
            return {
                "text": transcription,
                "skipped": False,
                "processing_time": processing_time,
                "vad_metadata": vad_metadata,
                "grammar_metadata": grammar_metadata,
                "stats": self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Processing error: {type(e).__name__}: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.audio_buffer.clear()
            return None
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """
        Transcribe audio with Whisper
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
                logger.info(f"Raw: '{text}'")
                
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return ""
    
    async def flush_buffer(self) -> Optional[dict]:
        """
        Force process remaining buffer (called on disconnect)
        """
        if len(self.audio_buffer) > 10000:
            logger.info("Flushing remaining buffer...")
            return await self._process_buffer()
        return None
    
    def get_stats(self) -> dict:
        """Get processing statistics"""
        return self.stats.copy()
    
    def cleanup(self):
        """Clean up resources"""
        if self.grammar_corrector:
            self.grammar_corrector.close()

class ConnectionManager:
    """
    Manages WebSocket connections with shared resources
    Optimized for multiple concurrent users
    """
    def __init__(self, model_size="base", enable_vad=True, enable_grammar=True):
        self.active_connections = {}
        self.processors = {}
        self.connection_count = 0
        
        logger.info("=" * 70)
        logger.info(" Initializing Enhanced Speech-to-Text System")
        logger.info("=" * 70)
        
        # Load shared Whisper model (memory efficient)
        logger.info("Loading shared Whisper model...")
        self.shared_whisper_model = whisper.load_model(model_size)
        
        # Load shared VAD model
        self.shared_vad = VoiceActivityDetector() if enable_vad else None
        
        # Load shared Grammar Corrector
        self.shared_grammar = FastGrammarCorrector() if enable_grammar else None
        
        self.enable_vad = enable_vad
        self.enable_grammar = enable_grammar
        
        logger.info("=" * 70)
        logger.info(" System Ready!")
        logger.info("=" * 70)
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        # Create processor for this client (shares models)
        processor = EnhancedWhisperProcessor(
            "base", 
            enable_vad=self.enable_vad, 
            enable_grammar=self.enable_grammar
        )
        processor.model = self.shared_whisper_model
        processor.vad = self.shared_vad
        processor.grammar_corrector = self.shared_grammar
        
        self.processors[client_id] = processor
        self.connection_count += 1
        logger.info(f"✓ Client {client_id} connected | Total: {self.connection_count}")
    
    async def disconnect(self, client_id: str):
        # Process any remaining audio
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
                        "metadata": {
                            "processing_time": final_result.get('processing_time', 0),
                            "stats": final_result.get('stats', {})
                        }
                    })
                except:
                    pass
            
            # Log final statistics
            stats = processor.get_stats()
            logger.info(f"Session stats for {client_id}:")
            logger.info(f"  Total chunks: {stats['total_chunks']}")
            logger.info(f"  Processed: {stats['processed_chunks']}")
            logger.info(f"  Skipped: {stats['skipped_chunks']}")
            logger.info(f"  Total corrections: {stats['total_corrections']}")
        
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
                    # Optionally send skip notification
                    await websocket.send_json({
                        "type": "info",
                        "message": f"Audio skipped: {result.get('reason')}",
                        "metadata": result
                    })
                elif result.get('text'):
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result['text'],
                        "is_final": True,
                        "metadata": {
                            "processing_time": result.get('processing_time', 0),
                            "corrections": result.get('grammar_metadata', {}).get('corrections', 0),
                            "stats": result.get('stats', {})
                        }
                    })
        except Exception as e:
            logger.error(f"Error in process_and_send: {str(e)}")

# Initialize system with VAD and Grammar enabled
manager = ConnectionManager(
    model_size="base", 
    enable_vad=True, 
    enable_grammar=True
)

@app.get("/")
async def root():
    return {
        "service": "Enhanced Speech-to-Text API",
        "version": "2.0.0",
        "engine": "Whisper + Silero VAD + LanguageTool",
        "features": [
            "Voice Activity Detection (filters non-speech audio)",
            "Grammar & punctuation correction",
            "Spelling correction",
            "100% offline & private",
            "CPU optimized",
            "Production-grade architecture"
        ],
        "status": "running"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "active_connections": manager.connection_count,
        "vad_enabled": manager.shared_vad is not None and manager.shared_vad.enabled,
        "grammar_enabled": manager.shared_grammar is not None and manager.shared_grammar.enabled,
        "whisper_model": "base"
    }

@app.get("/stats")
async def get_stats():
    """Get aggregated statistics across all connections"""
    total_stats = {
        "active_connections": manager.connection_count,
        "processors": {}
    }
    
    for client_id, processor in manager.processors.items():
        total_stats["processors"][client_id] = processor.get_stats()
    
    return total_stats

@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Enhanced system ready - VAD + Grammar correction enabled",
            "features": {
                "vad": manager.enable_vad,
                "grammar": manager.enable_grammar
            },
            "client_id": client_id
        })
        
        while True:
            data = await websocket.receive_bytes()
            await manager.process_and_send(client_id, data)
            
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected normally")
    except Exception as e:
        logger.error(f"WebSocket error: {type(e).__name__}: {str(e)}")
        await manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    print()
    print("=" * 80)
    print("  Enhanced Real-Time Speech-to-Text Server")
    print("=" * 80)
    print("  ✓ Whisper (base model) - Speech recognition")
    print("  ✓ Silero VAD - Voice activity detection (filters noise)")
    print("  ✓ LanguageTool - Fast grammar correction (50-200ms)")
    print("  ✓ 100% Offline & Private")
    print("=" * 80)
    print()
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
