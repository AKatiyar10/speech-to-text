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
import ollama
import subprocess

torch.set_num_threads(1)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Enhanced Speech-to-Text API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SileroVAD:
    """
    Voice Activity Detection with proper audio format conversion
    """
    def __init__(self):
        logger.info("Loading Silero VAD...")
        try:
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
            logger.info("✓ Silero VAD loaded!")
        except Exception as e:
            logger.error(f"VAD load failed: {e}")
            self.enabled = False
    
    def contains_speech(self, audio_path: str) -> tuple[bool, dict]:
        """
        Check if audio contains speech
        Converts to WAV first for proper VAD processing
        """
        if not self.enabled:
            return True, {"reason": "vad_disabled"}
        
        try:
            # Convert WebM to WAV for VAD processing
            wav_path = audio_path.replace('.webm', '_vad.wav')
            
            try:
                # Use FFmpeg to convert
                result = subprocess.run([
                    'ffmpeg', '-i', audio_path,
                    '-ar', '16000',  # 16kHz sample rate
                    '-ac', '1',       # Mono
                    '-y',             # Overwrite
                    wav_path
                ], capture_output=True, text=True, timeout=5)
                
                if result.returncode != 0:
                    logger.warning("FFmpeg conversion failed, skipping VAD")
                    return True, {"reason": "conversion_failed"}
                
            except Exception as conv_error:
                logger.warning(f"Audio conversion error: {conv_error}")
                return True, {"reason": "conversion_error"}
            
            # Now process WAV file with VAD
            wav = self.read_audio(wav_path, sampling_rate=self.sample_rate)
            
            speech_timestamps = self.get_speech_timestamps(
                wav, 
                self.model,
                threshold=0.5,
                sampling_rate=self.sample_rate,
                min_speech_duration_ms=250,
                min_silence_duration_ms=100
            )
            
            # Clean up WAV file
            try:
                os.remove(wav_path)
            except:
                pass
            
            if len(speech_timestamps) > 0:
                duration = sum((ts['end'] - ts['start']) / self.sample_rate for ts in speech_timestamps)
                logger.info(f"✓ Speech detected: {duration:.2f}s ({len(speech_timestamps)} segments)")
                return True, {
                    "speech_duration": duration,
                    "segments": len(speech_timestamps)
                }
            else:
                logger.info("✗ No speech detected (noise/silence)")
                return False, {"reason": "no_speech"}
                
        except Exception as e:
            logger.error(f"VAD error: {str(e)}")
            return True, {"reason": "vad_error"}

class GrammarCorrector:
    """
    Gemma-based grammar correction - simplified and robust
    """
    def __init__(self, model_name="gemma2:2b"):
        self.model_name = model_name
        logger.info(f"Initializing Grammar Corrector with {model_name}")
        
        try:
            self.client = ollama.Client(host='http://localhost:11434')
            
            # Simple connection test
            test_response = self.client.generate(
                model=self.model_name,
                prompt="test",
                options={'num_predict': 1}
            )
            
            logger.info("✓ Ollama connected and model working!")
            
        except Exception as e:
            logger.error(f"Ollama initialization failed: {e}")
            logger.error("Grammar correction will be disabled")
            self.client = None
    
    async def correct_text(self, text: str) -> tuple[str, dict]:
        """
        Correct grammar with Gemma
        """
        if not self.client or not text or len(text.strip()) < 3:
            return text, {"corrections": 0}
        
        try:
            start_time = time.time()
            logger.info(f"Correcting: '{text}'")
            
            # Optimized prompt
            prompt = f"""Fix grammar, punctuation, and capitalization. Keep original words and meaning. Output only the corrected text.

Input: {text}
Output:"""
            
            response = await asyncio.to_thread(
                self.client.generate,
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.0,
                    'top_p': 0.9,
                    'num_predict': 100,
                    'stop': ['\n\n', 'Input:', 'Output:']
                }
            )
            
            corrected_text = response['response'].strip()
            corrected_text = corrected_text.replace('Output:', '').strip().strip('"\'')
            
            processing_time = time.time() - start_time
            
            if corrected_text != text:
                logger.info(f"✓ Corrected in {processing_time:.2f}s: '{corrected_text}'")
            else:
                logger.info(f"✓ No changes ({processing_time:.2f}s)")
            
            return corrected_text, {
                "corrections": 1 if corrected_text != text else 0,
                "processing_time": processing_time
            }
            
        except Exception as e:
            logger.error(f"Grammar error: {str(e)}")
            return text, {"corrections": 0, "error": str(e)}


class EnhancedWhisperProcessor:
    """
    Optimized processor for better timing and accuracy
    """
    def __init__(self, model_size="base", enable_vad=True, enable_grammar=True):
        logger.info("Initializing Enhanced Processor...")
        
        self.model = whisper.load_model(model_size)
        logger.info(f"✓ Whisper {model_size} loaded!")
        
        self.vad = SileroVAD() if enable_vad else None
        self.vad_enabled = enable_vad and (self.vad.enabled if self.vad else False)
        
        self.grammar = GrammarCorrector() if enable_grammar else None
        self.grammar_enabled = enable_grammar and (self.grammar.client is not None if self.grammar else False)
        
        # OPTIMIZED BUFFERING for faster response
        self.audio_buffer = bytearray()
        self.buffer_size_threshold = 60000      # 60KB (medium size)
        self.last_process_time = time.time()
        self.min_time_between_transcriptions = 2.0  # 2 seconds
        
        self.stats = {
            "total_chunks": 0,
            "processed": 0,
            "skipped": 0,
            "corrections": 0
        }
        
        logger.info(f"VAD: {'✓ Enabled' if self.vad_enabled else '✗ Disabled'}")
        logger.info(f"Grammar: {'✓ Enabled (Gemma)' if self.grammar_enabled else '✗ Disabled'}")
        logger.info(f"Buffer: {self.buffer_size_threshold/1000:.0f}KB every {self.min_time_between_transcriptions}s")
    
    async def add_audio_chunk(self, audio_data: bytes) -> Optional[dict]:
        """Add chunk and process when ready"""
        self.audio_buffer.extend(audio_data)
        self.stats["total_chunks"] += 1
        
        current_time = time.time()
        time_since_last = current_time - self.last_process_time
        
        # Process when threshold reached
        if (len(self.audio_buffer) >= self.buffer_size_threshold and 
            time_since_last >= self.min_time_between_transcriptions):
            
            logger.info(f"Buffer ready: {len(self.audio_buffer)/1024:.1f}KB after {time_since_last:.1f}s")
            return await self._process_buffer()
        
        return None
    
    async def _process_buffer(self) -> Optional[dict]:
        """Process buffer through complete pipeline"""
        if len(self.audio_buffer) == 0:
            return None
        
        pipeline_start = time.time()
        
        try:
            logger.info("="*70)
            logger.info(f"PROCESSING: {len(self.audio_buffer)/1024:.1f}KB")
            
            # Save to temp file
            with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
                temp_path = f.name
                f.write(bytes(self.audio_buffer))
            
            file_size = os.path.getsize(temp_path)
            logger.info(f"Temp file: {file_size/1024:.1f}KB")
            
            # STAGE 1: Voice Activity Detection
            vad_start = time.time()
            has_speech = True
            vad_metadata = {}
            
            if self.vad_enabled:
                has_speech, vad_metadata = self.vad.contains_speech(temp_path)
                vad_time = time.time() - vad_start
                logger.info(f"VAD: {vad_time:.3f}s - Speech: {has_speech}")
                
                if not has_speech:
                    logger.info("⚠️ Skipping - no speech detected")
                    self.stats["skipped"] += 1
                    self.audio_buffer.clear()
                    self.last_process_time = time.time()
                    
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                    
                    return {
                        "text": "",
                        "skipped": True,
                        "reason": "no_speech",
                        "vad_metadata": vad_metadata
                    }
            
            # STAGE 2: Whisper Transcription
            whisper_start = time.time()
            transcription = await asyncio.to_thread(self._transcribe_audio, temp_path)
            whisper_time = time.time() - whisper_start
            logger.info(f"Whisper: {whisper_time:.3f}s")
            
            if not transcription or len(transcription.strip()) < 2:
                logger.info("⚠️ Empty transcription")
                self.stats["skipped"] += 1
                self.audio_buffer.clear()
                self.last_process_time = time.time()
                
                try:
                    os.remove(temp_path)
                except:
                    pass
                
                return {
                    "text": "",
                    "skipped": True,
                    "reason": "empty_transcription"
                }
            
            # STAGE 3: Grammar Correction
            grammar_metadata = {}
            if self.grammar_enabled:
                grammar_start = time.time()
                corrected, grammar_metadata = await self.grammar.correct_text(transcription)
                grammar_time = time.time() - grammar_start
                logger.info(f"Grammar: {grammar_time:.3f}s")
                
                transcription = corrected
                self.stats["corrections"] += grammar_metadata.get('corrections', 0)
            
            # Finalize
            self.stats["processed"] += 1
            total_time = time.time() - pipeline_start
            
            self.audio_buffer.clear()
            self.last_process_time = time.time()
            
            try:
                os.remove(temp_path)
            except:
                pass
            
            logger.info(f"✓ COMPLETE: {total_time:.2f}s | TEXT: '{transcription}'")
            logger.info("="*70)
            
            return {
                "text": transcription,
                "skipped": False,
                "processing_time": total_time,
                "vad_metadata": vad_metadata,
                "grammar_metadata": grammar_metadata,
                "stats": self.stats.copy()
            }
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            self.audio_buffer.clear()
            return None
    
    def _transcribe_audio(self, audio_path: str) -> str:
        """Transcribe with Whisper"""
        try:
            result = self.model.transcribe(
                audio_path,
                language="en",
                fp16=False,
                temperature=0.0,
                condition_on_previous_text=False,
                verbose=False,
                # IMPORTANT: Better accuracy settings
                beam_size=5,           # More accurate
                best_of=5,             # More accurate
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            
            text = result["text"].strip()
            
            if text:
                logger.info(f"Raw: '{text}'")
                
            return text
            
        except Exception as e:
            logger.error(f"Transcription error: {str(e)}")
            return ""
    
    async def flush_buffer(self) -> Optional[dict]:
        """Flush remaining audio"""
        if len(self.audio_buffer) > 20000:
            logger.info(f"Flushing: {len(self.audio_buffer)/1024:.1f}KB")
            return await self._process_buffer()
        return None
    
    def get_stats(self) -> dict:
        return self.stats.copy()

class ConnectionManager:
    """Manages WebSocket connections"""
    def __init__(self):
        self.active_connections = {}
        self.processors = {}
        self.connection_count = 0
        
        logger.info("="*70)
        logger.info(" INITIALIZING ENHANCED SYSTEM")
        logger.info("="*70)
        
        # Shared resources
        self.shared_model = whisper.load_model("base")
        self.shared_vad = SileroVAD()
        self.shared_grammar = GrammarCorrector()
        
        logger.info("="*70)
        logger.info(" SYSTEM READY")
        logger.info("="*70)
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        
        processor = EnhancedWhisperProcessor("base", True, True)
        processor.model = self.shared_model
        processor.vad = self.shared_vad
        processor.grammar = self.shared_grammar
        
        self.processors[client_id] = processor
        self.connection_count += 1
        logger.info(f"✓ Client connected: {client_id} | Total: {self.connection_count}")
    
    async def disconnect(self, client_id: str):
        if client_id in self.processors:
            processor = self.processors[client_id]
            websocket = self.active_connections.get(client_id)
            
            final = await processor.flush_buffer()
            if final and final.get('text') and websocket:
                try:
                    await websocket.send_json({
                        "type": "transcription",
                        "text": final['text'],
                        "is_final": True
                    })
                except:
                    pass
            
            stats = processor.get_stats()
            logger.info(f"Session stats: {stats}")
        
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            self.connection_count -= 1
        if client_id in self.processors:
            del self.processors[client_id]
        
        logger.info(f"✗ Client disconnected: {client_id}")
    
    async def process_and_send(self, client_id: str, audio_data: bytes):
        processor = self.processors.get(client_id)
        websocket = self.active_connections.get(client_id)
        
        if not processor or not websocket:
            return
        
        try:
            result = await processor.add_audio_chunk(audio_data)
            
            if result:
                if result.get('text'):
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
                    logger.info(f"✓ Sent to client: '{result['text']}'")
                elif result.get('skipped'):
                    logger.info(f"ℹ️ Skipped: {result.get('reason')}")
        except Exception as e:
            logger.error(f"Error: {str(e)}")

manager = ConnectionManager()

@app.get("/")
async def root():
    return {
        "service": "Enhanced Speech-to-Text",
        "engine": "Whisper + Silero VAD + Gemma 2B",
        "status": "running",
        "features": ["VAD", "Grammar Correction (Gemma)", "Smart Buffering"]
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "connections": manager.connection_count,
        "vad": manager.shared_vad.enabled,
        "grammar": manager.shared_grammar.client is not None
    }

@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": "Enhanced system ready - Speak clearly for 2-3 seconds",
            "client_id": client_id
        })
        
        while True:
            data = await websocket.receive_bytes()
            await manager.process_and_send(client_id, data)
            
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        await manager.disconnect(client_id)

if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*70)
    print(" ENHANCED SPEECH-TO-TEXT SERVER")
    print(" Whisper (Enhanced Accuracy) + Silero VAD + Gemma Grammar")
    print(" Optimized for Real-Time Response")
    print("="*70 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
