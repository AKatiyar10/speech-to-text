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

app = FastAPI(title="Speech-to-Text WITH Continuous Transcription")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimpleVAD:
    """Simple energy-based VAD"""
    def __init__(self, energy_threshold=0.01):
        self.energy_threshold = energy_threshold
        self.enabled = True
        logger.info(f"✓ Simple VAD loaded (threshold={energy_threshold})")
    
    def has_speech(self, wav_path: str) -> bool:
        """Check if audio contains speech based on energy"""
        if not self.enabled:
            return True
        
        try:
            with wave.open(wav_path, 'rb') as wf:
                audio_data = wf.readframes(wf.getnframes())
                sample_width = wf.getsampwidth()
            
            if sample_width == 2:
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
            else:
                return True
            
            audio_normalized = audio_array.astype(np.float32) / 32768.0
            rms_energy = np.sqrt(np.mean(audio_normalized ** 2))
            has_speech = rms_energy > self.energy_threshold
            
            logger.info(f"VAD: Energy={rms_energy:.4f} {'✓ Speech' if has_speech else '✗ Silence'}")
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
            prompt = f"""Refine this transcribed speech into polished, professional text.

REQUIREMENTS:
1. Improve sentence flow and natural rhythm
2. Use better word choices where appropriate
3. Maintain the exact same meaning and intent
4. Fix grammar, punctuation, and capitalization
5. Make it sound like professionally written text
6. Remove filler words (um, uh, like) if present

ORIGINAL TEXT:
{text}

REFINED TEXT:"""
            
            response = await asyncio.to_thread(
                self.client.generate,
                model="gemma2:2b",
                prompt=prompt,
                options={'temperature': 0.3, 'num_predict': 200, 'top_p': 0.9}
            )
            
            corrected = response['response'].strip()
            corrected = corrected.replace('REFINED TEXT:', '').strip()
            corrected = corrected.replace('Refined text:', '').strip()
            corrected = corrected.strip('"\'')
            
            if corrected and len(corrected) > 3:
                logger.info(f"Grammar: '{text}' → '{corrected}'")
                return corrected
            return text
        except Exception as e:
            logger.error(f"Grammar error: {e}")
            return text

class ContinuousAudioProcessor:
    """
    Continuous transcription processor
    KEY FEATURES:
    - Accumulates text from multiple audio chunks
    - Only sends complete text when silence is detected
    - Keeps building the sentence as person speaks
    """
    def __init__(self, whisper_model, vad, grammar):
        self.model = whisper_model
        self.vad = vad
        self.grammar = grammar
        
        # Audio buffering
        self.audio_buffer = []
        self.process_interval = 2.0  # Process every 2 seconds for interim updates
        
        # CONTINUOUS TRANSCRIPTION STATE
        self.accumulated_text = []  # Store all text segments
        self.silence_count = 0  # Count consecutive silence detections
        self.silence_threshold = 2  # Number of silent chunks before finalizing
        
        self.stats = {"processed": 0, "skipped_vad": 0, "finalized": 0}
        logger.info("ContinuousAudioProcessor ready - will accumulate text until silence")
    
    def add_audio(self, pcm_data: bytes):
        """Add PCM audio chunk"""
        self.audio_buffer.append(pcm_data)
        
        total_bytes = sum(len(chunk) for chunk in self.audio_buffer)
        duration = (total_bytes / 2) / 16000
        
        return duration >= self.process_interval
    
    async def process(self) -> dict:
        """Process audio buffer - CONTINUOUS MODE"""
        if not self.audio_buffer:
            return {"text": "", "skipped": True, "interim": True}
        
        try:
            logger.info("="*60)
            
            combined = b''.join(self.audio_buffer)
            duration = (len(combined) / 2) / 16000
            logger.info(f"Processing {duration:.1f}s ({len(combined)/1024:.1f}KB)")
            
            if len(combined) < 16000:
                self.audio_buffer.clear()
                return {"text": "", "skipped": True, "interim": True}
            
            # Save as WAV
            wav_path = tempfile.mktemp(suffix=".wav")
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(combined)
            
            # VAD check
            has_speech = self.vad.has_speech(wav_path)
            
            if not has_speech:
                logger.info("⚠️ Silence detected")
                self.silence_count += 1
                self.audio_buffer.clear()
                os.remove(wav_path)
                
                # If we have accumulated text and detected enough silence, finalize
                if len(self.accumulated_text) > 0 and self.silence_count >= self.silence_threshold:
                    return await self._finalize_text()
                
                self.stats["skipped_vad"] += 1
                return {"text": "", "skipped": True, "interim": True}
            
            # Reset silence counter when speech detected
            self.silence_count = 0
            
            # Transcribe
            start = time.time()
            transcription = await asyncio.to_thread(self._transcribe, wav_path)
            whisper_time = time.time() - start
            
            os.remove(wav_path)
            self.audio_buffer.clear()  # Clear buffer after processing
            
            if not transcription or len(transcription.strip()) < 3:
                logger.info(f"Empty transcription ({whisper_time:.2f}s)")
                return {"text": "", "skipped": True, "interim": True}
            
            logger.info(f"Whisper ({whisper_time:.2f}s): '{transcription}'")
            
            # ACCUMULATE text instead of sending immediately
            self.accumulated_text.append(transcription)
            self.stats["processed"] += 1
            
            # Build current accumulated sentence
            current_full_text = " ".join(self.accumulated_text)
            logger.info(f"📝 Accumulated so far: '{current_full_text}'")
            
            # Return interim result (can be displayed in real-time)
            return {
                "text": current_full_text,
                "skipped": False,
                "interim": True,  # Mark as interim (not final)
                "stats": self.stats
            }
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self.audio_buffer.clear()
            return {"text": "", "skipped": True, "interim": True}
    
    async def _finalize_text(self) -> dict:
        """
        Finalize the accumulated text when silence detected
        Apply grammar correction to the COMPLETE text
        """
        if not self.accumulated_text:
            return {"text": "", "skipped": True, "interim": False}
        
        try:
            # Combine all accumulated text
            full_text = " ".join(self.accumulated_text)
            logger.info("="*60)
            logger.info(f"🎯 FINALIZING: '{full_text}'")
            
            # Apply grammar correction to COMPLETE text
            if self.grammar.enabled:
                grammar_start = time.time()
                refined_text = await self.grammar.correct(full_text)
                grammar_time = time.time() - grammar_start
                logger.info(f"Grammar refinement ({grammar_time:.2f}s)")
                full_text = refined_text
            
            logger.info(f"✓ FINAL RESULT: '{full_text}'")
            logger.info("="*60)
            
            # Clear accumulated text
            self.accumulated_text.clear()
            self.silence_count = 0
            self.stats["finalized"] += 1
            
            return {
                "text": full_text,
                "skipped": False,
                "interim": False,  # Mark as FINAL
                "stats": self.stats
            }
            
        except Exception as e:
            logger.error(f"Finalization error: {e}")
            self.accumulated_text.clear()
            self.silence_count = 0
            return {"text": "", "skipped": True, "interim": False}
    
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
    
    async def force_finalize(self) -> dict:
        """Force finalize when client disconnects"""
        if len(self.accumulated_text) > 0:
            logger.info("🔚 Force finalizing on disconnect...")
            return await self._finalize_text()
        return {"text": "", "skipped": True}

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
        logger.info(" CONTINUOUS TRANSCRIPTION MODE READY")
        logger.info("="*60)
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.connections[client_id] = websocket
        
        processor = ContinuousAudioProcessor(self.whisper, self.vad, self.grammar)
        self.processors[client_id] = processor
        
        logger.info(f"✓ Client connected: {client_id}")
    
    async def disconnect(self, client_id: str):
        if client_id in self.processors:
            processor = self.processors[client_id]
            websocket = self.connections.get(client_id)
            
            # Force finalize any remaining text
            final_result = await processor.force_finalize()
            if final_result.get('text') and websocket:
                try:
                    await websocket.send_json({
                        "type": "transcription",
                        "text": final_result['text'],
                        "is_final": True,
                        "stats": final_result.get('stats', {})
                    })
                except:
                    pass
            
            logger.info(f"Stats: {processor.stats}")
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
                    # Send result with interim flag
                    await websocket.send_json({
                        "type": "transcription",
                        "text": result['text'],
                        "is_final": not result.get('interim', False),
                        "stats": result.get('stats', {})
                    })
                    
                    if not result.get('interim'):
                        logger.info(f"✓ Sent FINAL: '{result['text']}'")
                    else:
                        logger.info(f"📤 Sent INTERIM: '{result['text']}'")
                        
        except Exception as e:
            logger.error(f"Error: {e}")

manager = ConnectionManager()

@app.get("/")
async def root():
    return {
        "service": "Speech-to-Text WITH Continuous Transcription",
        "status": "running",
        "mode": "continuous",
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
            "message": "Ready - Continuous Mode (speak continuously, pause to finalize)"
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
    print(" CONTINUOUS SPEECH-TO-TEXT SYSTEM")
    print(" - Accumulates text while you speak")
    print(" - Sends complete text after pause")
    print(" - No broken sub-sentences")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
