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

app = FastAPI(title="Optimized Speech-to-Text")

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

class OptimizedGrammarCorrector:
    """
    Grammar correction - ONLY used for FINAL results
    Skipped for interim results to save CPU/GPU
    """
    def __init__(self):
        logger.info("Loading Grammar Corrector (finals only)...")
        try:
            self.client = ollama.Client(host='http://localhost:11434')
            self.client.generate(model="gemma2:2b", prompt="test", options={'num_predict': 1})
            self.enabled = True
            logger.info("✓ Grammar enabled (finals only)")
        except Exception as e:
            logger.warning(f"Grammar disabled: {e}")
            self.enabled = False
    
    async def refine_final(self, text: str) -> str:
        """
        ONLY called on final sentences
        Deep refinement: grammar, flow, word choice, professional polish
        """
        if not self.enabled or len(text.strip()) < 5:
            return text
        
        try:
            start = time.time()
            logger.info("🔧 Applying FINAL refinement...")
            
            prompt = f"""Transform this transcribed speech into polished, professional text.

REQUIREMENTS:
Refine this transcribed speech into polished, professional text.

REQUIREMENTS (OPTIMIZED FOR MINIMAL GPU COST, DO NOT SKIP ANY POINTS):

1. Correct language mechanics
   - Fix all grammar, punctuation, and capitalization errors.
   - Preserve the original language, tense, and person (I/you/we/etc.).

2. Improve sentence flow and readability
   - Restructure sentences only when needed to improve rhythm, clarity, and natural reading flow.
   - Avoid heavy rewrites; keep changes minimal but effective.

3. Context-aware word choice
   - Replace incorrectly pronounced, awkward, or imprecise words with more appropriate alternatives that fit the context and intent of the sentence.
   - Do NOT introduce new ideas or facts.

4. Remove disfluencies and fillers
   - Remove filler words and speech disfluencies such as “um”, “uh”, “like”, “you know”, obvious false starts, and repeated phrases.
   - Keep only what contributes to the intended message.

5. Maintain meaning and intent
   - Preserve the exact meaning, intent, and tone of the original speech.
   - Do NOT change the speaker’s opinion, attitude, or level of formality.

6. Length-preserving refinement
   - Keep the overall length of the text roughly the same (within about ±15% of the original character count).
   - Do not compress into a summary and do not expand with explanations.

7. Smooth transitions and coherence
   - Ensure transitions between clauses and sentences are smooth and coherent.
   - The final result should read as one unified, logically connected piece of text, not disjointed fragments.

8. Professional yet conversational style
   - Make the text sound polished and professional while still natural and conversational, as if spoken by a clear, confident speaker.
   - Avoid overly formal or academic language unless the input is already in that register.

CONSTRAINTS FOR EFFICIENCY (TO REDUCE GPU/CPU COST):
- Apply these refinements in a single pass over the FINAL text only (not on interim/sub-sentences).
- Do not generate alternative versions or explanations; return ONLY the final refined text.


TRANSCRIBED TEXT:
{text}

REFINED VERSION:"""
            
            response = await asyncio.to_thread(
                self.client.generate,
                model="gemma2:2b",
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'top_k': 40,
                    'num_predict': 200,
                    'repeat_penalty': 1.1,
                    'stop': ['\n\nTRANSCRIBED', '\n\nREQUIREMENTS']
                }
            )
            
            refined = response['response'].strip()
            
            # Clean up artifacts
            refined = refined.replace('REFINED VERSION:', '').strip()
            refined = refined.replace('Refined version:', '').strip()
            refined = refined.replace('REFINED TEXT:', '').strip()
            refined = refined.strip('"\'').strip()
            
            # Validate
            if not refined or len(refined) < 3:
                logger.warning("Refinement empty, using original")
                return text
            
            # Length check
            length_ratio = len(refined) / len(text)
            if length_ratio < 0.5 or length_ratio > 2.5:
                logger.warning(f"Length changed too much ({length_ratio:.1f}x), using original")
                return text
            
            elapsed = time.time() - start
            
            if refined != text:
                logger.info(f"✓ Refined ({elapsed:.2f}s):")
                logger.info(f"  Original: '{text}'")
                logger.info(f"  Refined:  '{refined}'")
            else:
                logger.info(f"No changes needed ({elapsed:.2f}s)")
            
            return refined
            
        except Exception as e:
            logger.error(f"Refinement error: {e}")
            return text

class OptimizedAudioProcessor:
    """
    Optimized processor:
    - Interim results: RAW Whisper output (fast, no grammar)
    - Final results: Whisper + Grammar refinement (slow, polished)
    """
    def __init__(self, whisper_model, vad, grammar):
        self.model = whisper_model
        self.vad = vad
        self.grammar = grammar
        
        self.audio_buffer = []
        self.process_interval = 2.0
        
        # Continuous transcription state
        self.accumulated_text = []
        self.silence_count = 0
        self.silence_threshold = 2
        
        self.stats = {
            "interim_count": 0,
            "final_count": 0,
            "skipped_vad": 0,
            "grammar_time_saved": 0  # Track CPU savings
        }
        logger.info("OptimizedAudioProcessor ready")
    
    def add_audio(self, pcm_data: bytes):
        """Add PCM audio chunk"""
        self.audio_buffer.append(pcm_data)
        
        total_bytes = sum(len(chunk) for chunk in self.audio_buffer)
        duration = (total_bytes / 2) / 16000
        
        return duration >= self.process_interval
    
    async def process(self) -> dict:
        """
        Process audio buffer
        Returns interim results WITHOUT grammar processing
        """
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
                
                # Finalize if enough silence
                if len(self.accumulated_text) > 0 and self.silence_count >= self.silence_threshold:
                    return await self._finalize_text()
                
                self.stats["skipped_vad"] += 1
                return {"text": "", "skipped": True, "interim": True}
            
            # Reset silence counter
            self.silence_count = 0
            
            # Transcribe with Whisper
            start = time.time()
            transcription = await asyncio.to_thread(self._transcribe, wav_path)
            whisper_time = time.time() - start
            
            os.remove(wav_path)
            self.audio_buffer.clear()
            
            if not transcription or len(transcription.strip()) < 3:
                logger.info(f"Empty transcription ({whisper_time:.2f}s)")
                return {"text": "", "skipped": True, "interim": True}
            
            logger.info(f"Whisper ({whisper_time:.2f}s): '{transcription}'")
            
            # Accumulate text
            self.accumulated_text.append(transcription)
            self.stats["interim_count"] += 1
            
            # Build current accumulated text (NO GRAMMAR)
            current_full_text = " ".join(self.accumulated_text)
            logger.info(f"📝 Interim (raw): '{current_full_text}'")
            
            # Estimate time saved by skipping grammar
            estimated_grammar_time = 2.0  # ~2 seconds per grammar call
            self.stats["grammar_time_saved"] += estimated_grammar_time
            logger.info(f"⚡ CPU saved: ~{estimated_grammar_time:.1f}s (total: {self.stats['grammar_time_saved']:.1f}s)")
            
            # Return RAW interim result
            return {
                "text": current_full_text,
                "skipped": False,
                "interim": True,
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
        Finalize accumulated text
        ONLY HERE we apply grammar refinement (expensive operation)
        """
        if not self.accumulated_text:
            return {"text": "", "skipped": True, "interim": False}
        
        try:
            # Combine all text
            full_text = " ".join(self.accumulated_text)
            logger.info("="*60)
            logger.info(f"🎯 FINALIZING: '{full_text}'")
            
            # Apply DEEP grammar refinement (ONLY on finals)
            if self.grammar.enabled:
                logger.info("⏳ Applying grammar refinement (this is expensive)...")
                refined_text = await self.grammar.refine_final(full_text)
                full_text = refined_text
            else:
                logger.info("⚠️ Grammar disabled, returning raw Whisper output")
            
            logger.info(f"✅ FINAL RESULT: '{full_text}'")
            logger.info("="*60)
            
            # Clear state
            self.accumulated_text.clear()
            self.silence_count = 0
            self.stats["final_count"] += 1
            
            return {
                "text": full_text,
                "skipped": False,
                "interim": False,
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
        """Force finalize on disconnect"""
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
        self.grammar = OptimizedGrammarCorrector()
        
        logger.info("="*60)
        logger.info(" OPTIMIZED MODE: Grammar only on finals")
        logger.info("="*60)
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.connections[client_id] = websocket
        
        processor = OptimizedAudioProcessor(self.whisper, self.vad, self.grammar)
        self.processors[client_id] = processor
        
        logger.info(f"✓ Client connected: {client_id}")
    
    async def disconnect(self, client_id: str):
        if client_id in self.processors:
            processor = self.processors[client_id]
            websocket = self.connections.get(client_id)
            
            # Force finalize
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
            
            logger.info(f"📊 Session stats: {processor.stats}")
            logger.info(f"⚡ Total CPU time saved: {processor.stats['grammar_time_saved']:.1f}s")
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
                        "is_final": not result.get('interim', False),
                        "stats": result.get('stats', {})
                    })
                    
                    if not result.get('interim'):
                        logger.info(f"✓ Sent FINAL (with grammar): '{result['text']}'")
                    else:
                        logger.info(f"⚡ Sent INTERIM (raw Whisper): '{result['text']}'")
                        
        except Exception as e:
            logger.error(f"Error: {e}")

manager = ConnectionManager()

@app.get("/")
async def root():
    return {
        "service": "Optimized Speech-to-Text",
        "status": "running",
        "optimization": "Grammar only on finals (CPU saving)",
        "interim": "Raw Whisper output (fast)",
        "final": "Whisper + Grammar refinement (polished)"
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
            "message": "Optimized mode: Raw interim, refined finals"
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
    print(" OPTIMIZED SPEECH-TO-TEXT SYSTEM")
    print(" ⚡ Interim: Raw Whisper (fast)")
    print(" 🔧 Finals: Whisper + Grammar (polished)")
    print(" 💰 CPU/GPU savings on interim results")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
