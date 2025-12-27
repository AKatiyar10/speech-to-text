from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
import asyncio
import os
import tempfile
import whisper
import logging
import time
import wave
import numpy as np
import ollama
from enum import Enum

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Speech-to-Text with Feature Flags")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OutputMode(str, Enum):
    """Feature flags for output control"""
    RAW_ONLY = "raw"                    # Only raw Whisper output
    REFINED_ONLY = "refined"            # Only refined text
    REFINED_WITH_FEEDBACK = "full"      # Refined + feedback (default)
    ALL = "all"                         # Raw + Refined + Feedback

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

class EnhancedRefinementEngine:
    """Advanced refinement with speaking feedback"""
    def __init__(self, model_name="phi3:mini"):
        self.model_name = model_name
        logger.info(f"Loading Refinement Engine (model={model_name})...")
        try:
            self.client = ollama.Client(host='http://localhost:11434')
            self.client.generate(model=self.model_name, prompt="test", options={'num_predict': 1})
            self.enabled = True
            logger.info(f"✓ Refinement Engine enabled with {model_name}")
        except Exception as e:
            logger.warning(f"Refinement disabled: {e}")
            self.enabled = False
    
    async def refine_with_feedback(self, raw_text: str) -> dict:
        """
        Refine text AND provide speaking feedback
        Returns: {'refined_text': str, 'speaking_feedback': str, 'raw_text': str}
        """
        if not self.enabled or len(raw_text.strip()) < 5:
            return {
                'refined_text': raw_text,
                'speaking_feedback': 'Refinement engine disabled.',
                'raw_text': raw_text
            }
        
        try:
            start = time.time()
            logger.info(f"🔧 Refining: '{raw_text}'")
            
            prompt = f"""SYSTEM ROLE:
You are a careful editor and speaking coach. You take raw transcribed speech and:
1) rewrite it into natural, human-sounding prose, and
2) give concise, constructive feedback on how the speaker can improve.

PART 1 — REWRITE THE TEXT (FINAL OUTPUT, HUMAN-LIKE, NOT OBVIOUSLY AI):

TASK:
Refine the following transcribed speech into polished, natural text that reads as if written or spoken by a human, not by an AI.

REQUIREMENTS (DO NOT SKIP ANY POINT):

1. Correct language mechanics
   - Fix all grammar, punctuation, and capitalization errors.
   - Preserve the original language, tense, and person (I / you / we / they).

2. Improve sentence flow and readability
   - Restructure sentences only when needed to improve rhythm, clarity, and natural reading flow.
   - Avoid heavy rewrites; keep changes minimal but effective so it still feels like the same person speaking.

3. Context-aware word choice
   - Replace mispronounced, awkward, or slightly wrong words with more appropriate ones that fit the context and meaning of the sentence.
   - Use everyday, natural phrasing instead of "AI-sounding" wording.
   - Do NOT introduce new facts, ideas, or examples that were not implied in the original.

4. Remove disfluencies and fillers
   - Remove filler words and disfluencies such as: "um", "uh", "like", "you know", obvious false starts, and repeated fragments.
   - Keep hesitations only if they clearly add emotional tone or intent.

5. Maintain meaning and intent
   - Preserve the exact meaning, intent, and emotional tone of the original speech.
   - Do NOT change the speaker's opinion, stance, level of confidence, or formality.

6. Length-preserving refinement
   - Keep the overall length roughly the same (within about ±15% of the original character count).
   - Do not turn the text into a summary.
   - Do not expand it with explanations, commentary, or extra details.

7. Smooth transitions and coherence
   - Make sure transitions between clauses and sentences are smooth and coherent.
   - The final text must read as one unified, logically connected piece of speech, not as disjointed fragments.

8. Professional yet conversational style
   - Make it sound polished and confident, but still natural and conversational, as if spoken by a real person.
   - Avoid overly formal, academic, or robotic language unless the original clearly uses that style.

EFFICIENCY CONSTRAINTS (FOR LOW GPU/CPU COST):
- Apply all refinements in a single pass over the FINAL combined text only (not over interim/sub-sentences).
- Do not generate multiple options or drafts.
- Return ONLY the final refined text for PART 1, nothing else.

PART 2 — FEEDBACK ON SPEAKING SKILLS (BRIEF, ACTIONABLE ANALYSIS):

After producing the refined text in PART 1, analyze the ORIGINAL (unrefined) text and give brief coaching feedback to help the speaker improve.

REQUIREMENTS FOR FEEDBACK:

1. Focus of feedback
   - Comment on specific areas such as:
     - Overuse of filler words
     - Repetition or rambling
     - Confusing or vague wording
     - Grammar patterns that repeatedly cause problems
     - Pronunciation issues that led to wrong words in the transcript
     - Very long sentences that are hard to follow

2. Point to concrete examples
   - Refer to 2–5 specific phrases or patterns from the ORIGINAL text that show where the speaker can improve.
   - For each, briefly explain:
     - What the issue is.
     - How the speaker could say it better next time (in 1 short suggested alternative).

3. Keep feedback short and encouraging
   - Total feedback length: about 3–6 short bullet points.
   - Use a friendly, coaching tone (supportive, not judgmental).
   - Focus on practical, immediately usable tips.

4. Do NOT redo the full rewrite here
   - Do not rewrite the entire text again in the feedback section.
   - Just highlight patterns and show short example fixes.

OUTPUT FORMAT (STRICT):

1. First, output the refined text only (no labels, no headings).
2. Then add a blank line.
3. Then output a heading line exactly like this:
   SPEAKING FEEDBACK:
4. Then output 3–6 bullet points of feedback based on the ORIGINAL text.

Do not include any explanations about what you are doing. Do not mention these instructions in the output.

ORIGINAL TEXT:
{raw_text}"""
            
            response = await asyncio.to_thread(
                self.client.generate,
                model=self.model_name,
                prompt=prompt,
                options={
                    'temperature': 0.3,
                    'top_p': 0.9,
                    'top_k': 40,
                    'num_predict': 300,
                    'repeat_penalty': 1.1
                }
            )
            
            full_response = response['response'].strip()
            
            # Parse response into refined text and feedback
            if "SPEAKING FEEDBACK:" in full_response:
                parts = full_response.split("SPEAKING FEEDBACK:", 1)
                refined_text = parts[0].strip()
                speaking_feedback = parts[1].strip()
            else:
                refined_text = full_response
                speaking_feedback = "No feedback generated."
            
            refined_text = refined_text.strip('"\'').strip()
            
            if not refined_text or len(refined_text) < 3:
                logger.warning("Refinement empty, using original")
                refined_text = raw_text
                speaking_feedback = "Text too short for feedback."
            
            elapsed = time.time() - start
            logger.info(f"✓ Refined in {elapsed:.2f}s")
            
            return {
                'refined_text': refined_text,
                'speaking_feedback': speaking_feedback,
                'raw_text': raw_text
            }
            
        except Exception as e:
            logger.error(f"Refinement error: {e}")
            return {
                'refined_text': raw_text,
                'speaking_feedback': f"Error: {str(e)}",
                'raw_text': raw_text
            }

class ContinuousAudioProcessor:
    """Continuous transcription with feature flags"""
    def __init__(self, whisper_model, vad, refinement_engine, output_mode: OutputMode):
        self.model = whisper_model
        self.vad = vad
        self.refinement = refinement_engine
        self.output_mode = output_mode
        
        self.audio_buffer = []
        self.process_interval = 2.0
        
        self.accumulated_text = []
        self.silence_count = 0
        self.silence_threshold = 2
        
        self.stats = {
            "interim_count": 0,
            "final_count": 0,
            "skipped_vad": 0
        }
        logger.info(f"ContinuousAudioProcessor ready (mode={output_mode.value})")
    
    def add_audio(self, pcm_data: bytes):
        """Add PCM audio chunk"""
        self.audio_buffer.append(pcm_data)
        
        total_bytes = sum(len(chunk) for chunk in self.audio_buffer)
        duration = (total_bytes / 2) / 16000
        
        return duration >= self.process_interval
    
    async def process(self) -> dict:
        """Process audio - return interim"""
        if not self.audio_buffer:
            return {"text": "", "skipped": True, "interim": True}
        
        try:
            logger.info("="*60)
            
            combined = b''.join(self.audio_buffer)
            duration = (len(combined) / 2) / 16000
            logger.info(f"Processing {duration:.1f}s")
            
            if len(combined) < 16000:
                self.audio_buffer.clear()
                return {"text": "", "skipped": True, "interim": True}
            
            wav_path = tempfile.mktemp(suffix=".wav")
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(combined)
            
            has_speech = self.vad.has_speech(wav_path)
            
            if not has_speech:
                logger.info("⚠️ Silence detected")
                self.silence_count += 1
                self.audio_buffer.clear()
                os.remove(wav_path)
                
                if len(self.accumulated_text) > 0 and self.silence_count >= self.silence_threshold:
                    return await self._finalize_with_mode()
                
                self.stats["skipped_vad"] += 1
                return {"text": "", "skipped": True, "interim": True}
            
            self.silence_count = 0
            
            start = time.time()
            transcription = await asyncio.to_thread(self._transcribe, wav_path)
            whisper_time = time.time() - start
            
            os.remove(wav_path)
            self.audio_buffer.clear()
            
            if not transcription or len(transcription.strip()) < 3:
                logger.info(f"Empty transcription ({whisper_time:.2f}s)")
                return {"text": "", "skipped": True, "interim": True}
            
            logger.info(f"Whisper ({whisper_time:.2f}s): '{transcription}'")
            
            self.accumulated_text.append(transcription)
            self.stats["interim_count"] += 1
            
            current_full_text = " ".join(self.accumulated_text)
            logger.info(f"📝 Interim: '{current_full_text}'")
            
            return {
                "text": current_full_text,
                "skipped": False,
                "interim": True,
                "stats": self.stats
            }
            
        except Exception as e:
            logger.error(f"Error: {e}")
            self.audio_buffer.clear()
            return {"text": "", "skipped": True, "interim": True}
    
    async def _finalize_with_mode(self) -> dict:
        """
        Finalize based on output_mode feature flag
        """
        if not self.accumulated_text:
            return {"text": "", "skipped": True, "interim": False}
        
        try:
            raw_text = " ".join(self.accumulated_text)
            logger.info("="*60)
            logger.info(f"🎯 FINALIZING (mode={self.output_mode.value}): '{raw_text}'")
            
            result = {
                "skipped": False,
                "interim": False,
                "stats": self.stats
            }
            
            # RAW_ONLY: Just return raw Whisper
            if self.output_mode == OutputMode.RAW_ONLY:
                result["text"] = raw_text
                logger.info(f"✅ RAW_ONLY: '{raw_text}'")
            
            # REFINED_ONLY: Only refined text
            elif self.output_mode == OutputMode.REFINED_ONLY:
                if self.refinement.enabled:
                    refined_result = await self.refinement.refine_with_feedback(raw_text)
                    result["text"] = refined_result['refined_text']
                else:
                    result["text"] = raw_text
                logger.info(f"✅ REFINED_ONLY: '{result['text']}'")
            
            # REFINED_WITH_FEEDBACK: Refined + Feedback (no raw)
            elif self.output_mode == OutputMode.REFINED_WITH_FEEDBACK:
                if self.refinement.enabled:
                    refined_result = await self.refinement.refine_with_feedback(raw_text)
                    result["refined_text"] = refined_result['refined_text']
                    result["speaking_feedback"] = refined_result['speaking_feedback']
                else:
                    result["refined_text"] = raw_text
                    result["speaking_feedback"] = "Refinement disabled."
                logger.info(f"✅ REFINED_WITH_FEEDBACK")
            
            # ALL: Raw + Refined + Feedback
            elif self.output_mode == OutputMode.ALL:
                result["raw_text"] = raw_text
                if self.refinement.enabled:
                    refined_result = await self.refinement.refine_with_feedback(raw_text)
                    result["refined_text"] = refined_result['refined_text']
                    result["speaking_feedback"] = refined_result['speaking_feedback']
                else:
                    result["refined_text"] = raw_text
                    result["speaking_feedback"] = "Refinement disabled."
                logger.info(f"✅ ALL")
            
            logger.info("="*60)
            
            self.accumulated_text.clear()
            self.silence_count = 0
            self.stats["final_count"] += 1
            
            return result
            
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
            logger.info("🔚 Force finalizing...")
            return await self._finalize_with_mode()
        return {"text": "", "skipped": True}

class ConnectionManager:
    """Manages connections with feature flags"""
    def __init__(self, model_name="phi3:mini"):
        self.connections = {}
        self.processors = {}
        
        logger.info("="*60)
        logger.info(" LOADING MODELS")
        logger.info("="*60)
        
        self.whisper = whisper.load_model("small")
        logger.info("✓ Whisper loaded")
        
        self.vad = SimpleVAD(energy_threshold=0.02)
        self.refinement = EnhancedRefinementEngine(model_name=model_name)
        
        logger.info("="*60)
        logger.info(" SYSTEM READY: Feature Flags Enabled")
        logger.info("="*60)
    
    async def connect(self, websocket: WebSocket, client_id: str, output_mode: OutputMode):
        await websocket.accept()
        self.connections[client_id] = websocket
        
        processor = ContinuousAudioProcessor(
            self.whisper, 
            self.vad, 
            self.refinement,
            output_mode
        )
        self.processors[client_id] = processor
        
        logger.info(f"✓ Client connected: {client_id} (mode={output_mode.value})")
    
    async def disconnect(self, client_id: str):
        if client_id in self.processors:
            processor = self.processors[client_id]
            websocket = self.connections.get(client_id)
            
            final_result = await processor.force_finalize()
            if not final_result.get('skipped') and websocket:
                try:
                    await websocket.send_json({
                        "type": "transcription",
                        **final_result,
                        "is_final": True
                    })
                except:
                    pass
            
            logger.info(f"📊 Stats: {processor.stats}")
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
                
                if not result.get('skipped'):
                    await websocket.send_json({
                        "type": "transcription",
                        **result
                    })
                        
        except Exception as e:
            logger.error(f"Error: {e}")

manager = ConnectionManager(model_name="gemma2:2b")  # Change model here

@app.get("/")
async def root():
    return {
        "service": "Speech-to-Text with Feature Flags",
        "status": "running",
        "output_modes": {
            "raw": "Only raw Whisper output",
            "refined": "Only refined text",
            "full": "Refined + speaking feedback (default)",
            "all": "Raw + Refined + Feedback"
        },
        "usage": "ws://localhost:8000/ws/audio/{client_id}?mode=raw|refined|full|all"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "connections": len(manager.connections)
    }

@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    client_id: str,
    mode: Optional[str] = Query(default="full", regex="^(raw|refined|full|all)$")
):
    """
    WebSocket endpoint with feature flag support
    
    Query Parameters:
    - mode: raw | refined | full | all (default: full)
    
    Examples:
    - ws://localhost:8000/ws/audio/client123?mode=raw
    - ws://localhost:8000/ws/audio/client123?mode=refined
    - ws://localhost:8000/ws/audio/client123?mode=full
    - ws://localhost:8000/ws/audio/client123?mode=all
    """
    output_mode = OutputMode(mode)
    await manager.connect(websocket, client_id, output_mode)
    
    try:
        await websocket.send_json({
            "type": "connection",
            "message": f"Ready (mode={output_mode.value})"
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
    print(" SPEECH-TO-TEXT WITH FEATURE FLAGS")
    print(" Modes:")
    print("   ?mode=raw      → Raw Whisper only")
    print("   ?mode=refined  → Refined text only")
    print("   ?mode=full     → Refined + Feedback (default)")
    print("   ?mode=all      → Raw + Refined + Feedback")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
