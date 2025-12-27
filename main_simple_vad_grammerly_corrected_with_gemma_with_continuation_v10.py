from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
from pydantic import BaseModel
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
from datetime import datetime
import json
from pathlib import Path
import threading


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = FastAPI(title="Speech-to-Text with On-Demand Feedback")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===== REQUEST/RESPONSE MODELS =====
class GenerateFeedbackRequest(BaseModel):
    session_id: int


class GenerateFeedbackResponse(BaseModel):
    session_id: int
    speaking_feedback: str
    timestamp: str
    status: str


# ===== SESSION HISTORY MANAGER =====
class SessionHistoryManager:
    """Manages session history in JSON file with thread-safe operations"""
    def __init__(self, history_file="session_history.json"):
        self.history_file = Path(history_file)
        self.sessions = []
        self._lock = threading.Lock()  # Thread-safe file operations
        self._session_cache = {}  # Cache for session lookups
        self._stats_cache = None
        self._cache_dirty = True
        self._load_history()
        self._migrate_old_sessions()
        logger.info(f"📁 Session history manager initialized: {self.history_file}")
    
    def _load_history(self):
        """Load existing history from file"""
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    self.sessions = json.load(f)
                # Build cache
                self._rebuild_cache()
                logger.info(f"✓ Loaded {len(self.sessions)} sessions from history")
            except Exception as e:
                logger.error(f"Error loading history: {e}")
                self.sessions = []
        else:
            logger.info("No existing history file, starting fresh")
            self.sessions = []
    
    def _rebuild_cache(self):
        """Rebuild session ID cache for O(1) lookups"""
        self._session_cache = {s['session_id']: s for s in self.sessions}
    
    def _migrate_old_sessions(self):
        """Migrate old sessions to new format"""
        migrated = False
        for session in self.sessions:
            if 'feedback_generated_at' not in session:
                session['feedback_generated_at'] = None
                migrated = True
            
            if session.get('speaking_feedback') and session.get('feedback_generated_at') is None:
                session['feedback_generated_at'] = session.get('timestamp', datetime.now().isoformat())
                migrated = True
        
        if migrated:
            self._save_history()
            logger.info(f"✓ Migrated {len(self.sessions)} sessions to new format")
    
    def add_session(self, raw_text: str, refined_text: str, client_id: str) -> Dict[str, Any]:
        """Add a new session to history WITHOUT feedback initially"""
        with self._lock:
            session_id = len(self.sessions) + 1
            now = datetime.now()
            
            session = {
                "session_id": session_id,
                "timestamp": now.isoformat(),
                "timestamp_readable": now.strftime("%Y-%m-%d %H:%M:%S"),
                "client_id": client_id,
                "raw_text": raw_text,
                "refined_text": refined_text,
                "speaking_feedback": None,
                "feedback_generated_at": None,
                "character_counts": {
                    "raw": len(raw_text),
                    "refined": len(refined_text),
                    "feedback": 0
                }
            }
            
            self.sessions.append(session)
            self._session_cache[session_id] = session  # Update cache
            self._save_history()
            
            logger.info(f"💾 Session #{session_id} saved (without feedback)")
            return session
    
    def update_session_feedback(self, session_id: int, feedback: str) -> Dict[str, Any]:
        """Update a session with generated feedback"""
        with self._lock:
            session = self._session_cache.get(session_id)
            if not session:
                raise ValueError(f"Session #{session_id} not found")
            
            session['speaking_feedback'] = feedback
            session['feedback_generated_at'] = datetime.now().isoformat()
            session['character_counts']['feedback'] = len(feedback)
            self._save_history()
            logger.info(f"✅ Session #{session_id} updated with feedback")
            return session
    
    def _save_history(self):
        """Save history to file (called within lock)"""
        try:
            with open(self.history_file, 'w', encoding='utf-8') as f:
                json.dump(self.sessions, f, indent=2, ensure_ascii=False)
            logger.info(f"✓ History saved ({len(self.sessions)} sessions)")
        except Exception as e:
            logger.error(f"Error saving history: {e}")
    
    def get_all_sessions(self):
        """Get all sessions"""
        return self.sessions
    
    def get_recent_sessions(self, limit=10):
        """Get recent N sessions (optimized slicing)"""
        return self.sessions[-limit:][::-1] if len(self.sessions) > 0 else []
    
    def get_session_by_id(self, session_id: int):
        """Get specific session by ID (O(1) lookup)"""
        return self._session_cache.get(session_id)
    
    def get_stats(self):
        """Get overall statistics (manual caching to avoid memory leak)"""
        if self._cache_dirty or self._stats_cache is None:
            if not self.sessions:
                self._stats_cache = {
                    "total_sessions": 0,
                    "total_words_spoken": 0,
                    "sessions_with_feedback": 0,
                    "average_session_length": 0,
                    "first_session": None,
                    "last_session": None
                }
            else:
                total_words = sum(len(s['raw_text'].split()) for s in self.sessions)
                avg_length = sum(s['character_counts']['raw'] for s in self.sessions) / len(self.sessions)
                sessions_with_feedback = sum(1 for s in self.sessions if s.get('speaking_feedback'))
                
                self._stats_cache = {
                    "total_sessions": len(self.sessions),
                    "total_words_spoken": total_words,
                    "sessions_with_feedback": sessions_with_feedback,
                    "average_session_length": round(avg_length, 2),
                    "first_session": self.sessions[0]['timestamp_readable'],
                    "last_session": self.sessions[-1]['timestamp_readable']
                }
            self._cache_dirty = False
        
        return self._stats_cache
    
    def invalidate_stats_cache(self):
        """Invalidate stats cache when sessions are modified"""
        self._cache_dirty = True


class OutputMode(str, Enum):
    """Feature flags for output control"""
    RAW_ONLY = "raw"
    REFINED_ONLY = "refined"
    REFINED_WITH_FEEDBACK = "full"
    ALL = "all"


class SimpleVAD:
    """Simple energy-based VAD with optimized audio processing"""
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
                sample_width = wf.getsampwidth()
                n_frames = wf.getnframes()
                audio_data = wf.readframes(n_frames)
            
            if sample_width != 2:
                return True
            
            # Optimized: Use numpy's efficient operations
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_normalized = audio_array.astype(np.float32) * (1.0 / 32768.0)
            rms_energy = np.sqrt(np.mean(np.square(audio_normalized)))
            has_speech = rms_energy > self.energy_threshold
            
            logger.info(f"VAD: Energy={rms_energy:.4f} {'✓ Speech' if has_speech else '✗ Silence'}")
            return has_speech
            
        except Exception as e:
            logger.error(f"VAD error: {e}")
            return True


class EnhancedRefinementEngine:
    """Advanced refinement with on-demand feedback and async lock for thread safety"""
    def __init__(self, model_name="phi3:mini"):
        self.model_name = model_name
        self.client = None
        self.enabled = False
        self._lock = None  # Will be initialized as asyncio.Lock() when event loop is available
        logger.info(f"Loading Refinement Engine (model={model_name})...")
        
        try:
            self.client = ollama.Client(host='http://localhost:11434')
            # Test connection
            self.client.generate(model=self.model_name, prompt="test", options={'num_predict': 1})
            self.enabled = True
            logger.info(f"✓ Refinement Engine enabled with {model_name}")
        except Exception as e:
            logger.warning(f"Refinement disabled: {e}")
            self.enabled = False
    
    def _get_lock(self):
        """Lazy initialization of asyncio.Lock (must be called from async context)"""
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock
    
    async def refine_text(self, raw_text: str) -> str:
        """Refine text ONLY (no feedback generation) - Thread-safe"""
        if not self.enabled or len(raw_text.strip()) < 5:
            return raw_text
        
        lock = self._get_lock()
        async with lock:  # Prevent concurrent access to ollama client
            try:
                start = time.time()
                logger.info(f"🔧 Refining text: '{raw_text[:50]}...'")
                
                prompt = f"""Refine this transcribed speech. without loosing Context, remove fillers (um, uh, like), keep meaning and tone. Output only refined text.

TEXT: {raw_text}"""
                
                response = await asyncio.to_thread(
                    self.client.generate,
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': 0.3,
                        'top_p': 0.9,
                        'top_k': 40,
                        'num_predict': 200,
                        'repeat_penalty': 1.1
                    }
                )
                
                refined_text = response['response'].strip().strip('"\'')
                
                if not refined_text or len(refined_text) < 3:
                    logger.warning("Refinement empty, using original")
                    refined_text = raw_text
                
                elapsed = time.time() - start
                logger.info(f"✓ Refined in {elapsed:.2f}s")
                
                return refined_text
                
            except Exception as e:
                logger.error(f"Refinement error: {e}")
                return raw_text
    
    async def generate_feedback(self, raw_text: str) -> str:
        """Generate speaking feedback ONLY (called on-demand) - Thread-safe"""
        if not self.enabled or len(raw_text.strip()) < 5:
            return "Text too short for meaningful feedback."
        
        lock = self._get_lock()
        async with lock:  # Prevent concurrent access
            try:
                start = time.time()
                logger.info(f"💡 Generating feedback for: '{raw_text[:50]}...'")
                
                prompt = f"""Analyze this speech and give 3-5 brief feedback points on filler words, clarity, effective. Be constructive.

TEXT: {raw_text}"""
                
                response = await asyncio.to_thread(
                    self.client.generate,
                    model=self.model_name,
                    prompt=prompt,
                    options={
                        'temperature': 0.4,
                        'top_p': 0.9,
                        'top_k': 40,
                        'num_predict': 250,
                        'repeat_penalty': 1.1
                    }
                )
                
                feedback = response['response'].strip()
                
                if not feedback or len(feedback) < 10:
                    feedback = "Great job! Your speech was clear and well-structured."
                
                elapsed = time.time() - start
                logger.info(f"✓ Feedback generated in {elapsed:.2f}s")
                
                return feedback
                
            except Exception as e:
                logger.error(f"Feedback generation error: {e}")
                return f"Error generating feedback: {str(e)}"


class ContinuousAudioProcessor:
    """Continuous transcription WITHOUT automatic feedback (optimized)"""
    def __init__(self, whisper_model, vad, refinement_engine, output_mode: OutputMode, 
                 history_manager: SessionHistoryManager, client_id: str):
        self.model = whisper_model
        self.vad = vad
        self.refinement = refinement_engine
        self.output_mode = output_mode
        self.history_manager = history_manager
        self.client_id = client_id
        
        # Pre-allocate buffer with reasonable size
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
        logger.info(f"ContinuousAudioProcessor ready (mode={output_mode.value}, client={client_id})")
    
    def add_audio(self, pcm_data: bytes) -> bool:
        """Add PCM audio chunk (optimized calculation)"""
        self.audio_buffer.append(pcm_data)
        
        # Optimized: Calculate duration only when needed
        total_bytes = sum(len(chunk) for chunk in self.audio_buffer)
        duration = total_bytes * 3.125e-5  # Equivalent to (total_bytes / 2) / 16000
        
        return duration >= self.process_interval
    
    async def process(self) -> Dict[str, Any]:
        """Process audio - return interim"""
        if not self.audio_buffer:
            return {"text": "", "skipped": True, "interim": True}
        
        wav_path = None
        try:
            logger.info("="*60)
            
            combined = b''.join(self.audio_buffer)
            duration = len(combined) * 3.125e-5
            logger.info(f"Processing {duration:.1f}s")
            
            if len(combined) < 16000:
                return {"text": "", "skipped": True, "interim": True}
            
            # Create temp file
            wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
            os.close(wav_fd)  # Close file descriptor immediately
            
            with wave.open(wav_path, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(combined)
            
            has_speech = self.vad.has_speech(wav_path)
            
            if not has_speech:
                logger.info("⚠️ Silence detected")
                self.silence_count += 1
                
                if len(self.accumulated_text) > 0 and self.silence_count >= self.silence_threshold:
                    return await self._finalize_with_mode()
                
                self.stats["skipped_vad"] += 1
                return {"text": "", "skipped": True, "interim": True}
            
            self.silence_count = 0
            
            start = time.time()
            transcription = await asyncio.to_thread(self._transcribe, wav_path)
            whisper_time = time.time() - start
            
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
            return {"text": "", "skipped": True, "interim": True}
        finally:
            # Always cleanup
            self.audio_buffer.clear()
            if wav_path:
                try:
                    os.remove(wav_path)
                except Exception as e:
                    logger.warning(f"Failed to remove temp file: {e}")
    
    async def _finalize_with_mode(self) -> Dict[str, Any]:
        """Finalize - Save to history WITHOUT feedback"""
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
            
            if self.refinement.enabled:
                refined_text = await self.refinement.refine_text(raw_text)
            else:
                refined_text = raw_text
            
            saved_session = self.history_manager.add_session(
                raw_text=raw_text,
                refined_text=refined_text,
                client_id=self.client_id
            )
            self.history_manager.invalidate_stats_cache()
            
            logger.info(f"💾 Session saved: #{saved_session['session_id']}")
            
            # Build result based on output mode
            result["session_id"] = saved_session['session_id']
            
            if self.output_mode == OutputMode.RAW_ONLY:
                result["text"] = raw_text
            elif self.output_mode == OutputMode.REFINED_ONLY:
                result["text"] = refined_text
            elif self.output_mode == OutputMode.REFINED_WITH_FEEDBACK:
                result["refined_text"] = refined_text
                result["speaking_feedback"] = None
            elif self.output_mode == OutputMode.ALL:
                result["raw_text"] = raw_text
                result["refined_text"] = refined_text
                result["speaking_feedback"] = None
            
            logger.info(f"✅ Session #{saved_session['session_id']} finalized")
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
        """Transcribe with Whisper (thread-safe, already wrapped with asyncio.to_thread in caller)"""
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
    
    async def force_finalize(self) -> Dict[str, Any]:
        """Force finalize on disconnect"""
        if len(self.accumulated_text) > 0:
            logger.info("🔚 Force finalizing...")
            return await self._finalize_with_mode()
        return {"text": "", "skipped": True, "interim": False}


class ConnectionManager:
    """Manages connections with optimized model loading"""
    def __init__(self, model_name="phi3:mini", history_file="session_history.json"):
        self.connections = {}
        self.processors = {}
        
        logger.info("="*60)
        logger.info(" LOADING MODELS")
        logger.info("="*60)
        
        # Load Whisper model once (blocking on startup is acceptable)
        self.whisper = whisper.load_model("base")
        logger.info("✓ Whisper loaded")
        
        self.vad = SimpleVAD(energy_threshold=0.02)
        self.refinement = EnhancedRefinementEngine(model_name=model_name)
        self.history_manager = SessionHistoryManager(history_file=history_file)
        
        logger.info("="*60)
        logger.info(" SYSTEM READY")
        logger.info("="*60)
    
    async def connect(self, websocket: WebSocket, client_id: str, output_mode: OutputMode):
        await websocket.accept()
        self.connections[client_id] = websocket
        
        processor = ContinuousAudioProcessor(
            self.whisper, 
            self.vad, 
            self.refinement,
            output_mode,
            self.history_manager,
            client_id
        )
        self.processors[client_id] = processor
        
        logger.info(f"✓ Client connected: {client_id}")
    
    async def disconnect(self, client_id: str):
        """Disconnect client and finalize session WITHOUT sending through websocket"""
        processor = self.processors.get(client_id)
        
        # Finalize session first (just save to history, don't try to send)
        if processor:
            try:
                final_result = await processor.force_finalize()
                if not final_result.get('skipped'):
                    logger.info(f"✅ Final session saved for {client_id}: #{final_result.get('session_id')}")
            except Exception as e:
                logger.error(f"Error finalizing session for {client_id}: {e}")
            finally:
                # Always cleanup processor
                del self.processors[client_id]
        
        # Cleanup websocket connection
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
                    try:
                        await websocket.send_json({
                            "type": "transcription",
                            **result
                        })
                    except Exception as e:
                        logger.warning(f"Failed to send message to {client_id}: {e}")
                        # Connection likely closed, will be handled by disconnect
                        
        except Exception as e:
            logger.error(f"Error handling audio for {client_id}: {e}")


manager = ConnectionManager(model_name="phi3:mini", history_file="session_history.json")


# ===== GRACEFUL SHUTDOWN =====
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down gracefully...")
    
    # Finalize all active sessions
    for client_id in list(manager.processors.keys()):
        await manager.disconnect(client_id)
    
    logger.info("Shutdown complete")


# ===== API ENDPOINTS =====
@app.get("/")
async def root():
    stats = manager.history_manager.get_stats()
    return {
        "service": "Speech-to-Text with On-Demand Feedback",
        "status": "running",
        "session_stats": stats
    }


@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "connections": len(manager.connections),
        "total_sessions": len(manager.history_manager.sessions)
    }


@app.get("/api/sessions")
async def get_all_sessions():
    """Get all sessions"""
    return {
        "total": len(manager.history_manager.sessions),
        "sessions": manager.history_manager.get_all_sessions()
    }


@app.get("/api/sessions/recent")
async def get_recent_sessions(limit: int = 10):
    """Get recent N sessions"""
    return {
        "limit": limit,
        "sessions": manager.history_manager.get_recent_sessions(limit)
    }


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: int):
    """Get specific session"""
    session = manager.history_manager.get_session_by_id(session_id)
    if session:
        return session
    raise HTTPException(status_code=404, detail=f"Session #{session_id} not found")


@app.post("/api/sessions/{session_id}/generate-feedback", response_model=GenerateFeedbackResponse)
async def generate_feedback_for_session(session_id: int):
    """
    🎯 Generate speaking feedback for a specific session on-demand
    """
    try:
        logger.info(f"="*60)
        logger.info(f"🎯 ON-DEMAND FEEDBACK REQUEST for session #{session_id}")
        
        session = manager.history_manager.get_session_by_id(session_id)
        if not session:
            raise HTTPException(status_code=404, detail=f"Session #{session_id} not found")
        
        # Check if feedback already exists
        if session.get('speaking_feedback'):
            logger.info(f"✓ Feedback already exists for session #{session_id}")
            timestamp = session.get('feedback_generated_at') or session.get('timestamp', datetime.now().isoformat())
            
            return GenerateFeedbackResponse(
                session_id=session_id,
                speaking_feedback=session['speaking_feedback'],
                timestamp=timestamp,
                status="already_exists"
            )
        
        raw_text = session['raw_text']
        logger.info(f"💡 Generating feedback for: '{raw_text[:50]}...'")
        
        feedback = await manager.refinement.generate_feedback(raw_text)
        
        updated_session = manager.history_manager.update_session_feedback(session_id, feedback)
        manager.history_manager.invalidate_stats_cache()
        
        logger.info(f"✅ Feedback generated and saved for session #{session_id}")
        logger.info(f"="*60)
        
        return GenerateFeedbackResponse(
            session_id=session_id,
            speaking_feedback=feedback,
            timestamp=updated_session['feedback_generated_at'],
            status="generated"
        )
        
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error generating feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating feedback: {str(e)}")


@app.get("/api/stats")
async def get_stats():
    """Get overall statistics"""
    return manager.history_manager.get_stats()


@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket, 
    client_id: str,
    mode: Optional[str] = Query(default="all", regex="^(raw|refined|full|all)$")
):
    """WebSocket endpoint with proper cleanup"""
    output_mode = OutputMode(mode)
    
    try:
        await manager.connect(websocket, client_id, output_mode)
        
        await websocket.send_json({
            "type": "connection",
            "message": f"Ready (mode={output_mode.value})"
        })
        
        while True:
            data = await websocket.receive_bytes()
            await manager.handle_audio(client_id, data)
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        # Always cleanup, even on unexpected errors
        await manager.disconnect(client_id)


if __name__ == "__main__":
    import uvicorn
    print("\n" + "="*60)
    print(" SPEECH-TO-TEXT WITH ON-DEMAND FEEDBACK")
    print(" 💾 Sessions: session_history.json")
    print(" 🔗 POST /api/sessions/{id}/generate-feedback")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
