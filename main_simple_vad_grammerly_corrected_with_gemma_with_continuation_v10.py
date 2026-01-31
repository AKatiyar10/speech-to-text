"""
Main application entry point - Speech-to-Text with Speaker Diarization.

This is the main server file that orchestrates all components.

Imports from separate modules for better organization:
- speaker_match.py: SpeakerMatch dataclass
- voice_embedding_engine.py: VoiceEmbeddingEngine class
- speaker_label_manager.py: SpeakerLabelManager class
- conversation_manager.py: ConversationManager class
- session_history_manager.py: SessionHistoryManager class
- simple_vad.py: SimpleVAD class
- refinement_engine.py: EnhancedRefinementEngine class
- audio_processor.py: ContinuousAudioProcessor class
- connection_manager.py: ConnectionManager class

To run the server:
    python main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10.py

The server will start on http://0.0.0.0:8000
"""
import sys
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict, Any
from pydantic import BaseModel
import asyncio
import os
import tempfile
import logging
import time
import uvicorn

# Import from separate modules
from speaker_match import SpeakerMatch
from voice_embedding_engine import VoiceEmbeddingEngine
from speaker_label_manager import SpeakerLabelManager
from conversation_manager import ConversationManager
from session_history_manager import SessionHistoryManager
from simple_vad import SimpleVAD
from refinement_engine import EnhancedRefinementEngine, OutputMode
from audio_processor import ContinuousAudioProcessor
from connection_manager import ConnectionManager

# ============ VERBOSE LOGGING ============
VERBOSE_LOGGING = True
# ===========================================

# Skip console encoding setup to avoid hanging
# if sys.platform == "win32":
#     import io
#     sys.stdout = io.TextIOWrapper(sys.stdout, encoding='utf-8', errors='replace')
#     sys.stderr = io.TextIOWrapper(sys.stderr, encoding='utf-8', errors='replace')

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


# Initialize connection manager (loads models)
print("Loading models and initializing components...", flush=True)
manager = ConnectionManager(model_name="phi3:mini", history_file="session_history.json")
print("[OK] All components initialized successfully", flush=True)


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
            return {
                "session_id": session_id,
                "speaking_feedback": session['speaking_feedback'],
                "timestamp": session.get('feedback_generated_at', ''),
                "status": "already_exists"
            }
        
        # Generate feedback
        if manager.refinement.enabled:
            feedback = await manager.refinement.generate_feedback(session['raw_text'])
        else:
            feedback = "Refinement engine disabled. Ollama must be running for feedback generation."
        
        # Update session
        updated = manager.history_manager.update_session_feedback(session_id, feedback)
        
        logger.info(f"✅ Feedback generated for session #{session_id}")
        logger.info("="*60)
        
        return {
            "session_id": session_id,
            "speaking_feedback": feedback,
            "timestamp": updated['feedback_generated_at'],
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/stats")
async def get_stats():
    """Get overall statistics"""
    return manager.history_manager.get_stats()


# ===== SPEAKER DIARIZATION API ENDPOINTS =====

@app.post("/api/speakers/enroll")
async def enroll_speaker(
    name: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """Enroll speaker with audio file"""
    try:
        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)

        with open(wav_path, 'wb') as f:
            content = await audio_file.read()
            f.write(content)

        embedding = await manager.voice_engine.extract_embedding(wav_path)
        result = await manager.voice_engine.enroll_speaker(name, wav_path)

        color = manager.speaker_manager.register_speaker(name, embedding)

        logger.info(f"✓ Speaker enrolled: {name}")

        return {
            "success": True,
            "speaker_name": name,
            "color": color,
            "enrolled": True,
            "confidence": 0.0
        }
    except Exception as e:
        logger.error(f"Enroll error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/speakers/enroll/{name}")
async def enroll_existing_speaker(name: str, audio_file: UploadFile = File(...)):
    """Enroll an existing speaker (from labels) with audio"""
    try:
        # Check if speaker exists in labels
        if name not in manager.speaker_manager.labels:
            raise HTTPException(status_code=404, detail=f"Speaker {name} not found in labels")

        wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
        os.close(wav_fd)

        with open(wav_path, 'wb') as f:
            content = await audio_file.read()
            f.write(content)

        embedding = await manager.voice_engine.extract_embedding(wav_path)
        result = await manager.voice_engine.enroll_speaker(name, wav_path)

        # Update speaker's last heard time
        manager.speaker_manager.update_last_heard(name)

        logger.info(f"✓ Speaker enrolled with audio: {name}")

        return {
            "success": True,
            "speaker_name": name,
            "enrolled": True,
            "color": manager.speaker_manager.get_label_color(name)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Enroll existing speaker error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/speakers/relabel")
async def relabel_speaker(
    old_name: str = Form(...),
    new_name: str = Form(...),
    color: str = Form(None)
):
    """Relabel speaker (UNKNOWN_XX -> ALPHA/Custom)"""
    try:
        success = manager.speaker_manager.relabel_speaker(
            old_name, new_name, color
        )
        if success:
            return {"success": True, "old_name": old_name, "new_name": new_name}
        else:
            raise HTTPException(status_code=404, detail=f"Speaker {old_name} not found")
    except Exception as e:
        logger.error(f"Relabel error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/speakers/list")
async def get_speakers():
    """Get all enrolled speakers"""
    speakers = manager.speaker_manager.get_all_labels()

    # Add enrollment status for each speaker
    enrolled_count = 0
    for name, data in speakers.items():
        enrolled = manager.voice_engine.is_speaker_enrolled(name)
        data["enrolled"] = enrolled
        if enrolled:
            enrolled_count += 1

    return {
        "speakers": speakers,
        "total_speakers": len(speakers),
        "total_enrolled": enrolled_count
    }


@app.delete("/api/speakers/{name}")
async def delete_speaker(name: str):
    """Delete a speaker"""
    try:
        success = manager.speaker_manager.delete_speaker(name)
        if success:
            return {"success": True, "name": name}
        else:
            raise HTTPException(status_code=404, detail=f"Speaker {name} not found")
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/conversations/recent")
async def get_recent_conversations(limit: int = 10):
    """Get recent conversation entries"""
    entries = manager.conversation_manager.get_conversation_history(limit=limit)
    return {"entries": entries}


@app.get("/api/conversations/context")
async def get_conversation_context(entries: int = 12):
    """Get conversation context for LLM"""
    context = manager.conversation_manager.get_recent_context(num_entries=entries)
    return {"context": context}


@app.get("/api/conversations/speaker/{name}")
async def get_conversations_by_speaker(name: str, limit: int = 50):
    """Get conversation history for a specific speaker"""
    entries = manager.conversation_manager.get_conversation_history(speaker=name, limit=limit)
    return {"entries": entries}


# ===== WEBSOCKET ENDPOINT =====
@app.websocket("/ws/audio/{client_id}")
async def websocket_endpoint(
    websocket: WebSocket,
    client_id: str,
    mode: Optional[str] = Query(default="all", regex="^(raw|refined|full|all)$")
):
    """WebSocket endpoint with proper cleanup"""
    output_mode = OutputMode(mode)

    try:
        if VERBOSE_LOGGING:
            logger.info(f"[VERBOSE] WebSocket connection request: client_id={client_id}, mode={output_mode.value}")

        await manager.connect(websocket, client_id, output_mode)

        if VERBOSE_LOGGING:
            logger.info(f"[VERBOSE] WebSocket connected: {client_id}")

        await websocket.send_json({
            "type": "connection",
            "message": f"Ready (mode={output_mode.value})"
        })

        while True:
            data = await websocket.receive_bytes()
            if VERBOSE_LOGGING:
                logger.info(f"[VERBOSE] Audio received from {client_id}: {len(data)} bytes")
            await manager.handle_audio(client_id, data)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected normally: {client_id}")
    except Exception as e:
        logger.error(f"WebSocket error for {client_id}: {e}")
    finally:
        # Always cleanup, even on unexpected errors
        await manager.disconnect(client_id)


if __name__ == "__main__":
    print("\n" + "="*60, flush=True)
    print(" SPEECH-TO-TEXT WITH ON-DEMAND FEEDBACK", flush=True)
    print(" Speaker Diarization Enabled", flush=True)
    print(f" Loaded {len(manager.history_manager.sessions)} sessions", flush=True)
    print(f" Voice engine: {'enabled' if manager.voice_engine.enabled else 'disabled'}", flush=True)
    print("="*60, flush=True)
    print("Starting server on http://0.0.0.0:8000", flush=True)
    print("Press Ctrl+C to stop", flush=True)
    print("", flush=True)

    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
