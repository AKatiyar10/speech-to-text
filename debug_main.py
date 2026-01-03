#!/usr/bin/env python3
"""
Debug script to test server startup and diagnose issues.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_main_file():
    """Test importing the main file step by step"""
    print("=== TESTING MAIN FILE IMPORT ===")

    try:
        print("1. Testing basic imports...")
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
        print("   [OK] Basic imports")

        print("2. Testing module imports...")
        from speaker_match import SpeakerMatch
        from voice_embedding_engine import VoiceEmbeddingEngine
        from speaker_label_manager import SpeakerLabelManager
        from conversation_manager import ConversationManager
        from session_history_manager import SessionHistoryManager
        from simple_vad import SimpleVAD
        from refinement_engine import EnhancedRefinementEngine, OutputMode
        from audio_processor import ContinuousAudioProcessor
        from connection_manager import ConnectionManager
        print("   [OK] Module imports")

        print("3. Setting up logging...")
        VERBOSE_LOGGING = True
        print("   VERBOSE_LOGGING set")
        # Skip encoding setup for now to isolate the issue
        print("   Skipping encoding setup")
        logging.basicConfig(level=logging.INFO)
        print("   [OK] Logging setup")

        print("4. Creating app...")
        app = FastAPI(title="Speech-to-Text with On-Demand Feedback")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        print("   [OK] App created")

        print("5. Creating manager...")
        manager = ConnectionManager(model_name="phi3:mini", history_file="session_history.json")
        print("   [OK] Manager created")

        print("6. Adding basic routes...")
        @app.get("/")
        async def root():
            stats = manager.history_manager.get_stats()
            return {
                "service": "Speech-to-Text with On-Demand Feedback",
                "status": "running",
                "session_stats": stats
            }
        print("   [OK] Basic routes added")

        print("7. Adding health route...")
        @app.get("/health")
        async def health():
            return {
                "status": "healthy",
                "connections": len(manager.connections),
                "total_sessions": len(manager.history_manager.sessions)
            }
        print("   [OK] Health route added")

        print("\n=== TESTING API ENDPOINTS ===")
        print("Adding session routes...")
        @app.get("/api/sessions")
        async def get_all_sessions():
            return {
                "total": len(manager.history_manager.sessions),
                "sessions": manager.history_manager.get_all_sessions()
            }
        print("   [OK] Session routes added")

        print("Adding feedback routes...")
        class GenerateFeedbackRequest(BaseModel):
            session_id: int

        class GenerateFeedbackResponse(BaseModel):
            session_id: int
            speaking_feedback: str
            timestamp: str
            status: str

        @app.post("/api/sessions/{session_id}/generate-feedback", response_model=GenerateFeedbackResponse)
        async def generate_feedback_for_session(session_id: int):
            try:
                session = manager.history_manager.get_session_by_id(session_id)
                if not session:
                    raise HTTPException(status_code=404, detail=f"Session #{session_id} not found")

                if session.get('speaking_feedback'):
                    return {
                        "session_id": session_id,
                        "speaking_feedback": session['speaking_feedback'],
                        "timestamp": session.get('feedback_generated_at', ''),
                        "status": "already_exists"
                    }

                if manager.refinement.enabled:
                    feedback = await manager.refinement.generate_feedback(session['raw_text'])
                else:
                    feedback = "Refinement engine disabled. Ollama must be running for feedback generation."

                updated = manager.history_manager.update_session_feedback(session_id, feedback)

                return {
                    "session_id": session_id,
                    "speaking_feedback": feedback,
                    "timestamp": updated['feedback_generated_at'],
                    "status": "success"
                }

            except HTTPException:
                raise
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        print("   [OK] Feedback routes added")

        print("Adding speaker routes...")
        @app.post("/api/speakers/enroll")
        async def enroll_speaker(
            name: str = Form(...),
            audio_file: UploadFile = File(...)
        ):
            try:
                wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
                os.close(wav_fd)

                with open(wav_path, 'wb') as f:
                    content = await audio_file.read()
                    f.write(content)

                embedding = await manager.voice_engine.extract_embedding(wav_path)
                result = await manager.voice_engine.enroll_speaker(name, wav_path)

                color = manager.speaker_manager.register_speaker(name, embedding)

                return {
                    "success": True,
                    "speaker_name": name,
                    "color": color,
                    "confidence": 0.0
                }
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.post("/api/speakers/relabel")
        async def relabel_speaker(
            old_name: str = Form(...),
            new_name: str = Form(...),
            color: str = Form(None)
        ):
            try:
                success = manager.speaker_manager.relabel_speaker(
                    old_name, new_name, color
                )
                if success:
                    return {"success": True, "old_name": old_name, "new_name": new_name}
                else:
                    raise HTTPException(status_code=404, detail=f"Speaker {old_name} not found")
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))

        @app.get("/api/speakers/list")
        async def get_speakers():
            return {
                "speakers": manager.speaker_manager.get_all_labels()
            }
        print("   [OK] Speaker routes added")

        print("Adding WebSocket route...")
        @app.websocket("/ws/audio/{client_id}")
        async def websocket_endpoint(
            websocket: WebSocket,
            client_id: str,
            mode: Optional[str] = Query(default="all", regex="^(raw|refined|full|all)$")
        ):
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
                pass
            except Exception as e:
                print(f"WebSocket error for {client_id}: {e}")
            finally:
                await manager.disconnect(client_id)
        print("   [OK] WebSocket route added")

        print("\n=== SUCCESS ===")
        print("All routes added successfully!")
        print("Starting server...")

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_main_file()