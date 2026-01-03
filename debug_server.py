#!/usr/bin/env python3
"""
Debug script to test server startup and diagnose issues.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_imports():
    """Test all imports step by step"""
    print("=== TESTING IMPORTS ===")

    try:
        print("1. Basic FastAPI imports...")
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
        print("✓ Basic imports OK")

        print("2. Module imports...")
        from speaker_match import SpeakerMatch
        from voice_embedding_engine import VoiceEmbeddingEngine
        from speaker_label_manager import SpeakerLabelManager
        from conversation_manager import ConversationManager
        from session_history_manager import SessionHistoryManager
        from simple_vad import SimpleVAD
        from refinement_engine import EnhancedRefinementEngine, OutputMode
        from audio_processor import ContinuousAudioProcessor
        from connection_manager import ConnectionManager
        print("✓ Module imports OK")

        print("3. App creation...")
        app = FastAPI(title="Speech-to-Text with On-Demand Feedback")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        print("✓ App created")

        print("4. Manager creation...")
        manager = ConnectionManager(model_name="phi3:mini", history_file="session_history.json")
        print("✓ Manager created")

        print("5. Route creation...")
        @app.get("/")
        async def root():
            stats = manager.history_manager.get_stats()
            return {
                "service": "Speech-to-Text with On-Demand Feedback",
                "status": "running",
                "session_stats": stats
            }
        print("✓ Routes added")

        print("\n=== STARTUP TEST ===")
        print("Starting server on http://0.0.0.0:8000")
        print("Press Ctrl+C to stop")

        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_imports()