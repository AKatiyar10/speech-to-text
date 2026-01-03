#!/usr/bin/env python3
"""
Minimal test to isolate the server startup issue
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def test_step_by_step():
    print("=== STEP BY STEP TESTING ===")

    try:
        print("Step 1: Basic imports...")
        import asyncio
        import os
        import tempfile
        import logging
        import time
        import uvicorn
        print("   [OK] Basic imports")

        print("Step 2: FastAPI imports...")
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query, HTTPException, UploadFile, File, Form
        from fastapi.middleware.cors import CORSMiddleware
        from typing import Optional, Dict, Any
        from pydantic import BaseModel
        print("   [OK] FastAPI imports")

        print("Step 3: Module imports...")
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

        print("Step 4: App creation...")
        app = FastAPI(title="Test App")
        print("   [OK] App created")

        print("Step 5: Middleware...")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        print("   [OK] Middleware added")

        print("Step 6: Manager creation...")
        manager = ConnectionManager(model_name="phi3:mini", history_file="session_history.json")
        print("   [OK] Manager created")

        print("Step 7: Basic routes...")
        @app.get("/")
        async def root():
            return {"status": "ok", "sessions": len(manager.history_manager.sessions)}

        @app.get("/health")
        async def health():
            return {"status": "healthy"}
        print("   [OK] Basic routes added")

        print("\n=== SUCCESS ===")
        print("Minimal setup works. Issue must be in main file routes or initialization.")
        print("Let's check the main file step by step...")

    except Exception as e:
        print(f"\n[ERROR] Failed at step: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_step_by_step()