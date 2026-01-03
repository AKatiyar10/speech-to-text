#!/usr/bin/env python3
"""
Simple test script to verify server can start
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

def main():
    print("=== SERVER STARTUP TEST ===")

    try:
        # Test all imports
        print("1. Testing imports...")
        from speaker_match import SpeakerMatch
        from voice_embedding_engine import VoiceEmbeddingEngine
        from speaker_label_manager import SpeakerLabelManager
        from conversation_manager import ConversationManager
        from session_history_manager import SessionHistoryManager
        from simple_vad import SimpleVAD
        from refinement_engine import EnhancedRefinementEngine, OutputMode
        from audio_processor import ContinuousAudioProcessor
        from connection_manager import ConnectionManager
        print("   [OK] All module imports successful")

        # Test FastAPI
        print("2. Testing FastAPI...")
        from fastapi import FastAPI
        app = FastAPI()
        print("   [OK] FastAPI working")

        # Test manager creation
        print("3. Testing manager creation...")
        manager = ConnectionManager()
        print(f"   [OK] Manager created - {len(manager.history_manager.sessions)} sessions")

        # Test app creation
        print("4. Testing app setup...")
        from main_simple_vad_grammerly_corrected_with_gemma_with_continuation_v10 import app as main_app, manager as main_manager
        print("   [OK] Main app imported successfully")

        print("\n=== SUCCESS ===")
        print("All components are working correctly!")
        print("The server should start properly now.")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()