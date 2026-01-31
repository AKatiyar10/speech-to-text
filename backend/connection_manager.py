"""
Connection manager for handling WebSocket connections.
"""
import asyncio
import logging
import whisper
from fastapi import WebSocket

from session_history_manager import SessionHistoryManager
from simple_vad import SimpleVAD
from refinement_engine import EnhancedRefinementEngine, OutputMode
from voice_embedding_engine import VoiceEmbeddingEngine
from speaker_label_manager import SpeakerLabelManager
from conversation_manager import ConversationManager
from audio_processor import ContinuousAudioProcessor

logger = logging.getLogger(__name__)

# ============ VERBOSE LOGGING ============
VERBOSE_LOGGING = True
# ===========================================


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

        # Speaker diarization components
        self.voice_engine = VoiceEmbeddingEngine()
        self.speaker_manager = SpeakerLabelManager()
        self.conversation_manager = ConversationManager()

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
            self.voice_engine,
            self.speaker_manager,
            self.conversation_manager,
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
                result = await processor.force_finalize()
                if not result.get('skipped'):
                    logger.info(f"💾 Force-finalized session #{result.get('session_id')}")
            except Exception as e:
                logger.error(f"Error force-finalizing: {e}")
        
        # Remove from connections
        if client_id in self.connections:
            try:
                await self.connections[client_id].close()
            except Exception:
                pass
            del self.connections[client_id]
        
        # Remove processor
        if client_id in self.processors:
            del self.processors[client_id]
        
        logger.info(f"✓ Client disconnected: {client_id}")
    
    async def handle_audio(self, client_id: str, audio_data: bytes):
        """Handle incoming audio"""
        processor = self.processors.get(client_id)
        websocket = self.connections.get(client_id)

        if not processor or not websocket:
            if VERBOSE_LOGGING:
                logger.warning(f"[VERBOSE] No processor or websocket for {client_id}")
            return

        try:
            should_process = processor.add_audio(audio_data)

            if VERBOSE_LOGGING and should_process:
                logger.info(f"[VERBOSE] Processing audio for {client_id}")

            if should_process:
                result = await processor.process()

                if VERBOSE_LOGGING:
                    logger.info(f"[VERBOSE] Process result for {client_id}: skipped={result.get('skipped')}, interim={result.get('interim')}, text={result.get('text', '')[:50]}")

                if not result.get('skipped'):
                    try:
                        await websocket.send_json({
                            "type": "transcription",
                            **result
                        })
                        if VERBOSE_LOGGING:
                            logger.info(f"[VERBOSE] Sent message to {client_id}: type=transcription, interim={result.get('interim')}")
                    except Exception as e:
                        logger.warning(f"Failed to send message to {client_id}: {e}")

        except Exception as e:
            logger.error(f"Error handling audio for {client_id}: {e}")
