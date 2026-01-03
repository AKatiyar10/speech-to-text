"""
Continuous audio processor with speaker diarization support.
"""
import asyncio
import logging
import os
import tempfile
import time
import wave
from datetime import datetime
from typing import Dict, Any

from simple_vad import SimpleVAD
from refinement_engine import OutputMode
from session_history_manager import SessionHistoryManager
from voice_embedding_engine import VoiceEmbeddingEngine
from speaker_label_manager import SpeakerLabelManager
from conversation_manager import ConversationManager

logger = logging.getLogger(__name__)

# ============ VERBOSE LOGGING ============
VERBOSE_LOGGING = True
# ===========================================


class ContinuousAudioProcessor:
    """Continuous transcription WITHOUT automatic feedback (optimized)"""
    def __init__(self, whisper_model, vad: SimpleVAD, refinement_engine, voice_engine: VoiceEmbeddingEngine,
                 speaker_manager: SpeakerLabelManager, conversation_manager: ConversationManager,
                 output_mode: OutputMode, history_manager: SessionHistoryManager, client_id: str):
        self.model = whisper_model
        self.vad = vad
        self.refinement = refinement_engine
        self.voice_engine = voice_engine
        self.speaker_manager = speaker_manager
        self.conversation_manager = conversation_manager
        self.output_mode = output_mode
        self.history_manager = history_manager
        self.client_id = client_id

        # Pre-allocate buffer with reasonable size
        self.audio_buffer = []
        self.process_interval = 2.0

        self.accumulated_text = []
        self.silence_count = 0
        self.silence_threshold = 2

        # Speaker information for current session
        self.current_speaker = "UNKNOWN"
        self.current_speaker_color = self.speaker_manager.UNKNOWN_COLOR if speaker_manager else '#3b82f6'
        self.current_speaker_confidence = 0.0

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
            if VERBOSE_LOGGING:
                logger.info("[VERBOSE] process(): No audio buffer, skipping")
            return {"text": "", "skipped": True, "interim": True}

        wav_path = None
        try:
            if VERBOSE_LOGGING:
                logger.info("[VERBOSE] process() called with audio buffer")
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

            speaker_name = "UNKNOWN"
            speaker_color = self.speaker_manager.UNKNOWN_COLOR if self.speaker_manager else '#3b82f6'
            speaker_confidence = 0.0

            if self.voice_engine.enabled:
                try:
                    speaker_match = await self.voice_engine.identify_speaker(wav_path, confidence_threshold=0.75)

                    if speaker_match.is_match:
                        speaker_name = speaker_match.name
                        speaker_color = self.speaker_manager.get_label_color(speaker_name)
                        if VERBOSE_LOGGING:
                            logger.info(f"[VERBOSE] Recognized speaker: {speaker_name} (color: {speaker_color})")
                    else:
                        speaker_name = self.speaker_manager.get_or_create_unknown(speaker_match.embedding)
                        speaker_color = self.speaker_manager.UNKNOWN_COLOR
                        if VERBOSE_LOGGING:
                            logger.info(f"[VERBOSE] Unknown speaker assigned: {speaker_name} (color: {speaker_color})")

                    speaker_confidence = speaker_match.confidence
                    self.speaker_manager.update_last_heard(speaker_name)

                    # Store for final session save
                    self.current_speaker = speaker_name
                    self.current_speaker_color = speaker_color
                    self.current_speaker_confidence = speaker_confidence

                    if VERBOSE_LOGGING:
                        logger.info(f"[VERBOSE] Stored speaker info: name={self.current_speaker}, confidence={self.current_speaker_confidence:.4f}")

                    self.conversation_manager.add_entry(
                        speaker=speaker_name,
                        text=transcription,
                        confidence=speaker_confidence,
                        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        session_id=0,
                        duration=duration
                    )
                except Exception as e:
                    logger.error(f"Speaker identification error: {e}")
            
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
            if VERBOSE_LOGGING:
                logger.info("[VERBOSE] _finalize_with_mode: No accumulated text, skipping")
            return {"text": "", "skipped": True, "interim": False}

        try:
            raw_text = " ".join(self.accumulated_text)
            logger.info("="*60)
            logger.info(f"🎯 FINALIZING (mode={self.output_mode.value}): '{raw_text}'")

            if VERBOSE_LOGGING:
                logger.info(f"[VERBOSE] Finalizing session: speaker={self.current_speaker}, color={self.current_speaker_color}, confidence={self.current_speaker_confidence}")

            result = {
                "skipped": False,
                "interim": False,
                "stats": self.stats
            }

            if self.refinement.enabled:
                context = self.conversation_manager.get_recent_context(num_entries=12)
                if VERBOSE_LOGGING:
                    logger.info(f"[VERBOSE] Refining text with context (length: {len(context)} chars)")
                refined_text = await self.refinement.refine_text(raw_text, context=context)
            else:
                refined_text = raw_text

            saved_session = self.history_manager.add_session(
                raw_text=raw_text,
                refined_text=refined_text,
                client_id=self.client_id,
                speaker=self.current_speaker,
                speaker_color=self.current_speaker_color,
                confidence=self.current_speaker_confidence
            )
            self.history_manager.invalidate_stats_cache()

            logger.info(f"💾 Session saved: #{saved_session['session_id']}")

            # Build result based on output mode
            result["session_id"] = saved_session['session_id']
            result["speaker"] = self.current_speaker
            result["speaker_color"] = self.current_speaker_color
            result["confidence"] = self.current_speaker_confidence

            if VERBOSE_LOGGING:
                logger.info(f"[VERBOSE] Final result session_id: {result['session_id']}, speaker: {result['speaker']}")

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

            if VERBOSE_LOGGING:
                logger.info(f"[VERBOSE] Final result fields: {list(result.keys())}")

            logger.info(f"✅ Session #{saved_session['session_id']} finalized (speaker: {self.current_speaker})")
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
