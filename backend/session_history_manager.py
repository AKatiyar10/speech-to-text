"""
Session history manager for storing transcriptions in JSON.
"""
import logging
import json
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

# ============ VERBOSE LOGGING ============
VERBOSE_LOGGING = True
# ===========================================


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

            if 'speaker' not in session:
                session['speaker'] = 'UNKNOWN'
                session['speaker_color'] = '#3b82f6'
                session['confidence'] = 0.0
                migrated = True

        if migrated:
            self._save_history()
            logger.info(f"✓ Migrated {len(self.sessions)} sessions to new format")
    
    def add_session(self, raw_text: str, refined_text: str, client_id: str,
                    speaker: str = "UNKNOWN", speaker_color: str = "#3b82f6",
                    confidence: float = 0.0) -> Dict[str, Any]:
        """Add a new session to history WITHOUT feedback initially"""
        with self._lock:
            session_id = len(self.sessions) + 1
            now = datetime.now()

            if VERBOSE_LOGGING:
                logger.info(f"[VERBOSE] add_session called: raw_text='{raw_text[:100]}...', refined_text='{refined_text[:100]}...', speaker={speaker}, confidence={confidence}")

            session = {
                "session_id": session_id,
                "timestamp": now.isoformat(),
                "timestamp_readable": now.strftime("%Y-%m-%d %H:%M:%S"),
                "client_id": client_id,
                "raw_text": raw_text,
                "refined_text": refined_text,
                "speaking_feedback": None,
                "feedback_generated_at": None,
                "speaker": speaker,
                "speaker_color": speaker_color,
                "confidence": confidence,
                "character_counts": {
                    "raw": len(raw_text),
                    "refined": len(refined_text),
                    "feedback": 0
                }
            }

            self.sessions.append(session)
            self._session_cache[session_id] = session  # Update cache
            self._save_history()

            if VERBOSE_LOGGING:
                logger.info(f"[VERBOSE] Session #{session_id} saved with speaker info: name={speaker}, color={speaker_color}, confidence={confidence}")

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
