"""
Stores conversations in markdown for context awareness.
"""
import logging
import threading
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# ============ VERBOSE LOGGING ============
VERBOSE_LOGGING = True
# ===========================================


class ConversationManager:
    """Stores conversations in markdown for context awareness"""
    
    def __init__(self, md_file="sessions/conversations.md"):
        self.md_file = Path(md_file)
        self._lock = threading.Lock()
        self._ensure_file_exists()
        logger.info(f"✓ Conversation manager initialized: {self.md_file}")
    
    def _ensure_file_exists(self):
        """Create markdown file if it doesn't exist"""
        if not self.md_file.exists():
            self.md_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.md_file, 'w', encoding='utf-8') as f:
                f.write("# Conversation History\n\n---\n\n")
            logger.info(f"✓ Created conversation file: {self.md_file}")
    
    def add_entry(self, speaker: str, text: str, confidence: float,
                  timestamp: str = None, session_id: int = None,
                  duration: float = None) -> None:
        """Add conversation entry to markdown"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        session_str = str(session_id) if session_id is not None else "0"
        duration_str = f"{duration:.1f}" if duration is not None else "0.0"
        
        entry = f"""## {timestamp}
**Speaker: {speaker} ({confidence:.0%})**  
{text}

*Metadata: Duration={duration_str}s, Session={session_str}*

---
"""
        
        with self._lock:
            with open(self.md_file, 'a', encoding='utf-8') as f:
                f.write(entry)
            logger.info(f"✓ Added conversation entry: {speaker} ({confidence:.0%})")
    
    def get_recent_context(self, num_entries: int = 12) -> str:
        """Get recent conversation entries as context string for LLM"""
        try:
            entries = self._parse_conversations()
            recent = entries[-num_entries:] if len(entries) > num_entries else entries
            
            context = "Past conversation:\n"
            for entry in recent:
                context += f"- [{entry['speaker']} ({entry['confidence']:.0%})]: {entry['text'][:100]}...\n"
            
            return context
        except Exception as e:
            logger.error(f"Error getting context: {e}")
            return ""
    
    def get_conversation_history(self, speaker: Optional[str] = None,
                               limit: int = 50) -> List[Dict[str, Any]]:
        """Get conversation history (optionally filtered by speaker)"""
        try:
            entries = self._parse_conversations()
            
            if speaker:
                entries = [e for e in entries if e['speaker'] == speaker]
            
            # Return most recent first
            entries = entries[-limit:][::-1]
            return entries
        except Exception as e:
            logger.error(f"Error getting conversation history: {e}")
            return []
    
    def _parse_conversations(self) -> List[Dict[str, Any]]:
        """Parse markdown file into structured data"""
        entries = []
        try:
            if not self.md_file.exists():
                return entries
            
            with open(self.md_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple parser - split by "## " and extract speaker/text
            lines = content.split('\n')
            current_entry = None
            
            for line in lines:
                if line.startswith('## '):
                    if current_entry:
                        entries.append(current_entry)
                    current_entry = {
                        "timestamp": line[3:],
                        "speaker": "",
                        "text": "",
                        "confidence": 0.0,
                        "metadata": ""
                    }
                elif current_entry and line.startswith('**Speaker:'):
                    # Extract speaker and confidence
                    speaker_part = line[len('**Speaker: '):]
                    speaker_part = speaker_part.split('**')[0]
                    if ' (' in speaker_part:
                        parts = speaker_part.split(' (')
                        current_entry['speaker'] = parts[0]
                        conf_str = parts[1].replace('%)', '')
                        try:
                            current_entry['confidence'] = float(conf_str) / 100
                        except:
                            pass
                elif current_entry and line.startswith('*Metadata:'):
                    current_entry['metadata'] = line[len('*Metadata: '):].strip('*')
                elif current_entry and line and not line.startswith('##') and not line.startswith('**') and not line.startswith('*'):
                    current_entry['text'] += line + ' '
            
            if current_entry:
                entries.append(current_entry)
                
        except Exception as e:
            logger.error(f"Error parsing conversations: {e}")
        
        return entries
