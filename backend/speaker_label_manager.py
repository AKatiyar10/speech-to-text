"""
Manages speaker labels (ALPHA, UNKNOWN_XX) and color coding.
"""
import logging
import json
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

# ============ VERBOSE LOGGING ============
VERBOSE_LOGGING = True
# ===========================================


class SpeakerLabelManager:
    """Manages speaker labels (ALPHA, UNKNOWN_XX) and color coding"""
    
    UNKNOWN_COLOR = "#3b82f6"
    ALPHA_COLOR = "#22c55e"
    
    def __init__(self, labels_file="sessions/speaker_labels.json"):
        self.labels_file = Path(labels_file)
        self.labels = {}
        self.unknown_count = 0
        self._lock = threading.Lock()
        self._load_labels()
        logger.info(f"✓ Loaded {len(self.labels)} speaker labels")
        logger.info(f"✓ Speaker label manager initialized: {self.labels_file}")
    
    def _load_labels(self):
        """Load existing labels from JSON"""
        if self.labels_file.exists():
            try:
                with open(self.labels_file, 'r', encoding='utf-8') as f:
                    self.labels = json.load(f)
                
                # Find max unknown count
                for name in self.labels.keys():
                    if name.startswith("UNKNOWN_"):
                        num = int(name.split("_")[1])
                        if num > self.unknown_count:
                            self.unknown_count = num
            except Exception as e:
                logger.error(f"Error loading labels: {e}")
                self.labels = {}
        else:
            self.labels = {}
    
    def _save_labels(self):
        """Save labels to JSON"""
        try:
            self.labels_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.labels_file, 'w', encoding='utf-8') as f:
                json.dump(self.labels, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving labels: {e}")
    
    def get_or_create_unknown(self, embedding: np.ndarray) -> str:
        """Get next UNKNOWN label or create new one"""
        with self._lock:
            # Try to find existing UNKNOWN with similar embedding
            # For now, just create new UNKNOWN label
            self.unknown_count += 1
            name = f"UNKNOWN_{self.unknown_count:02d}"
            
            self.labels[name] = {
                "name": name,
                "color": self.UNKNOWN_COLOR,
                "created_at": datetime.now().isoformat(),
                "last_heard": datetime.now().isoformat()
            }
            
            # Save embedding to disk
            emb_dir = Path("speakers/embeddings")
            emb_dir.mkdir(parents=True, exist_ok=True)
            emb_file = emb_dir / f"{name}.npy"
            np.save(emb_file, embedding)
            
            self._save_labels()
            
            if VERBOSE_LOGGING:
                logger.info(f"[VERBOSE] Created new UNKNOWN speaker: {name}")
            
            return name
    
    def relabel_speaker(self, old_name: str, new_name: str, color: str = None) -> bool:
        """Relabel speaker (UNKNOWN_XX -> ALPHA/Custom)"""
        if old_name not in self.labels:
            return False
        
        with self._lock:
            old_label = self.labels[old_name]
            
            # Rename embedding file
            old_emb_file = Path("speakers/embeddings") / f"{old_name}.npy"
            new_emb_file = Path("speakers/embeddings") / f"{new_name}.npy"
            if old_emb_file.exists():
                old_emb_file.rename(new_emb_file)
            
            # Update or create new label
            if new_name not in self.labels:
                self.labels[new_name] = {
                    "name": new_name,
                    "color": color or old_label.get("color", self.ALPHA_COLOR),
                    "created_at": old_label.get("created_at", datetime.now().isoformat()),
                    "last_heard": old_label.get("last_heard", datetime.now().isoformat())
                }
            
            del self.labels[old_name]
            self._save_labels()
            
            logger.info(f"✓ Relabeled: {old_name} → {new_name}")
            return True
    
    def get_label_color(self, name: str) -> str:
        """Return color for speaker"""
        return self.labels.get(name, {}).get("color", self.UNKNOWN_COLOR)
    
    def register_speaker(self, name: str, embedding: np.ndarray) -> str:
        """Register a speaker with given name and embedding (for enrollment)"""
        with self._lock:
            # Get or use default color
            color = self.ALPHA_COLOR if name.upper() == "ALPHA" else self.UNKNOWN_COLOR

            self.labels[name] = {
                "name": name,
                "color": color,
                "created_at": datetime.now().isoformat(),
                "last_heard": datetime.now().isoformat()
            }

            # Save embedding to disk
            emb_dir = Path("speakers/embeddings")
            emb_dir.mkdir(parents=True, exist_ok=True)
            emb_file = emb_dir / f"{name}.npy"
            np.save(emb_file, embedding)

            self._save_labels()

            if VERBOSE_LOGGING:
                logger.info(f"[VERBOSE] Registered speaker: {name} with color {color}")

            return color

    def get_all_labels(self) -> Dict[str, Dict]:
        """Return all speakers with metadata"""
        return self.labels
    
    def update_last_heard(self, name: str):
        """Update last heard timestamp"""
        with self._lock:
            if name in self.labels:
                self.labels[name]["last_heard"] = datetime.now().isoformat()
    
    def delete_speaker(self, name: str) -> bool:
        """Delete a speaker"""
        with self._lock:
            if name not in self.labels:
                return False
            
            emb_file = Path("speakers/embeddings") / f"{name}.npy"
            try:
                if emb_file.exists():
                    emb_file.unlink()
            except Exception as e:
                logger.error(f"Failed to delete embedding file: {e}")
            
            del self.labels[name]
            self._save_labels()
            logger.info(f"✓ Deleted speaker: {name}")
            return True
