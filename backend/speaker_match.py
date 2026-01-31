"""
Speaker match data class for voice identification results.
"""
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class SpeakerMatch:
    """Result of speaker identification"""
    name: str
    confidence: float
    is_match: bool
    embedding: np.ndarray
