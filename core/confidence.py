"""
DeepGuard v2 - Confidence Engine

5-tier confidence classification:
- REAL (high confidence)
- LIKELY REAL
- UNCERTAIN  
- LIKELY FAKE
- DEEPFAKE (high confidence)
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional
import numpy as np

from config import config


class ConfidenceLevel(Enum):
    """5-tier confidence levels"""
    REAL = "REAL"
    LIKELY_REAL = "LIKELY REAL"
    UNCERTAIN = "UNCERTAIN"
    LIKELY_FAKE = "LIKELY FAKE"
    DEEPFAKE = "DEEPFAKE"


@dataclass
class DetectionResult:
    """Single detection result"""
    score: float                           # Raw model score (0-1, higher = more fake)
    level: ConfidenceLevel                 # Classified confidence level
    confidence_pct: int                    # Display percentage
    color: str                             # Display color hex
    is_stable: bool = True                 # Whether prediction is stable


class ConfidenceEngine:
    """
    Converts raw model scores into 5-tier confidence levels.
    
    Thresholds are configured in config.py for easy tuning.
    """
    
    def __init__(self):
        self.cfg = config.confidence
        self.overlay_cfg = config.overlay
    
    def classify(self, score: float) -> DetectionResult:
        """
        Classify a single score into confidence level.
        
        Args:
            score: Model output (0-1, higher = more likely fake)
            
        Returns:
            DetectionResult with level, color, and display info
        """
        
        if score < self.cfg.real_high:
            level = ConfidenceLevel.REAL
            color = self.overlay_cfg.color_real
            # Confidence in "real" = inverse of fake score
            confidence_pct = int((1 - score) * 100)
            
        elif score < self.cfg.real_low:
            level = ConfidenceLevel.LIKELY_REAL
            color = self.overlay_cfg.color_likely_real
            confidence_pct = int((1 - score) * 100)
            
        elif score < self.cfg.uncertain_high:
            level = ConfidenceLevel.UNCERTAIN
            color = self.overlay_cfg.color_uncertain
            # For uncertain, show closeness to 50%
            confidence_pct = int(50 + abs(0.5 - score) * 100)
            
        elif score < self.cfg.fake_low:
            level = ConfidenceLevel.LIKELY_FAKE
            color = self.overlay_cfg.color_likely_fake
            confidence_pct = int(score * 100)
            
        else:
            level = ConfidenceLevel.DEEPFAKE
            color = self.overlay_cfg.color_deepfake
            confidence_pct = int(score * 100)
        
        return DetectionResult(
            score=round(score, 4),
            level=level,
            confidence_pct=min(99, max(1, confidence_pct)),  # Clamp to 1-99%
            color=color
        )
    
    def get_display_text(self, result: DetectionResult) -> str:
        """Get formatted display text for overlay"""
        
        if result.level in [ConfidenceLevel.REAL, ConfidenceLevel.LIKELY_REAL]:
            return f"{result.level.value}  {result.confidence_pct}%"
        elif result.level == ConfidenceLevel.UNCERTAIN:
            return f"{result.level.value}  ~{result.confidence_pct}%"
        else:
            return f"{result.level.value}  {result.confidence_pct}%"
    
    def get_emoji(self, level: ConfidenceLevel) -> str:
        """Get status emoji for level"""
        emoji_map = {
            ConfidenceLevel.REAL: "🟢",
            ConfidenceLevel.LIKELY_REAL: "🟢",
            ConfidenceLevel.UNCERTAIN: "🟡",
            ConfidenceLevel.LIKELY_FAKE: "🟠",
            ConfidenceLevel.DEEPFAKE: "🔴"
        }
        return emoji_map.get(level, "⚪")
