"""
DeepGuard v2 - Configuration
"""

from dataclasses import dataclass
from typing import Tuple
import os


@dataclass
class CaptureConfig:
    """Screen capture settings"""
    fps: int = 10                          # Frames per second to analyze
    capture_full_screen: bool = True       # Capture entire screen
    monitor_index: int = 1                 # Primary monitor


@dataclass
class DetectionConfig:
    """Detection model settings"""
    model_name: str = "mesonet"            # Model to use: mesonet, efficientnet
    weights_path: str = "mesonet/weights/Meso4_DF.h5"
    input_size: Tuple[int, int] = (256, 256)
    min_face_size: int = 40                # Minimum face size in pixels


@dataclass
class TemporalConfig:
    """Temporal aggregation settings"""
    window_size: int = 30                  # Number of frames to consider
    decay_factor: float = 0.95             # Weight decay for older frames
    stability_threshold: float = 0.15     # Min change to update display


@dataclass
class ConfidenceConfig:
    """5-tier confidence band thresholds"""
    # Thresholds for classification (score is probability of FAKE)
    real_high: float = 0.20        # Below this = REAL (high confidence)
    real_low: float = 0.35         # Below this = LIKELY REAL
    uncertain_low: float = 0.45    # Below this = UNCERTAIN
    uncertain_high: float = 0.55   # Below this = UNCERTAIN
    fake_low: float = 0.65         # Below this = LIKELY FAKE
    # Above fake_low = DEEPFAKE (high confidence)


@dataclass
class OverlayConfig:
    """Overlay UI settings"""
    width: int = 280
    height_collapsed: int = 60
    height_expanded: int = 160
    opacity: float = 0.92
    corner_radius: int = 15
    position: str = "top-right"            # top-right, top-left, bottom-right, bottom-left
    margin: int = 20                       # Distance from screen edge
    
    # Colors (RGB)
    color_real: str = "#22C55E"            # Green
    color_likely_real: str = "#84CC16"     # Lime
    color_uncertain: str = "#EAB308"       # Yellow
    color_likely_fake: str = "#F97316"     # Orange
    color_deepfake: str = "#EF4444"        # Red
    color_background: str = "#1F2937"      # Dark gray
    color_text: str = "#F9FAFB"            # White


@dataclass
class GeminiConfig:
    """
    Gemini API Configuration
    
    DeepGuard uses Google's Gemini API to provide human-readable explanations
    of detection results. This is central to the Gemini 3 Hackathon submission.
    
    The Gemini model converts technical ML signals into user-friendly text,
    making deepfake detection accessible to non-technical users.
    """
    api_key: str = os.getenv("GEMINI_API_KEY", "")
    model: str = "gemini-2.0-flash-exp"    # Latest Gemini model (experimental)
    enabled: bool = True
    cache_duration: int = 5                # Seconds to cache explanations
    fallback_enabled: bool = True          # Use deterministic fallback if API fails


@dataclass 
class DeepGuardConfig:
    """Main configuration container"""
    capture: CaptureConfig = None
    detection: DetectionConfig = None
    temporal: TemporalConfig = None
    confidence: ConfidenceConfig = None
    overlay: OverlayConfig = None
    gemini: GeminiConfig = None
    
    def __post_init__(self):
        self.capture = self.capture or CaptureConfig()
        self.detection = self.detection or DetectionConfig()
        self.temporal = self.temporal or TemporalConfig()
        self.confidence = self.confidence or ConfidenceConfig()
        self.overlay = self.overlay or OverlayConfig()
        self.gemini = self.gemini or GeminiConfig()


# Global config instance
config = DeepGuardConfig()
