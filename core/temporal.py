"""
DeepGuard v2 - Temporal Aggregation Engine

Solves the v1 flickering problem by:
- Maintaining sliding window of predictions
- Applying weighted averaging (recent frames matter more)
- Detecting stability vs rapid changes
- Smoothing transitions between states
"""

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import time

from config import config
from core.confidence import ConfidenceEngine, DetectionResult, ConfidenceLevel


@dataclass
class TemporalState:
    """Current temporal analysis state"""
    raw_score: float                       # Latest raw score
    smoothed_score: float                  # Temporally smoothed score
    result: DetectionResult                # Classified result
    is_stable: bool                        # Has prediction stabilized?
    trend: str                             # "rising", "falling", "stable"
    frames_analyzed: int                   # Total frames in window
    last_update: float                     # Timestamp


class TemporalEngine:
    """
    Temporal aggregation for stable predictions.
    
    Instead of showing frame-by-frame flickering:
    - Buffers last N predictions
    - Applies exponential weighted average
    - Only updates display when change is significant
    - Tracks trends for smoother UX
    """
    
    def __init__(self):
        self.cfg = config.temporal
        self.confidence_engine = ConfidenceEngine()
        
        # Sliding window of scores
        self.score_buffer: deque = deque(maxlen=self.cfg.window_size)
        
        # Weights for exponential decay (recent frames weighted higher)
        self._precompute_weights()
        
        # State tracking
        self.last_smoothed_score: Optional[float] = None
        self.last_result: Optional[DetectionResult] = None
        self.last_display_update: float = 0
        self.frames_since_change: int = 0
        
    def _precompute_weights(self):
        """Precompute exponential decay weights"""
        weights = []
        for i in range(self.cfg.window_size):
            # More recent frames (higher index) get higher weight
            weight = self.cfg.decay_factor ** (self.cfg.window_size - 1 - i)
            weights.append(weight)
        self.weights = np.array(weights)
        
    def add_score(self, score: float) -> TemporalState:
        """
        Add a new frame score and get updated temporal state.
        
        Args:
            score: Raw model score (0-1)
            
        Returns:
            TemporalState with smoothed prediction
        """
        
        self.score_buffer.append(score)
        
        # Need at least a few frames for meaningful averaging
        if len(self.score_buffer) < 3:
            result = self.confidence_engine.classify(score)
            return TemporalState(
                raw_score=score,
                smoothed_score=score,
                result=result,
                is_stable=False,
                trend="analyzing",
                frames_analyzed=len(self.score_buffer),
                last_update=time.time()
            )
        
        # Compute weighted average
        scores = np.array(list(self.score_buffer))
        weights = self.weights[-len(scores):]  # Use only relevant weights
        weights = weights / weights.sum()       # Normalize
        
        smoothed_score = np.average(scores, weights=weights)
        
        # Determine trend
        trend = self._compute_trend(scores)
        
        # Check stability
        is_stable = self._check_stability(smoothed_score)
        
        # Classify smoothed score
        result = self.confidence_engine.classify(smoothed_score)
        result.is_stable = is_stable
        
        # Update state
        self.last_smoothed_score = smoothed_score
        self.last_result = result
        
        return TemporalState(
            raw_score=score,
            smoothed_score=smoothed_score,
            result=result,
            is_stable=is_stable,
            trend=trend,
            frames_analyzed=len(self.score_buffer),
            last_update=time.time()
        )
    
    def _compute_trend(self, scores: np.ndarray) -> str:
        """Determine if scores are rising, falling, or stable"""
        if len(scores) < 5:
            return "analyzing"
            
        recent = scores[-5:].mean()
        older = scores[-10:-5].mean() if len(scores) >= 10 else scores[:-5].mean()
        
        diff = recent - older
        
        if diff > 0.1:
            return "rising"
        elif diff < -0.1:
            return "falling"
        else:
            return "stable"
    
    def _check_stability(self, smoothed_score: float) -> bool:
        """Check if prediction has stabilized"""
        if self.last_smoothed_score is None:
            return False
            
        change = abs(smoothed_score - self.last_smoothed_score)
        
        if change < self.cfg.stability_threshold:
            self.frames_since_change += 1
        else:
            self.frames_since_change = 0
            
        # Stable if no significant change for 5+ frames
        return self.frames_since_change >= 5
    
    def should_update_display(self, state: TemporalState) -> bool:
        """
        Determine if display should be updated.
        
        Avoids constant flickering by only updating when:
        - Confidence level changed
        - Significant score change
        - Enough time has passed
        """
        
        if self.last_result is None:
            return True
            
        # Always update if level changed
        if state.result.level != self.last_result.level:
            return True
            
        # Update if significant score change
        score_change = abs(state.smoothed_score - self.last_smoothed_score)
        if score_change > self.cfg.stability_threshold:
            return True
            
        # Throttle updates to max 2 per second for stability
        time_since_update = time.time() - self.last_display_update
        if time_since_update > 0.5:
            self.last_display_update = time.time()
            return True
            
        return False
    
    def reset(self):
        """Reset temporal state (e.g., when no faces detected)"""
        self.score_buffer.clear()
        self.last_smoothed_score = None
        self.last_result = None
        self.frames_since_change = 0
    
    def get_history(self) -> List[float]:
        """Get score history for visualization"""
        return list(self.score_buffer)
    
    def get_stability_info(self) -> str:
        """Get human-readable stability status"""
        if len(self.score_buffer) < 3:
            return "Analyzing..."
        elif not self.last_result or not self.last_result.is_stable:
            return "Stabilizing..."
        else:
            return "Stable"
