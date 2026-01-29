"""
DeepGuard v2 - Main Detection Engine

Orchestrates the entire detection pipeline:
Screen Capture → Face Detection → Model Inference → 
Temporal Aggregation → Confidence Classification → Explanation
"""

import threading
import time
from typing import Optional, Callable, List
from dataclasses import dataclass
import numpy as np

from config import config
from core.screen_capture import ScreenCapture
from core.face_detector_mtcnn import FaceDetectorMTCNN
from core.detector import DeepfakeDetector
from core.temporal import TemporalEngine, TemporalState
from core.confidence import ConfidenceEngine, DetectionResult, ConfidenceLevel
from core.explainer import GeminiExplainer, Explanation


@dataclass
class EngineState:
    """Current state of the detection engine"""
    is_running: bool
    faces_detected: int
    temporal_state: Optional[TemporalState]
    explanation: Optional[Explanation]
    fps: float
    status: str  # "running", "no_faces", "analyzing", "error"


class DeepGuardEngine:
    """
    Main detection engine for DeepGuard v2.
    
    Runs the complete pipeline and provides callbacks for UI updates.
    Thread-safe design for use with overlay.
    """
    
    def __init__(self, on_state_update: Optional[Callable[[EngineState], None]] = None):
        """
        Initialize the engine.
        
        Args:
            on_state_update: Callback function when state changes
        """
        self.on_state_update = on_state_update
        
        # Initialize components
        print("[ENGINE] Initializing components...")
        
        self.screen_capture = ScreenCapture()
        self.face_detector = FaceDetectorMTCNN()
        self.deepfake_detector = DeepfakeDetector(config.detection.weights_path)
        self.temporal_engine = TemporalEngine()
        self.confidence_engine = ConfidenceEngine()
        self.explainer = GeminiExplainer()
        
        # State
        self.is_running = False
        self.current_state: Optional[EngineState] = None
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # Stats
        self.frame_times: List[float] = []
        self.frames_processed = 0
        
        print("[ENGINE] Initialization complete")
    
    def start(self):
        """Start the detection engine in background thread"""
        if self.is_running:
            return
            
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        print("[ENGINE] Started")
    
    def stop(self):
        """Stop the detection engine"""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        self.screen_capture.close()
        print("[ENGINE] Stopped")
    
    def _run_loop(self):
        """Main detection loop"""
        
        last_explanation_time = 0
        explanation_interval = 2.0  # Generate explanation every 2 seconds max
        current_explanation: Optional[Explanation] = None
        
        while self.is_running:
            loop_start = time.time()
            
            try:
                # 1. Capture screen
                frame = self.screen_capture.capture_frame()
                if frame is None:
                    time.sleep(0.01)
                    continue
                
                # 2. Detect faces
                faces = self.face_detector.detect_faces(frame)
                
                if not faces:
                    # No faces - reset temporal state
                    self.temporal_engine.reset()
                    state = EngineState(
                        is_running=True,
                        faces_detected=0,
                        temporal_state=None,
                        explanation=None,
                        fps=self._calculate_fps(loop_start),
                        status="no_faces"
                    )
                    self._update_state(state)
                    continue
                
                # 3. Run detection on first/largest face
                # (Future: track multiple faces)
                face = faces[0]
                result = self.deepfake_detector.predict(face)
                score = result["score"]
                
                # 4. Temporal aggregation
                temporal_state = self.temporal_engine.add_score(score)
                
                # 5. Generate explanation (rate-limited)
                current_time = time.time()
                if current_time - last_explanation_time > explanation_interval:
                    if temporal_state.is_stable:
                        current_explanation = self.explainer.explain(
                            result=temporal_state.result,
                            context="video",
                            trend=temporal_state.trend,
                            frames_analyzed=temporal_state.frames_analyzed
                        )
                        last_explanation_time = current_time
                
                # 6. Create state
                state = EngineState(
                    is_running=True,
                    faces_detected=len(faces),
                    temporal_state=temporal_state,
                    explanation=current_explanation,
                    fps=self._calculate_fps(loop_start),
                    status="running"
                )
                
                self._update_state(state)
                self.frames_processed += 1
                
            except Exception as e:
                print(f"[ENGINE] Error in loop: {e}")
                state = EngineState(
                    is_running=True,
                    faces_detected=0,
                    temporal_state=None,
                    explanation=None,
                    fps=0,
                    status="error"
                )
                self._update_state(state)
                time.sleep(0.1)
    
    def _update_state(self, state: EngineState):
        """Update state and notify callback"""
        with self._lock:
            self.current_state = state
            
        if self.on_state_update:
            self.on_state_update(state)
    
    def _calculate_fps(self, loop_start: float) -> float:
        """Calculate running FPS"""
        elapsed = time.time() - loop_start
        self.frame_times.append(elapsed)
        
        # Keep last 30 frame times
        if len(self.frame_times) > 30:
            self.frame_times = self.frame_times[-30:]
        
        avg_time = sum(self.frame_times) / len(self.frame_times)
        return round(1.0 / max(avg_time, 0.001), 1)
    
    def get_current_state(self) -> Optional[EngineState]:
        """Get current engine state (thread-safe)"""
        with self._lock:
            return self.current_state
    
    def get_stats(self) -> dict:
        """Get engine statistics"""
        return {
            "frames_processed": self.frames_processed,
            "capture_stats": self.screen_capture.get_stats(),
            "temporal_history": self.temporal_engine.get_history()
        }
