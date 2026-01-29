"""
DeepGuard v2 - Smart Screen Capture

Captures screen content for real-time analysis.
Optimized for:
- Full screen capture
- Detecting video regions (reels, shorts, etc.)
- Handling different aspect ratios
- Performance (target: 10+ FPS)
"""

import numpy as np
import cv2
from mss import mss
from typing import Optional, Tuple, List
import time
import threading

from config import config


class ScreenCapture:
    """
    Smart screen capture system.
    
    Captures the entire screen and provides frames for analysis.
    Optimized for detecting faces in video content like reels/shorts.
    
    Note: mss is NOT thread-safe, so we use thread-local storage
    to create one mss instance per thread.
    """
    
    def __init__(self):
        self.cfg = config.capture
        
        # Thread-local storage for mss instances
        self._local = threading.local()
        
        # Frame timing
        self.target_interval = 1.0 / self.cfg.fps
        self.last_capture_time = 0
        
        # Stats
        self.frames_captured = 0
        self.avg_capture_time = 0
        
    def _get_sct(self):
        """Get thread-local mss instance"""
        if not hasattr(self._local, 'sct'):
            self._local.sct = mss()
        return self._local.sct
    
    @property
    def primary_monitor(self):
        """Get primary monitor info"""
        sct = self._get_sct()
        return sct.monitors[self.cfg.monitor_index]
        
    def get_monitor_info(self) -> dict:
        """Get information about the capture monitor"""
        return {
            "width": self.primary_monitor["width"],
            "height": self.primary_monitor["height"],
            "top": self.primary_monitor["top"],
            "left": self.primary_monitor["left"]
        }
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """
        Capture a single frame from the screen.
        
        Returns:
            BGR numpy array of the screen, or None if too soon
        """
        
        # Rate limiting
        current_time = time.time()
        if current_time - self.last_capture_time < self.target_interval:
            return None
            
        self.last_capture_time = current_time
        
        # Capture screen
        start_time = time.time()
        
        sct = self._get_sct()
        monitor = sct.monitors[self.cfg.monitor_index]
        screenshot = sct.grab(monitor)
        
        # Convert to numpy array (BGRA)
        frame = np.array(screenshot)
        
        # Convert BGRA to BGR for OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        
        # Update stats
        capture_time = time.time() - start_time
        self.avg_capture_time = 0.9 * self.avg_capture_time + 0.1 * capture_time
        self.frames_captured += 1
        
        return frame
    
    def capture_frame_forced(self) -> np.ndarray:
        """Capture a frame immediately, ignoring rate limiting"""
        sct = self._get_sct()
        monitor = sct.monitors[self.cfg.monitor_index]
        screenshot = sct.grab(monitor)
        frame = np.array(screenshot)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        self.frames_captured += 1
        return frame
    
    def get_stats(self) -> dict:
        """Get capture statistics"""
        return {
            "frames_captured": self.frames_captured,
            "avg_capture_time_ms": round(self.avg_capture_time * 1000, 2),
            "target_fps": self.cfg.fps,
            "actual_fps": round(1.0 / max(self.avg_capture_time, 0.001), 1)
        }
    
    def detect_video_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Attempt to detect regions that might contain video content.
        
        This is a heuristic approach - looks for rectangular regions
        with high motion or specific aspect ratios (9:16 for reels/shorts).
        
        Note: This is optional/experimental for v2.
        
        Returns:
            List of (x, y, width, height) tuples
        """
        # For v2, we analyze the entire frame
        # This method is here for future optimization
        h, w = frame.shape[:2]
        return [(0, 0, w, h)]
    
    def is_vertical_video_region(self, x: int, y: int, w: int, h: int) -> bool:
        """Check if a region has vertical video aspect ratio (9:16)"""
        if w == 0:
            return False
        aspect_ratio = h / w
        # 9:16 = 1.78, allow some tolerance
        return 1.5 < aspect_ratio < 2.0
    
    def close(self):
        """Clean up resources"""
        # Thread-local mss instances clean themselves up
        # Just mark as closed - no explicit cleanup needed
        pass


class ScreenCaptureThread:
    """
    Threaded screen capture for non-blocking operation.
    
    Runs capture in background, provides latest frame on demand.
    """
    
    def __init__(self):
        self.capture = ScreenCapture()
        self.latest_frame: Optional[np.ndarray] = None
        self.running = False
        self._thread = None
        
    def start(self):
        """Start background capture thread"""
        import threading
        self.running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        
    def _capture_loop(self):
        """Background capture loop"""
        while self.running:
            frame = self.capture.capture_frame()
            if frame is not None:
                self.latest_frame = frame
            time.sleep(0.01)  # Small sleep to prevent CPU spin
    
    def get_latest_frame(self) -> Optional[np.ndarray]:
        """Get the most recent captured frame"""
        return self.latest_frame
    
    def stop(self):
        """Stop background capture"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self.capture.close()
