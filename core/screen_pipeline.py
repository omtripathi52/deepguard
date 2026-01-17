"""
DeepGuard – Screen Capture Deepfake Detection Pipeline (FINAL)

Detects deepfakes live from ANY visible screen content:
- websites
- videos
- images

Runs headless (no preview window) to avoid recursive capture issues.
"""

import time
import cv2
import numpy as np
from mss import mss

from core.detector import DeepfakeDetector
from core.face_detector_mtcnn import FaceDetectorMTCNN


class ScreenDeepfakePipeline:
    def __init__(self, weights_path: str):
        print("[INFO] Initializing screen capture pipeline...")

        self.face_detector = FaceDetectorMTCNN()
        self.detector = DeepfakeDetector(weights_path)
        self.sct = mss()

        # Capture region (adjust if needed)
        self.monitor = {
            "top": 0,
            "left": 0,
            "width": 1600,
            "height": 900
        }

    def run(self):
        print("[INFO] Screen deepfake detection started")
        print("[INFO] Place a CLEAR HUMAN FACE inside the capture region")
        print("[INFO] Press Ctrl+C to stop\n")

        last_log_time = 0

        try:
            while True:
                # Capture screen frame
                screenshot = self.sct.grab(self.monitor)
                frame = np.array(screenshot)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

                faces = self.face_detector.detect_faces(frame)

                # Always show activity
                if not faces:
                    time.sleep(0.05)
                    continue

                scores = []

                for face in faces:
                    try:
                        result = self.detector.predict(face)
                        scores.append(result["score"])
                    except Exception:
                        continue

                if scores:
                    avg_score = float(np.mean(scores))
                    label = "DEEPFAKE" if avg_score >= 0.5 else "REAL"

                    now = time.time()
                    if now - last_log_time >= 1.0:
                        print(
                            f"[SCREEN DETECTION] "
                            f"Faces={len(scores)} | "
                            f"Verdict={label} | "
                            f"AvgScore={avg_score:.3f}"
                        )
                        last_log_time = now

                time.sleep(0.05)

        except KeyboardInterrupt:
            print("\n[INFO] Screen detection stopped by user")


if __name__ == "__main__":
    pipeline = ScreenDeepfakePipeline(
        weights_path="mesonet/weights/Meso4_DF.h5"
    )
    pipeline.run()
