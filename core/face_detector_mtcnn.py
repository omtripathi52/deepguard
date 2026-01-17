"""
DeepGuard – Face Detection Module (MTCNN)
"""

from typing import List
import numpy as np
import cv2
from mtcnn import MTCNN


class FaceDetectorMTCNN:
    """
    Face detector using MTCNN.
    """

    def __init__(self):
        self.detector = MTCNN()

    def detect_faces(self, frame: np.ndarray, min_face_size: int = 40) -> List[np.ndarray]:
        """
        Detect faces in a frame and return cropped face images.

        Args:
            frame (np.ndarray): BGR image (OpenCV format)
            min_face_size (int): Minimum face size (pixels)

        Returns:
            List of cropped face images (BGR)
        """

        if frame is None or frame.size == 0:
            return []

        # Convert BGR → RGB for MTCNN
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        detections = self.detector.detect_faces(rgb)
        faces = []

        for det in detections:
            x, y, w, h = det.get("box", (0, 0, 0, 0))

            # Filter tiny faces
            if w < min_face_size or h < min_face_size:
                continue

            # Ensure valid coordinates
            x, y = max(0, x), max(0, y)

            face = frame[y:y + h, x:x + w]

            if face.size > 0:
                faces.append(face)

        return faces
