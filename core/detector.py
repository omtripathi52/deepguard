"""
DeepGuard – Deepfake Detector Core

This module wraps the MesoNet (Meso4) model and exposes a clean,
reusable interface for deepfake detection on face images.

Responsibilities:
- Load pretrained MesoNet weights
- Preprocess face images correctly
- Run inference
- Return structured results

NOTE:
- This file does NOT do face detection
- This file does NOT do screen capture
- This file does NOT call Gemini
"""

from typing import Dict
import numpy as np
import cv2

# Import MesoNet model (kept isolated)
from mesonet.classifiers import Meso4


class DeepfakeDetector:
    """
    Core deepfake detector using MesoNet (Meso4).

    Input  : cropped face image (numpy array, BGR or RGB)
    Output : probability score + label + confidence
    """

    def __init__(self, weights_path: str):
        """
        Initialize detector and load pretrained weights.

        Args:
            weights_path (str): path to .h5 MesoNet weights
        """
        self.model = Meso4()
        self.model.load(weights_path)

    @staticmethod
    def _preprocess(face_img: np.ndarray) -> np.ndarray:
        """
        Preprocess face image for MesoNet.

        Steps:
        - Ensure valid image
        - Convert to RGB
        - Resize to 256x256
        - Normalize to [0, 1]
        - Add batch dimension
        """

        if face_img is None or face_img.size == 0:
            raise ValueError("Invalid or empty face image provided.")

        # Ensure 3 channels
        if len(face_img.shape) != 3 or face_img.shape[2] != 3:
            raise ValueError("Face image must be a color image (H, W, 3).")

        # Convert BGR → RGB if needed
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Resize to model input
        face_rgb = cv2.resize(face_rgb, (256, 256))

        # Normalize
        face_rgb = face_rgb.astype("float32") / 255.0

        # Add batch dimension
        face_rgb = np.expand_dims(face_rgb, axis=0)

        return face_rgb

    def predict(self, face_img: np.ndarray) -> Dict:
        """
        Predict whether a face image is a deepfake.

        Returns:
            dict with:
            - score      : float (0–1, higher = more fake)
            - label      : "real" | "deepfake"
            - confidence : "low" | "medium" | "high"
        """

        processed = self._preprocess(face_img)

        # Model prediction
        score = float(self.model.predict(processed)[0][0])

        # Label logic
        label = "deepfake" if score >= 0.5 else "real"

        # Confidence bands (simple, interpretable)
        if score >= 0.75 or score <= 0.25:
            confidence = "high"
        elif score >= 0.6 or score <= 0.4:
            confidence = "medium"
        else:
            confidence = "low"

        return {
            "score": round(score, 4),
            "label": label,
            "confidence": confidence
        }
