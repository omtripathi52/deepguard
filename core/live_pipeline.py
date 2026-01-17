"""
This file:
- Opens webcam safely
- Reads frames continuously
- Detects faces using MTCNN
- Detects deepfake using MesoNet
- Displays results in real time

Controls:
- Press 'q' to quit
"""

import cv2
import time

from core.detector import DeepfakeDetector
from core.face_detector_mtcnn import FaceDetectorMTCNN


class LiveDeepfakePipeline:
    def __init__(
        self,
        weights_path: str,
        camera_index: int = 0
    ):
        """
        Initialize live pipeline.

        Args:
            weights_path (str): Path to MesoNet weights
            camera_index (int): Webcam index (0 or 1)
        """

        print("[INFO] Initializing DeepGuard pipeline...")

        self.face_detector = FaceDetectorMTCNN()
        self.detector = DeepfakeDetector(weights_path)

        self.cap = cv2.VideoCapture(camera_index)
        time.sleep(1)

        print("[DEBUG] Camera opened:", self.cap.isOpened())

        if not self.cap.isOpened():
            raise RuntimeError(
                "Camera could not be opened. "
                "Try changing camera_index to 1."
            )

    def run(self):
        """
        Start live detection loop.
        """

        print("[INFO] DeepGuard live detection started.")
        print("[INFO] Press 'q' to quit.")

        while True:
            # Read frame
            ret, frame = self.cap.read()

            if not ret:
                print("[ERROR] Failed to read frame from camera.")
                break

            # Detect faces
            faces = self.face_detector.detect_faces(frame)

            # Process each detected face
            for face in faces:
                try:
                    result = self.detector.predict(face)

                    label = result["label"]
                    score = result["score"]
                    confidence = result["confidence"]

                    color = (0, 255, 0) if label == "real" else (0, 0, 255)
                    text = f"{label.upper()} | {score:.2f} | {confidence}"

                    # Draw simple overlay (top-left)
                    cv2.putText(
                        frame,
                        text,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2,
                        cv2.LINE_AA
                    )

                except Exception as e:
                    print("[WARN] Prediction error:", e)

            # Show window
            cv2.imshow("DeepGuard – Live Detection", frame)

            # IMPORTANT: waitKey is REQUIRED for window to render
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("[INFO] Quitting...")
                break

        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("[INFO] Camera released, windows closed.")


# ---- ENTRY POINT ----
if __name__ == "__main__":
    pipeline = LiveDeepfakePipeline(
        weights_path="mesonet/weights/Meso4_DF.h5",
        camera_index=0  # change to 1 if camera doesn't open
    )
    pipeline.run()
