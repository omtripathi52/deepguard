"""
DeepGuard – Video Deepfake Detection Pipeline

Usage:
python -m core.video_pipeline --path path/to/video.mp4
"""

import argparse
import cv2
import numpy as np

from core.detector import DeepfakeDetector
from core.face_detector_mtcnn import FaceDetectorMTCNN


def run_video_pipeline(
    video_path: str,
    weights_path: str,
    frame_interval: int = 15,
    max_frames: int = 300
):
    print("[INFO] Loading video:", video_path)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    face_detector = FaceDetectorMTCNN()
    detector = DeepfakeDetector(weights_path)

    frame_count = 0
    processed_frames = 0
    predictions = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Sample frames for efficiency
        if frame_count % frame_interval != 0:
            continue

        processed_frames += 1

        faces = face_detector.detect_faces(frame)

        for face in faces:
            result = detector.predict(face)
            predictions.append(result["score"])

        if processed_frames >= max_frames:
            break

    cap.release()

    print(f"[INFO] Processed frames: {processed_frames}")
    print(f"[INFO] Total face predictions: {len(predictions)}")

    if len(predictions) == 0:
        print("[RESULT] No faces detected. Cannot assess deepfake.")
        return

    avg_score = float(np.mean(predictions))

    label = "deepfake" if avg_score >= 0.5 else "real"

    print(
        "\n[FINAL RESULT]\n"
        f"Video verdict → {label.upper()}\n"
        f"Average deepfake score → {avg_score:.3f}"
    )


def main():
    parser = argparse.ArgumentParser(description="DeepGuard Video Detector")
    parser.add_argument(
        "--path", "--video",
        dest="path",
        type=str,
        required=True,
        help="Path to video file (.mp4, .avi, etc.)"
    )

    args = parser.parse_args()

    run_video_pipeline(
        video_path=args.path,
        weights_path="mesonet/weights/Meso4_DF.h5"
    )


if __name__ == "__main__":
    main()
