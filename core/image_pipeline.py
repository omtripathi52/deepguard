"""
DeepGuard – Image Deepfake Detection Pipeline

Usage:
python -m core.image_pipeline --path path/to/image.jpg
"""

import argparse
import cv2

from core.detector import DeepfakeDetector
from core.face_detector_mtcnn import FaceDetectorMTCNN


def run_image_pipeline(image_path: str, weights_path: str):
    print("[INFO] Loading image:", image_path)

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    face_detector = FaceDetectorMTCNN()
    detector = DeepfakeDetector(weights_path)

    faces = face_detector.detect_faces(image)

    print(f"[INFO] Faces detected: {len(faces)}")

    if len(faces) == 0:
        print("[RESULT] No faces found. Cannot assess deepfake.")
        return

    for idx, face in enumerate(faces, start=1):
        result = detector.predict(face)

        print(
            f"[RESULT] Face {idx} → "
            f"{result['label'].upper()} | "
            f"score={result['score']} | "
            f"confidence={result['confidence']}"
        )


def main():
    parser = argparse.ArgumentParser(description="DeepGuard Image Detector")
    parser.add_argument(
        "--path", "--image",
        dest="path",
        type=str,
        required=True,
        help="Path to image file"
    )

    args = parser.parse_args()

    run_image_pipeline(
        image_path=args.path,
        weights_path="mesonet/weights/Meso4_DF.h5"
    )


if __name__ == "__main__":
    main()
