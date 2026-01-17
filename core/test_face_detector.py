import cv2
from core.face_detector_mtcnn import FaceDetectorMTCNN

img = cv2.imread("mesonet/test_images/df/df00204.jpg")

detector = FaceDetectorMTCNN()
faces = detector.detect_faces(img)

print("Faces detected:", len(faces))

for i, face in enumerate(faces):
    cv2.imwrite(f"face_{i}.jpg", face)
