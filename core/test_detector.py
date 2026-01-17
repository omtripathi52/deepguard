import cv2
from core.detector import DeepfakeDetector

detector = DeepfakeDetector(
    weights_path="mesonet/weights/Meso4_DF.h5"
)

img_path = "mesonet/test_images/df/df00204.jpg"
img = cv2.imread(img_path)

print("Image loaded:", img is not None)
print("Image shape:", None if img is None else img.shape)

result = detector.predict(img)
print(result)

