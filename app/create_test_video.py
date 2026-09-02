import cv2
import os

IMAGE_PATH = "datasets/pipe_leak/pipe leak.v6i.yolov8/test/images/test_leak.jpg"
OUTPUT_PATH = "data/test/leak_test_video.mp4"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")

os.makedirs("data/test", exist_ok=True)

height, width = image.shape[:2]

writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*"mp4v"),
    10,
    (width, height)
)

# 5 seconds × 10 FPS = 50 frames
for _ in range(50):
    writer.write(image)

writer.release()

print(f"Test video created: {OUTPUT_PATH}")