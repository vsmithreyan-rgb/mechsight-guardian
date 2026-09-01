import cv2
import os
from ultralytics import YOLO

from vision.leak_analyzer import analyse_leak_region


MODEL_PATH = "runs/detect/train/weights/best.pt"
TEST_FOLDER = "datasets/pipe_leak/pipe leak.v6i.yolov8/test/images"
OUTPUT_PATH = "results/mechsight_detection.jpg"

model = YOLO(MODEL_PATH)

print("\nMECHSIGHT GUARDIAN")
print("=" * 50)

detection_found = False

for filename in os.listdir(TEST_FOLDER):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    image_path = os.path.join(TEST_FOLDER, filename)
    image = cv2.imread(image_path)

    if image is None:
        continue

    results = model(image, conf=0.25, verbose=False)

    for result in results:
        if len(result.boxes) == 0:
            continue

        box = result.boxes[0]

        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        analysis = analyse_leak_region(
            image,
            [x1, y1, x2, y2]
        )

        print(f"Image: {filename}")
        print("Leak detected: YES")
        print(f"Confidence: {confidence:.2%}")
        print(f"Area ratio: {analysis['area_ratio']:.2%}")
        print(f"Edge density: {analysis['edge_density']:.4f}")
        print(f"Brightness: {analysis['brightness']:.2f}")
        print(f"Visual severity: {analysis['severity']}")

        # Draw detection box
        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            (255, 0, 0),
            3
        )

        # Main detection label
        label = f"Leak {confidence:.0%} | {analysis['severity']}"

        cv2.putText(
            image,
            label,
            (x1, max(y1 - 10, 30)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2
        )

        # OpenCV analysis panel
        cv2.rectangle(
            image,
            (10, 10),
            (340, 135),
            (0, 0, 0),
            -1
        )

        cv2.putText(
            image,
            "MECHSIGHT GUARDIAN",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            image,
            f"Confidence: {confidence:.1%}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            image,
            f"Area ratio: {analysis['area_ratio']:.1%}",
            (20, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        cv2.putText(
            image,
            f"Severity: {analysis['severity']}",
            (20, 111),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1
        )

        os.makedirs("results", exist_ok=True)

        cv2.imwrite(
            OUTPUT_PATH,
            image
        )

        print(f"\nVisual result saved to: {OUTPUT_PATH}")

        detection_found = True
        break

    if detection_found:
        break


if not detection_found:
    print("No leak detections found in test folder.")