import cv2
from ultralytics import YOLO

from leak_analyzer import analyse_leak_region


MODEL_PATH = "runs/detect/train/weights/best.pt"


def monitor_video(video_path):
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_number = 0
    leak_frames = 0
    previous_area = None

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model(
            frame,
            conf=0.25,
            verbose=False
        )

        leak_detected = False

        for result in results:
            if len(result.boxes) == 0:
                continue

            box = result.boxes[0]

            confidence = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            analysis = analyse_leak_region(
                frame,
                [x1, y1, x2, y2]
            )

            leak_detected = True
            leak_frames += 1

            current_area = analysis["area_ratio"]

            if previous_area is None:
                trend = "STARTING"
            elif current_area > previous_area * 1.10:
                trend = "GROWING"
            elif current_area < previous_area * 0.90:
                trend = "SHRINKING"
            else:
                trend = "STABLE"

            previous_area = current_area

            print(
                f"Frame {frame_number} | "
                f"Leak {confidence:.1%} | "
                f"Area {current_area:.1%} | "
                f"Trend {trend}"
            )

        if not leak_detected:
            print(
                f"Frame {frame_number} | "
                f"No leak detected"
            )

    cap.release()

    print("\nMECHSIGHT VIDEO SUMMARY")
    print("=" * 50)
    print(f"Frames analysed: {frame_number}")
    print(f"Frames with leak: {leak_frames}")

    if frame_number > 0:
        persistence = leak_frames / frame_number
        print(f"Leak persistence: {persistence:.1%}")

if __name__ == "__main__":
    monitor_video("data/test/leak_test_video.mp4")