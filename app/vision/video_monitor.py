import cv2
from ultralytics import YOLO

from leak_analyzer import analyse_leak_region
from motion_analyzer import analyse_motion


MODEL_PATH = "runs/detect/train/weights/best.pt"


def monitor_video(video_path):
    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {video_path}")

    frame_number = 0
    leak_frames = 0

    area_history = []
    confidence_history = []
    motion_history = []

    previous_frame = None

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

            # Highest-confidence leak detection
            best_box = max(
                result.boxes,
                key=lambda box: float(box.conf[0])
            )

            confidence = float(best_box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                best_box.xyxy[0].tolist()
            )

            analysis = analyse_leak_region(
                frame,
                [x1, y1, x2, y2]
            )

            current_area = analysis["area_ratio"]

            # OpenCV motion analysis
            motion = {
                "motion_ratio": 0.0,
                "motion_pixels": 0,
                "motion_level": "NONE"
            }

            if previous_frame is not None:
                motion = analyse_motion(
                    previous_frame,
                    frame,
                    [x1, y1, x2, y2]
                )

            area_history.append(current_area)
            confidence_history.append(confidence)
            motion_history.append(
                motion["motion_ratio"]
            )

            leak_frames += 1
            leak_detected = True

            print(
                f"Frame {frame_number} | "
                f"Leak {confidence:.1%} | "
                f"Area {current_area:.1%} | "
                f"Motion {motion['motion_ratio']:.1%} | "
                f"{motion['motion_level']}"
            )

            break

        if not leak_detected:
            print(
                f"Frame {frame_number} | "
                f"No leak detected"
            )

        # Store this frame for comparison with next frame
        previous_frame = frame.copy()

    cap.release()

    print("\nMECHSIGHT VIDEO SUMMARY")
    print("=" * 50)

    print(f"Frames analysed: {frame_number}")
    print(f"Frames with leak: {leak_frames}")

    if frame_number == 0:
        print("No frames available.")
        return

    persistence = leak_frames / frame_number

    print(f"Leak persistence: {persistence:.1%}")

    if not area_history:
        print("Average leak area: 0.0%")
        print("Average confidence: 0.0%")
        print("Average motion: 0.0%")
        print("Trend: NONE")
        print("Risk level: LOW")
        return

    average_area = (
        sum(area_history)
        / len(area_history)
    )

    average_confidence = (
        sum(confidence_history)
        / len(confidence_history)
    )

    average_motion = (
        sum(motion_history)
        / len(motion_history)
    )

    print(f"Average leak area: {average_area:.1%}")
    print(f"Average confidence: {average_confidence:.1%}")
    print(f"Average motion: {average_motion:.1%}")

    # Compare early part of video with later part
    split_index = max(
        1,
        len(area_history) // 3
    )

    early_area = (
        sum(area_history[:split_index])
        / len(area_history[:split_index])
    )

    late_area = (
        sum(area_history[-split_index:])
        / len(area_history[-split_index:])
    )

    if late_area > early_area * 1.15:
        trend = "GROWING"

    elif late_area < early_area * 0.85:
        trend = "SHRINKING"

    else:
        trend = "STABLE"

    print(f"Trend: {trend}")

    # Prototype MechSight risk logic
    if (
        persistence >= 0.80
        and (
            trend == "GROWING"
            or average_area >= 0.20
            or average_motion >= 0.25
        )
    ):
        risk = "HIGH"

    elif (
        persistence >= 0.50
        or average_area >= 0.07
        or average_motion >= 0.08
    ):
        risk = "MEDIUM"

    else:
        risk = "LOW"

    print(f"Risk level: {risk}")


if __name__ == "__main__":
    monitor_video("data/test/growing_leak_video.mp4")