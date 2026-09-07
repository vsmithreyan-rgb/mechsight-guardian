import cv2
from ultralytics import YOLO

from leak_analyzer import analyse_leak_region
from motion_analyzer import analyse_motion
from growth_analyzer import analyse_growth, classify_growth


MODEL_PATH = "runs/detect/leak_v2/weights/best.pt"

CONFIDENCE_THRESHOLD = 0.25
MAX_MISSED_FRAMES = 5
MIN_MOTION_EVIDENCE = 0.01


def monitor_video(video_path):

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {video_path}"
        )

    frame_number = 0

    # ==================================================
    # TEMPORAL STATISTICS
    # ==================================================

    yolo_detection_frames = 0
    temporal_evidence_frames = 0
    held_detection_frames = 0

    area_history = []
    confidence_history = []
    motion_history = []

    # NEW:
    # OpenCV changing-region history
    growth_history = []

    previous_frame = None

    # Remember most recent leak location
    last_box = None
    missed_frames = 0

    # ==================================================
    # PROCESS VIDEO
    # ==================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False
        )

        detected_box = None
        confidence = None

        # ==================================================
        # 1. YOLO PERCEPTION
        # ==================================================

        for result in results:

            if len(result.boxes) == 0:
                continue

            best_box = max(
                result.boxes,
                key=lambda box: float(box.conf[0])
            )

            confidence = float(
                best_box.conf[0]
            )

            x1, y1, x2, y2 = map(
                int,
                best_box.xyxy[0].tolist()
            )

            detected_box = [
                x1,
                y1,
                x2,
                y2
            ]

            break

        # ==================================================
        # 2. YOLO DETECTED LEAK
        # ==================================================

        if detected_box is not None:

            last_box = detected_box
            missed_frames = 0

            yolo_detection_frames += 1
            temporal_evidence_frames += 1

            # ----------------------------------------------
            # LEAK REGION ANALYSIS
            # ----------------------------------------------

            analysis = analyse_leak_region(
                frame,
                detected_box
            )

            current_area = analysis[
                "area_ratio"
            ]

            area_history.append(
                (
                    frame_number,
                    current_area
                )
            )

            confidence_history.append(
                confidence
            )

            # ----------------------------------------------
            # OPENCV MOTION ANALYSIS
            # ----------------------------------------------

            motion_ratio = 0.0
            motion_level = "NONE"

            if previous_frame is not None:

                motion = analyse_motion(
                    previous_frame,
                    frame,
                    detected_box
                )

                motion_ratio = motion[
                    "motion_ratio"
                ]

                motion_level = motion[
                    "motion_level"
                ]

            motion_history.append(
                motion_ratio
            )

            # ----------------------------------------------
            # OPENCV GROWTH ANALYSIS
            # ----------------------------------------------

            growth_ratio = 0.0

            if previous_frame is not None:

                growth = analyse_growth(
                    previous_frame,
                    frame,
                    detected_box
                )

                growth_ratio = growth[
                    "largest_change_ratio"
                ]

                growth_history.append(
                    growth_ratio
                )

            print(
                f"Frame {frame_number} | "
                f"YOLO LEAK {confidence:.1%} | "
                f"Area {current_area:.1%} | "
                f"Motion {motion_ratio:.1%} | "
                f"Growth evidence {growth_ratio:.2%} | "
                f"{motion_level}"
            )

        # ==================================================
        # 3. YOLO MISSED LEAK
        # ==================================================

        else:

            missed_frames += 1

            if (
                last_box is not None
                and
                missed_frames <= MAX_MISSED_FRAMES
                and
                previous_frame is not None
            ):

                # ------------------------------------------
                # MOTION FROM PREVIOUS LEAK REGION
                # ------------------------------------------

                motion = analyse_motion(
                    previous_frame,
                    frame,
                    last_box
                )

                motion_ratio = motion[
                    "motion_ratio"
                ]

                motion_level = motion[
                    "motion_level"
                ]

                motion_history.append(
                    motion_ratio
                )

                # ------------------------------------------
                # GROWTH EVIDENCE DURING YOLO MISS
                # ------------------------------------------

                growth = analyse_growth(
                    previous_frame,
                    frame,
                    last_box
                )

                growth_ratio = growth[
                    "largest_change_ratio"
                ]

                growth_history.append(
                    growth_ratio
                )

                # ------------------------------------------
                # TEMPORAL SUPPORT
                # ------------------------------------------

                if (
                    motion_ratio
                    >= MIN_MOTION_EVIDENCE
                ):

                    temporal_evidence_frames += 1
                    held_detection_frames += 1

                    print(
                        f"Frame {frame_number} | "
                        f"YOLO MISS | "
                        f"TEMPORAL EVIDENCE | "
                        f"Motion {motion_ratio:.1%} | "
                        f"Growth evidence "
                        f"{growth_ratio:.2%} | "
                        f"{motion_level}"
                    )

                else:

                    print(
                        f"Frame {frame_number} | "
                        f"YOLO MISS | "
                        f"No strong temporal evidence | "
                        f"Growth evidence "
                        f"{growth_ratio:.2%}"
                    )

            else:

                print(
                    f"Frame {frame_number} | "
                    f"No leak evidence"
                )

            # Forget stale leak location
            if (
                missed_frames
                > MAX_MISSED_FRAMES
            ):

                last_box = None

        previous_frame = frame.copy()

    cap.release()

    # ==================================================
    # VIDEO SUMMARY
    # ==================================================

    print(
        "\nMECHSIGHT TEMPORAL VIDEO SUMMARY"
    )

    print(
        "=" * 55
    )

    print(
        f"Frames analysed: "
        f"{frame_number}"
    )

    if frame_number == 0:

        print(
            "No frames available."
        )

        return

    # ==================================================
    # PERSISTENCE
    # ==================================================

    yolo_persistence = (
        yolo_detection_frames
        / frame_number
    )

    temporal_persistence = (
        temporal_evidence_frames
        / frame_number
    )

    print(
        f"YOLO detection frames: "
        f"{yolo_detection_frames}"
    )

    print(
        f"YOLO persistence: "
        f"{yolo_persistence:.1%}"
    )

    print(
        f"Temporal support frames: "
        f"{held_detection_frames}"
    )

    print(
        f"Combined leak evidence: "
        f"{temporal_persistence:.1%}"
    )

    # ==================================================
    # AVERAGES
    # ==================================================

    if confidence_history:

        average_confidence = (
            sum(confidence_history)
            / len(confidence_history)
        )

    else:

        average_confidence = 0.0

    if area_history:

        average_area = (
            sum(
                area
                for _, area
                in area_history
            )
            / len(area_history)
        )

    else:

        average_area = 0.0

    if motion_history:

        average_motion = (
            sum(motion_history)
            / len(motion_history)
        )

    else:

        average_motion = 0.0

    print(
        f"Average detected area: "
        f"{average_area:.1%}"
    )

    print(
        f"Average YOLO confidence: "
        f"{average_confidence:.1%}"
    )

    print(
        f"Average OpenCV motion: "
        f"{average_motion:.1%}"
    )

    # ==================================================
    # ORIGINAL YOLO-AREA TREND
    # ==================================================

    if (
        len(area_history) < 6
        or
        yolo_persistence < 0.50
    ):

        trend = "UNCERTAIN"

    else:

        split_index = max(
            1,
            len(area_history) // 3
        )

        early_values = [
            area
            for _, area
            in area_history[
                :split_index
            ]
        ]

        late_values = [
            area
            for _, area
            in area_history[
                -split_index:
            ]
        ]

        early_area = (
            sum(early_values)
            / len(early_values)
        )

        late_area = (
            sum(late_values)
            / len(late_values)
        )

        if (
            late_area
            > early_area * 1.15
        ):

            trend = "GROWING"

        elif (
            late_area
            < early_area * 0.85
        ):

            trend = "SHRINKING"

        else:

            trend = "STABLE"

    print(
        f"YOLO-area trend: "
        f"{trend}"
    )

    # ==================================================
    # NEW OPENCV GROWTH TREND
    # ==================================================

    growth_result = classify_growth(
        growth_history
    )

    opencv_growth_trend = (
        growth_result["trend"]
    )

    early_growth = (
        growth_result[
            "early_growth"
        ]
    )

    late_growth = (
        growth_result[
            "late_growth"
        ]
    )

    print(
        f"OpenCV growth trend: "
        f"{opencv_growth_trend}"
    )

    print(
        f"Early changing-region size: "
        f"{early_growth:.2%}"
    )

    print(
        f"Late changing-region size: "
        f"{late_growth:.2%}"
    )

    # ==================================================
    # PROTOTYPE RISK REASONING
    # ==================================================

    if (
        temporal_persistence >= 0.80
        and
        (
            opencv_growth_trend
            == "GROWING"
            or
            average_area >= 0.20
            or
            average_motion >= 0.25
        )
    ):

        risk = "HIGH"

    elif (
        temporal_persistence >= 0.50
        or
        average_area >= 0.07
        or
        average_motion >= 0.08
    ):

        risk = "MEDIUM"

    else:

        risk = "LOW"

    print(
        f"Prototype risk level: "
        f"{risk}"
    )

    # ==================================================
    # DECISION TRACE
    # ==================================================

    print(
        "\nDECISION TRACE"
    )

    print(
        "=" * 55
    )

    print(
        f"Perception: YOLO detected leaks in "
        f"{yolo_persistence:.1%} of frames."
    )

    print(
        f"Temporal evidence increased leak "
        f"support to "
        f"{temporal_persistence:.1%}."
    )

    print(
        f"OpenCV measured average motion of "
        f"{average_motion:.1%}."
    )

    print(
        f"YOLO-area trend: "
        f"{trend}."
    )

    print(
        f"OpenCV growth analysis: "
        f"{opencv_growth_trend}."
    )

    print(
        f"Changing-region comparison: "
        f"{early_growth:.2%} early -> "
        f"{late_growth:.2%} late."
    )

    print(
        f"Decision: prototype risk = "
        f"{risk}."
    )


if __name__ == "__main__":

    monitor_video(
        "data/test/growing_leak_video.mp4"
    )