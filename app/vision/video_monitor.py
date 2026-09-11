import json
import os
import sys

import cv2
from ultralytics import YOLO


# ==========================================================
# PROJECT PATH SETUP
# ==========================================================

CURRENT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

APP_DIR = os.path.dirname(
    CURRENT_DIR
)

if APP_DIR not in sys.path:
    sys.path.insert(
        0,
        APP_DIR,
    )


# ==========================================================
# MECHSIGHT MODULES
# ==========================================================

from leak_analyzer import analyse_leak_region
from motion_analyzer import analyse_motion
from growth_analyzer import (
    analyse_growth,
    classify_growth,
)
from evidence_capture import save_evidence_frame

from decision_engine import make_decision
from database.incident_manager import create_incident
from cloud.dynamodb_manager import save_incident_to_dynamodb


# ==========================================================
# SETTINGS
# ==========================================================

MODEL_PATH = "runs/detect/leak_v2/weights/best.pt"

INCIDENT_FILE = "data/incidents/incidents.json"

CONFIDENCE_THRESHOLD = 0.25
MAX_MISSED_FRAMES = 5
MIN_MOTION_EVIDENCE = 0.01


# ==========================================================
# LOCAL INCIDENT UPDATE
# ==========================================================

def update_local_incident(updated_incident):
    """
    Update an already-created incident in the
    local JSON store after evidence images have
    been generated.
    """

    if not os.path.exists(INCIDENT_FILE):
        return

    try:
        with open(
            INCIDENT_FILE,
            "r",
            encoding="utf-8",
        ) as file:
            incidents = json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):
        return

    for index, incident in enumerate(incidents):

        if (
            incident.get("incident_id")
            == updated_incident.get("incident_id")
        ):
            incidents[index] = updated_incident
            break

    with open(
        INCIDENT_FILE,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            incidents,
            file,
            indent=4,
        )


# ==========================================================
# VIDEO MONITOR
# ==========================================================

def monitor_video(video_path):

    model = YOLO(MODEL_PATH)

    cap = cv2.VideoCapture(
        video_path
    )

    if not cap.isOpened():
        raise FileNotFoundError(
            f"Could not open video: {video_path}"
        )


    frame_number = 0


    # ======================================================
    # STATISTICS
    # ======================================================

    yolo_detection_frames = 0
    temporal_evidence_frames = 0
    held_detection_frames = 0

    area_history = []
    confidence_history = []
    motion_history = []
    growth_history = []

    previous_frame = None
    last_box = None
    missed_frames = 0


    # ======================================================
    # VISUAL EVIDENCE CAPTURE
    # ======================================================

    early_evidence_frame = None
    early_evidence_box = None

    late_evidence_frame = None
    late_evidence_box = None


    # ======================================================
    # PROCESS VIDEO
    # ======================================================

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1


        # --------------------------------------------------
        # YOLO INFERENCE
        # --------------------------------------------------

        results = model(
            frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
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
                key=lambda box: float(
                    box.conf[0]
                ),
            )


            confidence = float(
                best_box.conf[0]
            )


            x1, y1, x2, y2 = map(
                int,
                best_box.xyxy[0].tolist(),
            )


            detected_box = [
                x1,
                y1,
                x2,
                y2,
            ]

            break


        # ==================================================
        # 2. LEAK DETECTED
        # ==================================================

        if detected_box is not None:

            last_box = detected_box
            missed_frames = 0

            yolo_detection_frames += 1
            temporal_evidence_frames += 1


            # ----------------------------------------------
            # SAVE EARLY / LATE FRAME IN MEMORY
            # ----------------------------------------------

            if early_evidence_frame is None:

                early_evidence_frame = frame.copy()
                early_evidence_box = detected_box.copy()


            late_evidence_frame = frame.copy()
            late_evidence_box = detected_box.copy()


            # ----------------------------------------------
            # LEAK REGION ANALYSIS
            # ----------------------------------------------

            analysis = analyse_leak_region(
                frame,
                detected_box,
            )


            current_area = analysis[
                "area_ratio"
            ]


            area_history.append(
                (
                    frame_number,
                    current_area,
                )
            )


            confidence_history.append(
                confidence
            )


            # ----------------------------------------------
            # OPENCV MOTION
            # ----------------------------------------------

            motion_ratio = 0.0
            motion_level = "NONE"


            if previous_frame is not None:

                motion = analyse_motion(
                    previous_frame,
                    frame,
                    detected_box,
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
            # OPENCV GROWTH
            # ----------------------------------------------

            growth_ratio = 0.0


            if previous_frame is not None:

                growth = analyse_growth(
                    previous_frame,
                    frame,
                    detected_box,
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
                f"Growth {growth_ratio:.2%} | "
                f"{motion_level}"
            )


        # ==================================================
        # 3. YOLO MISS
        # ==================================================

        else:

            missed_frames += 1


            if (
                last_box is not None
                and missed_frames <= MAX_MISSED_FRAMES
                and previous_frame is not None
            ):

                motion = analyse_motion(
                    previous_frame,
                    frame,
                    last_box,
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


                growth = analyse_growth(
                    previous_frame,
                    frame,
                    last_box,
                )


                growth_ratio = growth[
                    "largest_change_ratio"
                ]


                growth_history.append(
                    growth_ratio
                )


                if motion_ratio >= MIN_MOTION_EVIDENCE:

                    temporal_evidence_frames += 1
                    held_detection_frames += 1


                    print(
                        f"Frame {frame_number} | "
                        f"YOLO MISS | "
                        f"TEMPORAL EVIDENCE | "
                        f"Motion {motion_ratio:.1%} | "
                        f"Growth {growth_ratio:.2%} | "
                        f"{motion_level}"
                    )

                else:

                    print(
                        f"Frame {frame_number} | "
                        f"YOLO MISS | "
                        f"No strong temporal evidence | "
                        f"Growth {growth_ratio:.2%}"
                    )

            else:

                print(
                    f"Frame {frame_number} | "
                    f"No leak evidence"
                )


            if missed_frames > MAX_MISSED_FRAMES:
                last_box = None


        previous_frame = frame.copy()


    cap.release()


    # ======================================================
    # VIDEO SUMMARY
    # ======================================================

    print(
        "\nMECHSIGHT TEMPORAL VIDEO SUMMARY"
    )

    print(
        "=" * 55
    )

    print(
        f"Frames analysed: {frame_number}"
    )


    if frame_number == 0:

        print(
            "No frames available."
        )

        return None


    # ======================================================
    # PERSISTENCE
    # ======================================================

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


    # ======================================================
    # AVERAGES
    # ======================================================

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


    # ======================================================
    # YOLO AREA TREND
    # ======================================================

    if (
        len(area_history) < 6
        or yolo_persistence < 0.50
    ):

        yolo_trend = "UNCERTAIN"

    else:

        split_index = max(
            1,
            len(area_history) // 3,
        )


        early_values = [
            area
            for _, area
            in area_history[:split_index]
        ]


        late_values = [
            area
            for _, area
            in area_history[-split_index:]
        ]


        early_area = (
            sum(early_values)
            / len(early_values)
        )


        late_area = (
            sum(late_values)
            / len(late_values)
        )


        if late_area > early_area * 1.15:

            yolo_trend = "GROWING"

        elif late_area < early_area * 0.85:

            yolo_trend = "SHRINKING"

        else:

            yolo_trend = "STABLE"


    print(
        f"YOLO-area trend: "
        f"{yolo_trend}"
    )


    # ======================================================
    # OPENCV GROWTH TREND
    # ======================================================

    growth_result = classify_growth(
        growth_history
    )


    opencv_growth_trend = (
        growth_result[
            "trend"
        ]
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


    # ======================================================
    # PROTOTYPE RISK
    # ======================================================

    if (
        temporal_persistence >= 0.80
        and
        (
            opencv_growth_trend == "GROWING"
            or average_area >= 0.20
            or average_motion >= 0.25
        )
    ):

        risk = "HIGH"

    elif (
        temporal_persistence >= 0.50
        or average_area >= 0.07
        or average_motion >= 0.08
    ):

        risk = "MEDIUM"

    else:

        risk = "LOW"


    print(
        f"Prototype risk level: "
        f"{risk}"
    )


    # ======================================================
    # AGENT DECISION
    # ======================================================

    decision = make_decision(
        risk=risk,
        growth_trend=opencv_growth_trend,
        persistence=temporal_persistence,
        average_confidence=average_confidence,
        average_motion=average_motion,
    )


    action = decision[
        "action"
    ]


    priority = decision[
        "priority"
    ]


    approval_required = decision[
        "human_approval_required"
    ]


    reason = decision[
        "reason"
    ]


    # ======================================================
    # DECISION TRACE
    # ======================================================

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
        f"Temporal evidence increased leak support "
        f"to {temporal_persistence:.1%}."
    )

    print(
        f"OpenCV measured average motion of "
        f"{average_motion:.1%}."
    )

    print(
        f"YOLO-area trend: "
        f"{yolo_trend}."
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


    # ======================================================
    # AGENT ACTION
    # ======================================================

    print(
        "\nAGENT ACTION"
    )

    print(
        "=" * 55
    )

    print(
        f"Prototype risk: "
        f"{risk}"
    )

    print(
        f"Recommended action: "
        f"{action}"
    )

    print(
        f"Priority: "
        f"{priority}"
    )

    print(
        f"Human approval required: "
        f"{'YES' if approval_required else 'NO'}"
    )

    print(
        f"Reason: "
        f"{reason}"
    )


    # ======================================================
    # INCIDENT WORKFLOW
    # ======================================================

    incident_actions = {
        "ESCALATE_AND_INSPECT",
        "REQUEST_HUMAN_INSPECTION",
        "SCHEDULE_REINSPECTION",
    }


    if action in incident_actions:

        evidence = {

            "video_source":
                video_path,

            "frames_analysed":
                frame_number,

            "yolo_persistence":
                round(
                    yolo_persistence,
                    4,
                ),

            "combined_leak_evidence":
                round(
                    temporal_persistence,
                    4,
                ),

            "average_yolo_confidence":
                round(
                    average_confidence,
                    4,
                ),

            "average_detected_area":
                round(
                    average_area,
                    4,
                ),

            "average_opencv_motion":
                round(
                    average_motion,
                    4,
                ),

            "yolo_area_trend":
                yolo_trend,

            "opencv_growth_trend":
                opencv_growth_trend,

            "early_change_region":
                round(
                    early_growth,
                    6,
                ),

            "late_change_region":
                round(
                    late_growth,
                    6,
                ),
        }


        # ==================================================
        # CREATE INCIDENT
        # ==================================================

        incident = create_incident(
            risk=risk,
            action=action,
            priority=priority,
            human_approval_required=(
                approval_required
            ),
            reason=reason,
            evidence=evidence,
        )


        incident_id = (
            incident[
                "incident_id"
            ]
        )


        # ==================================================
        # SAVE VISUAL EVIDENCE FRAMES
        # ==================================================

        if (
            early_evidence_frame
            is not None
        ):

            early_path = (
                save_evidence_frame(
                    early_evidence_frame,
                    incident_id,
                    "early",
                    early_evidence_box,
                )
            )


            incident[
                "evidence"
            ][
                "early_evidence_frame"
            ] = early_path


        if (
            late_evidence_frame
            is not None
        ):

            late_path = (
                save_evidence_frame(
                    late_evidence_frame,
                    incident_id,
                    "late",
                    late_evidence_box,
                )
            )


            incident[
                "evidence"
            ][
                "late_evidence_frame"
            ] = late_path


        # ==================================================
        # UPDATE LOCAL INCIDENT
        # ==================================================

        update_local_incident(
            incident
        )


        # ==================================================
        # UPDATE DYNAMODB AGAIN WITH FRAME PATHS
        # ==================================================

        try:

            save_incident_to_dynamodb(
                incident
            )

            print()
            print(
                "VISUAL EVIDENCE "
                "SYNCED TO DYNAMODB"
            )

        except Exception as error:

            print()
            print(
                "VISUAL EVIDENCE "
                "DYNAMODB SYNC ERROR"
            )

            print(
                str(error)
            )


        # ==================================================
        # INCIDENT SUMMARY
        # ==================================================

        print(
            "\nINCIDENT CREATED"
        )

        print(
            "=" * 55
        )

        print(
            f"Incident ID: "
            f"{incident_id}"
        )

        print(
            f"Status: "
            f"{incident['status']}"
        )

        print(
            "Stored evidence: YES"
        )


        if (
            early_evidence_frame
            is not None
        ):

            print(
                f"Early evidence frame: "
                f"{incident['evidence']['early_evidence_frame']}"
            )


        if (
            late_evidence_frame
            is not None
        ):

            print(
                f"Late evidence frame: "
                f"{incident['evidence']['late_evidence_frame']}"
            )


        print(
            "Incident record saved to:"
        )

        print(
            INCIDENT_FILE
        )


        return incident


    print(
        "\nNo incident record required."
    )

    return None


# ==========================================================
# DIRECT TERMINAL TEST
# ==========================================================

if __name__ == "__main__":

    monitor_video(
        "data/test/growing_leak_video.mp4"
    )