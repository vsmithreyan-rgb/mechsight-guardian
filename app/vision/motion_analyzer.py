import cv2
import numpy as np


def analyse_motion(previous_frame, current_frame, box):
    x1, y1, x2, y2 = map(int, box)

    previous_roi = previous_frame[y1:y2, x1:x2]
    current_roi = current_frame[y1:y2, x1:x2]

    if previous_roi.size == 0 or current_roi.size == 0:
        return {
            "motion_ratio": 0.0,
            "motion_pixels": 0,
            "motion_level": "NONE"
        }

    previous_gray = cv2.cvtColor(
        previous_roi,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_roi,
        cv2.COLOR_BGR2GRAY
    )

    # Reduce small camera/image noise
    previous_gray = cv2.GaussianBlur(
        previous_gray,
        (5, 5),
        0
    )

    current_gray = cv2.GaussianBlur(
        current_gray,
        (5, 5),
        0
    )

    # Difference between consecutive frames
    difference = cv2.absdiff(
        previous_gray,
        current_gray
    )

    # Keep only meaningful changes
    _, motion_mask = cv2.threshold(
        difference,
        25,
        255,
        cv2.THRESH_BINARY
    )

    # Remove tiny isolated noise
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    motion_mask = cv2.morphologyEx(
        motion_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    motion_pixels = int(
        np.count_nonzero(motion_mask)
    )

    total_pixels = motion_mask.size

    motion_ratio = (
        motion_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    if motion_ratio >= 0.25:
        motion_level = "HIGH"

    elif motion_ratio >= 0.08:
        motion_level = "MEDIUM"

    elif motion_ratio > 0.01:
        motion_level = "LOW"

    else:
        motion_level = "NONE"

    return {
        "motion_ratio": float(motion_ratio),
        "motion_pixels": motion_pixels,
        "motion_level": motion_level
    }