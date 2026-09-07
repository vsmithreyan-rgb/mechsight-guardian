import cv2
import numpy as np


def expand_box(box, frame_shape, expansion=0.60):
    """
    Expand the YOLO leak box so OpenCV analyses motion
    around the leak, not only inside the detector box.
    """

    x1, y1, x2, y2 = map(int, box)

    height, width = frame_shape[:2]

    box_width = x2 - x1
    box_height = y2 - y1

    expand_x = int(box_width * expansion)
    expand_y = int(box_height * expansion)

    new_x1 = max(0, x1 - expand_x)
    new_y1 = max(0, y1 - expand_y)

    new_x2 = min(width, x2 + expand_x)

    # Expand further downward because water often
    # flows/sprays below the leak origin.
    new_y2 = min(
        height,
        y2 + int(expand_y * 1.8)
    )

    return [
        new_x1,
        new_y1,
        new_x2,
        new_y2
    ]


def analyse_motion(previous_frame, current_frame, box):

    expanded_box = expand_box(
        box,
        current_frame.shape
    )

    x1, y1, x2, y2 = expanded_box

    previous_roi = previous_frame[
        y1:y2,
        x1:x2
    ]

    current_roi = current_frame[
        y1:y2,
        x1:x2
    ]

    if (
        previous_roi.size == 0
        or current_roi.size == 0
    ):

        return {
            "motion_ratio": 0.0,
            "motion_pixels": 0,
            "motion_level": "NONE",
            "expanded_box": expanded_box
        }

    previous_gray = cv2.cvtColor(
        previous_roi,
        cv2.COLOR_BGR2GRAY
    )

    current_gray = cv2.cvtColor(
        current_roi,
        cv2.COLOR_BGR2GRAY
    )

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

    difference = cv2.absdiff(
        previous_gray,
        current_gray
    )

    # Lower than previous threshold because
    # water motion can create subtle brightness changes.
    _, motion_mask = cv2.threshold(
        difference,
        15,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    motion_mask = cv2.morphologyEx(
        motion_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    motion_mask = cv2.morphologyEx(
        motion_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    motion_pixels = int(
        np.count_nonzero(
            motion_mask
        )
    )

    total_pixels = motion_mask.size

    motion_ratio = (
        motion_pixels / total_pixels
        if total_pixels > 0
        else 0.0
    )

    if motion_ratio >= 0.20:

        motion_level = "HIGH"

    elif motion_ratio >= 0.06:

        motion_level = "MEDIUM"

    elif motion_ratio >= 0.01:

        motion_level = "LOW"

    else:

        motion_level = "NONE"

    return {
        "motion_ratio": float(
            motion_ratio
        ),
        "motion_pixels": motion_pixels,
        "motion_level": motion_level,
        "expanded_box": expanded_box
    }