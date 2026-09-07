import cv2
import numpy as np

from motion_analyzer import expand_box


def analyse_growth(previous_frame, current_frame, box):
    """
    Measure how much of the leak region is actively changing.

    This does NOT use YOLO box size as leak size.
    Instead, OpenCV measures the changing pixel region
    between consecutive frames.
    """

    expanded_box = expand_box(
        box,
        current_frame.shape,
        expansion=0.60
    )

    x1, y1, x2, y2 = expanded_box

    previous_roi = previous_frame[y1:y2, x1:x2]
    current_roi = current_frame[y1:y2, x1:x2]

    if (
        previous_roi.size == 0
        or current_roi.size == 0
    ):
        return {
            "change_ratio": 0.0,
            "largest_change_ratio": 0.0,
            "contour_count": 0,
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

    _, change_mask = cv2.threshold(
        difference,
        12,
        255,
        cv2.THRESH_BINARY
    )

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_OPEN,
        kernel
    )

    change_mask = cv2.morphologyEx(
        change_mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        change_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    roi_area = change_mask.shape[0] * change_mask.shape[1]

    changed_pixels = int(
        np.count_nonzero(change_mask)
    )

    change_ratio = (
        changed_pixels / roi_area
        if roi_area > 0
        else 0.0
    )

    largest_contour_area = 0.0

    for contour in contours:
        area = cv2.contourArea(contour)

        if area > largest_contour_area:
            largest_contour_area = area

    largest_change_ratio = (
        largest_contour_area / roi_area
        if roi_area > 0
        else 0.0
    )

    return {
        "change_ratio": float(change_ratio),
        "largest_change_ratio": float(
            largest_change_ratio
        ),
        "contour_count": len(contours),
        "expanded_box": expanded_box
    }


def classify_growth(growth_history):
    """
    Compare early vs late changing-region size.
    """

    if len(growth_history) < 10:
        return {
            "trend": "UNCERTAIN",
            "early_growth": 0.0,
            "late_growth": 0.0
        }

    split_index = max(
        1,
        len(growth_history) // 3
    )

    early_values = growth_history[
        :split_index
    ]

    late_values = growth_history[
        -split_index:
    ]

    early_growth = (
        sum(early_values)
        / len(early_values)
    )

    late_growth = (
        sum(late_values)
        / len(late_values)
    )

    # Avoid making strong claims when almost
    # no changing region is visible.
    if early_growth < 0.001 and late_growth < 0.001:

        trend = "NO SIGNIFICANT CHANGE"

    elif late_growth > early_growth * 1.20:

        trend = "GROWING"

    elif late_growth < early_growth * 0.80:

        trend = "SHRINKING"

    else:

        trend = "STABLE"

    return {
        "trend": trend,
        "early_growth": float(early_growth),
        "late_growth": float(late_growth)
    }