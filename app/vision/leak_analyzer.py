import cv2
import numpy as np


def analyse_leak_region(image, box):
    x1, y1, x2, y2 = map(int, box)

    roi = image[y1:y2, x1:x2]

    if roi.size == 0:
        return {
            "area_ratio": 0.0,
            "edge_density": 0.0,
            "brightness": 0.0,
            "severity": "UNKNOWN"
        }

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray, 50, 150)

    edge_density = np.count_nonzero(edges) / edges.size

    brightness = float(np.mean(gray))

    image_area = image.shape[0] * image.shape[1]
    leak_area = (x2 - x1) * (y2 - y1)

    area_ratio = leak_area / image_area

    if area_ratio > 0.20:
        severity = "HIGH"
    elif area_ratio > 0.07:
        severity = "MEDIUM"
    else:
        severity = "LOW"

    return {
        "area_ratio": float(area_ratio),
        "edge_density": float(edge_density),
        "brightness": brightness,
        "severity": severity
    }