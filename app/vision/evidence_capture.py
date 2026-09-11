import os
from pathlib import Path

import cv2


EVIDENCE_FOLDER = Path("data/evidence")


def ensure_evidence_folder():
    EVIDENCE_FOLDER.mkdir(
        parents=True,
        exist_ok=True,
    )


def save_evidence_frame(
    frame,
    incident_id,
    label,
    box=None,
):
    """
    Save a visual evidence frame for an incident.

    label examples:
    - early
    - late
    """

    ensure_evidence_folder()

    output_frame = frame.copy()

    if box is not None:
        x1, y1, x2, y2 = map(
            int,
            box,
        )

        cv2.rectangle(
            output_frame,
            (x1, y1),
            (x2, y2),
            (0, 0, 255),
            3,
        )

        cv2.putText(
            output_frame,
            f"Leak Evidence - {label.upper()}",
            (x1, max(y1 - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )

    filename = (
        f"{incident_id}_{label}.jpg"
    )

    output_path = (
        EVIDENCE_FOLDER
        / filename
    )

    cv2.imwrite(
        str(output_path),
        output_frame,
    )

    return str(output_path)