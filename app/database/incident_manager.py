import json
import os
from datetime import datetime


INCIDENT_FILE = "data/incidents/incidents.json"


def create_incident(
    risk,
    action,
    priority,
    human_approval_required,
    reason,
    evidence
):
    """
    Create a local MechSight incident record.

    Prototype storage only.
    Later this can be replaced by AWS-backed storage.
    """

    os.makedirs(
        os.path.dirname(INCIDENT_FILE),
        exist_ok=True
    )

    # Load existing incidents
    if os.path.exists(INCIDENT_FILE):

        try:
            with open(
                INCIDENT_FILE,
                "r",
                encoding="utf-8"
            ) as file:

                incidents = json.load(file)

        except (
            json.JSONDecodeError,
            OSError
        ):
            incidents = []

    else:
        incidents = []

    # Generate simple incident ID
    incident_id = (
        f"INC-{len(incidents) + 1:04d}"
    )

    incident = {
        "incident_id": incident_id,

        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat(timespec="seconds")
        ),

        "status": (
            "AWAITING_APPROVAL"
            if human_approval_required
            else "OPEN"
        ),

        "risk": risk,
        "priority": priority,

        "recommended_action": action,

        "human_approval_required":
            human_approval_required,

        "reason": reason,

        "evidence": evidence
    }

    incidents.append(
        incident
    )

    with open(
        INCIDENT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            incidents,
            file,
            indent=4
        )

    return incident