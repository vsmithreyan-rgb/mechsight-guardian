import json
import os
from datetime import datetime

from cloud.dynamodb_manager import save_incident_to_dynamodb


INCIDENT_FILE = "data/incidents/incidents.json"


def create_incident(
    risk,
    action,
    priority,
    human_approval_required,
    reason,
    evidence,
):
    """
    Create a MechSight incident record.

    The incident is stored locally for the current
    dashboard workflow and also synchronized to
    Amazon DynamoDB when cloud storage is available.
    """

    os.makedirs(
        os.path.dirname(INCIDENT_FILE),
        exist_ok=True,
    )

    # ======================================================
    # LOAD EXISTING LOCAL INCIDENTS
    # ======================================================

    if os.path.exists(INCIDENT_FILE):

        try:

            with open(
                INCIDENT_FILE,
                "r",
                encoding="utf-8",
            ) as file:

                incidents = json.load(
                    file
                )

        except (
            json.JSONDecodeError,
            OSError,
        ):
            incidents = []

    else:
        incidents = []


    # ======================================================
    # GENERATE INCIDENT ID
    # ======================================================

    incident_id = (
        f"INC-{len(incidents) + 1:04d}"
    )


    # ======================================================
    # BUILD INCIDENT RECORD
    # ======================================================

    incident = {
        "incident_id": incident_id,

        "created_at": (
            datetime.now()
            .astimezone()
            .isoformat(
                timespec="seconds"
            )
        ),

        "status": (
            "AWAITING_APPROVAL"
            if human_approval_required
            else "OPEN"
        ),

        "risk": risk,

        "priority": priority,

        "recommended_action":
            action,

        "human_approval_required":
            human_approval_required,

        "reason": reason,

        "evidence": evidence,
    }


    # ======================================================
    # SAVE LOCALLY
    # ======================================================

    incidents.append(
        incident
    )

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


    # ======================================================
    # SAVE TO AWS DYNAMODB
    # ======================================================

    try:

        save_incident_to_dynamodb(
            incident
        )

        print()
        print(
            "DYNAMODB INCIDENT SAVE SUCCESS"
        )
        print(
            f"Incident ID: {incident_id}"
        )
        print()

    except Exception as error:

        print()
        print(
            "DYNAMODB INCIDENT SAVE ERROR"
        )
        print(
            str(error)
        )
        print()

        # Local incident remains valid even if AWS
        # is temporarily unavailable.


    return incident