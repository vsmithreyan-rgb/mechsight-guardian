import json
import os

from cloud.dynamodb_manager import update_incident_status


INCIDENT_FILE = "data/incidents/incidents.json"


def update_incident_approval(
    incident_id,
    approved,
    reviewer="Human Operator",
):
    """
    Update an incident approval decision locally
    and synchronize the decision to AWS DynamoDB.
    """

    if not os.path.exists(
        INCIDENT_FILE
    ):
        return None

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
        return None


    updated_incident = None

    for incident in incidents:

        if (
            incident.get(
                "incident_id"
            )
            == incident_id
        ):

            if approved:

                incident["status"] = (
                    "APPROVED"
                )

                incident[
                    "approval_decision"
                ] = "APPROVED"

            else:

                incident["status"] = (
                    "REJECTED"
                )

                incident[
                    "approval_decision"
                ] = "REJECTED"


            incident[
                "reviewed_by"
            ] = reviewer

            updated_incident = (
                incident
            )

            break


    if updated_incident is None:
        return None


    # ======================================================
    # SAVE LOCAL JSON
    # ======================================================

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
    # UPDATE AWS DYNAMODB
    # ======================================================

    try:

        update_incident_status(
            incident_id=incident_id,
            status=updated_incident[
                "status"
            ],
            reviewer=reviewer,
            approval_decision=
                updated_incident[
                    "approval_decision"
                ],
        )

        print()
        print(
            "DYNAMODB APPROVAL UPDATE SUCCESS"
        )

        print(
            f"Incident ID: {incident_id}"
        )

        print(
            f"Status: "
            f"{updated_incident['status']}"
        )

        print()

    except Exception as error:

        print()
        print(
            "DYNAMODB APPROVAL UPDATE ERROR"
        )
        print(
            str(error)
        )
        print()

        # Local approval remains valid even if
        # cloud synchronization temporarily fails.


    return updated_incident