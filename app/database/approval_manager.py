import json
import os


INCIDENT_FILE = "data/incidents/incidents.json"


def update_incident_approval(
    incident_id,
    approved,
    reviewer="Human Operator"
):
    """
    Update the approval status of an existing incident.

    approved=True  -> APPROVED
    approved=False -> REJECTED
    """

    if not os.path.exists(INCIDENT_FILE):

        raise FileNotFoundError(
            f"Incident file not found: {INCIDENT_FILE}"
        )

    with open(
        INCIDENT_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        incidents = json.load(file)

    matched_incident = None

    for incident in incidents:

        if (
            incident.get("incident_id")
            == incident_id
        ):

            matched_incident = incident

            incident["status"] = (
                "APPROVED"
                if approved
                else "REJECTED"
            )

            incident[
                "reviewed_by"
            ] = reviewer

            incident[
                "approval_decision"
            ] = (
                "APPROVED"
                if approved
                else "REJECTED"
            )

            break

    if matched_incident is None:

        raise ValueError(
            f"Incident not found: {incident_id}"
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

    return matched_incident