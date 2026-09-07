import json
import os

from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    abort
)

from database.approval_manager import (
    update_incident_approval
)


app = Flask(__name__)

INCIDENT_FILE = "data/incidents/incidents.json"


def load_incidents():

    if not os.path.exists(INCIDENT_FILE):
        return []

    try:

        with open(
            INCIDENT_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            incidents = json.load(file)

        return incidents

    except (json.JSONDecodeError, OSError):

        return []


def get_incident(incident_id):

    incidents = load_incidents()

    for incident in incidents:

        if (
            incident.get("incident_id")
            == incident_id
        ):
            return incident

    return None


# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/")
def dashboard():

    incidents = load_incidents()

    total_incidents = len(incidents)

    high_risk = sum(
        1
        for incident in incidents
        if incident.get("risk") == "HIGH"
    )

    awaiting_approval = sum(
        1
        for incident in incidents
        if (
            incident.get("status")
            == "AWAITING_APPROVAL"
        )
    )

    approved = sum(
        1
        for incident in incidents
        if incident.get("status") == "APPROVED"
    )

    return render_template(
        "dashboard.html",
        incidents=incidents,
        total_incidents=total_incidents,
        high_risk=high_risk,
        awaiting_approval=awaiting_approval,
        approved=approved
    )


# ==========================================================
# INCIDENT DETAIL
# ==========================================================

@app.route(
    "/incident/<incident_id>"
)
def incident_detail(incident_id):

    incident = get_incident(
        incident_id
    )

    if incident is None:
        abort(404)

    evidence = incident.get(
        "evidence",
        {}
    )

    return render_template(
        "incident.html",
        incident=incident,
        evidence=evidence
    )


# ==========================================================
# APPROVE INCIDENT
# ==========================================================

@app.route(
    "/incident/<incident_id>/approve",
    methods=["POST"]
)
def approve_incident(incident_id):

    incident = get_incident(
        incident_id
    )

    if incident is None:
        abort(404)

    update_incident_approval(
        incident_id=incident_id,
        approved=True,
        reviewer="Dashboard Operator"
    )

    return redirect(
        url_for(
            "incident_detail",
            incident_id=incident_id
        )
    )


# ==========================================================
# REJECT INCIDENT
# ==========================================================

@app.route(
    "/incident/<incident_id>/reject",
    methods=["POST"]
)
def reject_incident(incident_id):

    incident = get_incident(
        incident_id
    )

    if incident is None:
        abort(404)

    update_incident_approval(
        incident_id=incident_id,
        approved=False,
        reviewer="Dashboard Operator"
    )

    return redirect(
        url_for(
            "incident_detail",
            incident_id=incident_id
        )
    )


# ==========================================================
# RUN APP
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )