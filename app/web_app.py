import json
import os
import sys

from flask import (
    Flask,
    render_template,
    redirect,
    url_for,
    abort,
    request,
    flash,
)

from werkzeug.utils import secure_filename

from database.approval_manager import update_incident_approval


# ==========================================================
# PATH SETUP
# ==========================================================

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
VISION_DIR = os.path.join(CURRENT_DIR, "vision")

if VISION_DIR not in sys.path:
    sys.path.insert(0, VISION_DIR)


from video_monitor import monitor_video


# ==========================================================
# APP SETUP
# ==========================================================

app = Flask(__name__)

app.secret_key = "mechsight-dev-secret"

INCIDENT_FILE = "data/incidents/incidents.json"
UPLOAD_FOLDER = "data/uploads"

ALLOWED_VIDEO_EXTENSIONS = {
    "mp4",
    "avi",
    "mov",
}

os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True,
)


# ==========================================================
# HELPERS
# ==========================================================

def load_incidents():

    if not os.path.exists(INCIDENT_FILE):
        return []

    try:

        with open(
            INCIDENT_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            return json.load(file)

    except (
        json.JSONDecodeError,
        OSError,
    ):

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


def allowed_video(filename):

    return (
        "." in filename
        and
        filename.rsplit(
            ".",
            1,
        )[1].lower()
        in ALLOWED_VIDEO_EXTENSIONS
    )


# ==========================================================
# DASHBOARD
# ==========================================================

@app.route("/")
def dashboard():

    incidents = load_incidents()

    total_incidents = len(
        incidents
    )

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
        if (
            incident.get("status")
            == "APPROVED"
        )
    )

    return render_template(
        "dashboard.html",
        incidents=incidents,
        total_incidents=total_incidents,
        high_risk=high_risk,
        awaiting_approval=awaiting_approval,
        approved=approved,
    )


# ==========================================================
# VIDEO UPLOAD + ANALYSIS
# ==========================================================

@app.route(
    "/upload",
    methods=["POST"],
)
def upload_video():

    if "video" not in request.files:

        flash(
            "No video file was provided."
        )

        return redirect(
            url_for("dashboard")
        )

    video = request.files["video"]

    if video.filename == "":

        flash(
            "Please choose a video file."
        )

        return redirect(
            url_for("dashboard")
        )

    if not allowed_video(
        video.filename
    ):

        flash(
            "Unsupported file type. "
            "Please upload MP4, AVI or MOV."
        )

        return redirect(
            url_for("dashboard")
        )

    filename = secure_filename(
        video.filename
    )

    save_path = os.path.join(
        UPLOAD_FOLDER,
        filename,
    )

    video.save(
        save_path
    )

    print()
    print(
        "VIDEO UPLOAD RECEIVED"
    )
    print(
        f"Filename: {filename}"
    )
    print(
        f"Saved to: {save_path}"
    )
    print()

    # ======================================================
    # RUN MECHSIGHT ANALYSIS
    # ======================================================

    try:

        incident = monitor_video(
            save_path
        )

    except Exception as error:

        print()
        print(
            "MECHSIGHT ANALYSIS ERROR"
        )
        print(
            str(error)
        )
        print()

        flash(
            "The video was uploaded, "
            "but analysis failed. "
            "Check the terminal for details."
        )

        return redirect(
            url_for("dashboard")
        )

    # ======================================================
    # INCIDENT CREATED
    # ======================================================

    if incident is not None:

        incident_id = incident[
            "incident_id"
        ]

        print()
        print(
            "WEB WORKFLOW COMPLETE"
        )
        print(
            f"Opening incident: "
            f"{incident_id}"
        )
        print()

        return redirect(
            url_for(
                "incident_detail",
                incident_id=incident_id,
            )
        )

    # ======================================================
    # NO INCIDENT REQUIRED
    # ======================================================

    flash(
        "Video analysis completed. "
        "No incident was created."
    )

    return redirect(
        url_for("dashboard")
    )


# ==========================================================
# INCIDENT DETAIL
# ==========================================================

@app.route(
    "/incident/<incident_id>"
)
def incident_detail(
    incident_id
):

    incident = get_incident(
        incident_id
    )

    if incident is None:
        abort(404)

    evidence = incident.get(
        "evidence",
        {},
    )

    return render_template(
        "incident.html",
        incident=incident,
        evidence=evidence,
    )


# ==========================================================
# APPROVE INCIDENT
# ==========================================================

@app.route(
    "/incident/<incident_id>/approve",
    methods=["POST"],
)
def approve_incident(
    incident_id
):

    incident = get_incident(
        incident_id
    )

    if incident is None:
        abort(404)

    update_incident_approval(
        incident_id=incident_id,
        approved=True,
        reviewer="Dashboard Operator",
    )

    return redirect(
        url_for(
            "incident_detail",
            incident_id=incident_id,
        )
    )


# ==========================================================
# REJECT INCIDENT
# ==========================================================

@app.route(
    "/incident/<incident_id>/reject",
    methods=["POST"],
)
def reject_incident(
    incident_id
):

    incident = get_incident(
        incident_id
    )

    if incident is None:
        abort(404)

    update_incident_approval(
        incident_id=incident_id,
        approved=False,
        reviewer="Dashboard Operator",
    )

    return redirect(
        url_for(
            "incident_detail",
            incident_id=incident_id,
        )
    )


# ==========================================================
# RUN APP
# ==========================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000,
    )