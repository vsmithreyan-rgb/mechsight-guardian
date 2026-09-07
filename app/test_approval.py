from database.approval_manager import update_incident_approval


incident = update_incident_approval(
    incident_id="INC-0001",
    approved=True
)

print("HUMAN APPROVAL UPDATE")
print("=" * 50)

print(
    f"Incident ID: "
    f"{incident['incident_id']}"
)

print(
    f"Status: "
    f"{incident['status']}"
)

print(
    f"Reviewed by: "
    f"{incident['reviewed_by']}"
)

print(
    f"Decision: "
    f"{incident['approval_decision']}"
)