def make_decision(
    risk,
    growth_trend,
    persistence,
    average_confidence,
    average_motion
):
    """
    Prototype MechSight Guardian decision engine.

    Converts visual evidence into a recommended next action.

    IMPORTANT:
    These rules are for prototype/demo use only.
    They are not industrial safety thresholds.
    """

    # --------------------------------------------------
    # HIGH-RISK PROTOTYPE CONDITION
    # --------------------------------------------------

    if risk == "HIGH":

        if growth_trend == "GROWING":

            return {
                "action": "ESCALATE_AND_INSPECT",
                "priority": "URGENT",
                "human_approval_required": True,
                "reason": (
                    "Persistent leak evidence with an "
                    "expanding changing region was detected."
                )
            }

        return {
            "action": "REQUEST_HUMAN_INSPECTION",
            "priority": "HIGH",
            "human_approval_required": True,
            "reason": (
                "High prototype risk was detected, "
                "but growth evidence is not conclusive."
            )
        }

    # --------------------------------------------------
    # MEDIUM-RISK PROTOTYPE CONDITION
    # --------------------------------------------------

    if risk == "MEDIUM":

        if persistence >= 0.80:

            return {
                "action": "SCHEDULE_REINSPECTION",
                "priority": "MEDIUM",
                "human_approval_required": False,
                "reason": (
                    "Leak evidence persisted through most "
                    "of the inspected video."
                )
            }

        return {
            "action": "CONTINUE_MONITORING",
            "priority": "MEDIUM",
            "human_approval_required": False,
            "reason": (
                "Leak evidence exists but is not yet "
                "strong enough for escalation."
            )
        }

    # --------------------------------------------------
    # LOW-RISK / LOW-EVIDENCE CONDITION
    # --------------------------------------------------

    if (
        average_confidence < 0.30
        and
        average_motion < 0.01
    ):

        return {
            "action": "RECHECK_VISUAL_INPUT",
            "priority": "LOW",
            "human_approval_required": False,
            "reason": (
                "Visual evidence is weak. A clearer image "
                "or another inspection may be required."
            )
        }

    return {
        "action": "CONTINUE_MONITORING",
        "priority": "LOW",
        "human_approval_required": False,
        "reason": (
            "Current visual evidence does not justify "
            "escalation."
        )
    }