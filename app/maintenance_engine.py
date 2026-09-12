from statistics import mean


def build_maintenance_assessment(
    incident_history
):
    """
    Build a simple evidence-based maintenance assessment
    from previous MechSight incidents.

    This is a prototype health/trend model.
    It does NOT claim certified remaining useful life.
    """

    if not incident_history:
        return {
            "health_score": 100,
            "degradation_trend": "NO_HISTORY",
            "maintenance_priority": "LOW",
            "recommended_action": "CONTINUE_MONITORING",
            "reason": (
                "No previous inspection history is available."
            ),
        }


    # ======================================================
    # EXTRACT HISTORICAL SIGNALS
    # ======================================================

    growth_values = []

    motion_values = []

    confidence_values = []

    high_risk_count = 0


    for incident in incident_history:

        evidence = incident.get(
            "evidence",
            {},
        )

        growth_values.append(
            evidence.get(
                "late_change_region",
                0,
            )
        )

        motion_values.append(
            evidence.get(
                "average_opencv_motion",
                0,
            )
        )

        confidence_values.append(
            evidence.get(
                "average_yolo_confidence",
                0,
            )
        )

        if (
            incident.get("risk")
            == "HIGH"
        ):
            high_risk_count += 1


    # ======================================================
    # AVERAGES
    # ======================================================

    average_growth = mean(
        growth_values
    )

    average_motion = mean(
        motion_values
    )

    average_confidence = mean(
        confidence_values
    )


    # ======================================================
    # SIMPLE TREND ANALYSIS
    # ======================================================

    degradation_trend = (
        "STABLE"
    )


    if len(
        growth_values
    ) >= 2:

        first_half = (
            growth_values[
                :max(
                    1,
                    len(growth_values) // 2,
                )
            ]
        )

        second_half = (
            growth_values[
                len(growth_values) // 2:
            ]
        )


        early_average = mean(
            first_half
        )

        late_average = mean(
            second_half
        )


        if (
            late_average
            > early_average * 1.15
        ):

            degradation_trend = (
                "WORSENING"
            )

        elif (
            late_average
            < early_average * 0.85
        ):

            degradation_trend = (
                "IMPROVING"
            )


    # ======================================================
    # HEALTH SCORE
    # ======================================================

    health_score = 100


    health_score -= min(
        high_risk_count * 8,
        32,
    )


    health_score -= min(
        average_growth * 1000,
        25,
    )


    health_score -= min(
        average_motion * 100,
        15,
    )


    if (
        degradation_trend
        == "WORSENING"
    ):

        health_score -= 15


    health_score = round(
        max(
            0,
            min(
                100,
                health_score,
            ),
        ),
        1,
    )


    # ======================================================
    # MAINTENANCE DECISION
    # ======================================================

    if (
        health_score < 45
        or degradation_trend
        == "WORSENING"
    ):

        maintenance_priority = (
            "URGENT"
        )

        recommended_action = (
            "INSPECT_AND_PLAN_MAINTENANCE"
        )

        reason = (
            "Historical inspection evidence indicates "
            "declining equipment condition."
        )


    elif (
        health_score < 70
        or high_risk_count >= 2
    ):

        maintenance_priority = (
            "HIGH"
        )

        recommended_action = (
            "SCHEDULE_MAINTENANCE_REVIEW"
        )

        reason = (
            "Repeated abnormal inspection evidence "
            "suggests maintenance should be reviewed."
        )


    else:

        maintenance_priority = (
            "LOW"
        )

        recommended_action = (
            "CONTINUE_MONITORING"
        )

        reason = (
            "Historical evidence does not currently "
            "show severe deterioration."
        )


    return {
        "health_score":
            health_score,

        "degradation_trend":
            degradation_trend,

        "maintenance_priority":
            maintenance_priority,

        "recommended_action":
            recommended_action,

        "reason":
            reason,

        "history_count":
            len(
                incident_history
            ),

        "average_growth":
            round(
                average_growth,
                6,
            ),

        "average_motion":
            round(
                average_motion,
                6,
            ),

        "average_yolo_confidence":
            round(
                average_confidence,
                4,
            ),
    }