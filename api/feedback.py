"""
Feedback Module
================
Handles prediction logging, confirmed SLA outcomes, and agent feedback.

Key design decisions:
- Predictions are logged BEFORE outcomes are known
- Outcomes are recorded separately when SLA deadline passes or ticket resolves
- Agent feedback rates the recommendation quality, not the SLA outcome
- An unresolved ticket is not treated as a successful prediction
"""

import json
from datetime import datetime, timezone
from api.database import (
    insert_prediction,
    insert_outcome,
    insert_feedback,
    get_prediction,
    get_predictions_for_ticket,
    get_feedback_summary,
    get_ticket,
    update_ticket,
    insert_ticket_activity,
    get_ticket_activities,
    acknowledge_ticket,
    correct_outcome_record,
    get_outcome_audits,
    get_db,
)


def log_prediction(prediction_result, explanation=None):
    """
    Log a prediction with its feature snapshot and optional explanation.

    Args:
        prediction_result: dict from sla_predictor.predict()
        explanation: dict from explainability.explain_prediction()

    Returns:
        prediction_id
    """
    prediction_data = {
        "prediction_id": prediction_result["prediction_id"],
        "ticket_id": prediction_result["ticket_id"],
        "predicted_at": prediction_result.get(
            "predicted_at", datetime.now(timezone.utc).isoformat()
        ),
        "risk_score": prediction_result["risk_score"],
        "risk_label": prediction_result["risk_label"],
        "decision_threshold": prediction_result.get("decision_threshold", 0.5),
        "model_version": prediction_result["model_version"],
        "feature_snapshot": prediction_result.get("feature_snapshot", {}),
        "shap_explanation": explanation,
        "recommendation": explanation.get("recommendation", "") if explanation else "",
        "prediction_uncertainty": prediction_result.get("prediction_uncertainty"),
        "data_quality_score": prediction_result.get("data_quality_score"),
        "data_quality_warnings": prediction_result.get("data_quality_warnings"),
    }

    return insert_prediction(prediction_data)


def record_outcome(ticket_id, actual_breach, prediction_id=None, confirmed_by=None, notes=None):
    """
    Record a confirmed SLA outcome with validation.

    CRITICAL RULE: An unresolved ticket CANNOT be confirmed as a non-breach (met),
    because open tickets can still breach in the future.
    """
    ticket = get_ticket(ticket_id)
    actual_breach_int = 1 if actual_breach else 0

    if actual_breach_int == 0 and ticket:
        curr_status = (ticket.get("status") or "").strip().lower()
        if curr_status not in ("resolved", "closed"):
            raise ValueError(
                f"Cannot confirm SLA as met for an open ticket (status: '{ticket.get('status')}'). "
                f"Tickets must be Resolved or Closed before confirming SLA compliance."
            )

    if prediction_id is None:
        preds = get_predictions_for_ticket(ticket_id)
        if preds:
            prediction_id = preds[0]["prediction_id"]

    if actual_breach_int == 1:
        outcome_status = "confirmed_breach"
    else:
        outcome_status = "confirmed_met"

    outcome_data = {
        "ticket_id": ticket_id,
        "prediction_id": prediction_id,
        "actual_breach": actual_breach_int,
        "outcome_status": outcome_status,
        "outcome_confirmed_at": datetime.now(timezone.utc).isoformat(),
        "confirmed_by": confirmed_by,
        "notes": notes,
    }

    insert_outcome(outcome_data)
    # Sync with ticket record
    update_ticket(ticket_id, {"sla_breach": actual_breach_int})
    return outcome_status


def correct_outcome(ticket_id, new_actual_breach, corrected_by, reason):
    """
    Formally correct an SLA outcome with full append-only audit tracking.
    """
    if not reason or len(reason.strip()) < 5:
        raise ValueError("A substantial justification reason is required to correct an SLA outcome.")
    return correct_outcome_record(ticket_id, new_actual_breach, corrected_by, reason)


def record_ticket_activity(ticket_id, user_name, activity_type, note, metadata=None):
    """Log an agent activity event (ack, work note, status update, escalation request)."""
    return insert_ticket_activity(ticket_id, user_name, activity_type, note, metadata)


def get_ticket_activity_history(ticket_id):
    """Retrieve chronological activity events for a ticket."""
    return get_ticket_activities(ticket_id)


def get_outcome_audit_history(ticket_id=None):
    """Retrieve outcome correction audit trail."""
    return get_outcome_audits(ticket_id)


def record_feedback(ticket_id, feedback_label, prediction_id=None,
                    feedback_reason=None, reviewer_name=None):
    """
    Record agent feedback on a recommendation.

    Valid labels: 'helpful', 'not_helpful', 'incorrect'

    Agent feedback is about recommendation quality, NOT ground-truth SLA outcome.
    """
    valid_labels = {"helpful", "not_helpful", "incorrect"}
    if feedback_label not in valid_labels:
        raise ValueError(
            f"Invalid feedback_label '{feedback_label}'. Must be one of: {valid_labels}"
        )

    feedback_data = {
        "ticket_id": ticket_id,
        "prediction_id": prediction_id,
        "feedback_label": feedback_label,
        "feedback_reason": feedback_reason,
        "reviewer_name": reviewer_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    insert_feedback(feedback_data)
    return feedback_label


def get_prediction_detail(prediction_id):
    """Get full prediction detail including feature snapshot and explanation."""
    return get_prediction(prediction_id)


def get_ticket_predictions(ticket_id):
    """Get all predictions for a ticket, newest first."""
    return get_predictions_for_ticket(ticket_id)


def get_feedback_stats():
    """Get aggregated feedback statistics."""
    return get_feedback_summary()


def get_outcome_stats():
    """Get aggregated outcome statistics."""
    with get_db() as conn:
        total = conn.execute("SELECT COUNT(*) FROM sla_outcomes").fetchone()[0]
        breached = conn.execute(
            "SELECT COUNT(*) FROM sla_outcomes WHERE actual_breach = 1"
        ).fetchone()[0]
        met = conn.execute(
            "SELECT COUNT(*) FROM sla_outcomes WHERE actual_breach = 0"
        ).fetchone()[0]

    return {
        "total_outcomes": total,
        "breached": breached,
        "met": met,
        "breach_rate": round(breached / total, 3) if total > 0 else 0.0,
    }
