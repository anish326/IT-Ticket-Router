"""
Human-in-the-Loop Escalation Engine
=====================================
Evaluates escalation policies, routes to reviewers, manages approval
workflows, and maintains a full audit trail.

The AI recommends actions; it does not execute consequential changes
without human approval.

Features:
- Configurable policy rules loaded from escalation_policies.json
- Reviewer routing by role and team
- Deduplication of active escalations per ticket
- Escalation chain with timeout-based auto-escalation
- In-app notifications (no external services required)
- Full audit trail with decision history
"""

import json
import os
from datetime import datetime, timedelta, timezone

from api.database import (
    insert_escalation,
    get_escalation,
    get_escalations,
    get_escalation_counts,
    update_escalation_decision,
    acknowledge_escalation,
    snooze_escalation,
    reopen_escalation,
    update_escalation_action_effectiveness,
    check_duplicate_escalation,
    get_expired_escalations,
    get_users_by_role,
    get_db,
)

POLICIES_PATH = os.path.join(os.path.dirname(__file__), "escalation_policies.json")

# Cached policies
_policies = None


def _load_policies():
    """Load escalation policies from JSON configuration."""
    global _policies
    if _policies is not None:
        return _policies

    if not os.path.exists(POLICIES_PATH):
        _policies = {"rules": [], "escalation_chain": [], "deduplication_window_minutes": 120, "max_escalation_level": 3}
        return _policies

    with open(POLICIES_PATH, "r") as f:
        _policies = json.load(f)
    return _policies


def reload_policies():
    """Force reload policies from disk."""
    global _policies
    _policies = None
    return _load_policies()


def evaluate_escalation(prediction_result, ticket_data=None, explanation=None):
    """
    Evaluate escalation policies against a prediction result.

    Args:
        prediction_result: dict from sla_predictor.predict()
        ticket_data: original ticket data (priority, time remaining, etc.)
        explanation: dict from explainability.explain_prediction()

    Returns:
        dict with triggered (bool), escalation details, or None if no escalation needed
    """
    policies = _load_policies()
    risk_score = prediction_result.get("risk_score", 0)
    confidence = prediction_result.get("confidence", 1.0)
    ticket_id = prediction_result.get("ticket_id", "unknown")

    # Merge feature data
    features = prediction_result.get("feature_snapshot", {}).copy()
    if ticket_data:
        features.update({
            "priority": ticket_data.get("priority", "Medium"),
            "time_remaining_hours": ticket_data.get(
                "time_remaining_hours",
                features.get("time_remaining_hours", 24.0)
            ),
            "response_delay_hours": ticket_data.get(
                "response_delay_hours",
                features.get("response_delay_hours", 0)
            ),
        })

    triggered_rules = []

    for rule in policies.get("rules", []):
        conditions = rule.get("conditions", {})
        matches = True

        # Check min_risk_score
        if "min_risk_score" in conditions:
            if risk_score < conditions["min_risk_score"]:
                matches = False

        # Check max_time_remaining_hours
        if "max_time_remaining_hours" in conditions and matches:
            time_remaining = features.get("time_remaining_hours", float("inf"))
            if time_remaining > conditions["max_time_remaining_hours"]:
                matches = False

        # Check priority
        if "priority" in conditions and matches:
            ticket_priority = features.get("priority", "")
            if ticket_priority != conditions["priority"]:
                matches = False

        # Check max_confidence (low confidence trigger)
        if "max_confidence" in conditions and matches:
            if confidence > conditions["max_confidence"]:
                matches = False

        # Check min_response_delay_hours
        if "min_response_delay_hours" in conditions and matches:
            delay = features.get("response_delay_hours", 0)
            if delay < conditions["min_response_delay_hours"]:
                matches = False

        if matches:
            triggered_rules.append(rule)

    if not triggered_rules:
        return {
            "triggered": False,
            "reason": "No escalation policy triggered",
            "risk_score": risk_score,
        }

    # Use the highest-severity triggered rule
    level_priority = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    triggered_rules.sort(
        key=lambda r: level_priority.get(r.get("escalation_level", "medium"), 2),
        reverse=True,
    )
    primary_rule = triggered_rules[0]

    # Check deduplication
    dedup_window = policies.get("deduplication_window_minutes", 120)
    if check_duplicate_escalation(ticket_id, dedup_window):
        return {
            "triggered": False,
            "reason": "Active escalation already exists for this ticket",
            "risk_score": risk_score,
            "deduplicated": True,
        }

    # Determine reviewer
    reviewer_role = primary_rule.get("reviewer_role", "reviewer")
    assigned_reviewer = _route_to_reviewer(reviewer_role, features)

    # Calculate timeout
    timeout_minutes = primary_rule.get("timeout_minutes", 60)
    timeout_at = (datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)).isoformat()

    # Build explanation summary
    explanation_summary = ""
    if explanation and explanation.get("top_factors"):
        factor_labels = [f["label"] for f in explanation["top_factors"][:3]]
        explanation_summary = "; ".join(factor_labels)

    # Combine trigger conditions from all matching rules
    trigger_descriptions = [r.get("description", r.get("name", "")) for r in triggered_rules]
    trigger_conditions = " | ".join(trigger_descriptions)

    # Evidence snapshot captures full AI context at moment of escalation
    evidence_snapshot = {
        "prediction_id": prediction_result.get("prediction_id"),
        "model_version": prediction_result.get("model_version"),
        "risk_score": risk_score,
        "breach_probability": prediction_result.get("breach_probability", risk_score),
        "prediction_uncertainty": prediction_result.get("prediction_uncertainty"),
        "data_quality_score": prediction_result.get("data_quality_score"),
        "top_factors": explanation.get("top_factors", []) if explanation else [],
        "recommendation": explanation.get("recommendation", "") if explanation else "",
        "triggered_rules": [r.get("name") for r in triggered_rules],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    # Create escalation
    escalation_data = {
        "ticket_id": ticket_id,
        "prediction_id": prediction_result.get("prediction_id"),
        "risk_score": risk_score,
        "trigger_conditions": trigger_conditions,
        "assigned_reviewer": assigned_reviewer,
        "escalation_level": primary_rule.get("escalation_level", "medium"),
        "status": "pending_ack",
        "timeout_at": timeout_at,
        "escalation_chain_level": 1,
        "explanation_summary": explanation_summary,
        "evidence_snapshot": evidence_snapshot,
    }

    escalation_id = insert_escalation(escalation_data)

    return {
        "triggered": True,
        "escalation_id": escalation_id,
        "level": primary_rule.get("escalation_level", "medium"),
        "reason": trigger_conditions,
        "assigned_reviewer": assigned_reviewer,
        "timeout_at": timeout_at,
        "action": primary_rule.get("action", "review"),
        "risk_score": risk_score,
        "evidence_snapshot": evidence_snapshot,
        "triggered_rules": [r.get("name") for r in triggered_rules],
    }


def _route_to_reviewer(role, features=None):
    """
    Determine the appropriate reviewer based on role and ticket context.
    """
    users = get_users_by_role(role)
    if not users:
        # Fallback to admin if no users with requested role
        users = get_users_by_role("admin")

    if not users:
        return "Unassigned (no reviewers available)"

    # Simple round-robin or team-based routing
    if features and features.get("assigned_team"):
        team = features.get("assigned_team", "")
        team_users = [u for u in users if u.get("team", "") == team]
        if team_users:
            return team_users[0]["display_name"]

    # Default to first available reviewer
    return users[0]["display_name"]


def acknowledge_escalation_flow(escalation_id, reviewer_name):
    """Reviewer acknowledges escalation, shifting status from pending_ack to under_review."""
    esc = get_escalation(escalation_id)
    if not esc:
        raise ValueError(f"Escalation {escalation_id} not found")
    success = acknowledge_escalation(escalation_id, reviewer_name)
    if not success:
        raise ValueError(f"Could not acknowledge escalation {escalation_id}. Current status: {esc.get('status')}")
    return {"escalation_id": escalation_id, "status": "under_review", "reviewer": reviewer_name}


def snooze_escalation_flow(escalation_id, duration_minutes, snooze_reason, reviewer_name):
    """
    Temporarily defer an escalation.
    POLICY ENFORCEMENT: Critical escalations cannot be snoozed for more than 15 minutes.
    """
    esc = get_escalation(escalation_id)
    if not esc:
        raise ValueError(f"Escalation {escalation_id} not found")

    level = (esc.get("escalation_level") or "").lower()
    if level == "critical" and duration_minutes > 15:
        raise ValueError("Critical escalations cannot be snoozed for more than 15 minutes per operational SLA policy.")

    if not snooze_reason or len(snooze_reason.strip()) < 5:
        raise ValueError("A clear snooze reason is required to defer an escalation.")

    snoozed_until_iso = (datetime.now(timezone.utc) + timedelta(minutes=duration_minutes)).isoformat()
    success = snooze_escalation(escalation_id, snoozed_until_iso, snooze_reason, reviewer_name)
    if not success:
        raise ValueError(f"Failed to snooze escalation {escalation_id}.")
    return {
        "escalation_id": escalation_id,
        "status": "snoozed",
        "snoozed_until": snoozed_until_iso,
        "snooze_reason": snooze_reason,
    }


def reopen_escalation_flow(escalation_id, reopen_reason, reviewer_name):
    """Reopen an escalation if risk remains unresolved."""
    esc = get_escalation(escalation_id)
    if not esc:
        raise ValueError(f"Escalation {escalation_id} not found")

    if not reopen_reason or len(reopen_reason.strip()) < 5:
        raise ValueError("A clear justification reason is required to reopen an escalation.")

    success = reopen_escalation(escalation_id, reopen_reason, reviewer_name)
    if not success:
        raise ValueError(f"Failed to reopen escalation {escalation_id}.")
    return {"escalation_id": escalation_id, "status": "reopened", "reopened_by": reviewer_name}


def record_action_effectiveness_flow(escalation_id, completed, effective, notes, reviewer_name):
    """Record whether approved action was completed and whether it helped."""
    esc = get_escalation(escalation_id)
    if not esc:
        raise ValueError(f"Escalation {escalation_id} not found")

    success = update_escalation_action_effectiveness(
        escalation_id=escalation_id,
        completed=completed,
        effective=effective,
        notes=notes,
        reviewer_name=reviewer_name,
    )
    if not success:
        raise ValueError(f"Failed to update action effectiveness for escalation {escalation_id}.")
    return {
        "escalation_id": escalation_id,
        "action_completed": bool(completed),
        "action_effective": effective,
        "notes": notes,
    }


def decide_escalation(escalation_id, decision, reason, reviewer_name, expected_version=None):
    """
    Record a reviewer's decision on an escalation with optimistic concurrency protection.

    Valid decisions: approve, reject, reassign, investigate
    """
    valid_decisions = {"approve", "reject", "reassign", "investigate"}
    if decision not in valid_decisions:
        raise ValueError(
            f"Invalid decision '{decision}'. Must be one of: {valid_decisions}"
        )

    # Verify escalation exists and can be decided
    esc = get_escalation(escalation_id)
    if not esc:
        raise ValueError(f"Escalation {escalation_id} not found")

    allowed_decide_statuses = {"pending", "pending_ack", "under_review", "snoozed", "reopened"}
    if esc["status"] not in allowed_decide_statuses:
        raise ValueError(
            f"Escalation {escalation_id} is already in final status '{esc['status']}', cannot decide"
        )

    update_escalation_decision(escalation_id, decision, reason, reviewer_name, expected_version=expected_version)
    return {
        "escalation_id": escalation_id,
        "decision": decision,
        "status": "approved" if decision == "approve" else (decision if decision != "investigate" else "investigating"),
        "decided_at": datetime.now(timezone.utc).isoformat(),
    }


def process_timeouts():
    """
    Process expired escalations by advancing them through the escalation chain.

    Returns a list of escalations that were auto-escalated.
    """
    policies = _load_policies()
    chain = policies.get("escalation_chain", [])
    max_level = policies.get("max_escalation_level", 3)

    expired = get_expired_escalations()
    auto_escalated = []

    for esc in expired:
        current_level = esc.get("escalation_chain_level", 1)

        if current_level >= max_level:
            # Max level reached — mark as expired
            with get_db() as conn:
                conn.execute(
                    """UPDATE escalation_logs
                       SET status = 'expired',
                           decision = 'timeout_expired',
                           decision_reason = 'Maximum escalation level reached without response',
                           decided_at = ?
                       WHERE id = ?""",
                    (datetime.now(timezone.utc).isoformat(), esc["id"]),
                )
            auto_escalated.append({
                "original_id": esc["id"],
                "action": "expired",
                "level": current_level,
            })
            continue

        # Find next level in chain
        next_level = current_level + 1
        next_chain = None
        for c in chain:
            if c.get("level") == next_level:
                next_chain = c
                break

        if not next_chain:
            continue

        # Mark current as timed out
        with get_db() as conn:
            conn.execute(
                """UPDATE escalation_logs
                   SET status = 'expired',
                       decision = 'timeout_auto_escalated',
                       decision_reason = 'No response within timeout period. Auto-escalated to next level.',
                       decided_at = ?
                   WHERE id = ?""",
                (datetime.now(timezone.utc).isoformat(), esc["id"]),
            )

        # Create new escalation at next level
        reviewer_role = next_chain.get("role", "admin")
        assigned_reviewer = _route_to_reviewer(reviewer_role)
        timeout_minutes = 60  # Default for escalated items

        timeout_at = (datetime.now(timezone.utc) + timedelta(minutes=timeout_minutes)).isoformat()

        new_esc_data = {
            "ticket_id": esc["ticket_id"],
            "prediction_id": esc.get("prediction_id"),
            "risk_score": esc["risk_score"],
            "trigger_conditions": f"Auto-escalated from level {current_level}: {esc['trigger_conditions']}",
            "assigned_reviewer": assigned_reviewer,
            "escalation_level": esc.get("escalation_level", "high"),
            "timeout_at": timeout_at,
            "escalation_chain_level": next_level,
            "parent_escalation_id": esc["id"],
            "explanation_summary": esc.get("explanation_summary"),
        }

        new_id = insert_escalation(new_esc_data)
        auto_escalated.append({
            "original_id": esc["id"],
            "new_id": new_id,
            "action": "auto_escalated",
            "from_level": current_level,
            "to_level": next_level,
            "new_reviewer": assigned_reviewer,
        })

    return auto_escalated


def get_escalation_detail(escalation_id):
    """Get full escalation detail including linked prediction explanation."""
    esc = get_escalation(escalation_id)
    if not esc:
        return None

    # Fetch linked prediction explanation if available
    if esc.get("prediction_id"):
        from api.database import get_prediction
        prediction = get_prediction(esc["prediction_id"])
        if prediction:
            esc["prediction_detail"] = prediction

    # Fetch child escalations (chain history)
    with get_db() as conn:
        children = conn.execute(
            """SELECT * FROM escalation_logs
               WHERE parent_escalation_id = ?
               ORDER BY created_at ASC""",
            (escalation_id,),
        ).fetchall()
        esc["escalation_history"] = [dict(c) for c in children]

    return esc


def get_notifications(reviewer_name=None):
    """
    Get pending in-app notifications for escalations.
    Shows pending escalations assigned to the given reviewer or all if None.
    """
    with get_db() as conn:
        if reviewer_name:
            rows = conn.execute(
                """SELECT * FROM escalation_logs
                   WHERE status = 'pending' AND assigned_reviewer = ?
                   ORDER BY created_at DESC""",
                (reviewer_name,),
            ).fetchall()
        else:
            rows = conn.execute(
                """SELECT * FROM escalation_logs
                   WHERE status = 'pending'
                   ORDER BY created_at DESC"""
            ).fetchall()

    return [dict(r) for r in rows]
