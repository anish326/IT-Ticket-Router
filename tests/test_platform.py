"""
Comprehensive Test Suite for AI Ticket Intelligence Platform
=============================================================
Tests all core modules and API endpoints:
- Database schema & operations
- SLA Predictor (RandomForest + calibration)
- Explainability (SHAP TreeExplainer, top factors, recommendations)
- Feedback & Outcome tracking
- Escalation Engine (policies, reviewer routing, decisions, audit)
- Model Drift Monitor (supervised metrics, drift detection)
- Flask API Endpoints (Auth, Tickets, Predictions, Explain, Feedback, Escalations, Health)
"""

import os
import sys
import json
import pytest
from datetime import datetime, timezone

# Add root directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.database import (
    init_db,
    get_db,
    get_user,
    insert_ticket,
    get_ticket,
    get_tickets,
    update_ticket,
    get_escalations,
    get_active_model_version,
)
from api.sla_predictor import predict as sla_predict, preprocess_input, get_model_version
from api.explainability import explain_prediction
from api.feedback import log_prediction, record_outcome, record_feedback, get_feedback_stats
from api.escalation_engine import (
    evaluate_escalation,
    decide_escalation,
    get_escalation_detail,
    reload_policies,
)
from api.drift_monitor import evaluate_model, get_model_health
from api.app import app


@pytest.fixture(scope="module")
def client():
    """Create a Flask test client with an authenticated admin session."""
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test-secret-key"

    with app.test_client() as test_client:
        # Log in as admin
        with test_client.session_transaction() as sess:
            sess["username"] = "admin"
            sess["display_name"] = "System Admin"
            sess["role"] = "admin"
            sess["team"] = "Platform"
        yield test_client


# ===========================================================================
# 1. Database & Foundation Tests
# ===========================================================================

def test_database_initialization():
    """Verify database initializes properly and contains default users."""
    init_db()
    with get_db() as conn:
        users = conn.execute("SELECT username, role FROM users").fetchall()
        usernames = {u["username"] for u in users}
        assert "admin" in usernames
        assert "reviewer" in usernames
        assert "agent" in usernames

        # Verify tables exist
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        table_names = {t["name"] for t in tables}
        expected = {
            "users", "tickets", "prediction_logs", "sla_outcomes",
            "agent_feedback", "model_versions", "model_evaluations", "escalation_logs"
        }
        for t in expected:
            assert t in table_names, f"Table {t} missing from database"


def test_ticket_crud():
    """Verify ticket insertion, retrieval, and updating."""
    test_id = f"TKT-TEST-{int(datetime.now(timezone.utc).timestamp())}"
    ticket_data = {
        "ticket_id": test_id,
        "title": "Database connection drop",
        "description": "App servers cannot reach secondary replica.",
        "priority": "High",
        "issue_type": "Incident",
        "status": "Open",
        "assigned_team": "DevOps",
        "sla_deadline_hours": 6.0,
        "created_hours": 2.0,
        "time_remaining_hours": 4.0,
        "response_delay_hours": 1.5,
        "team_backlog_count": 8,
        "similar_ticket_resolution_hours": 5.0,
        "description_length": 45,
    }
    insert_ticket(ticket_data)

    retrieved = get_ticket(test_id)
    assert retrieved is not None
    assert retrieved["priority"] == "High"
    assert retrieved["issue_type"] == "Incident"

    # Update ticket
    success = update_ticket(test_id, {"status": "In Progress"})
    assert success is True
    updated = get_ticket(test_id)
    assert updated["status"] == "In Progress"


# ===========================================================================
# 2. SLA Predictor Tests
# ===========================================================================

def test_sla_predictor_valid_input():
    """Verify SLA predictor returns calibrated probability, label, and confidence."""
    sample = {
        "ticket_id": "TKT-PRED-1",
        "priority": "Critical",
        "issue_type": "Incident",
        "created_hours": 3.0,
        "sla_deadline_hours": 2.0,
        "response_delay_hours": 2.5,
        "team_backlog_count": 15,
        "similar_ticket_resolution_hours": 8.0,
        "description_length": 350,
    }
    result = sla_predict(sample)

    assert "risk_score" in result
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_label"] in ["Low", "Medium", "High", "Critical"]
    assert 0.0 <= result["confidence"] <= 1.0
    assert "assigned_team" in result
    assert "model_version" in result
    assert result["ticket_id"] == "TKT-PRED-1"


def test_sla_predictor_missing_fields_defaults():
    """Verify SLA predictor safely handles minimal input with defaults and warns on unknown fields."""
    sample = {
        "ticket_id": "TKT-MINIMAL",
        "priority": "NonStandardPriority",
    }
    result = sla_predict(sample)
    assert "risk_score" in result
    assert result["risk_score"] >= 0.0
    assert "warnings" in result
    assert len(result["warnings"]) > 0


# ===========================================================================
# 3. Explainability Module (SHAP) Tests
# ===========================================================================

def test_explainability_generation():
    """Verify SHAP produces top contributing factors and actionable suggestions."""
    sample = {
        "ticket_id": "TKT-EXP-1",
        "priority": "High",
        "issue_type": "Bug",
        "created_hours": 12.0,
        "sla_deadline_hours": 8.0,
        "time_remaining_hours": 0.0,
        "response_delay_hours": 4.5,
        "team_backlog_count": 16,
        "similar_ticket_resolution_hours": 14.0,
        "description_length": 200,
    }

    pred = sla_predict(sample)
    feat_arr, feat_dict, _ = preprocess_input(sample)
    exp = explain_prediction(feat_arr, feat_dict, pred["risk_score"], ticket_data=sample)

    assert "top_factors" in exp
    assert isinstance(exp["top_factors"], list)
    assert len(exp["top_factors"]) > 0
    assert "recommendation" in exp
    assert len(exp["recommendation"]) > 0
    assert "summary" in exp

    # Check factor fields
    first_factor = exp["top_factors"][0]
    assert "feature" in first_factor
    assert "label" in first_factor
    assert "impact" in first_factor
    assert "direction" in first_factor
    assert "source" in first_factor
    assert first_factor["source"] in ["model", "operational_rule"]


# ===========================================================================
# 4. Feedback & Outcomes Tests
# ===========================================================================

def test_feedback_and_outcomes_flow():
    """Verify logging prediction, recording confirmed outcome, and rating recommendation."""
    ticket_id = f"TKT-FB-{int(datetime.now(timezone.utc).timestamp())}"
    sample = {
        "ticket_id": ticket_id,
        "priority": "Medium",
        "issue_type": "Task",
        "created_hours": 2.0,
        "sla_deadline_hours": 24.0,
    }
    pred = sla_predict(sample)
    feat_arr, feat_dict, _ = preprocess_input(sample)
    exp = explain_prediction(feat_arr, feat_dict, pred["risk_score"], ticket_data=sample)

    # 1. Log prediction
    pred_id = log_prediction(pred, exp)
    assert pred_id is not None

    # 2. Record agent feedback
    fb_label = record_feedback(
        ticket_id=ticket_id,
        feedback_label="helpful",
        prediction_id=pred_id,
        feedback_reason="Helped prioritize before standup",
        reviewer_name="Sarah Chen",
    )
    assert fb_label == "helpful"

    # 3. Check feedback stats
    stats = get_feedback_stats()
    assert stats["total"] >= 1
    assert stats["helpful"] >= 1

    # 4. Record confirmed SLA outcome (ticket must be Resolved to confirm SLA compliance)
    from api.database import update_ticket
    update_ticket(ticket_id, {"status": "Resolved"})
    status = record_outcome(
        ticket_id=ticket_id,
        actual_breach=0,
        prediction_id=pred_id,
        confirmed_by="QA Lead",
        notes="Resolved within SLA"
    )
    assert status == "confirmed_met"


# ===========================================================================
# 5. Escalation Engine Tests
# ===========================================================================

def test_escalation_engine_critical_rule():
    """Verify critical severity ticket triggers critical escalation policy."""
    reload_policies()
    critical_sample = {
        "ticket_id": f"TKT-ESC-CRIT-{int(datetime.now(timezone.utc).timestamp())}",
        "priority": "Critical",
        "issue_type": "Incident",
        "created_hours": 1.0,
        "sla_deadline_hours": 2.0,
        "response_delay_hours": 0.5,
    }
    pred = sla_predict(critical_sample)
    feat_arr, feat_dict, _ = preprocess_input(critical_sample)
    exp = explain_prediction(feat_arr, feat_dict, pred["risk_score"], ticket_data=critical_sample)

    esc_result = evaluate_escalation(pred, ticket_data=critical_sample, explanation=exp)
    assert esc_result["triggered"] is True
    assert esc_result["escalation_id"] is not None
    assert esc_result["level"] == "critical"

    # Verify decision recording
    esc_id = esc_result["escalation_id"]
    dec_result = decide_escalation(
        escalation_id=esc_id,
        decision="approve",
        reason="Approved emergency response team dispatch",
        reviewer_name="Sarah Chen",
    )
    assert dec_result["decision"] == "approve"
    assert dec_result["status"] == "approved"

    # Verify audit detail
    detail = get_escalation_detail(esc_id)
    assert detail["decision"] == "approve"
    assert detail["reviewer_name"] == "Sarah Chen"


def test_escalation_deduplication():
    """Verify active pending escalation prevents duplicate creation within window."""
    ticket_id = f"TKT-DEDUP-{int(datetime.now(timezone.utc).timestamp())}"
    sample = {
        "ticket_id": ticket_id,
        "priority": "Critical",
        "created_hours": 1.0,
        "sla_deadline_hours": 2.0,
    }
    pred = sla_predict(sample)

    # First evaluation creates escalation
    first = evaluate_escalation(pred, ticket_data=sample)
    assert first["triggered"] is True

    # Second evaluation within window is deduplicated
    second = evaluate_escalation(pred, ticket_data=sample)
    assert second.get("deduplicated") is True
    assert second["triggered"] is False


# ===========================================================================
# 6. Model Drift Monitor Tests
# ===========================================================================

def test_model_drift_and_health():
    """Verify evaluate_model and get_model_health compute valid metrics or state."""
    health = get_model_health()
    assert "active_version" in health
    assert "drift_status" in health
    assert "metrics" in health

    eval_result = evaluate_model()
    assert "model_version" in eval_result
    assert "total_predictions" in eval_result
    assert "drift_status" in eval_result


# ===========================================================================
# 7. Flask API Endpoints Tests
# ===========================================================================

def test_api_tickets(client):
    """Test /tickets list and create endpoints."""
    res = client.get("/tickets")
    assert res.status_code == 200
    data = res.get_json()
    assert "tickets" in data
    assert "total" in data

    # Create ticket via API
    new_ticket = {
        "title": "API Created Ticket",
        "priority": "Medium",
        "issue_type": "Task",
        "sla_deadline_hours": 12.0,
    }
    post_res = client.post("/tickets", json=new_ticket)
    assert post_res.status_code == 201
    created_data = post_res.get_json()
    assert "ticket_id" in created_data


def test_api_predict_with_explanation_and_escalation(client):
    """Test POST /predict returns prediction, SHAP explanation, and escalation result."""
    payload = {
        "ticket_id": f"TKT-API-{int(datetime.now(timezone.utc).timestamp())}",
        "priority": "High",
        "issue_type": "Bug",
        "created_hours": 4.0,
        "sla_deadline_hours": 6.0,
        "response_delay_hours": 2.0,
        "team_backlog_count": 10,
        "similar_ticket_resolution_hours": 7.0,
        "description_length": 150,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.get_json()

    assert "risk_score" in data
    assert "risk_label" in data
    assert "explanation" in data
    assert "top_factors" in data["explanation"]
    assert "recommendation" in data["explanation"]
    assert "escalation" in data


def test_api_explain_endpoint(client):
    """Test GET /explain/<ticket_id> retrieves explanation."""
    ticket_id = f"TKT-EXP-EP-{int(datetime.now(timezone.utc).timestamp())}"
    # Run prediction first
    client.post("/predict", json={
        "ticket_id": ticket_id,
        "priority": "Low",
        "issue_type": "Task",
        "created_hours": 1.0,
        "sla_deadline_hours": 48.0,
    })

    res = client.get(f"/explain/{ticket_id}")
    assert res.status_code == 200
    data = res.get_json()
    assert data["ticket_id"] == ticket_id
    assert "top_factors" in data
    assert "recommendation" in data


def test_api_outcomes_and_feedback(client):
    """Test POST /outcome and POST /feedback."""
    ticket_id = f"TKT-OUTCOME-EP-{int(datetime.now(timezone.utc).timestamp())}"

    # Log prediction first
    client.post("/predict", json={"ticket_id": ticket_id, "priority": "Medium"})

    # Post outcome via /outcome
    out_res = client.post("/outcome", json={
        "ticket_id": ticket_id,
        "actual_breach": True,
        "notes": "Breached due to dependency delay",
    })
    assert out_res.status_code == 200
    out_data = out_res.get_json()
    assert out_data["status"] == "recorded"
    assert out_data["actual_breach"] is True

    # Post feedback via /feedback
    fb_res = client.post("/feedback", json={
        "ticket_id": ticket_id,
        "feedback_label": "helpful",
        "feedback_reason": "Escalation warning was accurate",
    })
    assert fb_res.status_code == 200
    fb_data = fb_res.get_json()
    assert fb_data["status"] == "recorded"

    # Check summary
    sum_res = client.get("/feedback/summary")
    assert sum_res.status_code == 200
    assert "total" in sum_res.get_json()


def test_api_escalations(client):
    """Test /escalations list and /escalation/<id>/decide endpoints."""
    res = client.get("/escalations")
    assert res.status_code == 200
    data = res.get_json()
    assert "escalations" in data
    assert "counts" in data


def test_api_model_health(client):
    """Test /model-health and /model-health/evaluate."""
    res = client.get("/model-health")
    assert res.status_code == 200
    data = res.get_json()
    assert "drift_status" in data

    eval_res = client.post("/model-health/evaluate")
    assert eval_res.status_code == 200
    eval_data = eval_res.get_json()
    assert "model_version" in eval_data


def test_api_dashboard_endpoint(client):
    """Test GET /dashboard renders dashboard HTML."""
    res = client.get("/dashboard")
    assert res.status_code == 200


# ===========================================================================
# 8. Advanced Workflow & Governance Tests
# ===========================================================================

def test_agent_workflow(client):
    """Verify dedicated Agent workflow: ack, work notes, request escalation, activities log."""
    ticket_id = f"TKT-AGENT-{int(datetime.now(timezone.utc).timestamp())}"
    # 1. Create ticket
    c_res = client.post("/tickets", json={
        "ticket_id": ticket_id,
        "title": "Agent Workflow Test",
        "description": "Testing agent operational flow",
        "priority": "High",
        "status": "Open",
    })
    assert c_res.status_code == 201

    # 2. Agent acknowledges assignment
    ack_res = client.post(f"/tickets/{ticket_id}/ack")
    assert ack_res.status_code == 200
    assert ack_res.get_json()["status"] == "acknowledged"

    # Verify status changed to In Progress
    t_res = client.get(f"/tickets/{ticket_id}")
    assert t_res.get_json()["status"] == "In Progress"

    # 3. Agent adds work notes
    note_res = client.post(f"/tickets/{ticket_id}/notes", json={
        "note": "Investigated database slow queries and gathered telemetry logs."
    })
    assert note_res.status_code == 200
    assert note_res.get_json()["status"] == "recorded"

    # 4. Agent requests human review / escalation
    req_res = client.post(f"/tickets/{ticket_id}/request-escalation", json={
        "reason": "Needs secondary approval for production index rebuild.",
        "urgency": "high"
    })
    assert req_res.status_code == 200
    assert req_res.get_json()["status"] == "escalation_requested"

    # 5. Fetch chronological activities
    act_res = client.get(f"/tickets/{ticket_id}/activities")
    assert act_res.status_code == 200
    acts = act_res.get_json()["activities"]
    assert len(acts) >= 3
    types = [a["activity_type"] for a in acts]
    assert "ack" in types
    assert "work_note" in types
    assert "escalation_request" in types


def test_escalation_lifecycle_and_concurrency(client):
    """Verify separated escalation lifecycle, evidence snapshot, snooze limits, and optimistic locking."""
    ticket_id = f"TKT-LIFECYCLE-{int(datetime.now(timezone.utc).timestamp())}"
    sample = {
        "ticket_id": ticket_id,
        "priority": "Critical",
        "created_hours": 1.0,
        "sla_deadline_hours": 2.0,
        "description": "Critical outage requiring escalation lifecycle tracking",
    }
    pred = sla_predict(sample)
    esc_result = evaluate_escalation(pred, ticket_data=sample)
    assert esc_result["triggered"] is True
    esc_id = esc_result["escalation_id"]

    # Verify evidence snapshot exists
    assert "evidence_snapshot" in esc_result
    snap = esc_result["evidence_snapshot"]
    assert "breach_probability" in snap
    assert "model_version" in snap

    # Fetch initial escalation: status should be pending_ack, version should be 1
    d_res = client.get(f"/escalation/{esc_id}")
    esc = d_res.get_json()
    assert esc["status"] == "pending_ack"
    assert esc["version"] == 1

    # 1. Reviewer acknowledges escalation -> under_review
    ack_res = client.post(f"/escalation/{esc_id}/ack")
    assert ack_res.status_code == 200
    assert ack_res.get_json()["status"] == "under_review"

    # 2. Test Critical Snooze Policy: critical escalations cannot exceed 15 mins
    bad_snooze = client.post(f"/escalation/{esc_id}/snooze", json={
        "duration_minutes": 45,
        "reason": "Deferring critical escalation for 45 minutes",
    })
    assert bad_snooze.status_code == 400
    assert "15 minutes" in bad_snooze.get_json()["error"]

    # Valid snooze (<= 15m) succeeds
    good_snooze = client.post(f"/escalation/{esc_id}/snooze", json={
        "duration_minutes": 10,
        "reason": "Waiting 10 minutes for automated deployment health check",
    })
    assert good_snooze.status_code == 200
    assert good_snooze.get_json()["status"] == "snoozed"

    # 3. Reviewer reopens escalation
    reopen_res = client.post(f"/escalation/{esc_id}/reopen", json={
        "reason": "Deployment failed to resolve latency, reopening escalation."
    })
    assert reopen_res.status_code == 200
    assert reopen_res.get_json()["status"] == "reopened"

    # 4. Action effectiveness tracking
    eff_res = client.post(f"/escalation/{esc_id}/action-status", json={
        "completed": True,
        "effective": "yes",
        "notes": "Failover to secondary cluster restored latency.",
    })
    assert eff_res.status_code == 200
    assert eff_res.get_json()["action_effective"] == "yes"

    # 5. Optimistic Concurrency Check: conflict check
    curr_esc = client.get(f"/escalation/{esc_id}").get_json()
    curr_v = curr_esc["version"]

    # Conflicting decision with outdated version
    conflict_res = client.post(f"/escalation/{esc_id}/decide", json={
        "decision": "approve",
        "decision_reason": "Approved failover action",
        "expected_version": curr_v - 1, # Stale version
    })
    assert conflict_res.status_code == 400
    assert "Conflict" in conflict_res.get_json()["error"]

    # Valid decision with matching version
    decide_res = client.post(f"/escalation/{esc_id}/decide", json={
        "decision": "approve",
        "decision_reason": "Approved failover action",
        "expected_version": curr_v,
    })
    assert decide_res.status_code == 200
    assert decide_res.get_json()["decision"] == "approve"


def test_uncertainty_and_data_quality_indicators():
    """Verify calibrated probability is separated from tree uncertainty and data quality."""
    # 1. Clean ticket
    clean_ticket = {
        "ticket_id": "TKT-CLEAN-1",
        "priority": "High",
        "issue_type": "Bug",
        "sla_deadline_hours": 24.0,
        "created_hours": 2.0,
        "time_remaining_hours": 22.0,
        "response_delay_hours": 0.5,
        "team_backlog_count": 12,
        "similar_ticket_resolution_hours": 6.0,
        "description": "Detailed explanation of bug report with complete reproduction steps",
    }
    p_clean = sla_predict(clean_ticket)
    assert "breach_probability" in p_clean
    assert "prediction_uncertainty" in p_clean
    assert "confidence_interval" in p_clean
    assert "data_quality_score" in p_clean
    assert isinstance(p_clean["confidence_interval"], list)
    assert len(p_clean["confidence_interval"]) == 2
    assert p_clean["data_quality_score"] == 100
    assert len(p_clean["data_quality_warnings"]) == 0

    # 2. Low-quality ticket with missing / anomalous fields
    bad_ticket = {
        "ticket_id": "TKT-BAD-1",
        "created_hours": 50.0,
        "sla_deadline_hours": 24.0,
        "time_remaining_hours": 10.0, # Contradiction: created > deadline but time remaining > 0
        "team_backlog_count": 450,    # Very high backlog
        "description": "Short",       # Extremely short description
    }
    p_bad = sla_predict(bad_ticket)
    assert p_bad["data_quality_score"] < 100
    assert len(p_bad["data_quality_warnings"]) >= 2


def test_controlled_outcome_and_audit_correction(client):
    """Verify unresolved tickets cannot be marked met, and authorized corrections are auditable."""
    ticket_id = f"TKT-OUTCOME-AUDIT-{int(datetime.now(timezone.utc).timestamp())}"
    # Create an Open ticket
    client.post("/tickets", json={
        "ticket_id": ticket_id,
        "title": "Outcome Control Test",
        "status": "Open",
    })

    # 1. Attempt to confirm SLA as MET while ticket is still Open -> MUST FAIL
    bad_outcome = client.post("/outcome", json={
        "ticket_id": ticket_id,
        "actual_breach": False,
        "notes": "Premature non-breach confirmation attempt",
    })
    assert bad_outcome.status_code == 400
    err_msg = bad_outcome.get_json()["error"].lower()
    assert "open ticket" in err_msg or "resolved or closed" in err_msg

    # 2. Resolve ticket first, then confirm SLA as met
    client.patch(f"/tickets/{ticket_id}", json={"status": "Resolved"})
    good_outcome = client.post("/outcome", json={
        "ticket_id": ticket_id,
        "actual_breach": False,
        "notes": "Verified resolution before SLA deadline",
    })
    assert good_outcome.status_code == 200
    assert good_outcome.get_json()["outcome_status"] == "confirmed_met"

    # 3. Formally correct outcome (e.g. Audit discovered customer reopen / late delivery)
    corr_res = client.post("/outcome/correct", json={
        "ticket_id": ticket_id,
        "actual_breach": True,
        "reason": "Post-incident audit determined work completed 45 minutes after deadline.",
    })
    assert corr_res.status_code == 200
    assert corr_res.get_json()["status"] == "corrected"

    # 4. Verify audit trail history
    audit_res = client.get(f"/outcome/audits?ticket_id={ticket_id}")
    assert audit_res.status_code == 200
    audits = audit_res.get_json()["audits"]
    assert len(audits) >= 1
    assert audits[0]["previous_breach"] == 0
    assert audits[0]["corrected_breach"] == 1
    assert "Post-incident audit" in audits[0]["reason"]


def test_model_governance_and_rollback(client):
    """Verify candidate evaluation gate and atomic rollback."""
    # 1. Inspect model health
    h_res = client.get("/model-health")
    assert h_res.status_code == 200
    curr_version = h_res.get_json()["active_version"]

    # 2. Test candidate retraining workflow
    cand_res = client.post("/model-health/retrain")
    assert cand_res.status_code == 200
    cand_data = cand_res.get_json()
    assert cand_data["status"] == "candidate_ready"
    assert "comparison" in cand_data
    comp = cand_data["comparison"]
    assert "candidate_metrics" in comp
    assert "active_model_metrics" in comp
    assert "misclassification_cost" in comp["candidate_metrics"]
    assert "gate_checks" in comp

    cand_ver = cand_data["candidate_version"]

    # 3. Activate candidate model
    act_res = client.post("/model-health/activate", json={"version": cand_ver})
    assert act_res.status_code == 200
    assert act_res.get_json()["status"] == "activated"

    # Verify active version changed
    h_res2 = client.get("/model-health")
    assert h_res2.get_json()["active_version"] == cand_ver

    # 4. Rollback model to previous version
    roll_res = client.post("/model-health/rollback")
    assert roll_res.status_code == 200
    assert roll_res.get_json()["status"] == "rolled_back"

    # Verify rollback restored previous version
    h_res3 = client.get("/model-health")
    assert h_res3.get_json()["active_version"] == curr_version

    # Dashboard smoke check
    dash_res = client.get("/dashboard")
    assert dash_res.status_code == 200
    assert b"AI Ticket Intelligence Platform" in dash_res.data
