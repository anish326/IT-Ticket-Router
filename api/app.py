"""
AI Ticket Intelligence Platform — Flask API
=============================================
Endpoints for ticket management, SLA prediction with SHAP explanations,
feedback/outcome tracking, model health monitoring, human-in-the-loop
escalation, and an interactive dashboard.

No external services required. Runs fully offline with SQLite persistence.
"""

import sys
import os

# Ensure project root is in sys.path regardless of how the script is launched
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import io
import csv
import json
import secrets
from functools import wraps
from datetime import datetime, timezone, timedelta

from flask import (
    Flask,
    request,
    jsonify,
    render_template,
    session,
    redirect,
    url_for,
)
from werkzeug.security import check_password_hash

from api.database import (
    init_db,
    get_db,
    get_user,
    get_ticket,
    get_tickets,
    insert_ticket,
    update_ticket,
    acknowledge_ticket,
    insert_ticket_activity,
    get_ticket_activities,
    insert_escalation,
    get_escalations as db_get_escalations,
    get_escalation_counts,
    get_model_versions,
    activate_model_version,
    get_prediction,
)
from api.sla_predictor import predict as sla_predict, preprocess_input, get_model_version, reload_model
from api.explainability import explain_prediction, reset_explainer
from api.feedback import (
    log_prediction,
    record_outcome,
    correct_outcome,
    get_outcome_audit_history,
    record_feedback,
    get_prediction_detail,
    get_ticket_predictions,
    get_feedback_stats,
    get_outcome_stats,
)
from api.drift_monitor import (
    evaluate_model,
    get_model_health,
    prepare_candidate_model,
    activate_candidate_model,
    rollback_active_model,
)
from api.escalation_engine import (
    evaluate_escalation,
    decide_escalation,
    acknowledge_escalation_flow,
    snooze_escalation_flow,
    reopen_escalation_flow,
    record_action_effectiveness_flow,
    process_timeouts,
    get_escalation_detail,
    get_notifications,
)

# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "ticket-intelligence-prod-secret-key-326-secure-fixed-2026"
)
app.config["SESSION_COOKIE_NAME"] = "ticket_intel_session"
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_PATH"] = "/"
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

# Initialize database on startup
init_db()


# ---------------------------------------------------------------------------
# Authentication helpers
# ---------------------------------------------------------------------------
def login_required(f):
    """Decorator requiring an authenticated session or valid HTTP Basic auth."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if "username" not in session:
            auth = request.authorization
            if auth and auth.username and auth.password:
                user = get_user(auth.username)
                if user and check_password_hash(user["password_hash"], auth.password):
                    session.permanent = True
                    session["username"] = user["username"]
                    session["display_name"] = user["display_name"]
                    session["role"] = user["role"]
                    session["team"] = user.get("team", "")
                    return f(*args, **kwargs)
            # Web pages redirect to /login
            if request.path in ["/dashboard", "/"]:
                return redirect(url_for("login_page"))
            # API endpoints return 401 JSON
            return jsonify({"error": "Authentication required"}), 401
        return f(*args, **kwargs)
    return decorated


def role_required(*roles):
    """Decorator requiring specific role(s)."""
    def decorator(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if "role" not in session or session["role"] not in roles:
                return jsonify({"error": "Insufficient permissions"}), 403
            return f(*args, **kwargs)
        return decorated
    return decorator


def get_current_user():
    """Return current session user info."""
    return {
        "username": session.get("username"),
        "display_name": session.get("display_name"),
        "role": session.get("role"),
    }


# ---------------------------------------------------------------------------
# Auth endpoints
# ---------------------------------------------------------------------------

@app.route("/login", methods=["GET"])
def login_page():
    """Serve login page (handled by dashboard template)."""
    if "username" in session:
        return redirect(url_for("dashboard"))
    return render_template("dashboard.html", page="login")


@app.route("/api/login", methods=["POST"])
def api_login():
    """Authenticate user and create session."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    username = data.get("username", "").strip()
    password = data.get("password", "")

    user = get_user(username)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid username or password"}), 401

    session.permanent = True
    session["username"] = user["username"]
    session["display_name"] = user["display_name"]
    session["role"] = user["role"]
    session["team"] = user.get("team", "")

    return jsonify({
        "status": "authenticated",
        "username": user["username"],
        "display_name": user["display_name"],
        "role": user["role"],
    })


@app.route("/api/logout", methods=["POST"])
def api_logout():
    """Clear session."""
    session.clear()
    return jsonify({"status": "logged_out"})


@app.route("/api/me", methods=["GET"])
def api_me():
    """Return current user info."""
    if "username" not in session:
        return jsonify({"authenticated": False}), 401
    return jsonify({
        "authenticated": True,
        "username": session["username"],
        "display_name": session.get("display_name"),
        "role": session.get("role"),
    })


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return redirect(url_for("dashboard"))


@app.route("/dashboard")
@login_required
def dashboard():
    """Serve the interactive dashboard."""
    return render_template("dashboard.html", page="dashboard", user=get_current_user())


# ---------------------------------------------------------------------------
# Ticket CRUD
# ---------------------------------------------------------------------------

@app.route("/tickets", methods=["GET"])
@login_required
def list_tickets():
    """List tickets with optional filters."""
    filters = {}
    for key in ["status", "priority", "issue_type", "assigned_team", "search"]:
        val = request.args.get(key)
        if val:
            filters[key] = val

    limit = min(int(request.args.get("limit", 50)), 200)
    offset = int(request.args.get("offset", 0))

    tickets, total = get_tickets(filters=filters, limit=limit, offset=offset)
    return jsonify({"tickets": tickets, "total": total, "limit": limit, "offset": offset})


@app.route("/tickets", methods=["POST"])
@login_required
def create_ticket():
    """Create a new ticket."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    ticket_id = data.get("ticket_id")
    if not ticket_id:
        # Auto-generate ticket ID
        with get_db() as conn:
            count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        ticket_id = f"TKT-{2000 + count}"
        data["ticket_id"] = ticket_id

    # Calculate description_length if description provided
    if "description" in data and "description_length" not in data:
        data["description_length"] = len(data.get("description", ""))

    # Calculate time_remaining if not provided
    if "time_remaining_hours" not in data:
        sla = float(data.get("sla_deadline_hours", 24.0))
        created = float(data.get("created_hours", 0.0))
        data["time_remaining_hours"] = max(0.0, sla - created)

    try:
        result_id = insert_ticket(data)
        return jsonify({"status": "created", "ticket_id": result_id}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/tickets/<ticket_id>", methods=["GET"])
@login_required
def get_ticket_endpoint(ticket_id):
    """Get a single ticket with its prediction history."""
    ticket = get_ticket(ticket_id)
    if not ticket:
        return jsonify({"error": "Ticket not found"}), 404

    # Include prediction history
    predictions = get_ticket_predictions(ticket_id)
    ticket["predictions"] = predictions

    return jsonify(ticket)


@app.route("/tickets/<ticket_id>", methods=["PATCH"])
@login_required
def patch_ticket(ticket_id):
    """Update ticket fields."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    ticket = get_ticket(ticket_id)
    if not ticket:
        return jsonify({"error": "Ticket not found"}), 404

    success = update_ticket(ticket_id, data)
    if success:
        return jsonify({"status": "updated", "ticket_id": ticket_id})
    return jsonify({"error": "No valid fields to update"}), 400


@app.route("/tickets/<ticket_id>/ack", methods=["POST"])
@login_required
def acknowledge_ticket_endpoint(ticket_id):
    """Agent acknowledges ticket assignment, moving status to 'In Progress'."""
    user = session.get("display_name", "agent")
    success = acknowledge_ticket(ticket_id, user)
    if not success:
        return jsonify({"error": "Failed to acknowledge ticket"}), 400
    return jsonify({"status": "acknowledged", "ticket_id": ticket_id, "acknowledged_by": user})


@app.route("/tickets/<ticket_id>/notes", methods=["POST"])
@login_required
def add_ticket_note(ticket_id):
    """Add a work note to a ticket activity log."""
    data = request.get_json() or {}
    note = data.get("note", "").strip()
    if not note:
        return jsonify({"error": "note content is required"}), 400
    user = session.get("display_name", "agent")
    activity_id = insert_ticket_activity(ticket_id, user, "work_note", note)
    return jsonify({"status": "recorded", "activity_id": activity_id, "ticket_id": ticket_id})


@app.route("/tickets/<ticket_id>/request-escalation", methods=["POST"])
@login_required
def request_ticket_escalation(ticket_id):
    """Explicitly request human review / escalation for a ticket."""
    data = request.get_json() or {}
    reason = data.get("reason", "").strip()
    urgency = data.get("urgency", "high")
    if not reason:
        return jsonify({"error": "Justification reason is required to request escalation"}), 400
    user = session.get("display_name", "agent")

    # 1. Log activity
    insert_ticket_activity(ticket_id, user, "escalation_request", f"Escalation requested ({urgency}): {reason}")

    # 2. Trigger escalation record
    from datetime import timedelta, timezone
    preds = get_ticket_predictions(ticket_id)
    pred_id = preds[0]["prediction_id"] if preds else None
    risk = preds[0]["risk_score"] if preds else (0.85 if urgency.lower() == "critical" else 0.65)
    esc_id = insert_escalation({
        "ticket_id": ticket_id,
        "prediction_id": pred_id,
        "risk_score": risk,
        "trigger_conditions": f"Manual Escalation Request by {user}: {reason}",
        "assigned_reviewer": "Reviewer Lead",
        "escalation_level": urgency.lower(),
        "status": "pending_ack",
        "timeout_at": (datetime.now(timezone.utc) + timedelta(minutes=60)).isoformat(),
        "explanation_summary": f"Agent requested review: {reason}",
        "evidence_snapshot": {
            "requested_by": user,
            "urgency": urgency,
            "reason": reason,
            "requested_at": datetime.now(timezone.utc).isoformat(),
        }
    })
    return jsonify({"status": "escalation_requested", "escalation_id": esc_id, "ticket_id": ticket_id})


@app.route("/tickets/<ticket_id>/activities", methods=["GET"])
@login_required
def get_ticket_activities_endpoint(ticket_id):
    """Retrieve chronological activity log for a ticket."""
    activities = get_ticket_activities(ticket_id)
    return jsonify({"ticket_id": ticket_id, "activities": activities})


@app.route("/tickets/import", methods=["POST"])
@login_required
@role_required("admin", "reviewer")
def import_tickets():
    """Import tickets from CSV data (JSON body with array of ticket objects)."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    tickets_data = data.get("tickets", [])
    if not tickets_data:
        return jsonify({"error": "No tickets provided"}), 400

    imported = 0
    errors = []
    for i, t in enumerate(tickets_data):
        try:
            if "description" in t and "description_length" not in t:
                t["description_length"] = len(t.get("description", ""))
            if "time_remaining_hours" not in t:
                sla = float(t.get("sla_deadline_hours", 24.0))
                created = float(t.get("created_hours", 0.0))
                t["time_remaining_hours"] = max(0.0, sla - created)
            insert_ticket(t)
            imported += 1
        except Exception as e:
            errors.append({"index": i, "error": str(e)})

    return jsonify({"imported": imported, "errors": errors})


@app.route("/tickets/export", methods=["GET"])
@login_required
def export_tickets():
    """Export tickets as JSON."""
    tickets, total = get_tickets(limit=10000, offset=0)
    return jsonify({"tickets": tickets, "total": total})


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

@app.route("/predict", methods=["POST"])
@login_required
def predict_endpoint():
    """
    Predict SLA breach risk with full explanation and escalation evaluation.

    Returns risk score, SHAP explanation, recommendation, and escalation status.
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    try:
        # Run prediction
        prediction = sla_predict(data)

        # Generate SHAP explanation
        feature_array, feature_dict, *_ = preprocess_input(data)
        explanation = explain_prediction(
            feature_array, feature_dict, prediction["risk_score"], ticket_data=data
        )

        # Log prediction with explanation
        log_prediction(prediction, explanation)

        # Evaluate escalation policies
        escalation = evaluate_escalation(prediction, ticket_data=data, explanation=explanation)

        # Build response
        response = {
            "ticket_id": prediction["ticket_id"],
            "prediction_id": prediction["prediction_id"],
            "risk_score": prediction["risk_score"],
            "risk_label": prediction["risk_label"],
            "confidence": prediction["confidence"],
            "assigned_team": prediction["assigned_team"],
            "model_version": prediction["model_version"],
            "predicted_at": prediction["predicted_at"],
            "explanation": explanation,
            "escalation": escalation,
        }

        if prediction.get("warnings"):
            response["warnings"] = prediction["warnings"]

        return jsonify(response)

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        return jsonify({"error": f"Prediction failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Prediction detail
# ---------------------------------------------------------------------------

@app.route("/predictions/<prediction_id>", methods=["GET"])
@login_required
def get_prediction_endpoint(prediction_id):
    """Get full prediction detail."""
    pred = get_prediction_detail(prediction_id)
    if not pred:
        return jsonify({"error": "Prediction not found"}), 404
    return jsonify(pred)


@app.route("/predictions/<prediction_id>/explanation", methods=["GET"])
@login_required
def get_explanation_endpoint(prediction_id):
    """Get the stored explanation for a prediction."""
    pred = get_prediction_detail(prediction_id)
    if not pred:
        return jsonify({"error": "Prediction not found"}), 404

    return jsonify({
        "prediction_id": prediction_id,
        "ticket_id": pred.get("ticket_id"),
        "risk_score": pred.get("risk_score"),
        "risk_label": pred.get("risk_label"),
        "predicted_at": pred.get("predicted_at"),
        "explanation": pred.get("shap_explanation"),
        "recommendation": pred.get("recommendation"),
        "feature_snapshot": pred.get("feature_snapshot"),
    })


@app.route("/explain/<ticket_id>", methods=["GET"])
@login_required
def explain_ticket_endpoint(ticket_id):
    """Retrieve stored or on-demand explanation for a ticket."""
    preds = get_ticket_predictions(ticket_id)
    if not preds:
        ticket = get_ticket(ticket_id)
        if not ticket:
            return jsonify({"error": "Ticket not found"}), 404
        try:
            pred = sla_predict(ticket)
            feature_array, feature_dict, *_ = preprocess_input(ticket)
            exp = explain_prediction(
                feature_array, feature_dict, pred["risk_score"], ticket_data=ticket
            )
            log_prediction(pred, exp)
            return jsonify({
                "ticket_id": ticket_id,
                "prediction_id": pred["prediction_id"],
                "risk_score": pred["risk_score"],
                "risk_label": pred["risk_label"],
                "predicted_at": pred["predicted_at"],
                "explanation": exp,
                "top_factors": exp.get("top_factors", []),
                "recommendation": exp.get("recommendation", ""),
                "feature_snapshot": pred.get("feature_snapshot", {}),
            })
        except Exception as e:
            return jsonify({"error": f"Failed to generate explanation: {str(e)}"}), 500

    latest_pred = preds[0]
    exp = latest_pred.get("shap_explanation") or {}
    return jsonify({
        "ticket_id": ticket_id,
        "prediction_id": latest_pred.get("prediction_id"),
        "risk_score": latest_pred.get("risk_score"),
        "risk_label": latest_pred.get("risk_label"),
        "predicted_at": latest_pred.get("predicted_at"),
        "explanation": exp,
        "top_factors": exp.get("top_factors", []),
        "recommendation": exp.get("recommendation") or latest_pred.get("recommendation", ""),
        "feature_snapshot": latest_pred.get("feature_snapshot", {}),
    })


# ---------------------------------------------------------------------------
# Feedback and outcomes
# ---------------------------------------------------------------------------

@app.route("/feedback", methods=["POST"])
@login_required
def submit_feedback():
    """Submit agent feedback on a recommendation."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    ticket_id = data.get("ticket_id")
    feedback_label = data.get("feedback_label")
    if not ticket_id or not feedback_label:
        return jsonify({"error": "ticket_id and feedback_label required"}), 400

    try:
        label = record_feedback(
            ticket_id=ticket_id,
            feedback_label=feedback_label,
            prediction_id=data.get("prediction_id"),
            feedback_reason=data.get("feedback_reason"),
            reviewer_name=session.get("display_name", "unknown"),
        )
        return jsonify({"status": "recorded", "ticket_id": ticket_id, "feedback_label": label})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/feedback/summary", methods=["GET"])
@login_required
def feedback_summary():
    """Get aggregated feedback statistics."""
    return jsonify(get_feedback_stats())


@app.route("/outcomes", methods=["POST"])
@app.route("/outcome", methods=["POST"])
@login_required
@role_required("admin", "reviewer")
def submit_outcome():
    """Record a confirmed SLA outcome with validation."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "JSON body required"}), 400

    ticket_id = data.get("ticket_id")
    actual_breach = data.get("actual_breach")
    if not ticket_id or actual_breach is None:
        return jsonify({"error": "ticket_id and actual_breach required"}), 400

    try:
        status = record_outcome(
            ticket_id=ticket_id,
            actual_breach=actual_breach,
            prediction_id=data.get("prediction_id"),
            confirmed_by=session.get("display_name", "unknown"),
            notes=data.get("notes"),
        )
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    try:
        update_ticket(ticket_id, {
            "sla_breach": 1 if actual_breach else 0,
            "status": "Resolved",
            "resolved_at": datetime.now(timezone.utc).isoformat(),
        })
    except Exception:
        pass

    return jsonify({
        "status": "recorded",
        "ticket_id": ticket_id,
        "actual_breach": bool(actual_breach),
        "outcome_status": status,
        "outcome_confirmed_at": datetime.now(timezone.utc).isoformat(),
    })


@app.route("/outcome/correct", methods=["POST"])
@app.route("/outcomes/correct", methods=["POST"])
@login_required
@role_required("admin", "reviewer")
def correct_outcome_endpoint():
    """Formally correct an SLA outcome record with audit history."""
    data = request.get_json() or {}
    ticket_id = data.get("ticket_id")
    new_breach = data.get("actual_breach")
    reason = data.get("reason", "").strip()
    if not ticket_id or new_breach is None or not reason:
        return jsonify({"error": "ticket_id, actual_breach, and justification reason are required"}), 400

    reviewer = session.get("display_name", "reviewer")
    try:
        changed = correct_outcome(ticket_id, bool(new_breach), reviewer, reason)
        return jsonify({
            "status": "corrected" if changed else "unchanged",
            "ticket_id": ticket_id,
            "actual_breach": bool(new_breach),
            "corrected_by": reviewer,
            "reason": reason,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/outcome/audits", methods=["GET"])
@app.route("/outcomes/audits", methods=["GET"])
@login_required
def get_outcome_audits_endpoint():
    """Retrieve audit history of SLA outcome corrections."""
    ticket_id = request.args.get("ticket_id")
    audits = get_outcome_audit_history(ticket_id)
    return jsonify({"audits": audits})


# ---------------------------------------------------------------------------
# Model health & Governance
# ---------------------------------------------------------------------------

@app.route("/model-health", methods=["GET"])
@login_required
def model_health():
    """Get current model performance metrics and drift status."""
    health = get_model_health()
    return jsonify(health)


@app.route("/model-health/evaluate", methods=["POST"])
@login_required
@role_required("admin", "reviewer")
def trigger_evaluation():
    """Manually trigger a model evaluation."""
    result = evaluate_model()
    return jsonify(result)


@app.route("/model-health/retrain", methods=["POST"])
@login_required
@role_required("admin")
def trigger_retrain():
    """Prepare a candidate model from confirmed outcomes and evaluate against active model."""
    result = prepare_candidate_model()
    return jsonify(result)


@app.route("/model-health/activate", methods=["POST"])
@login_required
@role_required("admin")
def activate_model():
    """Activate a candidate model version with validation and hot reload (requires admin)."""
    data = request.get_json() or {}
    version = data.get("version")
    if not version:
        return jsonify({"error": "version required"}), 400

    admin = session.get("display_name", "admin")
    try:
        res = activate_candidate_model(version, approved_by=admin)
        reset_explainer()
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/model-health/rollback", methods=["POST"])
@login_required
@role_required("admin")
def rollback_model_endpoint():
    """Roll back active model to the previous model version atomically."""
    admin = session.get("display_name", "admin")
    try:
        res = rollback_active_model(admin_name=admin)
        reset_explainer()
        return jsonify(res)
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/model-health/versions", methods=["GET"])
@login_required
def list_model_versions():
    """List all registered model versions."""
    versions = get_model_versions()
    return jsonify({"versions": versions})


# ---------------------------------------------------------------------------
# Escalations & Reviewer Queue
# ---------------------------------------------------------------------------

@app.route("/escalations", methods=["GET"])
@login_required
def list_escalations():
    """List escalation requests with optional status filter."""
    status = request.args.get("status")
    limit = min(int(request.args.get("limit", 50)), 200)
    escalations = db_get_escalations(status=status, limit=limit)
    counts = get_escalation_counts()
    return jsonify({"escalations": escalations, "counts": counts})


@app.route("/escalations/<int:escalation_id>", methods=["GET"])
@app.route("/escalation/<int:escalation_id>", methods=["GET"])
@login_required
def get_escalation_endpoint(escalation_id):
    """Get full escalation detail with audit trail and evidence snapshot."""
    detail = get_escalation_detail(escalation_id)
    if not detail:
        return jsonify({"error": "Escalation not found"}), 404
    return jsonify(detail)


@app.route("/escalations/<int:escalation_id>/ack", methods=["POST"])
@app.route("/escalation/<int:escalation_id>/ack", methods=["POST"])
@login_required
@role_required("admin", "reviewer")
def ack_escalation_endpoint(escalation_id):
    """Reviewer acknowledges an escalation, changing status to under_review."""
    reviewer = session.get("display_name", "reviewer")
    try:
        res = acknowledge_escalation_flow(escalation_id, reviewer)
        return jsonify(res)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/escalations/<int:escalation_id>/snooze", methods=["POST"])
@app.route("/escalation/<int:escalation_id>/snooze", methods=["POST"])
@login_required
@role_required("admin", "reviewer")
def snooze_escalation_endpoint(escalation_id):
    """Temporarily defer an escalation with reason and duration."""
    data = request.get_json() or {}
    duration = int(data.get("duration_minutes", 30))
    reason = data.get("reason", "").strip()
    reviewer = session.get("display_name", "reviewer")
    try:
        res = snooze_escalation_flow(escalation_id, duration, reason, reviewer)
        return jsonify(res)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/escalations/<int:escalation_id>/reopen", methods=["POST"])
@app.route("/escalation/<int:escalation_id>/reopen", methods=["POST"])
@login_required
@role_required("admin", "reviewer")
def reopen_escalation_endpoint(escalation_id):
    """Reopen an escalation when risk remains unresolved."""
    data = request.get_json() or {}
    reason = data.get("reason", "").strip()
    reviewer = session.get("display_name", "reviewer")
    try:
        res = reopen_escalation_flow(escalation_id, reason, reviewer)
        return jsonify(res)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/escalations/<int:escalation_id>/action-status", methods=["POST"])
@app.route("/escalation/<int:escalation_id>/action-status", methods=["POST"])
@login_required
@role_required("admin", "reviewer")
def action_status_endpoint(escalation_id):
    """Record whether approved action was completed and whether it helped."""
    data = request.get_json() or {}
    completed = bool(data.get("completed", True))
    effective = data.get("effective", "yes")
    notes = data.get("notes", "")
    reviewer = session.get("display_name", "reviewer")
    try:
        res = record_action_effectiveness_flow(escalation_id, completed, effective, notes, reviewer)
        return jsonify(res)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/escalations/<int:escalation_id>/decide", methods=["POST"])
@app.route("/escalation/<int:escalation_id>/decide", methods=["POST"])
@login_required
@role_required("admin", "reviewer")
def decide_escalation_endpoint(escalation_id):
    """Submit a reviewer decision on an escalation with optimistic locking."""
    data = request.get_json() or {}
    decision = data.get("decision")
    reason = data.get("decision_reason", "")
    version = data.get("expected_version")
    if not decision:
        return jsonify({"error": "decision required"}), 400

    try:
        result = decide_escalation(
            escalation_id=escalation_id,
            decision=decision,
            reason=reason,
            reviewer_name=session.get("display_name", "unknown"),
            expected_version=int(version) if version is not None else None,
        )
        return jsonify({"status": "decision_recorded", **result})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/escalations/check", methods=["POST"])
@login_required
@role_required("admin", "reviewer")
def check_escalation_timeouts():
    """Process timeout checks and auto-escalations."""
    results = process_timeouts()
    return jsonify({
        "status": "checked",
        "processed": len(results),
        "results": results,
    })


@app.route("/audit-logs", methods=["GET"])
@login_required
def get_audit_logs():
    """Audit logs endpoint filtered by role (Agent: own; Reviewer: operational; Admin: full)."""
    role = session.get("role", "agent")
    user = session.get("display_name", "")
    with get_db() as conn:
        outcome_audits = [dict(r) for r in conn.execute("SELECT * FROM outcome_audit_logs ORDER BY created_at DESC LIMIT 50").fetchall()]
        escalation_audits = [dict(r) for r in conn.execute("SELECT id, ticket_id, status, decision, decision_reason, reviewer_name, decided_at, version FROM escalation_logs WHERE decision IS NOT NULL ORDER BY decided_at DESC LIMIT 50").fetchall()]
        if role == "agent":
            ticket_activities = [dict(r) for r in conn.execute("SELECT * FROM ticket_activity_logs WHERE user_name = ? ORDER BY created_at DESC LIMIT 50", (user,)).fetchall()]
        else:
            ticket_activities = [dict(r) for r in conn.execute("SELECT * FROM ticket_activity_logs ORDER BY created_at DESC LIMIT 50").fetchall()]
    return jsonify({
        "role": role,
        "outcome_corrections": outcome_audits if role in ("admin", "reviewer") else [],
        "escalation_decisions": escalation_audits if role in ("admin", "reviewer") else [],
        "ticket_activities": ticket_activities,
    })


@app.route("/escalations/notifications", methods=["GET"])
@login_required
def get_escalation_notifications():
    """Get pending escalation notifications for the current user."""
    reviewer_name = session.get("display_name")
    notifications = get_notifications(reviewer_name)
    return jsonify({"notifications": notifications, "count": len(notifications)})


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------

@app.errorhandler(404)
def not_found(e):
    if request.is_json or request.path.startswith("/api"):
        return jsonify({"error": "Not found"}), 404
    return redirect(url_for("dashboard"))


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Starting AI Ticket Intelligence Platform...")
    print(f"Model version: {get_model_version()}")
    print("Dashboard: http://127.0.0.1:5000/dashboard")
    print("Default users: admin/admin123, reviewer/reviewer123, agent/agent123")
    app.run(debug=True, host="127.0.0.1", port=5000)
