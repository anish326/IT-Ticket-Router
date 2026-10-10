"""
Database Foundation
====================
SQLite persistence layer for the AI Ticket Intelligence Platform.

Tables:
- users: Local authentication and reviewer authorization
- tickets: Ticket management and feature storage
- prediction_logs: Every prediction with feature snapshot and explanation
- sla_outcomes: Confirmed SLA breach/met outcomes (separate from predictions)
- agent_feedback: Agent ratings on recommendations (separate from outcomes)
- model_versions: Model artifacts, training metadata, activation status
- model_evaluations: Performance metrics and drift assessments over time
- escalation_logs: Full escalation audit trail with decisions

All timestamps stored as ISO-8601 strings. Parameterized queries used throughout.
"""

import sqlite3
import os
import json
from datetime import datetime, timezone
from contextlib import contextmanager
from werkzeug.security import generate_password_hash

if os.environ.get("VERCEL"):
    import shutil
    TMP_DB = "/tmp/ticket_intelligence.db"
    bundled_db = os.path.join(os.path.dirname(__file__), "ticket_intelligence.db")
    if not os.path.exists(TMP_DB) and os.path.exists(bundled_db):
        try:
            shutil.copy2(bundled_db, TMP_DB)
        except Exception:
            pass
    DB_PATH = TMP_DB
else:
    DB_PATH = os.path.join(os.path.dirname(__file__), "ticket_intelligence.db")

SCHEMA_VERSION = 1

SCHEMA_SQL = """
-- Schema version tracking
CREATE TABLE IF NOT EXISTS schema_info (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Local users for authentication and reviewer authorization
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    display_name TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'agent',
    team TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    active INTEGER NOT NULL DEFAULT 1
);

-- Ticket management and feature storage
CREATE TABLE IF NOT EXISTS tickets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL DEFAULT 'Untitled Ticket',
    description TEXT DEFAULT '',
    priority TEXT NOT NULL DEFAULT 'Medium',
    issue_type TEXT NOT NULL DEFAULT 'Task',
    status TEXT NOT NULL DEFAULT 'Open',
    assigned_team TEXT DEFAULT '',
    assignee TEXT DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    sla_deadline_hours REAL NOT NULL DEFAULT 24.0,
    created_hours REAL NOT NULL DEFAULT 0.0,
    time_remaining_hours REAL NOT NULL DEFAULT 24.0,
    response_delay_hours REAL NOT NULL DEFAULT 0.0,
    team_backlog_count INTEGER NOT NULL DEFAULT 0,
    similar_ticket_resolution_hours REAL NOT NULL DEFAULT 0.0,
    description_length INTEGER NOT NULL DEFAULT 0,
    sla_breach INTEGER DEFAULT NULL,
    resolved_at TEXT DEFAULT NULL,
    acknowledged_at TEXT DEFAULT NULL,
    acknowledged_by TEXT DEFAULT NULL,
    synthetic INTEGER NOT NULL DEFAULT 0
);

-- Ticket activity logs for agent workflow
CREATE TABLE IF NOT EXISTS ticket_activity_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    user_name TEXT NOT NULL,
    activity_type TEXT NOT NULL, -- 'ack', 'work_note', 'status_change', 'escalation_request'
    note TEXT NOT NULL,
    metadata TEXT DEFAULT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
);

-- Prediction logs with feature snapshots
CREATE TABLE IF NOT EXISTS prediction_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE NOT NULL,
    ticket_id TEXT NOT NULL,
    predicted_at TEXT NOT NULL DEFAULT (datetime('now')),
    risk_score REAL NOT NULL,
    risk_label TEXT NOT NULL,
    decision_threshold REAL NOT NULL DEFAULT 0.5,
    model_version TEXT NOT NULL,
    feature_snapshot TEXT NOT NULL,
    shap_explanation TEXT DEFAULT NULL,
    recommendation TEXT DEFAULT NULL,
    uncertainty REAL DEFAULT NULL,
    data_quality_score REAL DEFAULT NULL,
    data_quality_warnings TEXT DEFAULT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
);

-- Confirmed SLA outcomes (separate from predictions)
CREATE TABLE IF NOT EXISTS sla_outcomes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    prediction_id TEXT DEFAULT NULL,
    actual_breach INTEGER NOT NULL,
    outcome_status TEXT NOT NULL DEFAULT 'confirmed',
    outcome_confirmed_at TEXT NOT NULL DEFAULT (datetime('now')),
    confirmed_by TEXT DEFAULT NULL,
    notes TEXT DEFAULT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id),
    FOREIGN KEY (prediction_id) REFERENCES prediction_logs(prediction_id)
);

-- Outcome correction audit trail
CREATE TABLE IF NOT EXISTS outcome_audit_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    previous_breach INTEGER NOT NULL,
    corrected_breach INTEGER NOT NULL,
    corrected_by TEXT NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id)
);

-- Agent feedback on recommendations
CREATE TABLE IF NOT EXISTS agent_feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    prediction_id TEXT DEFAULT NULL,
    feedback_label TEXT NOT NULL,
    feedback_reason TEXT DEFAULT NULL,
    reviewer_name TEXT DEFAULT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id),
    FOREIGN KEY (prediction_id) REFERENCES prediction_logs(prediction_id)
);

-- Model version registry
CREATE TABLE IF NOT EXISTS model_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version TEXT UNIQUE NOT NULL,
    model_path TEXT NOT NULL,
    feature_columns TEXT NOT NULL,
    training_date TEXT NOT NULL DEFAULT (datetime('now')),
    training_samples INTEGER DEFAULT NULL,
    test_samples INTEGER DEFAULT NULL,
    precision_score REAL DEFAULT NULL,
    recall_score REAL DEFAULT NULL,
    f1_score REAL DEFAULT NULL,
    brier_score REAL DEFAULT NULL,
    baseline_accuracy REAL DEFAULT NULL,
    is_active INTEGER NOT NULL DEFAULT 0,
    is_candidate INTEGER NOT NULL DEFAULT 0,
    approved_by TEXT DEFAULT NULL,
    approved_at TEXT DEFAULT NULL,
    rollback_version TEXT DEFAULT NULL,
    evaluation_metrics TEXT DEFAULT NULL,
    notes TEXT DEFAULT NULL
);

-- Model evaluation history
CREATE TABLE IF NOT EXISTS model_evaluations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_version TEXT NOT NULL,
    evaluation_date TEXT NOT NULL DEFAULT (datetime('now')),
    precision_score REAL DEFAULT NULL,
    recall_score REAL DEFAULT NULL,
    f1_score REAL DEFAULT NULL,
    brier_score REAL DEFAULT NULL,
    false_positives INTEGER DEFAULT 0,
    false_negatives INTEGER DEFAULT 0,
    true_positives INTEGER DEFAULT 0,
    true_negatives INTEGER DEFAULT 0,
    total_predictions INTEGER DEFAULT 0,
    confirmed_outcomes INTEGER DEFAULT 0,
    drift_status TEXT DEFAULT 'stable',
    drift_details TEXT DEFAULT NULL,
    notes TEXT DEFAULT NULL,
    FOREIGN KEY (model_version) REFERENCES model_versions(version)
);

-- Escalation audit trail
CREATE TABLE IF NOT EXISTS escalation_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    version INTEGER NOT NULL DEFAULT 1,
    ticket_id TEXT NOT NULL,
    prediction_id TEXT DEFAULT NULL,
    risk_score REAL NOT NULL,
    trigger_conditions TEXT NOT NULL,
    assigned_reviewer TEXT DEFAULT NULL,
    escalation_level TEXT NOT NULL DEFAULT 'medium',
    status TEXT NOT NULL DEFAULT 'pending',
    decision TEXT DEFAULT NULL,
    decision_reason TEXT DEFAULT NULL,
    reviewer_name TEXT DEFAULT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    acknowledged_at TEXT DEFAULT NULL,
    acknowledged_by TEXT DEFAULT NULL,
    decided_at TEXT DEFAULT NULL,
    resolved_at TEXT DEFAULT NULL,
    timeout_at TEXT DEFAULT NULL,
    snoozed_until TEXT DEFAULT NULL,
    snooze_reason TEXT DEFAULT NULL,
    reopened_at TEXT DEFAULT NULL,
    reopened_by TEXT DEFAULT NULL,
    reopen_reason TEXT DEFAULT NULL,
    action_completed INTEGER DEFAULT 0,
    action_effective TEXT DEFAULT NULL,
    action_notes TEXT DEFAULT NULL,
    evidence_snapshot TEXT DEFAULT NULL,
    escalation_chain_level INTEGER NOT NULL DEFAULT 1,
    parent_escalation_id INTEGER DEFAULT NULL,
    explanation_summary TEXT DEFAULT NULL,
    FOREIGN KEY (ticket_id) REFERENCES tickets(ticket_id),
    FOREIGN KEY (prediction_id) REFERENCES prediction_logs(prediction_id),
    FOREIGN KEY (parent_escalation_id) REFERENCES escalation_logs(id)
);

-- Performance indexes
CREATE INDEX IF NOT EXISTS idx_tickets_ticket_id ON tickets(ticket_id);
CREATE INDEX IF NOT EXISTS idx_tickets_status ON tickets(status);
CREATE INDEX IF NOT EXISTS idx_tickets_priority ON tickets(priority);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_ticket_id ON prediction_logs(ticket_id);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_prediction_id ON prediction_logs(prediction_id);
CREATE INDEX IF NOT EXISTS idx_prediction_logs_model_version ON prediction_logs(model_version);
CREATE INDEX IF NOT EXISTS idx_sla_outcomes_ticket_id ON sla_outcomes(ticket_id);
CREATE INDEX IF NOT EXISTS idx_sla_outcomes_prediction_id ON sla_outcomes(prediction_id);
CREATE INDEX IF NOT EXISTS idx_agent_feedback_ticket_id ON agent_feedback(ticket_id);
CREATE INDEX IF NOT EXISTS idx_escalation_logs_ticket_id ON escalation_logs(ticket_id);
CREATE INDEX IF NOT EXISTS idx_escalation_logs_status ON escalation_logs(status);
CREATE INDEX IF NOT EXISTS idx_model_versions_active ON model_versions(is_active);
"""

DEFAULT_USERS = [
    {
        "username": "admin",
        "password": "admin123",
        "display_name": "System Admin",
        "role": "admin",
        "team": "Platform",
    },
    {
        "username": "reviewer",
        "password": "reviewer123",
        "display_name": "Sarah Chen",
        "role": "reviewer",
        "team": "L2 Support",
    },
    {
        "username": "agent",
        "password": "agent123",
        "display_name": "Alex Johnson",
        "role": "agent",
        "team": "L1 Support",
    },
]


def get_db_path():
    """Return the configured database path."""
    return os.environ.get("TICKET_DB_PATH", DB_PATH)


@contextmanager
def get_db():
    """Context manager for database connections with WAL mode and foreign keys."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path=None):
    """Initialize the database schema and default users.

    Safe to call multiple times — uses CREATE IF NOT EXISTS.
    """
    if db_path:
        os.environ["TICKET_DB_PATH"] = db_path

    actual_path = get_db_path()
    os.makedirs(os.path.dirname(actual_path) if os.path.dirname(actual_path) else ".", exist_ok=True)

    with get_db() as conn:
        conn.executescript(SCHEMA_SQL)

        # Ensure new columns exist on legacy tables (safe schema migration)
        migrations = [
            "ALTER TABLE tickets ADD COLUMN acknowledged_at TEXT DEFAULT NULL",
            "ALTER TABLE tickets ADD COLUMN acknowledged_by TEXT DEFAULT NULL",
            "ALTER TABLE prediction_logs ADD COLUMN uncertainty REAL DEFAULT NULL",
            "ALTER TABLE prediction_logs ADD COLUMN data_quality_score REAL DEFAULT NULL",
            "ALTER TABLE prediction_logs ADD COLUMN data_quality_warnings TEXT DEFAULT NULL",
            "ALTER TABLE model_versions ADD COLUMN rollback_version TEXT DEFAULT NULL",
            "ALTER TABLE model_versions ADD COLUMN evaluation_metrics TEXT DEFAULT NULL",
            "ALTER TABLE escalation_logs ADD COLUMN version INTEGER NOT NULL DEFAULT 1",
            "ALTER TABLE escalation_logs ADD COLUMN acknowledged_at TEXT DEFAULT NULL",
            "ALTER TABLE escalation_logs ADD COLUMN acknowledged_by TEXT DEFAULT NULL",
            "ALTER TABLE escalation_logs ADD COLUMN snoozed_until TEXT DEFAULT NULL",
            "ALTER TABLE escalation_logs ADD COLUMN snooze_reason TEXT DEFAULT NULL",
            "ALTER TABLE escalation_logs ADD COLUMN reopened_at TEXT DEFAULT NULL",
            "ALTER TABLE escalation_logs ADD COLUMN reopened_by TEXT DEFAULT NULL",
            "ALTER TABLE escalation_logs ADD COLUMN reopen_reason TEXT DEFAULT NULL",
            "ALTER TABLE escalation_logs ADD COLUMN action_completed INTEGER DEFAULT 0",
            "ALTER TABLE escalation_logs ADD COLUMN action_effective TEXT DEFAULT NULL",
            "ALTER TABLE escalation_logs ADD COLUMN action_notes TEXT DEFAULT NULL",
            "ALTER TABLE escalation_logs ADD COLUMN evidence_snapshot TEXT DEFAULT NULL",
        ]
        for sql in migrations:
            try:
                conn.execute(sql)
            except sqlite3.OperationalError:
                pass  # column already exists

        # Check schema version
        try:
            row = conn.execute(
                "SELECT value FROM schema_info WHERE key = 'version'"
            ).fetchone()
            if row is None:
                conn.execute(
                    "INSERT INTO schema_info (key, value) VALUES (?, ?)",
                    ("version", str(SCHEMA_VERSION)),
                )
        except sqlite3.OperationalError:
            pass

        # Seed default users if table is empty
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        if user_count == 0:
            for user in DEFAULT_USERS:
                conn.execute(
                    """INSERT INTO users (username, password_hash, display_name, role, team)
                       VALUES (?, ?, ?, ?, ?)""",
                    (
                        user["username"],
                        generate_password_hash(user["password"]),
                        user["display_name"],
                        user["role"],
                        user["team"],
                    ),
                )

    return actual_path


# ---------------------------------------------------------------------------
# Ticket helpers
# ---------------------------------------------------------------------------

def insert_ticket(ticket_data):
    """Insert a single ticket record. Returns the ticket_id."""
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO tickets
               (ticket_id, title, description, priority, issue_type, status,
                assigned_team, assignee, created_at, updated_at,
                sla_deadline_hours, created_hours, time_remaining_hours,
                response_delay_hours, team_backlog_count,
                similar_ticket_resolution_hours, description_length,
                sla_breach, synthetic)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                ticket_data.get("ticket_id"),
                ticket_data.get("title", "Untitled Ticket"),
                ticket_data.get("description", ""),
                ticket_data.get("priority", "Medium"),
                ticket_data.get("issue_type", "Task"),
                ticket_data.get("status", "Open"),
                ticket_data.get("assigned_team", ""),
                ticket_data.get("assignee", ""),
                ticket_data.get("created_at", datetime.now(timezone.utc).isoformat()),
                datetime.now(timezone.utc).isoformat(),
                ticket_data.get("sla_deadline_hours", 24.0),
                ticket_data.get("created_hours", 0.0),
                ticket_data.get("time_remaining_hours", 24.0),
                ticket_data.get("response_delay_hours", 0.0),
                ticket_data.get("team_backlog_count", 0),
                ticket_data.get("similar_ticket_resolution_hours", 0.0),
                ticket_data.get("description_length", 0),
                ticket_data.get("sla_breach"),
                ticket_data.get("synthetic", 0),
            ),
        )
    return ticket_data.get("ticket_id")


def get_ticket(ticket_id):
    """Fetch a single ticket by ticket_id."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)
        ).fetchone()
        return dict(row) if row else None


def get_tickets(filters=None, limit=100, offset=0):
    """Fetch tickets with optional filters."""
    query = "SELECT * FROM tickets WHERE 1=1"
    params = []

    if filters:
        if filters.get("status"):
            query += " AND status = ?"
            params.append(filters["status"])
        if filters.get("priority"):
            query += " AND priority = ?"
            params.append(filters["priority"])
        if filters.get("issue_type"):
            query += " AND issue_type = ?"
            params.append(filters["issue_type"])
        if filters.get("assigned_team"):
            query += " AND assigned_team = ?"
            params.append(filters["assigned_team"])
        if filters.get("search"):
            query += " AND (title LIKE ? OR ticket_id LIKE ?)"
            search_term = f"%{filters['search']}%"
            params.extend([search_term, search_term])

    query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        total = conn.execute(
            "SELECT COUNT(*) FROM tickets WHERE 1=1", []
        ).fetchone()[0]
        return [dict(r) for r in rows], total


def update_ticket(ticket_id, updates):
    """Update specific fields of a ticket."""
    allowed_fields = [
        "title", "description", "priority", "issue_type", "status",
        "assigned_team", "assignee", "sla_deadline_hours", "created_hours",
        "time_remaining_hours", "response_delay_hours", "team_backlog_count",
        "similar_ticket_resolution_hours", "description_length",
        "sla_breach", "resolved_at", "acknowledged_at", "acknowledged_by",
    ]

    set_clauses = []
    params = []
    for field, value in updates.items():
        if field in allowed_fields:
            set_clauses.append(f"{field} = ?")
            params.append(value)

    if not set_clauses:
        return False

    set_clauses.append("updated_at = ?")
    params.append(datetime.now(timezone.utc).isoformat())
    params.append(ticket_id)

    with get_db() as conn:
        result = conn.execute(
            f"UPDATE tickets SET {', '.join(set_clauses)} WHERE ticket_id = ?",
            params,
        )
        return result.rowcount > 0


def insert_ticket_activity(ticket_id, user_name, activity_type, note, metadata=None):
    """Log an activity event for a ticket (ack, work note, status change, escalation request)."""
    with get_db() as conn:
        _ensure_ticket_exists(conn, ticket_id)
        cursor = conn.execute(
            """INSERT INTO ticket_activity_logs (ticket_id, user_name, activity_type, note, metadata, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                ticket_id,
                user_name,
                activity_type,
                note,
                json.dumps(metadata) if metadata else None,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        return cursor.lastrowid


def get_ticket_activities(ticket_id):
    """Retrieve all chronological activity events for a ticket."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM ticket_activity_logs
               WHERE ticket_id = ?
               ORDER BY created_at ASC""",
            (ticket_id,),
        ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("metadata"):
                try:
                    d["metadata"] = json.loads(d["metadata"])
                except Exception:
                    pass
            results.append(d)
        return results


def acknowledge_ticket(ticket_id, user_name):
    """Acknowledge ticket assignment by an agent."""
    now = datetime.now(timezone.utc).isoformat()
    success = update_ticket(ticket_id, {
        "status": "In Progress",
        "acknowledged_at": now,
        "acknowledged_by": user_name,
    })
    if success:
        insert_ticket_activity(
            ticket_id=ticket_id,
            user_name=user_name,
            activity_type="ack",
            note=f"Ticket acknowledged by {user_name}. Status changed to In Progress.",
        )
    return success


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def _ensure_ticket_exists(conn, ticket_id):
    """Ensure a ticket record exists to satisfy foreign key constraints."""
    if not ticket_id:
        return
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT OR IGNORE INTO tickets (ticket_id, title, status, created_at, updated_at)
           VALUES (?, ?, 'Open', ?, ?)""",
        (ticket_id, f"Ticket {ticket_id}", now, now),
    )


def insert_prediction(prediction_data):
    """Insert a prediction log entry."""
    with get_db() as conn:
        _ensure_ticket_exists(conn, prediction_data.get("ticket_id"))
        warnings = prediction_data.get("data_quality_warnings") or prediction_data.get("warnings") or []
        conn.execute(
            """INSERT INTO prediction_logs
               (prediction_id, ticket_id, predicted_at, risk_score, risk_label,
                decision_threshold, model_version, feature_snapshot,
                shap_explanation, recommendation, uncertainty, data_quality_score, data_quality_warnings)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                prediction_data["prediction_id"],
                prediction_data["ticket_id"],
                prediction_data.get("predicted_at", datetime.now(timezone.utc).isoformat()),
                prediction_data["risk_score"],
                prediction_data["risk_label"],
                prediction_data.get("decision_threshold", 0.5),
                prediction_data["model_version"],
                json.dumps(prediction_data.get("feature_snapshot", {})),
                json.dumps(prediction_data.get("shap_explanation")),
                prediction_data.get("recommendation"),
                prediction_data.get("prediction_uncertainty") or prediction_data.get("uncertainty"),
                prediction_data.get("data_quality_score"),
                json.dumps(warnings) if warnings else None,
            ),
        )
    return prediction_data["prediction_id"]


def get_prediction(prediction_id):
    """Fetch a single prediction by ID."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM prediction_logs WHERE prediction_id = ?", (prediction_id,)
        ).fetchone()
        if row:
            result = dict(row)
            result["feature_snapshot"] = json.loads(result.get("feature_snapshot", "{}"))
            result["shap_explanation"] = json.loads(result.get("shap_explanation") or "null")
            if result.get("data_quality_warnings"):
                try:
                    result["data_quality_warnings"] = json.loads(result["data_quality_warnings"])
                except Exception:
                    pass
            return result
        return None


def get_predictions_for_ticket(ticket_id):
    """Fetch all predictions for a ticket, newest first."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM prediction_logs WHERE ticket_id = ? ORDER BY predicted_at DESC",
            (ticket_id,),
        ).fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["feature_snapshot"] = json.loads(r.get("feature_snapshot", "{}"))
            r["shap_explanation"] = json.loads(r.get("shap_explanation") or "null")
            if r.get("data_quality_warnings"):
                try:
                    r["data_quality_warnings"] = json.loads(r["data_quality_warnings"])
                except Exception:
                    pass
            results.append(r)
        return results


def _valid_prediction_id(conn, pred_id):
    """Return pred_id if it exists in prediction_logs, else None."""
    if not pred_id:
        return None
    row = conn.execute(
        "SELECT prediction_id FROM prediction_logs WHERE prediction_id = ?",
        (pred_id,)
    ).fetchone()
    return pred_id if row else None


# ---------------------------------------------------------------------------
# Outcome and feedback helpers
# ---------------------------------------------------------------------------

def insert_outcome(outcome_data):
    """Record a confirmed SLA outcome."""
    with get_db() as conn:
        _ensure_ticket_exists(conn, outcome_data.get("ticket_id"))
        pred_id = _valid_prediction_id(conn, outcome_data.get("prediction_id"))
        conn.execute(
            """INSERT INTO sla_outcomes
               (ticket_id, prediction_id, actual_breach, outcome_status,
                outcome_confirmed_at, confirmed_by, notes)
               VALUES (?,?,?,?,?,?,?)""",
            (
                outcome_data["ticket_id"],
                pred_id,
                outcome_data["actual_breach"],
                outcome_data.get("outcome_status", "confirmed"),
                outcome_data.get(
                    "outcome_confirmed_at", datetime.now(timezone.utc).isoformat()
                ),
                outcome_data.get("confirmed_by"),
                outcome_data.get("notes"),
            ),
        )


def get_outcome(ticket_id):
    """Fetch the latest confirmed outcome for a ticket."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT * FROM sla_outcomes WHERE ticket_id = ?
               ORDER BY outcome_confirmed_at DESC LIMIT 1""",
            (ticket_id,),
        ).fetchone()
        return dict(row) if row else None


def correct_outcome_record(ticket_id, new_actual_breach, corrected_by, reason):
    """
    Formally correct a previously recorded SLA outcome with append-only audit trail.
    """
    now = datetime.now(timezone.utc).isoformat()
    new_breach_int = 1 if new_actual_breach else 0
    with get_db() as conn:
        existing = conn.execute(
            "SELECT * FROM sla_outcomes WHERE ticket_id = ? ORDER BY outcome_confirmed_at DESC LIMIT 1",
            (ticket_id,),
        ).fetchone()
        if not existing:
            raise ValueError(f"No existing outcome found for ticket {ticket_id} to correct.")

        old_breach = int(existing["actual_breach"])
        if old_breach == new_breach_int:
            return False  # No change

        # 1. Append to audit trail
        conn.execute(
            """INSERT INTO outcome_audit_logs (ticket_id, previous_breach, corrected_breach, corrected_by, reason, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (ticket_id, old_breach, new_breach_int, corrected_by, reason, now),
        )

        # 2. Update existing outcome record
        new_status = "confirmed_breach" if new_breach_int == 1 else "confirmed_met"
        conn.execute(
            """UPDATE sla_outcomes
               SET actual_breach = ?, outcome_status = ?, confirmed_by = ?,
                   outcome_confirmed_at = ?, notes = ?
               WHERE id = ?""",
            (
                new_breach_int,
                new_status,
                corrected_by,
                now,
                f"Corrected from {old_breach} to {new_breach_int}. Reason: {reason}",
                existing["id"],
            ),
        )

        # 3. Synchronize ticket record
        conn.execute(
            "UPDATE tickets SET sla_breach = ?, updated_at = ? WHERE ticket_id = ?",
            (new_breach_int, now, ticket_id),
        )
        return True


def get_outcome_audits(ticket_id=None):
    """Retrieve audit trail of outcome corrections."""
    with get_db() as conn:
        if ticket_id:
            rows = conn.execute(
                "SELECT * FROM outcome_audit_logs WHERE ticket_id = ? ORDER BY created_at DESC",
                (ticket_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM outcome_audit_logs ORDER BY created_at DESC LIMIT 100"
            ).fetchall()
        return [dict(r) for r in rows]


def insert_feedback(feedback_data):
    """Record agent feedback on a recommendation."""
    with get_db() as conn:
        _ensure_ticket_exists(conn, feedback_data.get("ticket_id"))
        pred_id = _valid_prediction_id(conn, feedback_data.get("prediction_id"))
        conn.execute(
            """INSERT INTO agent_feedback
               (ticket_id, prediction_id, feedback_label, feedback_reason,
                reviewer_name, created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                feedback_data["ticket_id"],
                pred_id,
                feedback_data["feedback_label"],
                feedback_data.get("feedback_reason"),
                feedback_data.get("reviewer_name"),
                feedback_data.get("created_at", datetime.now(timezone.utc).isoformat()),
            ),
        )


def get_feedback_summary():
    """Aggregate feedback statistics."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT feedback_label, COUNT(*) as cnt FROM agent_feedback GROUP BY feedback_label"
        ).fetchall()
        summary = {"total": 0, "helpful": 0, "not_helpful": 0, "incorrect": 0}
        for row in rows:
            label = row["feedback_label"]
            count = row["cnt"]
            summary["total"] += count
            if label in summary:
                summary[label] = count
        summary["helpful_rate"] = (
            round(summary["helpful"] / summary["total"], 3)
            if summary["total"] > 0
            else 0.0
        )
        return summary


# ---------------------------------------------------------------------------
# Model version helpers
# ---------------------------------------------------------------------------

def insert_model_version(version_data):
    """Register a new model version."""
    with get_db() as conn:
        conn.execute(
            """INSERT OR REPLACE INTO model_versions
               (version, model_path, feature_columns, training_date,
                training_samples, test_samples, precision_score, recall_score,
                f1_score, brier_score, baseline_accuracy,
                is_active, is_candidate, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                version_data["version"],
                version_data["model_path"],
                json.dumps(version_data.get("feature_columns", [])),
                version_data.get("training_date", datetime.now(timezone.utc).isoformat()),
                version_data.get("training_samples"),
                version_data.get("test_samples"),
                version_data.get("precision_score"),
                version_data.get("recall_score"),
                version_data.get("f1_score"),
                version_data.get("brier_score"),
                version_data.get("baseline_accuracy"),
                version_data.get("is_active", 0),
                version_data.get("is_candidate", 0),
                version_data.get("notes"),
            ),
        )


def get_active_model_version():
    """Get the currently active model version."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM model_versions WHERE is_active = 1 ORDER BY training_date DESC LIMIT 1"
        ).fetchone()
        if row:
            result = dict(row)
            result["feature_columns"] = json.loads(result.get("feature_columns", "[]"))
            return result
        return None


def activate_model_version(version, approved_by):
    """Activate a model version and deactivate all others, recording previous version for rollback."""
    with get_db() as conn:
        current_active = conn.execute(
            "SELECT version FROM model_versions WHERE is_active = 1 LIMIT 1"
        ).fetchone()
        prev_version = current_active["version"] if current_active and current_active["version"] != version else None

        conn.execute("UPDATE model_versions SET is_active = 0")
        conn.execute(
            """UPDATE model_versions
               SET is_active = 1, is_candidate = 0,
                   approved_by = ?, approved_at = ?,
                   rollback_version = COALESCE(?, rollback_version)
               WHERE version = ?""",
            (approved_by, datetime.now(timezone.utc).isoformat(), prev_version, version),
        )


def get_previous_model_version(current_version=None):
    """Find the previous model version suitable for rollback."""
    with get_db() as conn:
        if not current_version:
            active = conn.execute("SELECT version, rollback_version FROM model_versions WHERE is_active = 1 LIMIT 1").fetchone()
            if active and active["rollback_version"]:
                target_ver = active["rollback_version"]
                row = conn.execute("SELECT * FROM model_versions WHERE version = ?", (target_ver,)).fetchone()
                if row:
                    res = dict(row)
                    res["feature_columns"] = json.loads(res.get("feature_columns", "[]"))
                    return res
            if active:
                current_version = active["version"]

        # Fallback: most recently trained previous version
        query = "SELECT * FROM model_versions WHERE version != ? ORDER BY training_date DESC LIMIT 1" if current_version else "SELECT * FROM model_versions ORDER BY training_date DESC LIMIT 1"
        params = (current_version,) if current_version else ()
        row = conn.execute(query, params).fetchone()
        if row:
            res = dict(row)
            res["feature_columns"] = json.loads(res.get("feature_columns", "[]"))
            return res
        return None


def rollback_model_version(approved_by="admin"):
    """Roll back to the previous model version atomically."""
    prev = get_previous_model_version()
    if not prev:
        raise ValueError("No previous model version available for rollback.")
    prev_version = prev["version"]
    activate_model_version(prev_version, approved_by=f"rollback_by_{approved_by}")
    return prev_version


def get_model_versions():
    """List all model versions."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM model_versions ORDER BY training_date DESC"
        ).fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["feature_columns"] = json.loads(r.get("feature_columns", "[]"))
            results.append(r)
        return results


# ---------------------------------------------------------------------------
# Model evaluation helpers
# ---------------------------------------------------------------------------

def insert_evaluation(eval_data):
    """Record a model evaluation."""
    with get_db() as conn:
        conn.execute(
            """INSERT INTO model_evaluations
               (model_version, evaluation_date, precision_score, recall_score,
                f1_score, brier_score, false_positives, false_negatives,
                true_positives, true_negatives, total_predictions,
                confirmed_outcomes, drift_status, drift_details, notes)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                eval_data["model_version"],
                eval_data.get("evaluation_date", datetime.now(timezone.utc).isoformat()),
                eval_data.get("precision_score"),
                eval_data.get("recall_score"),
                eval_data.get("f1_score"),
                eval_data.get("brier_score"),
                eval_data.get("false_positives", 0),
                eval_data.get("false_negatives", 0),
                eval_data.get("true_positives", 0),
                eval_data.get("true_negatives", 0),
                eval_data.get("total_predictions", 0),
                eval_data.get("confirmed_outcomes", 0),
                eval_data.get("drift_status", "stable"),
                json.dumps(eval_data.get("drift_details")),
                eval_data.get("notes"),
            ),
        )


def get_evaluations(model_version=None, limit=20):
    """Get model evaluation history."""
    with get_db() as conn:
        if model_version:
            rows = conn.execute(
                """SELECT * FROM model_evaluations
                   WHERE model_version = ?
                   ORDER BY evaluation_date DESC LIMIT ?""",
                (model_version, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM model_evaluations ORDER BY evaluation_date DESC LIMIT ?",
                (limit,),
            ).fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["drift_details"] = json.loads(r.get("drift_details") or "null")
            results.append(r)
        return results


# ---------------------------------------------------------------------------
# Escalation helpers
# ---------------------------------------------------------------------------

def insert_escalation(esc_data):
    """Create a new escalation log entry. Returns the escalation id."""
    with get_db() as conn:
        _ensure_ticket_exists(conn, esc_data.get("ticket_id"))
        pred_id = _valid_prediction_id(conn, esc_data.get("prediction_id"))
        cursor = conn.execute(
            """INSERT INTO escalation_logs
               (ticket_id, prediction_id, risk_score, trigger_conditions,
                assigned_reviewer, escalation_level, status, timeout_at,
                snoozed_until, snooze_reason, evidence_snapshot,
                escalation_chain_level, parent_escalation_id, explanation_summary, version)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,1)""",
            (
                esc_data["ticket_id"],
                pred_id,
                esc_data["risk_score"],
                esc_data["trigger_conditions"],
                esc_data.get("assigned_reviewer"),
                esc_data.get("escalation_level", "medium"),
                esc_data.get("status", "pending"),
                esc_data.get("timeout_at"),
                esc_data.get("snoozed_until"),
                esc_data.get("snooze_reason"),
                json.dumps(esc_data.get("evidence_snapshot")) if esc_data.get("evidence_snapshot") else None,
                esc_data.get("escalation_chain_level", 1),
                esc_data.get("parent_escalation_id"),
                esc_data.get("explanation_summary"),
            ),
        )
        return cursor.lastrowid


def get_escalation(escalation_id):
    """Fetch a single escalation by ID with evidence snapshot decoded."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM escalation_logs WHERE id = ?", (escalation_id,)
        ).fetchone()
        if not row:
            return None
        res = dict(row)
        if res.get("evidence_snapshot"):
            try:
                res["evidence_snapshot"] = json.loads(res["evidence_snapshot"])
            except Exception:
                pass
        return res


def get_escalations(status=None, limit=50):
    """List escalations with optional status filter."""
    with get_db() as conn:
        if status:
            rows = conn.execute(
                """SELECT * FROM escalation_logs
                   WHERE status = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (status, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM escalation_logs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        results = []
        for r in rows:
            d = dict(r)
            if d.get("evidence_snapshot"):
                try:
                    d["evidence_snapshot"] = json.loads(d["evidence_snapshot"])
                except Exception:
                    pass
            results.append(d)
        return results


def get_escalation_counts():
    """Get count of escalations by status."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM escalation_logs GROUP BY status"
        ).fetchall()
        counts = {}
        for row in rows:
            counts[row["status"]] = row["cnt"]
        return counts


def acknowledge_escalation(escalation_id, reviewer_name):
    """Reviewer acknowledges an escalation, changing status to under_review."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        result = conn.execute(
            """UPDATE escalation_logs
               SET status = 'under_review',
                   acknowledged_at = ?,
                   acknowledged_by = ?,
                   reviewer_name = COALESCE(reviewer_name, ?),
                   version = version + 1
               WHERE id = ? AND status IN ('pending', 'pending_ack')""",
            (now, reviewer_name, reviewer_name, escalation_id),
        )
        return result.rowcount > 0


def snooze_escalation(escalation_id, snoozed_until_iso, snooze_reason, reviewer_name):
    """Temporarily defer an escalation with reviewer reason."""
    with get_db() as conn:
        result = conn.execute(
            """UPDATE escalation_logs
               SET status = 'snoozed',
                   snoozed_until = ?,
                   snooze_reason = ?,
                   reviewer_name = ?,
                   version = version + 1
               WHERE id = ?""",
            (snoozed_until_iso, snooze_reason, reviewer_name, escalation_id),
        )
        return result.rowcount > 0


def reopen_escalation(escalation_id, reopen_reason, reviewer_name):
    """Reopen an escalation if risk remains unresolved."""
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as conn:
        result = conn.execute(
            """UPDATE escalation_logs
               SET status = 'reopened',
                   reopened_at = ?,
                   reopened_by = ?,
                   reopen_reason = ?,
                   version = version + 1
               WHERE id = ?""",
            (now, reviewer_name, reopen_reason, escalation_id),
        )
        return result.rowcount > 0


def update_escalation_action_effectiveness(escalation_id, completed, effective, notes, reviewer_name):
    """Record whether approved action was completed and whether it helped."""
    with get_db() as conn:
        result = conn.execute(
            """UPDATE escalation_logs
               SET action_completed = ?,
                   action_effective = ?,
                   action_notes = ?,
                   version = version + 1
               WHERE id = ?""",
            (1 if completed else 0, effective, notes, escalation_id),
        )
        return result.rowcount > 0


def update_escalation_decision(escalation_id, decision, reason, reviewer_name, expected_version=None):
    """Record a reviewer's decision on an escalation with optimistic concurrency protection."""
    now = datetime.now(timezone.utc).isoformat()
    status_map = {
        "approve": "approved",
        "reject": "rejected",
        "reassign": "reassigned",
        "investigate": "investigating",
    }
    new_status = status_map.get(decision, decision)

    with get_db() as conn:
        # Check current record and optimistic lock
        row = conn.execute("SELECT version, status FROM escalation_logs WHERE id = ?", (escalation_id,)).fetchone()
        if not row:
            raise ValueError(f"Escalation {escalation_id} not found.")

        current_ver = row["version"]
        if expected_version is not None and current_ver != expected_version:
            raise ValueError(
                f"Conflict: Escalation {escalation_id} was modified by another reviewer (current v{current_ver}, expected v{expected_version}). Please refresh."
            )

        conn.execute(
            """UPDATE escalation_logs
               SET status = ?, decision = ?, decision_reason = ?,
                   reviewer_name = ?, decided_at = ?, version = version + 1
               WHERE id = ?""",
            (new_status, decision, reason, reviewer_name, now, escalation_id),
        )


def check_duplicate_escalation(ticket_id, window_minutes=120):
    """Check if there is an active (pending or under review) escalation for this ticket within the dedup window."""
    with get_db() as conn:
        row = conn.execute(
            """SELECT id FROM escalation_logs
               WHERE ticket_id = ? AND status IN ('pending', 'pending_ack', 'under_review')
               AND datetime(created_at, '+' || ? || ' minutes') > datetime('now')
               LIMIT 1""",
            (ticket_id, window_minutes),
        ).fetchone()
        return row is not None


def get_expired_escalations():
    """Get pending escalations that have exceeded their timeout."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT * FROM escalation_logs
               WHERE status IN ('pending', 'pending_ack')
               AND timeout_at IS NOT NULL
               AND datetime(timeout_at) < datetime('now')"""
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# User helpers
# ---------------------------------------------------------------------------

def get_user(username):
    """Fetch a user by username."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ? AND active = 1", (username,)
        ).fetchone()
        return dict(row) if row else None


def get_users_by_role(role):
    """Fetch all active users with a given role."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM users WHERE role = ? AND active = 1", (role,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_predictions_with_outcomes():
    """Join predictions with confirmed outcomes for metric calculation."""
    with get_db() as conn:
        rows = conn.execute(
            """SELECT p.prediction_id, p.ticket_id, p.risk_score, p.risk_label,
                      p.decision_threshold, p.model_version, p.feature_snapshot,
                      p.predicted_at,
                      o.actual_breach, o.outcome_status
               FROM prediction_logs p
               INNER JOIN sla_outcomes o ON p.prediction_id = o.prediction_id
               WHERE o.outcome_status IN ('confirmed', 'confirmed_breach', 'confirmed_met')
               ORDER BY p.predicted_at DESC"""
        ).fetchall()
        results = []
        for row in rows:
            r = dict(row)
            r["feature_snapshot"] = json.loads(r.get("feature_snapshot", "{}"))
            results.append(r)
        return results


if __name__ == "__main__":
    path = init_db()
    print(f"Database initialized at: {path}")
