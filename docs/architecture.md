# AI Ticket Intelligence Platform — Architecture & Technical Design

## 1. Executive Summary

The **AI Ticket Intelligence Platform** is a local-first, self-contained support operations platform that automates SLA breach prediction, provides explainable insights via SHAP (SHapley Additive exPlanations), continuously monitors model health and data drift, and orchestrates human-in-the-loop escalation workflows.

The system requires **zero cloud dependencies, zero external APIs, and zero Jira credentials**. All model training, inference, explainability calculations, and persistence operate fully offline on the local host.

---

## 2. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                        TICKET INTAKE & OPERATIONS                      │
│        (Web Dashboard UI / REST API: tickets, predictions, triage)     │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FLASK APPLICATION LAYER                         │
│   • Session Auth & RBAC (Admin, Reviewer, Agent)                       │
│   • RESTful API Endpoints (/predict, /explain, /outcomes, etc.)        │
│   • In-App Reviewer Notifications & Audit Logging                      │
└───────┬───────────────────────────┬────────────────────────────┬───────┘
        │                           │                            │
        ▼                           ▼                            ▼
┌──────────────────┐       ┌─────────────────┐       ┌──────────────────┐
│  SLA PREDICTOR   │       │ EXPLAINABILITY  │       │ ESCALATION ENGINE│
│  (sla_predictor) │       │(explainability) │       │(escalation_engine)
│                  │       │                 │       │                  │
│ • Calibrated     │       │ • TreeExplainer │       │ • 5 Policy Rules │
│   RandomForest   │──────▶│ • Top-3 Factors │──────▶│ • Reviewer Route │
│ • 9 Features     │       │ • Model vs Rule │       │ • Deduplication  │
│ • Breach Prob    │       │ • Actionable    │       │ • Timeout Chain  │
│ • Safe Defaults  │       │   Suggestions   │       │ • Human Decision │
└────────┬─────────┘       └────────┬────────┘       └────────┬─────────┘
         │                          │                         │
         └──────────────────────────┼─────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FEEDBACK & DRIFT MONITOR                        │
│   • Prediction Logging with Feature Snapshots (feedback.py)            │
│   • Ground-Truth Outcome Confirmation (Met vs Breached)                │
│   • Supervised Performance Metrics (Precision, Recall, F1, Brier)      │
│   • Distribution Drift Detection (Feature shifts & Outcome shifts)     │
│   • Controlled Model Retraining & Candidate Promotion (drift_monitor)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        SQLITE PERSISTENCE LAYER                        │
│   • users              • prediction_logs        • escalation_logs      │
│   • tickets            • sla_outcomes           • model_versions       │
│   • agent_feedback     • model_evaluations      • WAL mode + FKs       │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Core Modules

### 3.1 SLA Predictor (`api/sla_predictor.py`)
- **Model**: `RandomForestClassifier` calibrated with isotonic regression (`CalibratedClassifierCV`) to output well-calibrated probabilities rather than raw ranking scores.
- **Features (9)**:
  1. `priority_encoded`: Low (0), Medium (1), High (2), Critical (3)
  2. `issue_type_encoded`: Task (0), Bug (1), Service Request (2), Incident (3), Change Request (4)
  3. `created_hours`: Ticket age in hours
  4. `sla_deadline_hours`: SLA target window in hours
  5. `time_remaining_hours`: Effective hours remaining until breach deadline
  6. `response_delay_hours`: Elapsed time without agent acknowledgement
  7. `team_backlog_count`: Current open queue size for assigned team
  8. `similar_ticket_resolution_hours`: Historical resolution benchmark
  9. `description_length`: Ticket description character count
- **Confidence Metric**: Distance from the decision boundary ($|p - \text{threshold}| + 0.5$).
- **Safe Defaults**: Handles missing or unknown inputs gracefully with schema defaults and non-fatal warnings.

### 3.2 Explainability Module (`api/explainability.py`)
- **SHAP Integration**: Employs `shap.TreeExplainer` on the underlying tree ensemble to compute exact local Shapley values for class 1 (SLA breach).
- **Factor Ranking**: Extracts top contributing features, computing magnitude, directional impact (`increases_risk` vs `decreases_risk`), and human-friendly explanation text.
- **Source Transparency**: Explicitly tags factors as `model` (derived from SHAP feature contributions) or `operational_rule` (policy-derived constraints).
- **Actionable Recommendations**: Translates top risk drivers into concrete operational guidance (e.g., immediate reassignment, manager notification, workload triage).

### 3.3 Model Drift & Feedback Learning (`api/drift_monitor.py` & `api/feedback.py`)
- **Separation of Concerns**:
  - `prediction_logs`: Logs predictions at inference time before outcomes occur.
  - `sla_outcomes`: Ground-truth breach status recorded when ticket resolves or deadline expires.
  - `agent_feedback`: Staff ratings on recommendation quality (`helpful`, `not_helpful`, `incorrect`), independent of breach status.
- **Supervised Metrics**: Evaluates active model against confirmed outcomes:
  - Precision, Recall, F1 Score
  - Brier Score Loss (probabilistic calibration error)
  - Confusion Matrix ($TN, FP, FN, TP$)
- **Drift Detection**:
  - **Feature Drift**: Identifies statistical distribution shifts in continuous features (> 1.5 standard deviations).
  - **Outcome Drift**: Flags sudden shifts in the historical breach rate (> 15% delta).
  - **Performance Degradation**: Warns when F1 drops by > 10% from baseline.
- **Retraining Workflow**: Trains candidate models on newly confirmed outcomes; requires explicit admin promotion to activate.

### 3.4 Human-in-the-Loop Escalation Engine (`api/escalation_engine.py`)
- **Policy Rules**: Loaded dynamically from `api/escalation_policies.json`:
  1. *Urgent High Risk*: Risk $\ge 80\% \to$ Critical review (30 min timeout)
  2. *High Risk / Tight Window*: Risk $\ge 60\%$ and time remaining $< 1\text{h} \to$ Team lead review
  3. *Critical Severity*: Priority Critical $\to$ Emergency incident policy
  4. *Low Model Confidence*: Confidence $< 0.55$ and risk $\ge 40\% \to$ Human triage
  5. *Unacknowledged Ticket*: Delay $\ge 4\text{h}$ and risk $\ge 40\% \to$ Escalation chain
- **Reviewer Routing**: Automatically routes tickets to designated teams and roles.
- **Deduplication**: Prevents duplicate escalation alerts for the same ticket within a configurable window (120 minutes).
- **Approval Workflow**: Human reviewers can `approve`, `reject`, `reassign`, or `investigate`.
- **Escalation Chain**: Pending escalations exceeding their SLA timeout automatically escalate to higher authority tiers (Lead $\to$ Senior $\to$ Manager).

---

## 4. Database Schema (SQLite)

The system uses SQLite in WAL mode with enforced foreign keys:

1. `users`: Credentials, password hashes (Werkzeug PBKDF2), roles (`admin`, `reviewer`, `agent`), teams.
2. `tickets`: Ticket metadata, lifecycle state, feature values, confirmed resolution status.
3. `prediction_logs`: Immutable record of every inference, feature snapshot, SHAP factors, and recommendations.
4. `sla_outcomes`: Ground truth outcomes linked to tickets and prediction logs.
5. `agent_feedback`: Staff feedback on recommendation utility.
6. `escalation_logs`: Full audit trail of escalations, reviewer assignments, decisions, and timestamps.
7. `model_versions`: Registry of trained model artifacts, baseline metrics, and activation status.
8. `model_evaluations`: Historical drift evaluations and validation metrics over time.

---

## 5. Security & Access Control

- **Session Authentication**: Secure HTTP-only session cookies with Lax SameSite protection.
- **HTTP Basic Auth**: Seamlessly supported on all API endpoints for programmatic script and automated integration access.
- **Role-Based Access Control (RBAC)**:
  - `agent`: Submit tickets, view predictions, submit feedback.
  - `reviewer`: Decide on escalations, confirm SLA outcomes, view health metrics.
  - `admin`: Activate retrained models, configure policies, full system management.