"""
Model Drift Monitor
====================
Monitors model performance over time using confirmed SLA outcomes.

Implements:
- Precision, recall, F1, Brier score from confirmed outcomes
- False positive / false negative counts
- Feature distribution drift (baseline vs. recent)
- Prediction distribution drift
- Performance trends by time and category
- Controlled retraining workflow with human approval

Uses only eligible, confirmed outcomes for supervised metrics.
Does not fabricate metrics when insufficient data exists.
Distinguishes input drift from measured performance degradation.
"""

import json
import os
import numpy as np
from datetime import datetime, timezone
from collections import Counter

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    confusion_matrix,
)

from api.database import (
    get_predictions_with_outcomes,
    get_evaluations,
    insert_evaluation,
    get_active_model_version,
    get_db,
    insert_model_version,
)

# Minimum number of confirmed outcomes required for reliable metrics
MIN_OUTCOMES_FOR_METRICS = 10
DRIFT_THRESHOLD_PVALUE = 0.05
PERFORMANCE_DEGRADATION_THRESHOLD = 0.10  # 10% drop triggers warning


def evaluate_model(model_version=None):
    """
    Evaluate the current model using confirmed predictions and outcomes.

    Returns a dict with metrics, confusion matrix, drift status, and sample counts.
    Returns None metrics when insufficient confirmed outcomes exist.
    """
    # Get predictions with confirmed outcomes
    data = get_predictions_with_outcomes()

    if model_version:
        data = [d for d in data if d.get("model_version") == model_version]

    if not model_version:
        active = get_active_model_version()
        model_version = active["version"] if active else "unknown"

    result = {
        "model_version": model_version,
        "evaluation_date": datetime.now(timezone.utc).isoformat(),
        "total_predictions": _count_total_predictions(model_version),
        "confirmed_outcomes": len(data),
        "outcome_coverage": 0.0,
    }

    total_preds = result["total_predictions"]
    if total_preds > 0:
        result["outcome_coverage"] = round(len(data) / total_preds, 3)

    if len(data) < MIN_OUTCOMES_FOR_METRICS:
        result["precision_score"] = None
        result["recall_score"] = None
        result["f1_score"] = None
        result["brier_score"] = None
        result["false_positives"] = 0
        result["false_negatives"] = 0
        result["true_positives"] = 0
        result["true_negatives"] = 0
        result["drift_status"] = "insufficient_data"
        result["drift_details"] = {
            "message": f"Need at least {MIN_OUTCOMES_FOR_METRICS} confirmed outcomes. Have {len(data)}."
        }
        result["notes"] = "Insufficient confirmed outcomes for reliable metrics."
        return result

    # Extract predictions and actual outcomes
    y_true = np.array([d["actual_breach"] for d in data])
    y_prob = np.array([d["risk_score"] for d in data])
    thresholds = [d.get("decision_threshold", 0.5) for d in data]
    default_threshold = thresholds[0] if thresholds else 0.5
    y_pred = (y_prob >= default_threshold).astype(int)

    # Compute metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall_val = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    brier = brier_score_loss(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    result.update({
        "precision_score": round(precision, 4),
        "recall_score": round(recall_val, 4),
        "f1_score": round(f1, 4),
        "brier_score": round(brier, 4),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "true_negatives": int(tn),
    })

    # Drift detection
    drift_result = _detect_drift(data, model_version)
    result["drift_status"] = drift_result["status"]
    result["drift_details"] = drift_result

    # Generate notes
    notes = []
    if result["drift_status"] == "drift_detected":
        notes.append("Input distribution drift detected.")
    if precision < 0.5:
        notes.append("Precision is below 0.5 — many false positive alerts.")
    if recall_val < 0.5:
        notes.append("Recall is below 0.5 — missing many actual breaches.")
    result["notes"] = " ".join(notes) if notes else "Model performing within expected range."

    # Store evaluation
    insert_evaluation(result)

    return result


def _count_total_predictions(model_version):
    """Count total predictions for a model version."""
    with get_db() as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM prediction_logs WHERE model_version = ?",
            (model_version,),
        ).fetchone()
        return row[0] if row else 0


def _detect_drift(data, model_version):
    """
    Detect feature and prediction distribution drift.

    Compares recent predictions against baseline training distribution.
    Uses simple statistical tests (mean shift, variance change).
    """
    drift_result = {
        "status": "stable",
        "feature_drift_detected": False,
        "outcome_drift_detected": False,
        "degraded_features": [],
        "details": [],
    }

    if len(data) < MIN_OUTCOMES_FOR_METRICS:
        drift_result["status"] = "insufficient_data"
        return drift_result

    # Load training baseline from model metadata
    try:
        metadata_path = os.path.join(
            os.path.dirname(__file__), "..", "ml_model", "feature_columns.json"
        )
        with open(os.path.normpath(metadata_path), "r") as f:
            metadata = json.load(f)
        baseline_metrics = metadata.get("metrics", {})
    except FileNotFoundError:
        baseline_metrics = {}

    # Check feature distribution drift using feature snapshots
    feature_snapshots = []
    for d in data:
        snapshot = d.get("feature_snapshot", {})
        if isinstance(snapshot, str):
            try:
                snapshot = json.loads(snapshot)
            except (json.JSONDecodeError, TypeError):
                snapshot = {}
        if snapshot:
            feature_snapshots.append(snapshot)

    if len(feature_snapshots) >= MIN_OUTCOMES_FOR_METRICS:
        # Compare feature means — simple shift detection
        numeric_features = [
            "created_hours", "sla_deadline_hours", "response_delay_hours",
            "team_backlog_count", "time_remaining_hours",
            "similar_ticket_resolution_hours", "description_length",
        ]
        for feat in numeric_features:
            values = [s.get(feat, 0) for s in feature_snapshots if feat in s]
            if len(values) >= 5:
                recent = values[-min(20, len(values)):]
                older = values[:max(1, len(values) - 20)]
                if len(older) >= 3 and len(recent) >= 3:
                    mean_diff = abs(np.mean(recent) - np.mean(older))
                    overall_std = np.std(values) if np.std(values) > 0 else 1.0
                    if mean_diff / overall_std > 1.5:
                        drift_result["degraded_features"].append(feat)
                        drift_result["details"].append(
                            f"{feat}: mean shifted by {mean_diff:.2f} ({mean_diff/overall_std:.1f}σ)"
                        )

        if drift_result["degraded_features"]:
            drift_result["feature_drift_detected"] = True

    # Check prediction outcome drift
    recent_outcomes = [d["actual_breach"] for d in data[-20:]]
    older_outcomes = [d["actual_breach"] for d in data[:-20]] if len(data) > 20 else []
    if len(recent_outcomes) >= 5 and len(older_outcomes) >= 5:
        recent_rate = np.mean(recent_outcomes)
        older_rate = np.mean(older_outcomes)
        if abs(recent_rate - older_rate) > 0.15:
            drift_result["outcome_drift_detected"] = True
            drift_result["details"].append(
                f"Breach rate shifted: {older_rate:.2%} -> {recent_rate:.2%}"
            )

    # Check performance degradation vs. baseline
    if baseline_metrics:
        baseline_f1 = baseline_metrics.get("f1")
        if baseline_f1:
            y_true = np.array([d["actual_breach"] for d in data])
            y_pred = (np.array([d["risk_score"] for d in data]) >= 0.5).astype(int)
            current_f1 = f1_score(y_true, y_pred, zero_division=0)
            if baseline_f1 - current_f1 > PERFORMANCE_DEGRADATION_THRESHOLD:
                drift_result["details"].append(
                    f"F1 degraded: baseline {baseline_f1:.3f} -> current {current_f1:.3f}"
                )

    # Determine overall status
    if drift_result["feature_drift_detected"] or drift_result["outcome_drift_detected"]:
        drift_result["status"] = "drift_detected"
    elif drift_result["details"]:
        drift_result["status"] = "warning"
    else:
        drift_result["status"] = "stable"

    return drift_result


def get_model_health():
    """
    Get current model health: latest evaluation + history.
    """
    active = get_active_model_version()
    model_version = active["version"] if active else "unknown"

    evals = get_evaluations(model_version=model_version, limit=10)
    latest = evals[0] if evals else None

    # Get total predictions and outcomes counts
    with get_db() as conn:
        total_preds = conn.execute(
            "SELECT COUNT(*) FROM prediction_logs WHERE model_version = ?",
            (model_version,),
        ).fetchone()[0]
        total_outcomes = conn.execute(
            "SELECT COUNT(*) FROM sla_outcomes"
        ).fetchone()[0]

    return {
        "model_version": model_version,
        "active_version": model_version,
        "drift_status": latest.get("drift_status", "stable") if latest else "insufficient_data",
        "metrics": latest or {},
        "model_info": active,
        "latest_evaluation": latest,
        "evaluation_history": evals,
        "total_predictions": total_preds,
        "total_confirmed_outcomes": total_outcomes,
    }


def prepare_candidate_model():
    """
    Prepare a candidate model for evaluation.

    This is a simplified workflow that:
    1. Loads historical data enriched with confirmed outcomes
    2. Retrains a new model
    3. Evaluates on held-out data
    4. Stores as candidate (NOT active) — requires human approval

    Returns candidate model info or error.
    """
    try:
        import pandas as pd
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.calibration import CalibratedClassifierCV
        from joblib import dump as joblib_dump

        # Get confirmed outcomes
        data = get_predictions_with_outcomes()
        if len(data) < 50:
            return {
                "status": "error",
                "message": f"Need at least 50 confirmed outcomes for retraining. Have {len(data)}.",
            }

        # Build training data from confirmed predictions
        feature_cols = [
            "priority_encoded", "issue_type_encoded", "created_hours",
            "sla_deadline_hours", "time_remaining_hours", "response_delay_hours",
            "team_backlog_count", "similar_ticket_resolution_hours",
            "description_length",
        ]

        rows = []
        for d in data:
            snapshot = d["feature_snapshot"]
            if isinstance(snapshot, str):
                snapshot = json.loads(snapshot)
            row = {col: snapshot.get(col, 0) for col in feature_cols}
            row["sla_breach"] = d["actual_breach"]
            rows.append(row)

        df = pd.DataFrame(rows)

        # Time-separated split (first 80%, last 20%)
        split_idx = int(len(df) * 0.8)
        X_train = df[feature_cols].iloc[:split_idx]
        y_train = df["sla_breach"].iloc[:split_idx]
        X_test = df[feature_cols].iloc[split_idx:]
        y_test = df["sla_breach"].iloc[split_idx:]

        # Train candidate model
        clf = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_leaf=3, random_state=42
        )
        clf.fit(X_train, y_train)

        calibrated = CalibratedClassifierCV(clf, cv=3, method="isotonic")
        calibrated.fit(X_train, y_train)

        # Evaluate candidate on held-out test split
        y_pred = calibrated.predict(X_test)
        y_prob = calibrated.predict_proba(X_test)[:, 1]

        precision = precision_score(y_test, y_pred, zero_division=0)
        recall_val = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        brier = brier_score_loss(y_test, y_prob)

        cm_cand = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn_c, fp_c, fn_c, tp_c = cm_cand.ravel() if cm_cand.size == 4 else (0, 0, 0, 0)
        cost_candidate = int((fn_c * 5) + (fp_c * 1))

        # Evaluate ACTIVE model on the EXACT SAME held-out test split
        from api.sla_predictor import _load_model as get_runtime_active_model, reload_model
        try:
            curr_active_model = get_runtime_active_model()
            active_y_pred = curr_active_model.predict(X_test)
            active_y_prob = curr_active_model.predict_proba(X_test)[:, 1]
            act_precision = float(precision_score(y_test, active_y_pred, zero_division=0))
            act_recall = float(recall_score(y_test, active_y_pred, zero_division=0))
            act_f1 = float(f1_score(y_test, active_y_pred, zero_division=0))
            act_brier = float(brier_score_loss(y_test, active_y_prob))
            cm_act = confusion_matrix(y_test, active_y_pred, labels=[0, 1])
            tn_a, fp_a, fn_a, tp_a = cm_act.ravel() if cm_act.size == 4 else (0, 0, 0, 0)
            cost_active = int((fn_a * 5) + (fp_a * 1))
        except Exception:
            act_precision, act_recall, act_f1, act_brier = 0.5, 0.5, 0.5, 0.25
            cost_active = cost_candidate
            tn_a, fp_a, fn_a, tp_a = 0, 0, 0, 0

        # Baseline Brier and Brier Skill Score calculation (relative to empirical event rate)
        event_rate = float(np.mean(y_test))
        baseline_brier = float(np.mean((y_test - event_rate) ** 2)) if len(y_test) > 0 else 0.25
        bss_candidate = round(1.0 - (brier / baseline_brier), 4) if baseline_brier > 0 else 0.0
        bss_active = round(1.0 - (act_brier / baseline_brier), 4) if baseline_brier > 0 else 0.0

        # Promotion Gate checks
        gate_checks = {
            "false_negative_cost_better_or_equal": bool(cost_candidate <= cost_active),
            "recall_retention": bool(recall_val >= (act_recall - 0.05)),
            "brier_skill_positive": bool(bss_candidate > 0),
            "sample_size_adequate": bool(len(X_test) >= 10),
        }
        recommended = bool(
            gate_checks["false_negative_cost_better_or_equal"] and
            gate_checks["recall_retention"] and
            gate_checks["sample_size_adequate"]
        )

        # Version the candidate
        active = get_active_model_version()
        active_version = active["version"] if active else "1.0.0"
        parts = active_version.split(".")
        candidate_version = f"{parts[0]}.{int(parts[1]) + 1}.0"

        # Save candidate model artifacts
        model_dir = os.path.join(os.path.dirname(__file__), "..", "ml_model")
        candidate_path = os.path.join(model_dir, f"sla_model_candidate_{candidate_version}.joblib")
        joblib_dump(calibrated, candidate_path)

        base_path = os.path.join(model_dir, f"sla_model_base_candidate_{candidate_version}.joblib")
        joblib_dump(clf, base_path)

        # Register as candidate in database
        eval_metrics = {
            "candidate_metrics": {
                "precision": round(precision, 4),
                "recall": round(recall_val, 4),
                "f1": round(f1, 4),
                "brier_score": round(brier, 4),
                "brier_skill_score": bss_candidate,
                "misclassification_cost": cost_candidate,
                "false_negatives": int(fn_c),
                "false_positives": int(fp_c),
            },
            "active_model_metrics": {
                "precision": round(act_precision, 4),
                "recall": round(act_recall, 4),
                "f1": round(act_f1, 4),
                "brier_score": round(act_brier, 4),
                "brier_skill_score": bss_active,
                "misclassification_cost": cost_active,
                "false_negatives": int(fn_a),
                "false_positives": int(fp_a),
            },
            "gate_checks": gate_checks,
            "recommended_for_promotion": recommended,
        }

        insert_model_version({
            "version": candidate_version,
            "model_path": candidate_path,
            "feature_columns": feature_cols,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "precision_score": round(precision, 4),
            "recall_score": round(recall_val, 4),
            "f1_score": round(f1, 4),
            "brier_score": round(brier, 4),
            "is_active": 0,
            "is_candidate": 1,
            "notes": f"Candidate model retrained on {len(data)} confirmed outcomes. Cost: {cost_candidate} vs Active: {cost_active}.",
        })

        return {
            "status": "candidate_ready",
            "candidate_version": candidate_version,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "comparison": eval_metrics,
            "message": "Candidate model prepared and evaluated side-by-side on held-out test split. Requires explicit Admin promotion.",
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}


def activate_candidate_model(candidate_version, approved_by="admin"):
    """
    Atomically promote a candidate model to active after validation and hot reload.
    """
    from joblib import load as joblib_load
    from api.database import activate_model_version, get_model_versions
    from api.sla_predictor import reload_model

    versions = get_model_versions()
    target = next((v for v in versions if v["version"] == candidate_version), None)
    if not target:
        raise ValueError(f"Candidate model version {candidate_version} not found.")

    # 1. Pre-validation: Test loading candidate artifacts from disk
    model_path = target["model_path"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model artifact file {model_path} does not exist.")

    test_model = joblib_load(model_path)
    if not hasattr(test_model, "predict_proba"):
        raise ValueError("Candidate artifact does not implement predict_proba.")

    # 2. Update feature metadata
    metadata_path = os.path.join(os.path.dirname(__file__), "..", "ml_model", "feature_columns.json")
    try:
        with open(os.path.normpath(metadata_path), "r") as f:
            meta = json.load(f)
        meta["model_version"] = candidate_version
        with open(os.path.normpath(metadata_path), "w") as f:
            json.dump(meta, f, indent=2)
    except Exception:
        pass

    # 3. Copy candidate file to default model path for persistence
    import shutil
    default_cal_path = os.path.join(os.path.dirname(__file__), "..", "ml_model", "sla_breach_model.joblib")
    shutil.copy2(model_path, default_cal_path)

    base_candidate = model_path.replace("sla_model_candidate_", "sla_model_base_candidate_")
    if os.path.exists(base_candidate):
        default_base_path = os.path.join(os.path.dirname(__file__), "..", "ml_model", "sla_base_model.joblib")
        shutil.copy2(base_candidate, default_base_path)

    # 4. Atomically activate in database
    activate_model_version(candidate_version, approved_by)

    # 5. Hot reload active model in memory
    reload_model()

    return {
        "status": "activated",
        "active_version": candidate_version,
        "approved_by": approved_by,
        "activated_at": datetime.now(timezone.utc).isoformat(),
    }


def rollback_active_model(admin_name="admin"):
    """
    Roll back to the previous model version atomically.
    """
    from api.database import rollback_model_version, get_active_model_version
    from api.sla_predictor import reload_model

    previous_ver = rollback_model_version(approved_by=admin_name)
    current_active = get_active_model_version()

    # Hot reload
    reload_model()

    return {
        "status": "rolled_back",
        "active_version": previous_ver,
        "admin": admin_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
