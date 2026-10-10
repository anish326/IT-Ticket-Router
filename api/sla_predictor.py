"""
SLA Predictor Module
=====================
Loads the active model and feature metadata, validates input,
applies training-time preprocessing, and returns calibrated
breach probabilities.
"""

import os
import json
import uuid
import numpy as np
from datetime import datetime, timezone
from joblib import load

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "ml_model")
def _find_path(filename):
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "ml_model", filename),
        os.path.join(os.path.dirname(__file__), "ml_model", filename),
        os.path.join(os.getcwd(), "ml_model", filename),
        os.path.join(os.path.dirname(__file__), filename)
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.normpath(c)
    return os.path.normpath(candidates[0])

DEFAULT_MODEL_PATH = _find_path("sla_model.joblib")
DEFAULT_BASE_MODEL_PATH = _find_path("sla_model_base.joblib")
DEFAULT_METADATA_PATH = _find_path("feature_columns.json")

# Cached model and metadata
_model = None
_base_model = None
_metadata = None


def _load_metadata():
    """Load feature column metadata from JSON."""
    global _metadata
    if _metadata is not None:
        return _metadata

    path = _find_path("feature_columns.json")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Feature metadata not found at {path}. Run train_model.py first."
        )

    with open(path, "r") as f:
        _metadata = json.load(f)
    return _metadata


def _load_model():
    """Load the calibrated model from joblib."""
    global _model
    if _model is not None:
        return _model

    path = _find_path("sla_model.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run train_model.py first."
        )

    _model = load(path)
    return _model


def _load_base_model():
    """Load the uncalibrated base RandomForest model for SHAP TreeExplainer."""
    global _base_model
    if _base_model is not None:
        return _base_model

    path = _find_path("sla_model_base.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Base model not found at {path}. Run train_model.py first."
        )

    _base_model = load(path)
    return _base_model


def get_model():
    """Return the loaded calibrated model."""
    return _load_model()


def get_base_model():
    """Return the base model (for SHAP TreeExplainer)."""
    return _load_base_model()


def get_metadata():
    """Return feature metadata."""
    return _load_metadata()


def get_feature_columns():
    """Return the ordered list of feature column names."""
    metadata = _load_metadata()
    return metadata["columns"]


def get_model_version():
    """Return the current model version string."""
    metadata = _load_metadata()
    return metadata.get("model_version", "unknown")


def reload_model():
    """Force reload of model and metadata (after retraining)."""
    global _model, _base_model, _metadata
    _model = None
    _base_model = None
    _metadata = None


def preprocess_input(ticket_data):
    """
    Validate and preprocess ticket input data for prediction.

    Applies the exact same transformations as training time.
    Returns (feature_array, feature_dict, warnings).
    """
    metadata = _load_metadata()
    priority_map = metadata.get("priority_map", {})
    issue_type_map = metadata.get("issue_type_map", {})
    feature_columns = metadata["columns"]
    warnings = []

    # Encode priority
    priority_raw = ticket_data.get("priority")
    if not priority_raw:
        warnings.append("Missing priority field, defaulted to Medium")
        priority_encoded = 1
    else:
        priority_encoded = priority_map.get(priority_raw)
        if priority_encoded is None:
            warnings.append(f"Unknown priority '{priority_raw}', defaulting to Medium (1)")
            priority_encoded = 1

    # Encode issue type
    issue_type_raw = ticket_data.get("issue_type")
    if not issue_type_raw:
        warnings.append("Missing issue_type field, defaulted to Task")
        issue_type_encoded = 0
    else:
        issue_type_encoded = issue_type_map.get(issue_type_raw)
        if issue_type_encoded is None:
            warnings.append(f"Unknown issue_type '{issue_type_raw}', defaulting to Task (0)")
            issue_type_encoded = 0

    # Numeric features with defaults and validation
    def safe_float(key, default=0.0, min_val=0.0):
        val = ticket_data.get(key, default)
        try:
            val = float(val)
            if val < min_val:
                warnings.append(f"{key} = {val} is below minimum {min_val}, clamped")
                val = min_val
            return val
        except (TypeError, ValueError):
            warnings.append(f"Invalid {key} = {val}, using default {default}")
            return float(default)

    created_hours = safe_float("created_hours", 0.0)
    sla_deadline_hours = safe_float("sla_deadline_hours", 24.0, min_val=0.1)
    response_delay_hours = safe_float("response_delay_hours", 0.0)
    team_backlog_count = int(safe_float("team_backlog_count", 0))
    similar_ticket_resolution_hours = safe_float(
        "similar_ticket_resolution_hours", 0.0
    )
    desc_raw = ticket_data.get("description", "")
    if "description_length" in ticket_data:
        description_length = int(safe_float("description_length", len(desc_raw)))
    else:
        description_length = len(desc_raw)

    if description_length < 10:
        warnings.append("Short or missing description may limit context quality")
    if team_backlog_count > 300:
        warnings.append(f"High backlog count ({team_backlog_count}) may indicate system stress")

    # Compute time remaining
    time_remaining_hours = ticket_data.get("time_remaining_hours")
    if time_remaining_hours is None:
        time_remaining_hours = max(0.0, sla_deadline_hours - created_hours)
    else:
        time_remaining_hours = max(0.0, float(time_remaining_hours))

    if created_hours > sla_deadline_hours and time_remaining_hours > 0:
        warnings.append("Inconsistency detected: created_hours exceeds sla_deadline_hours but time_remaining > 0")

    # Data quality score: starts at 100, drops with warnings
    data_quality_score = max(10, 100 - (len(warnings) * 15))

    # Build feature dict in column order
    feature_dict = {
        "priority_encoded": priority_encoded,
        "issue_type_encoded": issue_type_encoded,
        "created_hours": created_hours,
        "sla_deadline_hours": sla_deadline_hours,
        "time_remaining_hours": time_remaining_hours,
        "response_delay_hours": response_delay_hours,
        "team_backlog_count": team_backlog_count,
        "similar_ticket_resolution_hours": similar_ticket_resolution_hours,
        "description_length": description_length,
    }

    # Build feature array in the correct column order
    feature_array = np.array(
        [[feature_dict[col] for col in feature_columns]], dtype=np.float64
    )

    return feature_array, feature_dict, warnings


def predict(ticket_data, decision_threshold=0.5):
    """
    Generate an SLA breach prediction for a ticket.

    Args:
        ticket_data: dict with ticket features
        decision_threshold: probability threshold for breach label (default 0.5)

    Returns:
        dict with prediction_id, breach_probability, risk_score, risk_label,
        model_version, feature_snapshot, assigned_team, prediction_uncertainty,
        confidence_interval, data_quality_score, data_quality_warnings
    """
    model = _load_model()
    base_model = _load_base_model()
    metadata = _load_metadata()

    feature_array, feature_dict, warnings = preprocess_input(ticket_data)
    data_quality_score = max(10, 100 - (len(warnings) * 15))

    # Get calibrated probability using DataFrame to retain feature names
    import pandas as pd
    df_features = pd.DataFrame(feature_array, columns=metadata["columns"])
    proba = model.predict_proba(df_features)
    risk_score = float(proba[0][1])  # calibrated breach probability

    # Calculate ensemble dispersion (uncertainty) across trees
    if hasattr(base_model, "estimators_") and len(base_model.estimators_) > 0:
        tree_probs = np.array([
            tree.predict_proba(df_features.values)[0, 1]
            for tree in base_model.estimators_
        ])
        uncertainty = float(np.std(tree_probs))
        ci_lower = float(np.percentile(tree_probs, 5))
        ci_upper = float(np.percentile(tree_probs, 95))
        confidence_interval = [round(max(0.0, ci_lower), 4), round(min(1.0, ci_upper), 4)]
        # True epistemic model certainty: high when trees agree (low dispersion)
        model_confidence = float(max(0.1, min(1.0, 1.0 - (uncertainty * 2.0))))
    else:
        uncertainty = 0.05
        confidence_interval = [round(max(0.0, risk_score - 0.1), 4), round(min(1.0, risk_score + 0.1), 4)]
        model_confidence = 0.90

    # Determine risk label based on calibrated probability vs threshold
    if risk_score >= decision_threshold:
        risk_label = "High"
    elif risk_score >= 0.4:
        risk_label = "Medium"
    else:
        risk_label = "Low"

    # Team assignment based on priority
    priority_raw = ticket_data.get("priority", "Medium")
    if priority_raw == "Critical":
        assigned_team = "L3 Engineering"
    elif priority_raw == "High":
        assigned_team = "L2 Support"
    else:
        assigned_team = "L1 Support"

    prediction_id = str(uuid.uuid4())[:12]

    return {
        "prediction_id": prediction_id,
        "ticket_id": ticket_data.get("ticket_id", "unknown"),
        "risk_score": round(risk_score, 4),
        "breach_probability": round(risk_score, 4),
        "risk_label": risk_label,
        "decision_threshold": decision_threshold,
        "model_version": metadata.get("model_version", "unknown"),
        "feature_snapshot": feature_dict,
        "assigned_team": assigned_team,
        "prediction_uncertainty": round(uncertainty, 4),
        "confidence_interval": confidence_interval,
        "data_quality_score": data_quality_score,
        "data_quality_warnings": warnings,
        "confidence": round(model_confidence, 4),
        "warnings": warnings,
        "predicted_at": datetime.now(timezone.utc).isoformat(),
    }
