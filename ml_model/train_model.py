"""
Enhanced Model Training
========================
Trains a RandomForestClassifier on synthetic ticket data with seven features.

Uses chronological train/test split (first 80% for training, last 20% for testing).
Compares against a majority-class baseline.
Exports the model, feature metadata, and model version info.

Prediction target: eventual SLA breach (binary: 0 = met, 1 = breached).
All features are available at prediction time (no future leakage).

Evaluation results are from synthetic data and should not be presented
as real-world model performance.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

MODEL_VERSION = "1.0.0"

FEATURE_COLUMNS = [
    "priority_encoded",
    "issue_type_encoded",
    "created_hours",
    "sla_deadline_hours",
    "time_remaining_hours",
    "response_delay_hours",
    "team_backlog_count",
    "similar_ticket_resolution_hours",
    "description_length",
]

PRIORITY_MAP = {"Low": 0, "Medium": 1, "High": 2, "Critical": 3}
ISSUE_TYPE_MAP = {
    "Task": 0,
    "Bug": 1,
    "Service Request": 2,
    "Incident": 3,
    "Change Request": 4,
}

FEATURE_METADATA = {
    "columns": FEATURE_COLUMNS,
    "priority_map": PRIORITY_MAP,
    "issue_type_map": ISSUE_TYPE_MAP,
    "target": "sla_breach",
    "model_version": MODEL_VERSION,
    "feature_labels": {
        "priority_encoded": "Ticket priority",
        "issue_type_encoded": "Issue type",
        "created_hours": "Ticket age (hours)",
        "sla_deadline_hours": "SLA deadline (hours)",
        "time_remaining_hours": "Time remaining until SLA",
        "response_delay_hours": "Response delay (hours)",
        "team_backlog_count": "Team backlog size",
        "similar_ticket_resolution_hours": "Similar ticket resolution time",
        "description_length": "Description length",
    },
}


def preprocess(df):
    """Apply training-time preprocessing. Returns feature DataFrame."""
    df = df.copy()
    df["priority_encoded"] = df["priority"].map(PRIORITY_MAP).fillna(1).astype(int)
    df["issue_type_encoded"] = (
        df["issue_type"].map(ISSUE_TYPE_MAP).fillna(0).astype(int)
    )

    # Ensure time_remaining_hours exists
    if "time_remaining_hours" not in df.columns:
        df["time_remaining_hours"] = (
            df["sla_deadline_hours"] - df["created_hours"]
        ).clip(lower=0)

    return df[FEATURE_COLUMNS]


def train():
    """Train the model, evaluate, and export artifacts."""
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.csv")
    data_path = os.path.normpath(data_path)

    if not os.path.exists(data_path):
        print(f"Dataset not found at {data_path}")
        print("Run generate_data.py first: python ml_model/generate_data.py")
        sys.exit(1)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} tickets from {data_path}")
    print(f"Breach rate: {df['sla_breach'].mean():.2%}")

    # Chronological split: sort by created_at, first 80% train, last 20% test
    df = df.sort_values("created_at").reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    X_train = preprocess(train_df)
    y_train = train_df["sla_breach"].astype(int)
    X_test = preprocess(test_df)
    y_test = test_df["sla_breach"].astype(int)

    print(f"\nTraining set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Train breach rate: {y_train.mean():.2%}")
    print(f"Test breach rate: {y_test.mean():.2%}")

    # Train RandomForestClassifier
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # Calibrate probabilities using cross-validation on training data
    calibrated_clf = CalibratedClassifierCV(clf, cv=5, method="isotonic")
    calibrated_clf.fit(X_train, y_train)

    # Predictions
    y_pred = calibrated_clf.predict(X_test)
    y_prob = calibrated_clf.predict_proba(X_test)[:, 1]

    # Metrics
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    brier = brier_score_loss(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Model Evaluation (synthetic data)")
    print(f"{'='*50}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"Brier:     {brier:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['SLA Met', 'SLA Breached'])}")

    # Baseline: always predict majority class
    majority_class = int(y_train.mode()[0])
    baseline_accuracy = (y_test == majority_class).mean()
    model_accuracy = (y_pred == y_test).mean()
    print(f"Baseline accuracy (majority class): {baseline_accuracy:.4f}")
    print(f"Model accuracy: {model_accuracy:.4f}")
    print(f"Improvement over baseline: {model_accuracy - baseline_accuracy:+.4f}")

    # Feature importances (from the underlying RandomForest)
    print(f"\nFeature Importances:")
    importances = clf.feature_importances_
    for feat, imp in sorted(
        zip(FEATURE_COLUMNS, importances), key=lambda x: -x[1]
    ):
        print(f"  {feat}: {imp:.4f}")

    # Export model
    model_dir = os.path.dirname(__file__)
    model_path = os.path.join(model_dir, "sla_model.joblib")
    dump(calibrated_clf, model_path)
    print(f"\nModel saved to: {model_path}")

    # Also save the uncalibrated model for SHAP (TreeExplainer needs base estimator)
    base_model_path = os.path.join(model_dir, "sla_model_base.joblib")
    dump(clf, base_model_path)
    print(f"Base model saved to: {base_model_path}")

    # Export feature metadata
    metadata_path = os.path.join(model_dir, "feature_columns.json")
    metadata = FEATURE_METADATA.copy()
    metadata["training_samples"] = len(X_train)
    metadata["test_samples"] = len(X_test)
    metadata["metrics"] = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "brier_score": round(brier, 4),
        "baseline_accuracy": round(baseline_accuracy, 4),
        "model_accuracy": round(model_accuracy, 4),
    }
    metadata["confusion_matrix"] = {
        "tn": int(cm[0][0]),
        "fp": int(cm[0][1]),
        "fn": int(cm[1][0]),
        "tp": int(cm[1][1]),
    }

    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Feature metadata saved to: {metadata_path}")

    # Register model version in database if available
    try:
        from api.database import init_db, insert_model_version

        init_db()
        insert_model_version(
            {
                "version": MODEL_VERSION,
                "model_path": model_path,
                "feature_columns": FEATURE_COLUMNS,
                "training_samples": len(X_train),
                "test_samples": len(X_test),
                "precision_score": round(precision, 4),
                "recall_score": round(recall, 4),
                "f1_score": round(f1, 4),
                "brier_score": round(brier, 4),
                "baseline_accuracy": round(baseline_accuracy, 4),
                "is_active": 1,
                "notes": "Initial model trained on synthetic data",
            }
        )
        print("Model version registered in database")
    except Exception as e:
        print(f"Note: Could not register in database (non-critical): {e}")

    print(f"\nTraining complete. Model version: {MODEL_VERSION}")
    return calibrated_clf, clf


if __name__ == "__main__":
    train()
