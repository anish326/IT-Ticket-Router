"""
Explainability Module
=====================
Integrates SHAP with the trained RandomForest model to generate
per-prediction explanations.

Provides:
- Top contributing features (positive and negative)
- Human-readable labels and impact descriptions
- Transparent rule-based recommendations
- Clear distinction between model-derived and policy-based factors

SHAP identifies influential model features; it does not establish
that those factors caused the breach.
"""

import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

from api.sla_predictor import get_base_model, get_metadata, get_feature_columns

# Cached SHAP explainer
_explainer = None


# Human-readable feature labels and detail templates
FEATURE_INFO = {
    "priority_encoded": {
        "label": "Ticket priority level",
        "detail_high": "High/Critical priority increases urgency",
        "detail_low": "Low/Medium priority reduces urgency",
    },
    "issue_type_encoded": {
        "label": "Issue type classification",
        "detail_high": "Incident/Bug type correlates with longer resolution",
        "detail_low": "Task/Service Request type tends to resolve faster",
    },
    "created_hours": {
        "label": "Ticket age",
        "detail_high": "Ticket has been open for {value:.1f} hours",
        "detail_low": "Ticket is relatively new ({value:.1f} hours old)",
    },
    "sla_deadline_hours": {
        "label": "SLA deadline window",
        "detail_high": "Longer SLA window ({value:.1f} hours)",
        "detail_low": "Tight SLA deadline ({value:.1f} hours)",
    },
    "time_remaining_hours": {
        "label": "Time remaining until SLA",
        "detail_high": "Sufficient time remaining ({value:.1f} hours)",
        "detail_low": "Only {value:.1f} hours remaining until SLA deadline",
    },
    "response_delay_hours": {
        "label": "Response delay",
        "detail_high": "No agent response for {value:.1f} hours",
        "detail_low": "Quick response time ({value:.1f} hours)",
    },
    "team_backlog_count": {
        "label": "Team backlog size",
        "detail_high": "Assigned team has {value:.0f} open tickets",
        "detail_low": "Team backlog is manageable ({value:.0f} tickets)",
    },
    "similar_ticket_resolution_hours": {
        "label": "Historical resolution time",
        "detail_high": "Similar tickets took ~{value:.1f} hours to resolve",
        "detail_low": "Similar tickets resolved quickly (~{value:.1f} hours)",
    },
    "description_length": {
        "label": "Description detail level",
        "detail_high": "Detailed description ({value:.0f} chars) may indicate complexity",
        "detail_low": "Brief description ({value:.0f} chars)",
    },
}


def _get_explainer():
    """Get or create the cached SHAP TreeExplainer."""
    global _explainer
    if _explainer is not None:
        return _explainer

    if not SHAP_AVAILABLE:
        return None

    try:
        base_model = get_base_model()
        _explainer = shap.TreeExplainer(base_model)
        return _explainer
    except Exception as e:
        print(f"Warning: Could not create SHAP explainer: {e}")
        return None


def reset_explainer():
    """Reset the cached explainer (after model reload)."""
    global _explainer
    _explainer = None


def explain_prediction(feature_array, feature_dict, risk_score, ticket_data=None, top_n=3):
    """
    Generate a SHAP-based explanation for a single prediction.

    Args:
        feature_array: numpy array of shape (1, n_features) in column order
        feature_dict: dict mapping feature names to values
        risk_score: the predicted breach probability
        ticket_data: optional original ticket data for context
        top_n: number of top contributing features to return

    Returns:
        dict with top_factors, recommendation, explanation_type
    """
    feature_columns = get_feature_columns()
    explanation = {
        "top_factors": [],
        "recommendation": "",
        "explanation_type": "unavailable",
        "risk_score": risk_score,
    }

    # Try SHAP explanation
    explainer = _get_explainer()
    if explainer is not None:
        try:
            shap_values = explainer.shap_values(feature_array)

            # Handle multi-class output: use class-1 (breach) SHAP values
            if isinstance(shap_values, list):
                # For binary classification, shap_values is [class_0, class_1]
                sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
            elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
                # Shape (n_samples, n_features, n_classes) -> sample 0, class 1 (breach)
                sv = shap_values[0, :, 1] if shap_values.shape[2] > 1 else shap_values[0, :, 0]
            elif hasattr(shap_values, "ndim") and shap_values.ndim == 2:
                sv = shap_values[0]
            else:
                sv = np.array(shap_values).flatten()

            # Build factor list sorted by absolute impact
            factors = []
            for i, col in enumerate(feature_columns):
                impact = float(sv[i])
                value = feature_dict.get(col, 0)
                info = FEATURE_INFO.get(col, {})

                direction = "increases_risk" if impact > 0 else "decreases_risk"

                # Generate human-readable detail
                if impact > 0:
                    detail_template = info.get("detail_high", f"{col} = {{value}}")
                else:
                    detail_template = info.get("detail_low", f"{col} = {{value}}")

                try:
                    detail = detail_template.format(value=value)
                except (KeyError, ValueError):
                    detail = f"{info.get('label', col)}: {value}"

                factors.append(
                    {
                        "feature": col,
                        "label": info.get("label", col),
                        "impact": round(abs(impact), 4),
                        "direction": direction,
                        "detail": detail,
                        "value": value,
                        "source": "model",
                    }
                )

            # Sort by absolute impact, take top N
            factors.sort(key=lambda x: x["impact"], reverse=True)
            explanation["top_factors"] = factors[:top_n]
            explanation["explanation_type"] = "shap"

        except Exception as e:
            print(f"Warning: SHAP explanation failed: {e}")
            explanation["explanation_type"] = "fallback"
            explanation["top_factors"] = _fallback_explanation(
                feature_dict, feature_columns, top_n
            )
    else:
        explanation["explanation_type"] = "fallback"
        explanation["top_factors"] = _fallback_explanation(
            feature_dict, feature_columns, top_n
        )

    # Generate policy-based recommendation
    explanation["recommendation"] = _generate_recommendation(
        risk_score, feature_dict, explanation["top_factors"], ticket_data
    )

    # Add policy-based factors (distinguished from model-derived)
    policy_factors = _get_policy_factors(risk_score, feature_dict, ticket_data)
    if policy_factors:
        explanation["policy_factors"] = policy_factors

    # Human-readable summary of primary factors
    explanation["summary"] = ", ".join(f["label"] for f in explanation["top_factors"] if "label" in f)

    return explanation


def _fallback_explanation(feature_dict, feature_columns, top_n):
    """
    Generate a simple feature-importance-based explanation when SHAP is unavailable.
    Uses the model's built-in feature_importances_.
    """
    try:
        base_model = get_base_model()
        importances = base_model.feature_importances_
    except Exception:
        return []

    factors = []
    for i, col in enumerate(feature_columns):
        value = feature_dict.get(col, 0)
        info = FEATURE_INFO.get(col, {})
        importance = float(importances[i])

        factors.append(
            {
                "feature": col,
                "label": info.get("label", col),
                "impact": round(importance, 4),
                "direction": "contributes",
                "detail": f"{info.get('label', col)}: {value}",
                "value": value,
                "source": "feature_importance",
            }
        )

    factors.sort(key=lambda x: x["impact"], reverse=True)
    return factors[:top_n]


def _generate_recommendation(risk_score, feature_dict, top_factors, ticket_data=None):
    """
    Generate a transparent, rule-based recommendation based on risk and factors.
    These are suggestions, not causal claims.
    """
    recommendations = []

    # Risk-level-based suggestions
    if risk_score >= 0.80:
        recommendations.append(
            "Escalate for urgent review and consider reassignment to an available, qualified agent."
        )
    elif risk_score >= 0.60:
        recommendations.append(
            "Flag for team lead review. Monitor closely as SLA deadline approaches."
        )
    elif risk_score >= 0.40:
        recommendations.append(
            "Continue monitoring. Consider prioritizing if workload allows."
        )

    # Factor-specific suggestions
    for factor in top_factors:
        if factor["direction"] == "increases_risk":
            feature = factor["feature"]
            if feature == "response_delay_hours" and factor.get("value", 0) > 3:
                recommendations.append(
                    "Response delay is significant. Check if the assigned agent is available."
                )
            elif feature == "team_backlog_count" and factor.get("value", 0) > 12:
                recommendations.append(
                    "Team backlog is high. Consider redistributing workload or reassigning."
                )
            elif feature == "time_remaining_hours" and factor.get("value", 0) < 1:
                recommendations.append(
                    "Very little time remaining. Immediate attention required."
                )
            elif feature == "similar_ticket_resolution_hours" and factor.get("value", 0) > 8:
                recommendations.append(
                    "Similar tickets have taken long to resolve. Allocate additional resources if possible."
                )

    if not recommendations:
        recommendations.append("No immediate action required. Continue standard workflow.")

    return " ".join(recommendations[:3])  # Limit to 3 recommendations


def _get_policy_factors(risk_score, feature_dict, ticket_data=None):
    """
    Identify factors based on operational rules, not model output.
    These are clearly marked as rule-based, not model-derived.
    """
    policy_factors = []

    time_remaining = feature_dict.get("time_remaining_hours", float("inf"))
    response_delay = feature_dict.get("response_delay_hours", 0)

    if time_remaining < 1.0:
        policy_factors.append(
            {
                "label": "SLA deadline imminent",
                "detail": f"Less than {time_remaining:.1f} hours remaining",
                "source": "operational_rule",
                "severity": "critical",
            }
        )

    if response_delay > 4.0:
        policy_factors.append(
            {
                "label": "Extended response gap",
                "detail": f"No response for {response_delay:.1f} hours",
                "source": "operational_rule",
                "severity": "high",
            }
        )

    priority = ticket_data.get("priority", "") if ticket_data else ""
    if priority == "Critical":
        policy_factors.append(
            {
                "label": "Critical severity ticket",
                "detail": "Critical tickets require expedited handling per policy",
                "source": "operational_rule",
                "severity": "critical",
            }
        )

    return policy_factors
