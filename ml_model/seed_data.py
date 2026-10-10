"""
Database Seed Script
====================
Populates the SQLite database from data/tickets.csv and generates
realistic initial predictions, SHAP explanations, confirmed outcomes,
agent feedback, escalation records, and initial model evaluation metrics.

Usage:
    python ml_model/seed_data.py
"""

import os
import sys
import pandas as pd
from datetime import datetime

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from api.database import (
    init_db,
    get_db,
    insert_ticket,
    get_tickets,
)
from api.sla_predictor import predict as sla_predict, preprocess_input
from api.explainability import explain_prediction
from api.feedback import log_prediction, record_outcome, record_feedback
from api.escalation_engine import evaluate_escalation, decide_escalation
from api.drift_monitor import evaluate_model


def seed(data_path=None, limit_simulate=150):
    """Seed the database with tickets, predictions, outcomes, feedback, and escalations."""
    init_db()

    if data_path is None:
        data_path = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.csv")
    data_path = os.path.normpath(data_path)

    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        print("Please run: python ml_model/generate_data.py first.")
        return False

    print(f"Reading tickets from {data_path}...")
    df = pd.read_csv(data_path)

    with get_db() as conn:
        existing_count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]

    if existing_count == 0:
        print(f"Importing {len(df)} tickets into database...")
        for _, row in df.iterrows():
            ticket_data = {
                "ticket_id": str(row["ticket_id"]),
                "title": str(row["title"]),
                "description": f"Ticket {row['ticket_id']} - {row['issue_type']} issue affecting {row['assigned_team']}.",
                "priority": str(row["priority"]),
                "issue_type": str(row["issue_type"]),
                "status": str(row["status"]),
                "assigned_team": str(row["assigned_team"]),
                "assignee": f"agent_{hash(str(row['ticket_id'])) % 5 + 1}",
                "created_at": str(row["created_at"]),
                "sla_deadline_hours": float(row["sla_deadline_hours"]),
                "created_hours": float(row["created_hours"]),
                "time_remaining_hours": float(row["time_remaining_hours"]),
                "response_delay_hours": float(row["response_delay_hours"]),
                "team_backlog_count": int(row["team_backlog_count"]),
                "similar_ticket_resolution_hours": float(row["similar_ticket_resolution_hours"]),
                "description_length": int(row["description_length"]),
                "sla_breach": int(row["sla_breach"]) if pd.notnull(row["sla_breach"]) else None,
                "synthetic": 1,
            }
            insert_ticket(ticket_data)
        print(f"Successfully inserted {len(df)} tickets.")
    else:
        print(f"Database already contains {existing_count} tickets.")

    # Check if we already have prediction logs
    with get_db() as conn:
        pred_count = conn.execute("SELECT COUNT(*) FROM prediction_logs").fetchone()[0]

    if pred_count == 0:
        print(f"Simulating initial AI predictions and workflows for first {limit_simulate} tickets...")
        tickets_to_simulate = df.head(limit_simulate)

        escalation_ids = []
        simulated_count = 0

        for _, row in tickets_to_simulate.iterrows():
            t_dict = row.to_dict()
            try:
                # 1. Prediction
                pred = sla_predict(t_dict)

                # 2. SHAP Explanation
                feat_arr, feat_dict, _ = preprocess_input(t_dict)
                exp = explain_prediction(feat_arr, feat_dict, pred["risk_score"], ticket_data=t_dict)

                # 3. Log Prediction
                pred_id = log_prediction(pred, exp)

                # 4. Record Confirmed Outcome for resolved/closed tickets
                if t_dict.get("status") in ("Resolved", "Closed"):
                    actual_breach = int(t_dict["sla_breach"])
                    record_outcome(
                        ticket_id=t_dict["ticket_id"],
                        actual_breach=actual_breach,
                        prediction_id=pred_id,
                        confirmed_by="Automated SLA Tracker",
                        notes=f"Ticket lifecycle ended. Breach status confirmed ({actual_breach})."
                    )

                # 5. Agent Feedback on recommendation (sample)
                if simulated_count % 3 == 0:
                    feedback_choice = "helpful" if pred["risk_score"] > 0.4 else "not_helpful"
                    if simulated_count % 12 == 0:
                        feedback_choice = "incorrect"
                    record_feedback(
                        ticket_id=t_dict["ticket_id"],
                        feedback_label=feedback_choice,
                        prediction_id=pred_id,
                        feedback_reason=f"Actionable insight provided by {exp.get('recommendation', '')[:40]}...",
                        reviewer_name="Sarah Chen",
                    )

                # 6. Evaluate Escalation
                esc_result = evaluate_escalation(pred, ticket_data=t_dict, explanation=exp)
                if esc_result.get("triggered") and esc_result.get("escalation_id"):
                    escalation_ids.append(esc_result["escalation_id"])

                simulated_count += 1
            except Exception as e:
                print(f"Warning during simulation for {t_dict.get('ticket_id')}: {e}")

        print(f"Generated {simulated_count} predictions with explanations and outcomes.")

        # Simulate reviewer decisions on a few historical escalations
        if len(escalation_ids) >= 4:
            try:
                decide_escalation(escalation_ids[0], "approve", "Approved: Reassigned to Tier 2 specialist.", "Sarah Chen")
                decide_escalation(escalation_ids[1], "reject", "Rejected: False alarm, customer issue resolved.", "Alex Johnson")
                decide_escalation(escalation_ids[2], "reassign", "Reassigned to DevOps queue for immediate hotfix.", "Sarah Chen")
                decide_escalation(escalation_ids[3], "investigate", "Investigating root cause before committing resources.", "System Admin")
                print("Seeded escalation reviewer decisions.")
            except Exception as e:
                print(f"Note on escalation decision seeding: {e}")

        # Compute initial model evaluation
        try:
            eval_result = evaluate_model()
            print("Initial model evaluation computed:")
            print(f"  Precision: {eval_result.get('precision_score')}")
            print(f"  Recall:    {eval_result.get('recall_score')}")
            print(f"  F1:        {eval_result.get('f1_score')}")
            print(f"  Brier:     {eval_result.get('brier_score')}")
            print(f"  Drift:     {eval_result.get('drift_status')}")
        except Exception as e:
            print(f"Note on initial model evaluation: {e}")

    print("\nDatabase seeding complete!")
    return True


if __name__ == "__main__":
    seed()
