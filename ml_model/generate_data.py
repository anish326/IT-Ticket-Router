"""
Synthetic Ticket Data Generator
================================
Generates approximately 1,000 realistic support tickets with varied
priorities, issue types, response patterns, and SLA outcomes.

Uses a fixed random seed (42) for reproducibility.

Outcome generation logic:
- Breach probability is computed from a weighted combination of:
  - Priority urgency (25%): Critical > High > Medium > Low
  - Time pressure (30%): ratio of elapsed time to SLA deadline
  - Response delay (20%): longer delays increase risk
  - Team backlog (15%): higher backlog increases risk
  - Resolution complexity (10%): similar ticket resolution time vs deadline
- Gaussian noise (sigma=0.1) is added to prevent deterministic outcomes.
- The actual breach label is sampled from Bernoulli(breach_probability).

This is synthetic data for demonstration and testing purposes.
Evaluation results from this data should not be presented as
real-world model performance.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

SEED = 42
NUM_TICKETS = 1000


def generate_tickets(n=NUM_TICKETS, seed=SEED):
    """Generate n synthetic support tickets with realistic feature distributions."""
    np.random.seed(seed)

    priorities = ["Low", "Medium", "High", "Critical"]
    priority_weights = [0.20, 0.35, 0.30, 0.15]

    issue_types = ["Bug", "Task", "Incident", "Service Request", "Change Request"]
    issue_type_weights = [0.25, 0.20, 0.20, 0.20, 0.15]

    # SLA deadline ranges (hours) by priority
    sla_deadline_ranges = {
        "Low": (24.0, 72.0),
        "Medium": (8.0, 24.0),
        "High": (4.0, 12.0),
        "Critical": (1.0, 4.0),
    }

    teams = ["L1 Support", "L2 Support", "L3 Engineering", "DevOps", "Security"]
    statuses = ["Open", "In Progress", "Waiting", "Resolved", "Closed"]

    records = []
    base_time = datetime(2026, 1, 1, 8, 0, 0)

    for i in range(n):
        ticket_id = f"TKT-{1000 + i}"
        priority = np.random.choice(priorities, p=priority_weights)
        issue_type = np.random.choice(issue_types, p=issue_type_weights)

        # Created time spread over approximately six months
        created_at = base_time + timedelta(hours=float(np.random.uniform(0, 4320)))

        # SLA deadline based on priority
        sla_min, sla_max = sla_deadline_ranges[priority]
        sla_deadline_hours = float(np.random.uniform(sla_min, sla_max))

        # Ticket age at prediction time (can exceed deadline for resolved tickets)
        max_age = sla_deadline_hours * 1.5
        created_hours = float(np.random.uniform(0.5, max(1.0, max_age)))

        # Time remaining until SLA deadline
        time_remaining_hours = max(0.0, sla_deadline_hours - created_hours)

        # Response delay: exponential distribution, capped at ticket age
        response_delay_hours = float(np.random.exponential(scale=2.5))
        response_delay_hours = min(response_delay_hours, created_hours)

        # Team backlog: Poisson distributed
        team_backlog_count = int(np.random.poisson(lam=8)) + 1

        # Similar ticket resolution time: log-normal
        similar_ticket_resolution_hours = float(
            np.random.lognormal(mean=2.0, sigma=0.8)
        )

        # Description length: log-normal
        description_length = int(np.random.lognormal(mean=5.0, sigma=0.8))
        description_length = max(10, min(description_length, 3000))

        # Assigned team
        assigned_team = np.random.choice(teams)

        # Title generation
        title_templates = [
            f"{issue_type}: System issue #{i}",
            f"[{priority}] {issue_type} report",
            f"{assigned_team} - {issue_type} #{ticket_id}",
            f"Automated {issue_type.lower()} ticket",
            f"User-reported {issue_type.lower()}",
        ]
        title = np.random.choice(title_templates)

        # ---------------------------------------------------------------
        # Breach probability (used ONLY for label generation, not a feature)
        # ---------------------------------------------------------------
        priority_risk = {"Low": 0.10, "Medium": 0.25, "High": 0.50, "Critical": 0.70}[
            priority
        ]

        # Sigmoid-shaped time pressure
        if sla_deadline_hours > 0:
            ratio = created_hours / sla_deadline_hours
        else:
            ratio = 1.0
        time_pressure = 1.0 / (1.0 + np.exp(-(ratio - 0.6) * 5))

        delay_risk = min(1.0, response_delay_hours / 6.0)
        backlog_risk = min(1.0, team_backlog_count / 20.0)
        resolution_risk = min(
            1.0, similar_ticket_resolution_hours / (sla_deadline_hours + 1.0)
        )

        breach_prob = (
            0.25 * priority_risk
            + 0.30 * time_pressure
            + 0.20 * delay_risk
            + 0.15 * backlog_risk
            + 0.10 * resolution_risk
        )

        # Add noise to avoid deterministic label assignment
        breach_prob = float(np.clip(breach_prob + np.random.normal(0, 0.10), 0.0, 1.0))

        # Sample actual breach outcome
        sla_breach = 1 if np.random.random() < breach_prob else 0

        # Determine ticket status based on breach and age
        if created_hours > sla_deadline_hours:
            status = np.random.choice(["Resolved", "Closed"], p=[0.4, 0.6])
        elif created_hours > sla_deadline_hours * 0.7:
            status = np.random.choice(statuses, p=[0.15, 0.30, 0.25, 0.20, 0.10])
        else:
            status = np.random.choice(statuses, p=[0.35, 0.30, 0.20, 0.10, 0.05])

        records.append(
            {
                "ticket_id": ticket_id,
                "title": title,
                "priority": priority,
                "issue_type": issue_type,
                "created_at": created_at.isoformat(),
                "created_hours": round(created_hours, 2),
                "sla_deadline_hours": round(sla_deadline_hours, 2),
                "time_remaining_hours": round(time_remaining_hours, 2),
                "response_delay_hours": round(response_delay_hours, 2),
                "team_backlog_count": team_backlog_count,
                "similar_ticket_resolution_hours": round(
                    similar_ticket_resolution_hours, 2
                ),
                "description_length": description_length,
                "assigned_team": assigned_team,
                "status": status,
                "sla_breach": sla_breach,
                "synthetic": 1,
            }
        )

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    df = generate_tickets()

    output_path = os.path.join(os.path.dirname(__file__), "..", "data", "tickets.csv")
    output_path = os.path.normpath(output_path)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} synthetic tickets -> {output_path}")
    print(f"Breach rate: {df['sla_breach'].mean():.2%}")
    print(f"\nPriority distribution:\n{df['priority'].value_counts().to_string()}")
    print(f"\nIssue type distribution:\n{df['issue_type'].value_counts().to_string()}")
    print(f"\nStatus distribution:\n{df['status'].value_counts().to_string()}")
    print(f"\nFeature summary:")
    print(
        df[
            [
                "created_hours",
                "sla_deadline_hours",
                "response_delay_hours",
                "team_backlog_count",
                "similar_ticket_resolution_hours",
                "description_length",
            ]
        ]
        .describe()
        .round(2)
        .to_string()
    )
