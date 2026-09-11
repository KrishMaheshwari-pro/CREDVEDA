"""
Fairness audit: measures model outcomes across geography, gender and
business type, and reports where the model is systematically harsher --
exactly what PS #4 asks for. Uses the standard "four-fifths rule" adverse
impact test (a group's approval rate should be at least 80% of the
highest-approval-rate group's) alongside plain approval-rate and
average-score comparisons.

Protected attributes are never fed to the model (see config.FEATURE_COLS);
this module only ever reads them to slice already-computed scores, so a
disparity found here reflects either genuine, legitimate signal
differences (e.g. thinner files in a group) or a proxy effect -- not the
model being told the group directly.

# Updated on 2026-02-18
"""
import json
import sqlite3

import pandas as pd

import config


def load_scored_population() -> pd.DataFrame:
    conn = sqlite3.connect(config.DB_NAME)
    try:
        applicants = pd.read_sql_query("SELECT * FROM applicants", conn)
        scores = pd.read_sql_query("SELECT * FROM credit_scores", conn)
    finally:
        conn.close()
    return applicants.merge(scores, on="applicant_id", how="inner")


def audit_group(df: pd.DataFrame, group_col: str) -> list:
    rows = []
    grouped = df.groupby(group_col)
    for name, g in grouped:
        approval_rate = float((g["credit_score"] >= config.APPROVAL_SCORE_THRESHOLD).mean())
        rows.append({
            "group": str(name),
            "n": int(len(g)),
            "avg_score": round(float(g["credit_score"].mean()), 1),
            "approval_rate": round(approval_rate, 4),
            "avg_completeness": round(float(g["data_completeness"].mean()), 3),
            "thin_file_rate": round(float((g["data_completeness"] < 0.5).mean()), 4),
        })
    if not rows:
        return rows
    max_rate = max(r["approval_rate"] for r in rows) or 1e-9
    for r in rows:
        ratio = r["approval_rate"] / max_rate if max_rate > 0 else 1.0
        r["adverse_impact_ratio"] = round(ratio, 3)
        r["flagged"] = ratio < config.FAIRNESS_ADVERSE_IMPACT_THRESHOLD
    rows.sort(key=lambda r: r["approval_rate"], reverse=True)
    return rows


def run_fairness_audit() -> dict:
    df = load_scored_population()
    report = {"n_applicants": int(len(df)), "groups": {}}
    for group_col in config.FAIRNESS_GROUPS:
        report["groups"][group_col] = audit_group(df, group_col)

    flagged = []
    for group_col, rows in report["groups"].items():
        pretty_col = group_col.replace("_", " ").title()
        for r in rows:
            if r["flagged"]:
                flagged.append(
                    f"{pretty_col} · {r['group']} — approved {r['approval_rate']:.1%} of the time, "
                    f"only {r['adverse_impact_ratio']:.0%} of the best-performing group's rate "
                    f"(below the {config.FAIRNESS_ADVERSE_IMPACT_THRESHOLD:.0%} four-fifths threshold)."
                )
    report["flagged_disparities"] = flagged
    return report


if __name__ == "__main__":
    report = run_fairness_audit()
    print(json.dumps(report, indent=2))
    if report["flagged_disparities"]:
        print("\n--- Flagged disparities ---")
        for line in report["flagged_disparities"]:
            print(f" - {line}")
    else:
        print("\nNo group failed the four-fifths adverse-impact rule.")
# Updated on 2026-02-18
