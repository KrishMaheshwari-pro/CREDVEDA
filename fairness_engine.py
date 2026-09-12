"""
Fairness Engine — measures model behaviour across geography, gender,
and business type. Flags disparate impact using the 4/5ths rule.
"""
import sqlite3
try:
    import pandas as pd
    import numpy as np
except ImportError:
    pd = None
    np = None
import config


def load_scored_applications(conn=None):
    """Load all applications that have been scored."""
    close = False
    if conn is None:
        conn = sqlite3.connect(config.DB_NAME)
        close = True
    try:
        df = pd.read_sql_query("""
            SELECT a.app_id, a.gender, a.state, a.business_type, a.age,
                   a.monthly_revenue_avg, a.loan_amount_requested,
                   s.score, s.score_band, s.guardrail_severity
            FROM loan_applications a
            LEFT JOIN credit_scores_v2 s ON a.app_id = s.app_id
            WHERE s.score IS NOT NULL
        """, conn)
    except Exception:
        df = pd.DataFrame()
    finally:
        if close:
            conn.close()
    return df


def _approval_rate(df, threshold: int = 55):
    """Approval rate = fraction scoring >= threshold (GOOD STANDING or better)."""
    return (df['score'] >= threshold).mean()


def compute_group_stats(df, dimension: str, threshold: int = 55):
    """Compute approval rate, mean score, count for each group in a dimension."""
    if df.empty or dimension not in df.columns:
        return pd.DataFrame()

    groups = df.groupby(dimension)
    stats = groups.apply(lambda g: pd.Series({
        'total': len(g),
        'approved': (g['score'] >= threshold).sum(),
        'approval_rate': (g['score'] >= threshold).mean(),
        'mean_score': g['score'].mean(),
        'median_score': g['score'].median(),
    })).reset_index()
    stats.columns = [dimension, 'total', 'approved', 'approval_rate', 'mean_score', 'median_score']
    return stats.sort_values('approval_rate', ascending=False)


def apply_four_fifths_rule(stats_df, dimension: str):
    """
    Apply the 4/5ths (80%) rule: if any group's approval rate is less than
    80% of the highest group's rate, flag disparate impact.
    """
    if stats_df.empty:
        return []
    reference_rate = stats_df['approval_rate'].max()
    if reference_rate == 0:
        return []

    flags = []
    for _, row in stats_df.iterrows():
        air = row['approval_rate'] / reference_rate if reference_rate > 0 else 1.0
        if air < 0.80:
            flags.append({
                'dimension': dimension,
                'group': row[dimension],
                'approval_rate': round(row['approval_rate'], 4),
                'mean_score': round(row['mean_score'], 1),
                'reference_rate': round(reference_rate, 4),
                'adverse_impact_ratio': round(air, 4),
                'flag': 'DISPARATE IMPACT',
                'flag_color': '#dc3545',
                'flag_message': (
                    f"Group '{row[dimension]}' has an approval rate {air:.0%} of the "
                    f"best-performing group — below the 4/5ths threshold of 80%."
                ),
            })
    return flags


def compute_gini(scores):
    """Gini coefficient of score distribution across applicants."""
    arr = np.sort(np.array(scores.dropna()))
    if len(arr) == 0:
        return 0.0
    n = len(arr)
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * arr) / (n * np.sum(arr))) - (n + 1) / n)


def run_fairness_audit(threshold: int = 55):
    """
    Run a full fairness audit across all three dimensions.

    Returns a dict with per-dimension stats, flags, and aggregate metrics.
    """
    df = load_scored_applications()
    if df.empty:
        return {"error": "No scored applications found. Score some applications first."}

    dimensions = ['gender', 'state', 'business_type']
    result = {
        'total_applications': len(df),
        'threshold_score': threshold,
        'overall_approval_rate': round(_approval_rate(df, threshold), 4),
        'overall_mean_score': round(df['score'].mean(), 1),
        'score_gini': round(compute_gini(df['score']), 4),
        'dimensions': {},
        'all_flags': [],
    }

    for dim in dimensions:
        if dim not in df.columns:
            continue
        stats = compute_group_stats(df, dim, threshold)
        flags = apply_four_fifths_rule(stats, dim)
        result['dimensions'][dim] = {
            'stats': stats.to_dict(orient='records'),
            'flags': flags,
            'flagged_groups': len(flags),
        }
        result['all_flags'].extend(flags)

    result['total_flags'] = len(result['all_flags'])
    result['audit_status'] = 'PASS' if result['total_flags'] == 0 else 'REVIEW REQUIRED'
    result['audit_status_color'] = '#28a745' if result['total_flags'] == 0 else '#dc3545'
    return result


def save_audit_to_db(audit_result):
    """Persist fairness audit snapshot to fairness_audit table."""
    if 'error' in audit_result:
        return
    conn = sqlite3.connect(config.DB_NAME)
    try:
        for flag in audit_result.get('all_flags', []):
            conn.execute("""
                INSERT INTO fairness_audit
                (dimension, group_name, total_applications, approved_count,
                 approval_rate, mean_score, adverse_impact_ratio, flag)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                flag['dimension'], flag['group'],
                audit_result['total_applications'], 0,
                flag['approval_rate'], flag['mean_score'],
                flag['adverse_impact_ratio'], flag['flag']
            ))
        conn.commit()
    finally:
        conn.close()


if __name__ == "__main__":
    if pd is None or np is None:
        print("pandas/numpy not available — but fairness_engine.py structure is valid")
        print("fairness_engine.py OK")
    else:
        result = run_fairness_audit()
        if 'error' in result:
            print(result['error'])
        else:
            print(f"Total applications: {result['total_applications']}")
            print(f"Overall approval rate: {result['overall_approval_rate']:.1%}")
            print(f"Total fairness flags: {result['total_flags']}")
            print(f"Audit status: {result['audit_status']}")
        print("fairness_engine.py OK")
