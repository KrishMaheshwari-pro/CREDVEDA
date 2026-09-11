"""
Loan lifecycle: what happens after a score becomes a decision.

Scoring an applicant is only half the problem PS #4 describes. The reason
working capital stays out of reach is that nobody will take the first risk --
so the interesting question is what happens once somebody does. This module
disburses a loan, builds its EMI schedule, simulates repayment month by month,
and feeds the resulting behaviour back into how that borrower is assessed next
time.

Two design decisions matter here:

1. Simulated repayment behaviour is driven by each applicant's
   `latent_prob_good` -- the ground-truth creditworthiness the synthetic
   generator assigned them, which the model never sees. The model's prediction
   and the borrower's actual behaviour therefore come from independent
   sources, which is what makes the portfolio backtest in `backtest_buckets()`
   a real validation rather than a circular one.

2. Repayment history is applied as an explicit, auditable adjustment layered
   on top of the model score -- never silently folded into a model feature.
   That mirrors how lenders run a behavioural scorecard over an application
   score, and keeps the base model's explanation honest.

# Updated on 2026-02-18
"""
import random
import sqlite3
from datetime import datetime, timezone

import config
import scoring


def _connect():
    conn = sqlite3.connect(config.DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


# --- Disbursal ---------------------------------------------------------------
def _next_loan_id(conn) -> str:
    row = conn.execute("SELECT COUNT(*) AS n FROM loans").fetchone()
    return f"LN-{row['n'] + 1:05d}"


def active_loan_for(applicant_id: str):
    """A borrower may only hold one open loan at a time on the platform."""
    conn = _connect()
    try:
        return conn.execute(
            "SELECT * FROM loans WHERE applicant_id = ? AND status = 'active'",
            (applicant_id,),
        ).fetchone()
    finally:
        conn.close()


def loans_for(applicant_id: str):
    conn = _connect()
    try:
        return conn.execute(
            "SELECT * FROM loans WHERE applicant_id = ? ORDER BY disbursed_at DESC",
            (applicant_id,),
        ).fetchall()
    finally:
        conn.close()


def disburse(applicant_id: str, principal: float, tenure_months: int,
             score_at_funding: int, prob_good_at_funding: float) -> str:
    """Creates a loan and its full EMI schedule up front, the way an
    amortisation table is fixed at disbursal in a real loan agreement."""
    emi = scoring.emi(principal, tenure_months)
    conn = _connect()
    try:
        loan_id = _next_loan_id(conn)
        conn.execute(
            "INSERT INTO loans (loan_id, applicant_id, principal, tenure_months, annual_rate, "
            "monthly_emi, status, months_elapsed, score_at_funding, prob_good_at_funding, disbursed_at) "
            "VALUES (?,?,?,?,?,?,'active',0,?,?,?)",
            (loan_id, applicant_id, principal, tenure_months,
             config.ASSUMED_ANNUAL_INTEREST_RATE, emi,
             score_at_funding, prob_good_at_funding,
             datetime.now(timezone.utc).isoformat()),
        )
        conn.executemany(
            "INSERT INTO repayments (loan_id, instalment_no, amount_due, status, due_month) "
            "VALUES (?,?,?, 'pending', ?)",
            [(loan_id, i, emi, i) for i in range(1, tenure_months + 1)],
        )
        conn.commit()
        return loan_id
    finally:
        conn.close()


# --- Repayment simulation ----------------------------------------------------
def _ontime_probability(latent_prob_good: float) -> float:
    """Per-instalment probability of paying on time. Even a strong borrower
    misses occasionally; even a weak one usually pays early instalments."""
    if latent_prob_good is None:
        latent_prob_good = 0.6
    return 0.55 + 0.45 * float(latent_prob_good)


def advance_loan(loan_id: str, rng: random.Random | None = None) -> dict:
    """Advances a single loan by one month, settling the next instalment."""
    rng = rng or random
    conn = _connect()
    try:
        loan = conn.execute("SELECT * FROM loans WHERE loan_id = ?", (loan_id,)).fetchone()
        if loan is None or loan["status"] != "active":
            return {"changed": False}

        nxt = conn.execute(
            "SELECT * FROM repayments WHERE loan_id = ? AND status = 'pending' "
            "ORDER BY instalment_no LIMIT 1", (loan_id,)
        ).fetchone()
        if nxt is None:
            conn.execute("UPDATE loans SET status='closed', closed_at=? WHERE loan_id=?",
                         (datetime.now(timezone.utc).isoformat(), loan_id))
            conn.commit()
            return {"changed": True, "loan_closed": True}

        truth = conn.execute(
            "SELECT latent_prob_good FROM applicants WHERE applicant_id = ?",
            (loan["applicant_id"],)
        ).fetchone()
        p_ontime = _ontime_probability(truth["latent_prob_good"] if truth else None)

        draw = rng.random()
        if draw < p_ontime:
            status = "paid"
        elif rng.random() < config.LATE_RECOVERY_PROB:
            status = "late"      # paid, but behind schedule
        else:
            status = "missed"

        conn.execute(
            "UPDATE repayments SET status=?, paid_at=? WHERE loan_id=? AND instalment_no=?",
            (status, datetime.now(timezone.utc).isoformat() if status != "missed" else None,
             loan_id, nxt["instalment_no"]),
        )
        conn.execute("UPDATE loans SET months_elapsed = months_elapsed + 1 WHERE loan_id=?", (loan_id,))

        missed = conn.execute(
            "SELECT COUNT(*) AS n FROM repayments WHERE loan_id=? AND status='missed'", (loan_id,)
        ).fetchone()["n"]
        remaining = conn.execute(
            "SELECT COUNT(*) AS n FROM repayments WHERE loan_id=? AND status='pending'", (loan_id,)
        ).fetchone()["n"]

        outcome = {"changed": True, "instalment": nxt["instalment_no"], "status": status}
        if missed >= config.DEFAULT_AFTER_MISSED:
            conn.execute("UPDATE loans SET status='defaulted', closed_at=? WHERE loan_id=?",
                         (datetime.now(timezone.utc).isoformat(), loan_id))
            outcome["loan_defaulted"] = True
        elif remaining == 0:
            conn.execute("UPDATE loans SET status='closed', closed_at=? WHERE loan_id=?",
                         (datetime.now(timezone.utc).isoformat(), loan_id))
            outcome["loan_closed"] = True

        conn.commit()
        return outcome
    finally:
        conn.close()


def advance_all(months: int = 1) -> dict:
    """Fast-forwards every active loan. A live portfolio can't be demonstrated
    by waiting real months, so this is the clock."""
    summary = {"months": months, "instalments": 0, "paid": 0, "late": 0,
               "missed": 0, "closed": 0, "defaulted": 0}
    for _ in range(months):
        conn = _connect()
        try:
            ids = [r["loan_id"] for r in
                   conn.execute("SELECT loan_id FROM loans WHERE status='active'").fetchall()]
        finally:
            conn.close()
        for loan_id in ids:
            res = advance_loan(loan_id)
            if not res.get("changed"):
                continue
            if res.get("status"):
                summary["instalments"] += 1
                summary[res["status"]] = summary.get(res["status"], 0) + 1
            if res.get("loan_closed"):
                summary["closed"] += 1
            if res.get("loan_defaulted"):
                summary["defaulted"] += 1
    return summary


def schedule_for(loan_id: str):
    conn = _connect()
    try:
        return conn.execute(
            "SELECT * FROM repayments WHERE loan_id=? ORDER BY instalment_no", (loan_id,)
        ).fetchall()
    finally:
        conn.close()


def get_loan(loan_id: str):
    conn = _connect()
    try:
        return conn.execute("SELECT * FROM loans WHERE loan_id=?", (loan_id,)).fetchone()
    finally:
        conn.close()


# --- Repayment history feeding back into the score ---------------------------
def repayment_history(applicant_id: str) -> dict:
    """Aggregates this borrower's on-platform track record."""
    conn = _connect()
    try:
        row = conn.execute("""
            SELECT
                SUM(CASE WHEN r.status='paid'   THEN 1 ELSE 0 END) AS paid,
                SUM(CASE WHEN r.status='late'   THEN 1 ELSE 0 END) AS late,
                SUM(CASE WHEN r.status='missed' THEN 1 ELSE 0 END) AS missed
            FROM repayments r
            JOIN loans l ON l.loan_id = r.loan_id
            WHERE l.applicant_id = ? AND r.status != 'pending'
        """, (applicant_id,)).fetchone()
    finally:
        conn.close()

    paid = row["paid"] or 0
    late = row["late"] or 0
    missed = row["missed"] or 0
    settled = paid + late + missed
    return {
        "paid": paid, "late": late, "missed": missed, "settled": settled,
        "on_time_rate": (paid / settled) if settled else None,
    }


def repayment_adjustment(applicant_id: str) -> dict:
    """Converts the track record into an explicit score adjustment.

    Deliberately additive and capped, so it can be shown to the borrower as a
    line item ("+24 for four on-time EMIs") rather than disappearing into a
    model weight nobody can audit.
    """
    hist = repayment_history(applicant_id)
    if not hist["settled"]:
        return {"points": 0, "history": hist, "reason": None}

    bonus = min(hist["paid"] * config.REPAYMENT_BONUS_PER_ONTIME, config.REPAYMENT_BONUS_CAP)
    penalty = (hist["late"] * config.REPAYMENT_PENALTY_PER_LATE
               + hist["missed"] * config.REPAYMENT_PENALTY_PER_MISSED)
    points = max(bonus - penalty, config.REPAYMENT_PENALTY_CAP)

    bits = []
    if hist["paid"]:
        bits.append(f"{hist['paid']} on-time EMI{'s' if hist['paid'] != 1 else ''}")
    if hist["late"]:
        bits.append(f"{hist['late']} late")
    if hist["missed"]:
        bits.append(f"{hist['missed']} missed")
    return {
        "points": int(points),
        "history": hist,
        "reason": "Repayment track record on this platform: " + ", ".join(bits) + ".",
    }


def apply_repayment_history(applicant_id: str, result: dict) -> dict:
    """Layers the borrower's on-platform track record over their model score.

    The base model score is left intact and reported alongside the adjustment,
    so an underwriter (or the borrower) can always see both the statistical
    assessment and the behavioural override that moved it. Nothing here is
    hidden inside the model.
    """
    adj = repayment_adjustment(applicant_id)
    result["base_score"] = result["credit_score"]
    result["repayment_adjustment"] = adj
    if adj["points"] == 0:
        return result

    adjusted = int(max(config.SCORE_MIN, min(config.SCORE_MAX, result["credit_score"] + adj["points"])))
    shift = adjusted - result["credit_score"]
    result["credit_score"] = adjusted
    result["band_low"] = int(max(config.SCORE_MIN, result["band_low"] + shift))
    result["band_high"] = int(min(config.SCORE_MAX, result["band_high"] + shift))
    result["tier_label"], result["tier_tone"] = scoring.score_tier(adjusted)
    result["approved"] = adjusted >= config.APPROVAL_SCORE_THRESHOLD and not any(
        g["severity"] == "critical" for g in result.get("guardrail_flags", [])
    )
    return result


# --- Portfolio view ----------------------------------------------------------
def portfolio_stats() -> dict:
    conn = _connect()
    try:
        loans = conn.execute("SELECT * FROM loans").fetchall()
        rep = conn.execute("""
            SELECT status, COUNT(*) AS n, SUM(amount_due) AS amt
            FROM repayments GROUP BY status
        """).fetchall()
    finally:
        conn.close()

    by_status = {r["status"]: {"n": r["n"], "amt": r["amt"] or 0} for r in rep}
    collected = (by_status.get("paid", {}).get("amt", 0)
                 + by_status.get("late", {}).get("amt", 0))
    settled_n = sum(by_status.get(s, {}).get("n", 0) for s in ("paid", "late", "missed"))

    return {
        "n_loans": len(loans),
        "active": sum(1 for l in loans if l["status"] == "active"),
        "closed": sum(1 for l in loans if l["status"] == "closed"),
        "defaulted": sum(1 for l in loans if l["status"] == "defaulted"),
        "disbursed": sum(l["principal"] for l in loans),
        "collected": collected,
        "overdue_amt": by_status.get("missed", {}).get("amt", 0),
        "instalments_due": settled_n,
        "on_time_rate": (by_status.get("paid", {}).get("n", 0) / settled_n) if settled_n else None,
        "default_rate": (sum(1 for l in loans if l["status"] == "defaulted") / len(loans)) if loans else None,
    }


def portfolio_loans(limit: int = 200):
    conn = _connect()
    try:
        return conn.execute("""
            SELECT l.*, a.business_type, a.entity_type, a.geography_tier,
                   (SELECT COUNT(*) FROM repayments r WHERE r.loan_id=l.loan_id AND r.status='paid')   AS paid,
                   (SELECT COUNT(*) FROM repayments r WHERE r.loan_id=l.loan_id AND r.status='late')   AS late,
                   (SELECT COUNT(*) FROM repayments r WHERE r.loan_id=l.loan_id AND r.status='missed') AS missed
            FROM loans l
            JOIN applicants a ON a.applicant_id = l.applicant_id
            ORDER BY l.disbursed_at DESC LIMIT ?
        """, (limit,)).fetchall()
    finally:
        conn.close()


def alerts(limit: int = 12):
    """Borrowers who have missed an instalment -- the collections queue."""
    conn = _connect()
    try:
        return conn.execute("""
            SELECT l.loan_id, l.applicant_id, l.status, a.business_type, a.geography_tier,
                   COUNT(*) AS missed, MAX(r.instalment_no) AS last_instalment
            FROM repayments r
            JOIN loans l ON l.loan_id = r.loan_id
            JOIN applicants a ON a.applicant_id = l.applicant_id
            WHERE r.status = 'missed'
            GROUP BY l.loan_id
            ORDER BY missed DESC, last_instalment DESC
            LIMIT ?
        """, (limit,)).fetchall()
    finally:
        conn.close()


# --- Live model validation ---------------------------------------------------
def backtest_buckets(n_buckets: int = 5):
    """Groups funded loans by the probability the model gave them at funding
    time, then reports what those borrowers actually went on to do.

    This is the model's report card measured on operational outcomes rather
    than a held-out split: if the model is well calibrated, the observed
    on-time rate should climb monotonically across the buckets.
    """
    conn = _connect()
    try:
        rows = conn.execute("""
            SELECT l.loan_id, l.prob_good_at_funding, l.status,
                   (SELECT COUNT(*) FROM repayments r WHERE r.loan_id=l.loan_id AND r.status='paid')  AS paid,
                   (SELECT COUNT(*) FROM repayments r WHERE r.loan_id=l.loan_id
                                                       AND r.status IN ('paid','late','missed'))     AS settled
            FROM loans l
        """).fetchall()
    finally:
        conn.close()

    scored = [r for r in rows if r["settled"]]
    if not scored:
        return []

    buckets = []
    for i in range(n_buckets):
        lo = i / n_buckets
        hi = (i + 1) / n_buckets
        members = [r for r in scored
                   if lo <= (r["prob_good_at_funding"] or 0) < hi
                   or (i == n_buckets - 1 and (r["prob_good_at_funding"] or 0) == 1.0)]
        if not members:
            continue
        paid = sum(m["paid"] for m in members)
        settled = sum(m["settled"] for m in members)
        buckets.append({
            "label": f"{lo:.0%}-{hi:.0%}",
            "predicted": sum((m["prob_good_at_funding"] or 0) for m in members) / len(members),
            "actual_on_time": paid / settled if settled else 0,
            "n_loans": len(members),
            "defaulted": sum(1 for m in members if m["status"] == "defaulted"),
        })
    return buckets
# Updated on 2026-02-18
