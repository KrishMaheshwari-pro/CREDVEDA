"""
Core scoring library: turns a raw applicant record into a score, a
confidence band, plain-language reason codes, responsible-lending
guardrail flags, and a borrower-facing improvement path.

Used both by app.py (real-time scoring for the live "New Assessment" form
and for the stored applicant portfolio) and by this file's own CLI, which
batch-scores every applicant in the database.

# Updated on 2026-02-18
"""
import json
import sqlite3
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd

import config

_COMPONENTS = None

# Data Verification Index (DVI) — per-field weights and source confidence multipliers.
# Weights reflect how much each input drives the credit score. Source confidence
# reflects how trustworthy the data is based on how it was obtained.
_DVI_FIELDS = {
    # (applies_to, weight)
    "src_income":  ("both",     40),   # income / inflow — biggest FOIR driver
    "src_emi":     ("both",     25),   # existing debt obligations
    "src_gst":     ("business", 25),   # GST compliance — business only
    "src_utility": ("both",     10),   # utility / rent payment history
}
_SOURCE_CONFIDENCE = {"verified": 100, "estimated": 70, "self_declared": 30}


def blend_income_confidence(digital_amt: float, digital_src: str,
                            cash_amt: float, cash_src: str) -> float:
    """Amount-weighted confidence of the income figure.

    Digital/bank inflow is verifiable from statements; cash inflow is not, but
    can be corroborated (GST turnover, purchase invoices, a field visit). A
    kirana store that is 70% cash therefore is not automatically low-confidence
    — the verifiable 30% counts fully, and the cash 70% counts at whatever
    corroboration level it carries. This is exactly how NBFCs underwrite
    cash-heavy small businesses in practice."""
    total = (digital_amt or 0) + (cash_amt or 0)
    if total <= 0:
        return float(_SOURCE_CONFIDENCE.get(digital_src, 30))
    cd = _SOURCE_CONFIDENCE.get(digital_src, 100)
    cc = _SOURCE_CONFIDENCE.get(cash_src, 30)
    return (digital_amt * cd + cash_amt * cc) / total


def compute_dvi(field_sources: dict, entity_type: str, income_confidence: float = None) -> int:
    """Data Verification Index (0–100): proportion of score-driving data that is
    backed by a verifiable source rather than the borrower's own declaration.

    The income component is an amount-weighted blend of verifiable digital/bank
    inflow and (corroborable) cash inflow — passed in as `income_confidence`.
    The remaining components use their per-field source tags. This gives
    borrowers a direct incentive to share proof for whatever slice of income
    they can, rather than an all-or-nothing penalty for earning in cash."""
    total_w = 0
    weighted = 0
    for field, (applies_to, w) in _DVI_FIELDS.items():
        if applies_to == "business" and entity_type == "Individual":
            continue
        if field == "src_income" and income_confidence is not None:
            conf = income_confidence
        else:
            conf = _SOURCE_CONFIDENCE.get(field_sources.get(field, "self_declared"), 30)
        total_w += w
        weighted += w * conf
    return int(round(weighted / total_w)) if total_w else 30


def emi(principal: float, tenure_months: int, annual_rate: float = config.ASSUMED_ANNUAL_INTEREST_RATE) -> float:
    """Standard reducing-balance EMI formula."""
    if principal <= 0 or tenure_months <= 0:
        return 0.0
    r = annual_rate / 12.0
    factor = (1 + r) ** tenure_months
    return principal * r * factor / (factor - 1)


def _tree_bias_and_contributions(tree_estimator, x_row: np.ndarray, n_features: int):
    """Decomposes one fitted DecisionTreeClassifier's prediction for class 1
    into an additive root-bias + per-feature contribution, by walking the
    exact decision path taken by x_row and attributing the change in class-1
    probability at each split to the feature that split was made on. This is
    the same "TreeInterpreter" decomposition SHAP's TreeExplainer uses for
    tree ensembles, implemented directly to avoid a numba/NumPy version
    conflict with the `shap` package in this environment."""
    tree = tree_estimator.tree_
    node_indicator = tree_estimator.decision_path(x_row)
    node_path = node_indicator.indices[node_indicator.indptr[0]:node_indicator.indptr[1]]

    values = tree.value[:, 0, :]
    probs = values / values.sum(axis=1, keepdims=True)

    contributions = np.zeros(n_features)
    bias = probs[node_path[0], 1]
    for i in range(len(node_path) - 1):
        node, child = node_path[i], node_path[i + 1]
        feat = tree.feature[node]
        contributions[feat] += probs[child, 1] - probs[node, 1]
    return bias, contributions


def forest_explain(model, x_scaled_row: np.ndarray, n_features: int):
    """Averages the per-tree decomposition across the forest. bias + sum of
    contributions reproduces model.predict_proba(x)[:, 1] exactly, since
    RandomForestClassifier.predict_proba is itself the mean of each tree's
    leaf class-proportions."""
    biases, total = [], np.zeros(n_features)
    for estimator in model.estimators_:
        b, c = _tree_bias_and_contributions(estimator, x_scaled_row, n_features)
        biases.append(b)
        total += c
    n_trees = len(model.estimators_)
    return float(np.mean(biases)), total / n_trees


def load_components():
    """Loads (and caches) the trained model + preprocessing artefacts."""
    global _COMPONENTS
    if _COMPONENTS is not None:
        return _COMPONENTS
    model = joblib.load("model.joblib")
    scaler = joblib.load("scaler.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    impute_medians = joblib.load("impute_medians.joblib")
    percentiles = joblib.load("feature_percentiles.joblib")
    try:
        anomaly_model = joblib.load("anomaly_model.joblib")
    except FileNotFoundError:
        anomaly_model = None
    _COMPONENTS = {
        "model": model, "scaler": scaler, "feature_cols": feature_cols,
        "impute_medians": impute_medians, "percentiles": percentiles,
        "anomaly_model": anomaly_model,
    }
    return _COMPONENTS


def prediction_uncertainty(model, X_scaled: np.ndarray) -> float:
    """Standard error of the forest's predicted probability for this applicant.

    A Random Forest's output is the average of its trees. Where the trees
    agree, the model is recognising a pattern it has seen many times; where
    they diverge, it is extrapolating into a thinly-populated corner of the
    feature space. That divergence is genuine epistemic uncertainty, and a
    far more honest basis for a confidence band than counting how many form
    fields were filled in.

    We report the standard error (std / sqrt(n_trees)), not the raw spread of
    individual trees. Individual decision trees are deliberately high-variance
    -- their leaves sit near 0 or 1, so their raw spread is large and roughly
    constant regardless of how confident the ensemble actually is. The spread
    of the *mean* is what actually tightens as the trees converge, and is the
    quantity a confidence interval should be built from.
    """
    tree_probs = np.array([est.predict_proba(X_scaled)[0, 1] for est in model.estimators_])
    return float(tree_probs.std() / np.sqrt(len(tree_probs)))


def anomaly_check(components: dict, X_scaled: np.ndarray) -> dict:
    """Unsupervised data-consistency screen.

    Flags applications whose numbers do not hang together the way genuine
    ones do -- a large claimed inflow on a handful of transactions, GST
    filings that don't match turnover, and so on. It never blocks a decision
    by itself; it routes the file to a human, which is how a real lender
    treats a soft fraud signal.
    """
    model = components.get("anomaly_model")
    if model is None:
        return {"flagged": False, "score": None}
    raw_score = float(model.decision_function(X_scaled)[0])
    return {"flagged": bool(model.predict(X_scaled)[0] == -1), "score": round(raw_score, 4)}


def build_feature_row(raw: dict, components: dict) -> pd.DataFrame:
    """Builds a single-row, imputed, ordered feature DataFrame from a raw
    applicant dict (which may have missing/NaN business-only fields)."""
    feature_cols = components["feature_cols"]
    medians = components["impute_medians"]
    row = {}
    for col in feature_cols:
        val = raw.get(col, None)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = medians.get(col, 0.0)
        row[col] = float(val)
    return pd.DataFrame([row], columns=feature_cols)


def compute_completeness(raw: dict, entity_type: str) -> float:
    """Fraction of applicable alternative-data fields that are actually
    populated. Business-only fields are excluded from the denominator for
    Individual applicants so they aren't unfairly penalised for data that
    structurally does not exist for them."""
    fields = list(config.COMPLETENESS_FIELDS)
    if entity_type == "Individual":
        fields = [f for f in fields if f not in config.BUSINESS_ONLY_FIELDS]
    if not fields:
        return 1.0
    present = 0
    for f in fields:
        v = raw.get(f, None)
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            present += 1
    return present / len(fields)


def confidence_band(completeness: float):
    for min_c, label, width in config.CONFIDENCE_BANDS:
        if completeness >= min_c:
            return label, width
    return config.CONFIDENCE_BANDS[-1][1], config.CONFIDENCE_BANDS[-1][2]


def score_tier(score: float):
    """Maps a raw score to a human-readable tier + UI tone."""
    for min_score, label, tone in config.SCORE_TIERS:
        if score >= min_score:
            return label, tone
    return config.SCORE_TIERS[-1][1], config.SCORE_TIERS[-1][2]


def predict(components: dict, X: pd.DataFrame):
    X_scaled = components["scaler"].transform(X)
    prob_good = float(components["model"].predict_proba(X_scaled)[:, 1][0])
    score = int(round(config.SCORE_MIN + prob_good * (config.SCORE_MAX - config.SCORE_MIN)))
    return prob_good, score, X_scaled


def shap_factors(components: dict, X_scaled: np.ndarray, raw: dict, top_n=6):
    feature_cols = components["feature_cols"]
    _, contrib = forest_explain(components["model"], X_scaled, len(feature_cols))

    pairs = []
    for i, feat in enumerate(feature_cols):
        pairs.append({
            "feature": feat,
            "contribution": float(contrib[i]),
            "raw_value": raw.get(feat, None),
        })
    pairs.sort(key=lambda p: abs(p["contribution"]), reverse=True)
    return pairs[:top_n]


def map_reason_codes(factors: list, max_codes=5):
    """Maps the top SHAP-driving features to standard plain-language reason
    codes, skipping any feature that isn't applicable to this applicant
    (raw value missing, e.g. GST fields for an Individual)."""
    codes = []
    for f in factors:
        rc_key = next((k for k, v in config.REASON_CODES.items() if v["feature"] == f["feature"]), None)
        if rc_key is None:
            continue
        raw_val = f["raw_value"]
        if raw_val is None or (isinstance(raw_val, float) and np.isnan(raw_val)):
            continue
        spec = config.REASON_CODES[rc_key]
        is_positive_contribution = f["contribution"] >= 0
        template = spec["positive"] if is_positive_contribution else spec["negative"]
        try:
            text = template.format(value=raw_val)
        except (ValueError, KeyError):
            text = template
        codes.append({
            "code": rc_key,
            "label": spec["label"],
            "text": text,
            "impact": "positive" if is_positive_contribution else "negative",
            "contribution": round(f["contribution"], 4),
        })
        if len(codes) >= max_codes:
            break
    return codes


def foir_assessment(raw: dict) -> dict:
    """FOIR (Fixed Obligation to Income Ratio) -- the affordability metric
    Indian banks and NBFCs actually underwrite against -- plus an income-shock
    stress test.

    Checking affordability only against today's income is how lenders end up
    with a book that performs beautifully until the first bad quarter. The
    stress test re-runs the same ratio assuming income falls, which catches
    borrowers who are affordable only in good times.
    """
    foir = raw.get("repayment_burden_ratio", 0.0) or 0.0
    drop = config.STRESS_TEST_INCOME_DROP
    stressed = foir / (1 - drop) if drop < 1 else foir
    return {
        "foir": foir,
        "stressed_foir": stressed,
        "income_drop": drop,
        "passes": foir < config.FOIR_CAUTION,
        "survives_stress": stressed < config.FOIR_HARD_STOP,
    }


def guardrail_checks(raw: dict, completeness: float, dvi: int = 100):
    flags = []
    assessment = foir_assessment(raw)
    foir = assessment["foir"]
    stressed = assessment["stressed_foir"]

    if foir >= config.FOIR_HARD_STOP:
        flags.append({
            "severity": "critical",
            "message": (f"FOIR of {foir:.0%} breaches the {config.FOIR_HARD_STOP:.0%} hard limit — existing "
                        f"and proposed EMIs would consume {foir:.0%} of assessable income. Approving at the "
                        f"requested amount would create an unsustainable repayment burden; recommend decline "
                        f"or restructure to a lower amount / longer tenure."),
        })
    elif foir >= config.FOIR_CAUTION:
        flags.append({
            "severity": "warning",
            "message": (f"FOIR of {foir:.0%} is above the {config.FOIR_CAUTION:.0%} comfort threshold. "
                        f"Affordable today, but with little headroom — consider a smaller amount or longer tenure."),
        })
    elif stressed >= config.FOIR_HARD_STOP:
        # Passes today, fails the stress test: the case this check exists for.
        flags.append({
            "severity": "warning",
            "message": (f"Affordable today at {foir:.0%} FOIR, but a {assessment['income_drop']:.0%} income "
                        f"drop would push it to {stressed:.0%} — past the {config.FOIR_HARD_STOP:.0%} limit. "
                        f"This borrower has no buffer for a bad quarter; size the loan for the downside."),
        })

    requested = raw.get("requested_loan_amount", 0.0) or 0.0
    inflow = raw.get("avg_monthly_inflow", 1.0) or 1.0
    if completeness < 0.5 and requested > inflow * 6:
        flags.append({
            "severity": "warning",
            "message": ("This is a thin-file applicant requesting a large amount relative to observed "
                        "income. Recommend a smaller initial limit with a review after a short repayment "
                        "track record builds, rather than a single large-amount decision."),
        })
    # DVI guardrail — added after income/debt checks so it appears last in the list
    if dvi < 40:
        flags.append({
            "severity": "critical",
            "message": (
                f"Data Verification Index is {dvi}/100 — most key inputs (income, existing debt, "
                "payment history) are self-declared with no supporting document or linked data source. "
                "A conservative haircut has been applied to the score. Field verification is required "
                "before any credit decision or marketplace listing can proceed."
            ),
        })
    elif dvi < 60:
        flags.append({
            "severity": "warning",
            "message": (
                f"Data Verification Index is {dvi}/100 — several key inputs are self-declared. "
                "A conservative score adjustment has been applied. Recommend phone or document "
                "verification before marketplace listing to improve investor confidence."
            ),
        })
    return flags


def improvement_path(components: dict, raw: dict, current_score: int, entity_type: str, top_n=3):
    feature_cols = components["feature_cols"]
    percentiles = components["percentiles"]
    base_row = build_feature_row(raw, components)

    candidates = []
    for feat, spec in config.ACTIONABLE_FEATURES.items():
        if feat not in feature_cols:
            continue
        if entity_type == "Individual" and feat in config.BUSINESS_ONLY_FIELDS:
            continue
        current_val = float(base_row.at[0, feat])
        pct_key = int(spec["target_percentile"] * 100)
        target_val = percentiles.get(feat, {}).get(pct_key)
        if target_val is None:
            continue
        if spec["direction"] == "increase" and target_val <= current_val:
            continue
        if spec["direction"] == "decrease" and target_val >= current_val:
            continue

        sim_row = base_row.copy()
        sim_row.at[0, feat] = target_val
        prob, score, _ = predict(components, sim_row)
        delta = score - current_score
        if delta > 1:
            candidates.append({
                "feature": feat,
                "label": spec["label"],
                "current_value": current_val,
                "target_value": target_val,
                "estimated_point_gain": delta,
            })

    candidates.sort(key=lambda c: c["estimated_point_gain"], reverse=True)
    top = candidates[:top_n]
    for c in top:
        c["narrative"] = (
            f"{c['label']}: sustaining this for ~6 months could raise your score by "
            f"roughly {c['estimated_point_gain']} points."
        )
    return top


def score_core(raw: dict, entity_type: str = "Small Business", dvi: int = 100) -> dict:
    """Fast path: score + confidence band + guardrails, with no per-tree
    explanation walk. Used for batch-scoring the whole portfolio and for
    the fairness audit, where explanations for every single applicant
    aren't needed -- only for the one an underwriter is actually looking
    at (see score_full)."""
    components = load_components()
    X = build_feature_row(raw, components)
    prob_good, score, X_scaled = predict(components, X)

    completeness = compute_completeness(raw, entity_type)
    conf_label, completeness_width = confidence_band(completeness)

    # The band is the wider of two independent doubts: how much of the file we
    # actually have, and how much the forest's trees disagree with each other.
    # Taking the wider of the two means we never present more precision than
    # the weaker of the two signals supports.
    tree_std = prediction_uncertainty(components["model"], X_scaled)
    uncertainty_width = int(np.clip(tree_std * config.UNCERTAINTY_BAND_SCALE,
                                    config.UNCERTAINTY_BAND_MIN, config.UNCERTAINTY_BAND_MAX))
    band_width = max(completeness_width, uncertainty_width)

    band_low = max(config.SCORE_MIN, score - band_width // 2)
    band_high = min(config.SCORE_MAX, score + band_width // 2)
    guardrails = guardrail_checks(raw, completeness, dvi=dvi)
    tier_label, tier_tone = score_tier(score)
    anomaly = anomaly_check(components, X_scaled)
    if anomaly["flagged"]:
        guardrails.append({
            "severity": "warning",
            "message": ("This application's figures are statistically inconsistent with genuine ones in "
                        "the portfolio — the combination of values is unusual rather than any single "
                        "figure being extreme. Recommend manual verification of the underlying data "
                        "before any decision."),
        })

    return {
        "probability_good": round(prob_good, 4),
        "credit_score": score,
        "confidence_label": conf_label,
        "band_low": band_low,
        "band_high": band_high,
        "band_width": band_width,
        "tree_uncertainty": round(tree_std, 4),
        "uncertainty_width": uncertainty_width,
        "completeness_width": completeness_width,
        "foir": foir_assessment(raw),
        "anomaly": anomaly,
        # When the file is thin OR the forest is visibly unsure, the band --
        # not the point estimate -- becomes the headline the UI shows, per the
        # PS's "confidence band rather than a false precision score" rule.
        "band_first": completeness < config.BAND_FIRST_COMPLETENESS or band_width >= 100,
        "tier_label": tier_label,
        "tier_tone": tier_tone,
        "data_completeness": round(completeness, 3),
        "guardrail_flags": guardrails,
        "approved": score >= config.APPROVAL_SCORE_THRESHOLD and not any(
            g["severity"] == "critical" for g in guardrails
        ),
        "_X_scaled": X_scaled,
    }


def score_full(raw: dict, entity_type: str = "Small Business", field_sources: dict = None,
               income_confidence: float = None) -> dict:
    """Full pipeline for a single applicant: score + confidence band +
    guardrails + SHAP-style reason codes + improvement path. Used for the
    live "New Assessment" form and the applicant detail page."""
    if field_sources is None:
        field_sources = {}
    dvi = compute_dvi(field_sources, entity_type, income_confidence=income_confidence)
    components = load_components()
    result = score_core(raw, entity_type, dvi=dvi)
    X_scaled = result.pop("_X_scaled")

    factors = shap_factors(components, X_scaled, raw)
    result["shap_factors"] = factors
    result["reason_codes"] = map_reason_codes(factors)
    result["improvement_path"] = improvement_path(components, raw, result["credit_score"], entity_type)
    result["dvi"] = dvi
    result["field_sources"] = field_sources
    # Post-processing DVI penalty: transparent, explainable adjustment.
    # DVI ≥ 60 = no penalty; every 10 pts below 60 = ~4 pt score reduction.
    dvi_penalty = max(0, int((60 - dvi) * 0.4)) if dvi < 60 else 0
    result["dvi_penalty"] = dvi_penalty
    result["adjusted_score"] = max(config.SCORE_MIN, result["credit_score"] - dvi_penalty)
    return result


# Backwards-compatible alias
score_applicant = score_full


def get_or_compute_full(applicant_id: str, raw: dict, entity_type: str) -> dict:
    """Returns the cached full explanation for an applicant if one was
    already computed, otherwise computes it via score_full() and caches it
    back into 'credit_scores' so the next view of the same applicant is
    instant."""
    conn = sqlite3.connect(config.DB_NAME)
    try:
        row = conn.execute(
            "SELECT reason_codes, improvement_path, credit_score, probability_good, "
            "confidence_label, band_low, band_high, data_completeness, guardrail_flags, "
            "shap_factors FROM credit_scores WHERE applicant_id = ?", (applicant_id,)
        ).fetchone()

        if row and row[0] and row[0] != "[]":
            guardrails = json.loads(row[8])
            tier_label, tier_tone = score_tier(row[2])
            # This branch must return the same keys as score_full(), or a
            # template that renders fine on an applicant's first view breaks
            # on the second one (marketplace_listing.html reads
            # result.foir.foir unguarded). foir_assessment() is pure
            # arithmetic over raw, so it is recomputed rather than cached;
            # only the expensive tree walks come out of the DB.
            band_width = max(0, int(row[6]) - int(row[5]))
            return {
                "reason_codes": json.loads(row[0]),
                "improvement_path": json.loads(row[1]),
                "credit_score": row[2],
                "probability_good": row[3],
                "confidence_label": row[4],
                "band_low": row[5],
                "band_high": row[6],
                "data_completeness": row[7],
                "guardrail_flags": guardrails,
                "shap_factors": json.loads(row[9]) if row[9] else [],
                "band_first": row[7] < config.BAND_FIRST_COMPLETENESS,
                "tier_label": tier_label,
                "tier_tone": tier_tone,
                "foir": foir_assessment(raw),
                # None, not a clean verdict: the IsolationForest is not re-run
                # on a cache hit, so "no flag" would be a claim we can't make.
                # Templates guard with `result.anomaly and ...`.
                "anomaly": None,
                "band_width": band_width,
                "approved": row[2] >= config.APPROVAL_SCORE_THRESHOLD and not any(
                    g["severity"] == "critical" for g in guardrails
                ),
                "dvi": None,
                "dvi_penalty": 0,
                "adjusted_score": row[2],
                "field_sources": {},
            }

        result = score_full(raw, entity_type)
        conn.execute(
            "UPDATE credit_scores SET reason_codes = ?, improvement_path = ?, shap_factors = ? "
            "WHERE applicant_id = ?",
            (json.dumps(result["reason_codes"]), json.dumps(result["improvement_path"]),
             json.dumps(result["shap_factors"]), applicant_id),
        )
        conn.commit()
        return result
    finally:
        conn.close()


def generate_scores():
    """Batch-scores every applicant currently in the database in one
    vectorised pass (a single model.predict_proba() call over the whole
    population, not a Python loop calling the model once per row -- that
    was the original approach and it was far too slow at 4000 rows because
    of RandomForestClassifier's per-call parallel-dispatch overhead).
    Full SHAP-style reason codes and the improvement path walk every tree
    in the forest and are comparatively expensive; they're computed lazily,
    on demand, the first time an applicant's detail page is opened -- see
    score_full() / get_or_compute_full() and app.py."""
    components = load_components()
    feature_cols = components["feature_cols"]
    medians = components["impute_medians"]

    conn = sqlite3.connect(config.DB_NAME)
    try:
        df = pd.read_sql_query("SELECT * FROM applicants", conn)
    finally:
        conn.close()
    n = len(df)
    print(f"Scoring {n} applicants (vectorised fast path)...")

    X = df[feature_cols].copy()
    for col in feature_cols:
        X[col] = X[col].fillna(medians.get(col, 0.0))
    X_scaled = components["scaler"].transform(X)
    prob_good = components["model"].predict_proba(X_scaled)[:, 1]
    score = np.round(config.SCORE_MIN + prob_good * (config.SCORE_MAX - config.SCORE_MIN)).astype(int)

    completeness_fields_all = config.COMPLETENESS_FIELDS
    business_only = set(config.BUSINESS_ONLY_FIELDS)
    n_business_only_in_completeness = sum(1 for f in completeness_fields_all if f in business_only)
    is_individual = (df["entity_type"] == "Individual").to_numpy()
    applicable_counts = np.where(
        is_individual,
        len(completeness_fields_all) - n_business_only_in_completeness,
        len(completeness_fields_all),
    )
    present = np.zeros(n)
    for f in completeness_fields_all:
        col_present = df[f].notna().to_numpy()
        if f in business_only:
            present += np.where(is_individual, 0, col_present)
        else:
            present += col_present
    completeness = present / np.maximum(applicable_counts, 1)

    completeness_width = np.select(
        [completeness >= b[0] for b in config.CONFIDENCE_BANDS],
        [b[2] for b in config.CONFIDENCE_BANDS],
        default=config.CONFIDENCE_BANDS[-1][2],
    )
    conf_label = np.select(
        [completeness >= b[0] for b in config.CONFIDENCE_BANDS],
        [b[1] for b in config.CONFIDENCE_BANDS],
        default=config.CONFIDENCE_BANDS[-1][1],
    )

    # Model-uncertainty half of the band, vectorised: ask each tree for its
    # prediction on the whole population at once (160 calls over 4000 rows,
    # rather than 4000 walks over 160 trees) and take the standard error of
    # the ensemble mean per applicant. Must match score_core(), or the
    # portfolio table and the detail page would quote different bands.
    trees = components["model"].estimators_
    tree_probs = np.stack([est.predict_proba(X_scaled)[:, 1] for est in trees])
    standard_error = tree_probs.std(axis=0) / np.sqrt(len(trees))
    uncertainty_width = np.clip(standard_error * config.UNCERTAINTY_BAND_SCALE,
                                config.UNCERTAINTY_BAND_MIN, config.UNCERTAINTY_BAND_MAX)
    band_width = np.maximum(completeness_width, uncertainty_width).astype(int)
    band_low = np.clip(score - band_width // 2, config.SCORE_MIN, config.SCORE_MAX)
    band_high = np.clip(score + band_width // 2, config.SCORE_MIN, config.SCORE_MAX)

    now = datetime.now(timezone.utc).isoformat()
    rows = []
    for i in range(n):
        raw_row = {"repayment_burden_ratio": df["repayment_burden_ratio"].iat[i],
                   "requested_loan_amount": df["requested_loan_amount"].iat[i],
                   "avg_monthly_inflow": df["avg_monthly_inflow"].iat[i]}
        guardrails = guardrail_checks(raw_row, completeness[i])
        rows.append({
            "applicant_id": df["applicant_id"].iat[i],
            "credit_score": int(score[i]),
            "probability_good": round(float(prob_good[i]), 4),
            "confidence_label": conf_label[i],
            "band_low": int(band_low[i]),
            "band_high": int(band_high[i]),
            "data_completeness": round(float(completeness[i]), 3),
            "reason_codes": "[]",
            "guardrail_flags": json.dumps(guardrails),
            "improvement_path": "[]",
            "shap_factors": "[]",
            "scored_at": now,
        })

    scores_df = pd.DataFrame(rows)
    conn = sqlite3.connect(config.DB_NAME)
    try:
        scores_df.to_sql("credit_scores", conn, if_exists="replace", index=False)
        print(f"Saved {len(scores_df)} scores to 'credit_scores'.")
    finally:
        conn.close()


if __name__ == "__main__":
    generate_scores()
# Updated on 2026-02-18
