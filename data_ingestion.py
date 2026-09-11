"""
Alternative-data ingestion pipeline for CredVeda.

Real deployments would connect this module to:
  - UPI / Account Aggregator transaction feeds (digital payment flows)
  - the GST portal / e-invoice system (filing history, turnover, invoices)
  - utility & rent payment aggregators (BBPS, landlord/PG platforms)
  - supplier/buyer ledgers from accounting software (Vyapar, Tally, Khatabook)
  - credit bureaus, for the minority of applicants who already have a file

None of those are accessible without a regulated business partnership, so
this module instead generates a large, realistic SYNTHETIC population of
individuals and small businesses whose alternative-data signals are drawn
from distributions calibrated to plausible Indian MSME/gig-economy ranges,
with deliberate, non-random correlations to business type and geography
(e.g. rural/Tier-3 applicants genuinely show lower digital footprints and
lower bureau coverage) so that the fairness and thin-file modules downstream
have something real to detect and report on.

The output schema (see config.FEATURE_COLS) is the contract the rest of the
pipeline (model_training.py, scoring.py, fairness.py) depends on -- swapping
this generator for real connectors would not require changing anything else.

# Updated on 2026-02-18
"""
import sqlite3
import numpy as np
import pandas as pd
import config

# --- Per-category generation parameters -----------------------------------
# (revenue_mu, revenue_sigma, growth_mean, volatility_mean, outflow_mean,
#  gst_prob_base, supplier_hhi_mean)
CATEGORY_PARAMS = {
    "Retail / Kirana Store":        (11.92, 0.55,  0.03, 0.15, 0.78, 0.55, 0.35),
    "Textile & Apparel":            (12.20, 0.60,  0.03, 0.20, 0.75, 0.70, 0.40),
    "Food & Beverage":              (11.70, 0.50,  0.04, 0.18, 0.72, 0.50, 0.30),
    "Transport & Logistics":        (12.10, 0.55,  0.02, 0.25, 0.70, 0.45, 0.25),
    "Repair & Local Services":      (11.15, 0.50,  0.03, 0.20, 0.55, 0.30, 0.20),
    "Micro-Manufacturing":          (12.40, 0.60,  0.04, 0.22, 0.73, 0.75, 0.45),
    "Agri-allied Trading":          (11.98, 0.70, -0.01, 0.45, 0.80, 0.35, 0.50),
    "Wholesale Trading":            (12.90, 0.60,  0.03, 0.18, 0.82, 0.85, 0.30),
    "Salon & Personal Care":        (11.00, 0.45,  0.03, 0.15, 0.50, 0.25, 0.15),
    "Electronics & Mobile Repair":  (11.40, 0.50,  0.03, 0.18, 0.60, 0.40, 0.30),
    "Salaried Employee":            (10.55, 0.40,  0.02, 0.08, 0.65, 0.00, 0.30),
    "Gig / Platform Worker":        (9.99,  0.40,  0.06, 0.30, 0.60, 0.00, 0.30),
    "Freelance Professional":       (10.71, 0.55,  0.03, 0.28, 0.55, 0.00, 0.30),
    "Daily Wage Worker":            (9.62,  0.35, -0.01, 0.35, 0.80, 0.00, 0.30),
}

TIER_DIGITAL_ADOPTION = {"Tier 1": (6, 2), "Tier 2": (5, 3), "Tier 3": (4, 4), "Rural": (3, 5)}
TIER_BUREAU_PROB = {"Tier 1": 0.62, "Tier 2": 0.45, "Tier 3": 0.30, "Rural": 0.18}
TIER_VINTAGE_MULT = {"Tier 1": 1.15, "Tier 2": 1.0, "Tier 3": 0.9, "Rural": 0.8}

def _param(series, mapping, index, default):
    return series.map(lambda k: mapping.get(k, (default,) * 7)[index]).astype(float)


def _emi(principal, tenure_months, annual_rate=config.ASSUMED_ANNUAL_INTEREST_RATE):
    r = annual_rate / 12.0
    n = tenure_months
    factor = (1 + r) ** n
    return principal * r * factor / (factor - 1)


def generate_synthetic_applicants(n=None, seed=None) -> pd.DataFrame:
    n = n or config.N_APPLICANTS
    rng = np.random.default_rng(seed or config.RANDOM_SEED)

    # --- Identity / demographics (NEVER used as model features) ---
    entity_type = rng.choice(config.ENTITY_TYPES, size=n, p=[0.42, 0.58])
    business_type = np.array([
        rng.choice(config.BUSINESS_TYPES_BY_ENTITY[et]) for et in entity_type
    ])
    gender = rng.choice(config.GENDERS, size=n, p=config.GENDER_WEIGHTS)
    geography_tier = rng.choice(config.GEOGRAPHY_TIERS, size=n, p=config.GEOGRAPHY_TIER_WEIGHTS)
    geography_state = np.array([
        rng.choice(config.STATES_BY_TIER[t]) for t in geography_tier
    ])

    df = pd.DataFrame({
        "applicant_id": [f"APP-{i:05d}" for i in range(1, n + 1)],
        "entity_type": entity_type,
        "business_type": business_type,
        "gender": gender,
        "geography_tier": geography_tier,
        "geography_state": geography_state,
    })

    is_business = (df["entity_type"] == "Small Business").to_numpy()

    revenue_mu = _param(df["business_type"], CATEGORY_PARAMS, 0, 10.5).to_numpy()
    revenue_sigma = _param(df["business_type"], CATEGORY_PARAMS, 1, 0.5).to_numpy()
    growth_mean = _param(df["business_type"], CATEGORY_PARAMS, 2, 0.02).to_numpy()
    volatility_mean = _param(df["business_type"], CATEGORY_PARAMS, 3, 0.2).to_numpy()
    outflow_mean = _param(df["business_type"], CATEGORY_PARAMS, 4, 0.65).to_numpy()
    gst_prob_base = _param(df["business_type"], CATEGORY_PARAMS, 5, 0.0).to_numpy()
    supplier_hhi_mean = _param(df["business_type"], CATEGORY_PARAMS, 6, 0.3).to_numpy()

    # --- Business/relationship vintage (months) ---
    tier_mult = df["geography_tier"].map(TIER_VINTAGE_MULT).to_numpy()
    vintage_months = rng.gamma(shape=2.2, scale=16, size=n) * tier_mult
    vintage_months = np.clip(vintage_months, 1, 300)

    # --- Digital footprint ---
    tier_ab = np.array([TIER_DIGITAL_ADOPTION[t] for t in df["geography_tier"]])
    digital_adoption_ratio = rng.beta(tier_ab[:, 0], tier_ab[:, 1])
    digital_adoption_ratio = np.clip(digital_adoption_ratio, 0.05, 0.98)

    # --- True turnover / income (latent) -> observed digital inflow ---
    true_monthly_value = rng.lognormal(mean=revenue_mu, sigma=revenue_sigma)
    avg_monthly_inflow = true_monthly_value * digital_adoption_ratio * rng.normal(1.0, 0.08, size=n)
    avg_monthly_inflow = np.clip(avg_monthly_inflow, 2000, None)

    inflow_growth_rate_6m = rng.normal(growth_mean, 0.15, size=n)
    inflow_growth_rate_6m = np.clip(inflow_growth_rate_6m, -0.6, 1.5)

    inflow_volatility_cv = np.clip(rng.normal(volatility_mean, 0.08, size=n), 0.03, 0.95)

    monthly_txn_count = np.clip(
        rng.poisson(lam=np.clip(8 + (avg_monthly_inflow / 6000) * digital_adoption_ratio, 3, 400)), 1, None
    )

    txn_bounce_rate = np.clip(rng.beta(1.5, 22, size=n), 0.0, 0.5)

    # --- GST / invoicing (business only) ---
    gst_prob = gst_prob_base * np.clip(vintage_months / 24, 0.2, 1.0) * (0.6 + 0.4 * digital_adoption_ratio)
    gst_registered = (rng.random(n) < gst_prob) & is_business
    gst_filing_regularity = np.where(
        gst_registered, np.clip(rng.beta(6, 2, size=n), 0.05, 1.0), np.nan
    )
    overdue_invoice_ratio = np.where(
        gst_registered, np.clip(rng.beta(2, 9, size=n), 0.0, 0.9), np.nan
    )

    # --- Utility & rent payments (everyone) ---
    utility_ontime_ratio = np.clip(rng.beta(9, 2, size=n) - 0.03 * inflow_volatility_cv, 0.05, 1.0)
    owns_premises = rng.random(n) < 0.20
    rent_ontime_ratio = np.where(
        owns_premises, 1.0, np.clip(rng.beta(8, 2.5, size=n) - 0.05 * inflow_volatility_cv, 0.05, 1.0)
    )

    # --- Data-availability gaps (this is what actually produces "thin file"
    # applicants downstream): a largely cash-based applicant has no reliable
    # digital transaction trail to derive inflow/txn-count/bounce-rate from,
    # and a meaningful minority never had a utility connection or formal
    # rent record in their own name to draw on. These gaps concentrate in
    # Rural/Tier-3 geographies through digital_adoption_ratio, which is the
    # honest, non-injected source of the disparity fairness.py should catch.
    cash_heavy = digital_adoption_ratio < 0.22
    utility_data_gap = rng.random(n) < 0.10
    rent_data_gap = (rng.random(n) < 0.12) & ~owns_premises

    avg_monthly_inflow_reported = np.where(cash_heavy, np.nan, avg_monthly_inflow)
    monthly_txn_count_reported = np.where(cash_heavy, np.nan, monthly_txn_count)
    txn_bounce_rate_reported = np.where(cash_heavy, np.nan, txn_bounce_rate)
    utility_ontime_ratio_reported = np.where(utility_data_gap, np.nan, utility_ontime_ratio)
    rent_ontime_ratio_reported = np.where(rent_data_gap, np.nan, rent_ontime_ratio)

    # --- Supplier relationships (business only) ---
    supplier_concentration_hhi = np.where(
        is_business, np.clip(rng.beta(3, 6, size=n) + (supplier_hhi_mean - 0.3), 0.05, 0.95), np.nan
    )
    repeat_supplier_ratio = np.where(
        is_business, np.clip(rng.beta(5, 3, size=n), 0.05, 0.98), np.nan
    )

    # --- Estimated net cash flow (used to size both existing and requested debt) ---
    outflow_ratio = np.clip(rng.normal(outflow_mean, 0.08, size=n), 0.2, 0.97)
    net_cashflow = np.maximum(avg_monthly_inflow * (1 - outflow_ratio), 1500)

    # --- Existing credit obligations (sized relative to net cash flow, so a
    # single prior loan's EMI is a plausible, affordable fraction of it) ---
    existing_loan_count = rng.poisson(lam=0.5, size=n).clip(0, 6)
    per_loan_principal = net_cashflow * rng.uniform(2, 6, size=n)
    existing_monthly_emi = np.where(
        existing_loan_count > 0, _emi(per_loan_principal, 24) * existing_loan_count, 0.0
    )

    bureau_prob = df["geography_tier"].map(TIER_BUREAU_PROB).to_numpy() * np.clip(vintage_months / 36, 0.3, 1.0)
    bureau_score_available = rng.random(n) < bureau_prob
    bureau_score_raw = np.clip(rng.normal(650, 90, size=n), 300, 900)
    bureau_score_norm = np.where(bureau_score_available, (bureau_score_raw - 300) / 600, np.nan)

    # --- Requested loan & affordability (sized relative to net cash flow;
    # lognormal tail means most requests are modest but some over-reach,
    # giving the responsible-lending guardrail real cases to catch) ---
    requested_loan_amount = np.round(net_cashflow * rng.lognormal(mean=np.log(5.5), sigma=0.55, size=n), -3)
    requested_loan_amount = np.clip(requested_loan_amount, 5000, None)
    requested_tenure_months = rng.choice([12, 18, 24, 36, 48], size=n, p=[0.15, 0.2, 0.3, 0.25, 0.1])
    proposed_emi = _emi(requested_loan_amount, requested_tenure_months)

    repayment_burden_ratio = (existing_monthly_emi + proposed_emi) / net_cashflow
    repayment_burden_ratio = np.clip(repayment_burden_ratio, 0, 4)

    # --- Latent creditworthiness -> probabilistic repayment outcome label ---
    # IMPORTANT: this uses the TRUE underlying behavioural arrays, not the
    # masked "_reported" ones below. A borrower's real repayment risk does
    # not depend on whether our alternative-data feed happened to observe
    # every signal -- only the applicant's stored, model-facing record
    # (and therefore the score) should reflect what was actually observed.
    reliability = np.nanmean(
        np.vstack([
            utility_ontime_ratio,
            rent_ontime_ratio,
            np.where(np.isnan(gst_filing_regularity), 0.7, gst_filing_regularity),
        ]), axis=0
    )
    bounce_good = 1 - np.clip(txn_bounce_rate / 0.3, 0, 1)
    vol_good = 1 - np.clip(inflow_volatility_cv / 0.8, 0, 1)
    growth_good = np.clip((inflow_growth_rate_6m + 0.3) / 0.9, 0, 1)
    vintage_good = np.clip(vintage_months / 60, 0, 1)
    burden_good = 1 - np.clip(repayment_burden_ratio / 1.0, 0, 1)
    supplier_good = 1 - np.where(np.isnan(supplier_concentration_hhi), 0.3, supplier_concentration_hhi)
    overdue_good = 1 - np.where(np.isnan(overdue_invoice_ratio), 0.2, np.clip(overdue_invoice_ratio / 0.5, 0, 1))
    bureau_bonus = np.where(~bureau_score_available, 0.5, bureau_score_norm)
    bureau_flag = bureau_score_available.astype(float)

    latent = (
        0.15 * reliability + 0.15 * bounce_good + 0.10 * vol_good + 0.08 * growth_good
        + 0.12 * vintage_good + 0.20 * burden_good + 0.07 * supplier_good + 0.06 * overdue_good
        + 0.05 * bureau_bonus + 0.02 * bureau_flag
    )
    latent = np.clip(latent + rng.normal(0, 0.015, size=n), 0, 1)
    prob_good = 1 / (1 + np.exp(-16 * (latent - 0.62)))
    repayment_outcome = (rng.random(n) < prob_good).astype(int)

    df = df.assign(
        vintage_months=vintage_months,
        digital_adoption_ratio=digital_adoption_ratio,
        avg_monthly_inflow=avg_monthly_inflow_reported,
        inflow_growth_rate_6m=inflow_growth_rate_6m,
        inflow_volatility_cv=inflow_volatility_cv,
        monthly_txn_count=monthly_txn_count_reported,
        txn_bounce_rate=txn_bounce_rate_reported,
        gst_registered=gst_registered.astype(int),
        gst_filing_regularity=gst_filing_regularity,
        overdue_invoice_ratio=overdue_invoice_ratio,
        utility_ontime_ratio=utility_ontime_ratio_reported,
        rent_ontime_ratio=rent_ontime_ratio_reported,
        supplier_concentration_hhi=supplier_concentration_hhi,
        repeat_supplier_ratio=repeat_supplier_ratio,
        existing_loan_count=existing_loan_count,
        existing_monthly_emi=existing_monthly_emi,
        bureau_score_available=bureau_score_available.astype(int),
        bureau_score_raw=np.where(bureau_score_available, bureau_score_raw, np.nan),
        bureau_score_norm=bureau_score_norm,
        requested_loan_amount=requested_loan_amount,
        requested_tenure_months=requested_tenure_months,
        proposed_emi=proposed_emi,
        net_cashflow_estimate=net_cashflow,
        repayment_burden_ratio=repayment_burden_ratio,
        repayment_outcome=repayment_outcome,
        latent_prob_good=prob_good,
    )

    return df


def process_all_applicants():
    print(f"Generating {config.N_APPLICANTS} synthetic applicants (seed={config.RANDOM_SEED})...")
    df = generate_synthetic_applicants()
    print(f"Generated {len(df)} rows. Positive (good-repayment) rate: {df['repayment_outcome'].mean():.2%}")

    try:
        conn = sqlite3.connect(config.DB_NAME)
        df.to_sql("applicants", conn, if_exists="replace", index=False)
        print(f"Saved {len(df)} applicant rows to 'applicants' table in {config.DB_NAME}.")
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    finally:
        conn.close()


if __name__ == "__main__":
    process_all_applicants()
# Updated on 2026-02-18
