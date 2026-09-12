# config.py
"""
Central configuration for CredVeda: Alternative Credit Scoring for
Individuals and Small Businesses (Hackathon PS #4).

All "alternative data" here is realistic SYNTHETIC data (see data_ingestion.py).
Real deployments would swap the synthetic generator for live connectors to
UPI/Account-Aggregator transaction feeds, the GST portal, utility/rent
payment aggregators, and credit bureaus -- the rest of the pipeline
(feature engineering -> model -> scoring -> fairness -> guardrails) is
written to work unchanged against that real data since it consumes the
same feature schema defined below.

# Updated on 2026-02-18
"""

DB_NAME = "credit_intelligence.db"

# --- SYNTHETIC POPULATION ---
N_APPLICANTS = 4000
RANDOM_SEED = 42

ENTITY_TYPES = ["Individual", "Small Business"]

BUSINESS_TYPES_BY_ENTITY = {
    "Small Business": [
        "Retail / Kirana Store", "Textile & Apparel", "Food & Beverage",
        "Transport & Logistics", "Repair & Local Services", "Micro-Manufacturing",
        "Agri-allied Trading", "Wholesale Trading", "Salon & Personal Care",
        "Electronics & Mobile Repair",
    ],
    "Individual": [
        "Salaried Employee", "Gig / Platform Worker", "Freelance Professional",
        "Daily Wage Worker",
    ],
}

GENDERS = ["Male", "Female", "Other"]
GENDER_WEIGHTS = [0.56, 0.42, 0.02]

GEOGRAPHY_TIERS = ["Tier 1", "Tier 2", "Tier 3", "Rural"]
GEOGRAPHY_TIER_WEIGHTS = [0.22, 0.28, 0.28, 0.22]

STATES_BY_TIER = {
    "Tier 1": ["Maharashtra", "Karnataka", "Delhi NCR", "Tamil Nadu", "Telangana"],
    "Tier 2": ["Gujarat", "West Bengal", "Punjab", "Rajasthan", "Kerala", "Madhya Pradesh"],
    "Tier 3": ["Uttar Pradesh", "Bihar", "Odisha", "Chhattisgarh", "Jharkhand"],
    "Rural": ["Uttar Pradesh", "Bihar", "Madhya Pradesh", "Rajasthan", "Odisha"],
}

# --- MODEL FEATURES ---
# Behavioural / alternative-data features actually fed to the ML model.
# Protected attributes (gender, geography, business type, entity type) are
# deliberately EXCLUDED from this list -- they are used only downstream in
# fairness.py to audit the model's outcomes, never as a scoring input.
FEATURE_COLS = [
    "avg_monthly_inflow",
    "inflow_growth_rate_6m",
    "inflow_volatility_cv",
    "monthly_txn_count",
    "txn_bounce_rate",
    "digital_adoption_ratio",
    "gst_registered",
    "gst_filing_regularity",
    "overdue_invoice_ratio",
    "utility_ontime_ratio",
    "rent_ontime_ratio",
    "supplier_concentration_hhi",
    "repeat_supplier_ratio",
    "vintage_months",
    "existing_loan_count",
    "bureau_score_available",
    "bureau_score_norm",
    "repayment_burden_ratio",
]

# Fields that are structurally not applicable to an "Individual" applicant
# (no GST filings, no supplier book). Used to compute thin-file completeness
# fairly instead of penalising individuals for lacking business-only data.
BUSINESS_ONLY_FIELDS = [
    "gst_registered", "gst_filing_regularity", "overdue_invoice_ratio",
    "supplier_concentration_hhi", "repeat_supplier_ratio",
]

# Fields used to judge how "thin" a file is (data completeness -> confidence band)
COMPLETENESS_FIELDS = [
    "avg_monthly_inflow", "monthly_txn_count", "txn_bounce_rate",
    "gst_registered", "gst_filing_regularity", "utility_ontime_ratio",
    "rent_ontime_ratio", "supplier_concentration_hhi", "bureau_score_available",
]

# --- SCORING ---
SCORE_MIN = 300
SCORE_MAX = 900
APPROVAL_SCORE_THRESHOLD = 650  # illustrative cut-off used for fairness/approval-rate reporting

CONFIDENCE_BANDS = [
    # (min_completeness_inclusive, label, band_width_points)
    (0.75, "High confidence", 25),
    (0.50, "Medium confidence", 60),
    (0.0, "Low confidence (thin file)", 120),
]

# Below this completeness the PS's "confidence band rather than a false
# precision score" rule kicks in: the UI leads with the range, and the point
# estimate is demoted to a midpoint reference rather than being the headline.
BAND_FIRST_COMPLETENESS = 0.75

# Human-readable score tiers, so a number like 612 means something on sight.
# (min_score_inclusive, label, tone) -- tone maps to a UI colour.
SCORE_TIERS = [
    (750, "Excellent", "good"),
    (650, "Good", "good"),
    (550, "Fair", "warn"),
    (300, "Poor", "bad"),
]

ASSUMED_ANNUAL_INTEREST_RATE = 0.16  # illustrative reducing-balance rate used for EMI simulation

# --- RESPONSIBLE LENDING GUARDRAILS ---
# This ratio is the industry-standard FOIR (Fixed Obligation to Income Ratio)
# that Indian banks and NBFCs actually underwrite against: all fixed monthly
# obligations (existing EMIs + the proposed EMI) divided by assessable monthly
# income. Most lenders cap FOIR somewhere between 40% and 55%.
FOIR_CAUTION = 0.45
FOIR_HARD_STOP = 0.65
BURDEN_RATIO_CAUTION = FOIR_CAUTION      # retained aliases (same metric, older names)
BURDEN_RATIO_HARD_STOP = FOIR_HARD_STOP

# Income-shock stress test: real underwriting asks "does this loan survive a bad
# quarter?", not just "is it affordable today". We re-check FOIR assuming income
# falls by this much, and flag loans that are only affordable in good times.
STRESS_TEST_INCOME_DROP = 0.20

# --- PREDICTIVE UNCERTAINTY ---
# The confidence band blends two independent signals:
#   1. data completeness  -- how much of the applicant's file we actually have
#   2. model disagreement -- the spread of predictions across the forest's trees
# Tree disagreement is genuine epistemic uncertainty: where the trees diverge,
# the model is extrapolating rather than recognising a familiar pattern.
# Band width = standard error of the ensemble's predicted probability, widened
# to a ~95% interval (2 x 1.96 x SE) and mapped onto the 600-point score range:
#   2 * 1.96 * 600 ~= 2350
UNCERTAINTY_BAND_SCALE = 2350
UNCERTAINTY_BAND_MIN = 16         # never claim more precision than this
UNCERTAINTY_BAND_MAX = 220

# --- DATA CONSISTENCY / FRAUD SCREEN ---
# An unsupervised Isolation Forest flags applications whose numbers do not hang
# together the way genuine ones do (e.g. a large claimed inflow on a handful of
# monthly transactions). It never blocks a decision on its own -- it routes the
# file to a human, which is how a real lender treats a soft fraud signal.
ANOMALY_CONTAMINATION = 0.04

# --- LOAN LIFECYCLE ---
# What happens after approval: disbursal, an EMI schedule, and repayment
# behaviour that flows back into how the borrower is assessed next time.
LOAN_STATUSES = ["active", "closed", "defaulted"]
INSTALMENT_STATUSES = ["pending", "paid", "late", "missed"]

# A loan is written off once this many instalments have been missed outright.
DEFAULT_AFTER_MISSED = 3
# A missed payment is recoverable: borrowers catch up with this probability.
LATE_RECOVERY_PROB = 0.55

# --- REPAYMENT-HISTORY FEEDBACK ---
# On-platform repayment behaviour is applied as an explicit, auditable
# adjustment on top of the model score rather than being smuggled into a
# feature. This mirrors how lenders layer behavioural scorecards over an
# application score, and keeps the base model's explanation honest.
REPAYMENT_BONUS_PER_ONTIME = 6      # score points per on-time instalment
REPAYMENT_BONUS_CAP = 45
REPAYMENT_PENALTY_PER_LATE = 12
REPAYMENT_PENALTY_PER_MISSED = 35
REPAYMENT_PENALTY_CAP = -180

# --- REASON CODES ---
# Standard reason-code map: every SHAP driver resolves to one of these,
# each with a lending-ops-usable short label and a plain-language template
# for the applicant-facing explanation. `direction` says whether a HIGH
# value of the underlying feature is good (+) or bad (-) for the score.
REASON_CODES = {
    "RC01": {"feature": "utility_ontime_ratio", "direction": "+",
             "label": "Utility payment reliability",
             "positive": "Utility bills were paid on time {value:.0%} of the time over the last year.",
             "negative": "Utility bills were paid on time only {value:.0%} of the time over the last year."},
    "RC02": {"feature": "rent_ontime_ratio", "direction": "+",
             "label": "Rent payment reliability",
             "positive": "Rent/lease payments were made on time {value:.0%} of the time.",
             "negative": "Rent/lease payments were on time only {value:.0%} of the time."},
    "RC03": {"feature": "txn_bounce_rate", "direction": "-",
             "label": "Payment bounce rate",
             "positive": "Digital payments rarely bounce ({value:.1%} failure rate).",
             "negative": "A high share of digital payments bounced or failed ({value:.1%})."},
    "RC04": {"feature": "inflow_volatility_cv", "direction": "-",
             "label": "Cash flow stability",
             "positive": "Monthly cash inflows have been stable month to month.",
             "negative": "Monthly cash inflows swing significantly from month to month."},
    "RC05": {"feature": "inflow_growth_rate_6m", "direction": "+",
             "label": "Cash flow growth trend",
             "positive": "Monthly inflows grew {value:+.0%} over the last 6 months.",
             "negative": "The monthly inflow trend was negative over the last 6 months ({value:+.0%})."},
    "RC06": {"feature": "gst_filing_regularity", "direction": "+",
             "label": "GST filing regularity",
             "positive": "GST returns were filed on time {value:.0%} of the time.",
             "negative": "GST returns were filed on time only {value:.0%} of the time."},
    "RC07": {"feature": "overdue_invoice_ratio", "direction": "-",
             "label": "Invoice/receivables discipline",
             "positive": "Very few outstanding invoices are overdue ({value:.0%}).",
             "negative": "A large share of outstanding invoices are overdue ({value:.0%})."},
    "RC08": {"feature": "vintage_months", "direction": "+",
             "label": "Operating / relationship history",
             "positive": "{value:.0f} months of established operating history were on record.",
             "negative": "Only {value:.0f} months of operating history are on record (new to credit)."},
    "RC09": {"feature": "repayment_burden_ratio", "direction": "-",
             "label": "Existing debt burden",
             "positive": "Existing and proposed EMIs use a manageable {value:.0%} of estimated cash flow.",
             "negative": "Existing and proposed EMIs would use {value:.0%} of estimated cash flow, a heavy burden."},
    "RC10": {"feature": "supplier_concentration_hhi", "direction": "-",
             "label": "Supplier/buyer concentration",
             "positive": "Revenue is spread across a diversified set of suppliers/buyers.",
             "negative": "Revenue is concentrated in a small number of suppliers/buyers, a dependency risk."},
    "RC11": {"feature": "bureau_score_available", "direction": "+",
             "label": "Formal credit history availability",
             "positive": "A formal credit bureau record was available to corroborate this assessment.",
             "negative": "No formal credit bureau record exists; this is a thin-file applicant scored mainly on alternative data."},
    "RC12": {"feature": "digital_adoption_ratio", "direction": "+",
             "label": "Digital payment footprint",
             "positive": "A large share of transactions happen through traceable digital channels ({value:.0%}).",
             "negative": "Only {value:.0%} of transactions are traceable through digital channels, limiting visibility."},
    "RC13": {"feature": "existing_loan_count", "direction": "-",
             "label": "Existing loan count",
             "positive": "Few concurrent loan obligations are currently open ({value:.0f}).",
             "negative": "{value:.0f} concurrent loan obligations are already open, raising refinancing risk."},
    "RC14": {"feature": "monthly_txn_count", "direction": "+",
             "label": "Transaction activity level",
             "positive": "A healthy volume of monthly transactions ({value:.0f}) supports the assessment.",
             "negative": "Low monthly transaction volume ({value:.0f}) limits how much can be inferred."},
}

# Which FEATURE_COLS entries a borrower can realistically improve through
# their own behaviour (used by the "improvement path" simulator). Excludes
# structural facts like vintage_months or bureau_score_available.
ACTIONABLE_FEATURES = {
    "txn_bounce_rate":            {"direction": "decrease", "target_percentile": 0.10, "label": "Reduce bounced/failed digital payments"},
    "utility_ontime_ratio":       {"direction": "increase", "target_percentile": 0.90, "label": "Pay utility bills on time every month"},
    "rent_ontime_ratio":          {"direction": "increase", "target_percentile": 0.90, "label": "Pay rent/lease on time every month"},
    "gst_filing_regularity":      {"direction": "increase", "target_percentile": 0.90, "label": "File GST returns on schedule"},
    "overdue_invoice_ratio":      {"direction": "decrease", "target_percentile": 0.10, "label": "Collect outstanding invoices faster"},
    "inflow_volatility_cv":       {"direction": "decrease", "target_percentile": 0.20, "label": "Smooth out month-to-month cash flow"},
    "supplier_concentration_hhi": {"direction": "decrease", "target_percentile": 0.20, "label": "Diversify suppliers/buyers"},
    "repayment_burden_ratio":     {"direction": "decrease", "target_percentile": 0.20, "label": "Pay down existing loans before taking new debt"},
    "digital_adoption_ratio":     {"direction": "increase", "target_percentile": 0.90, "label": "Route more transactions through digital/UPI channels"},
}

# Groups audited for fairness (must be columns present on the applicants table
# but NEVER present in FEATURE_COLS).
FAIRNESS_GROUPS = ["gender", "geography_tier", "business_type", "entity_type"]
FAIRNESS_ADVERSE_IMPACT_THRESHOLD = 0.80  # four-fifths rule

# --- GUARDRAIL / DTI (used by guardrails.py) --------------------------------
DTI_WARNING_THRESHOLD = 0.45    # 45% total-obligation-to-income triggers WARNING
DTI_CRITICAL_THRESHOLD = 0.65   # 65% triggers CRITICAL (same as FOIR_HARD_STOP)

# --- THIN FILE (used by thin_file_handler.py) --------------------------------
THIN_FILE_COMPLETENESS_THRESHOLD = 0.50  # below this = thin file

# --- SCORE BANDS (for loan_product_matcher, lender_ops, score_report) -------
# Normalised 0-100 bands used by the new-module layer; the live scoring
# pipeline uses 300-900 (SCORE_MIN / SCORE_MAX).  Conversion: (x-300)/600*100
SCORE_BANDS = [
    {"min": 75, "max": 100, "label": "Excellent",         "color": "#22c55e"},
    {"min": 55, "max": 74,  "label": "Good",              "color": "#4f8cff"},
    {"min": 35, "max": 54,  "label": "Fair",              "color": "#f59e0b"},
    {"min": 0,  "max": 34,  "label": "Poor",              "color": "#ef4444"},
]

# --- CREDIT MARKETPLACE ------------------------------------------------------
# The marketplace lets lenders browse and co-fund vetted applicants, directly
# addressing the PS #4 problem: "working capital stays out of reach of viable
# businesses" because no single lender will take the first risk.
MARKETPLACE_MIN_SCORE = 620      # minimum 300-900 score to be listable
MARKETPLACE_MAX_LISTINGS = 200   # hard cap on open listings returned

# --- AI UNDERWRITER CHATBOT --------------------------------------------------
CHATBOT_MAX_TOKENS = 380         # Gemini response token budget per turn
