import sqlite3
import numpy as np
import pandas as pd
import uuid
import config

np.random.seed(42)
N = 50_000

STATES = list(config.STATE_CREDIT_GAP.keys())
STATE_WEIGHTS = [1 / (v + 0.1) for v in config.STATE_CREDIT_GAP.values()]
total_w = sum(STATE_WEIGHTS)
STATE_WEIGHTS = [w / total_w for w in STATE_WEIGHTS]

SEGMENTS = config.BORROWER_SEGMENTS

# Segment population weights
SEG_WEIGHTS = [0.28, 0.20, 0.20, 0.12, 0.10, 0.10]

GENDER_SPLIT = {"Male": 0.62, "Female": 0.32, "Other": 0.06}

def clip(arr, lo, hi):
    return np.clip(arr, lo, hi)

def generate_segment(seg, n):
    """Generate n records for a given borrower segment with realistic distributions."""
    rng = np.random.default_rng(abs(hash(seg)) % (2**31))

    # Segment-specific base parameters
    params = {
        "Street Vendor / Micro Retail": dict(
            vintage=(2, 1.5), revenue=(15000, 8000), upi_avg=(22, 12),
            gst_consistency=(0.3, 0.25), invoice_val=(2000, 1500),
            savings=(3000, 2500), digital_ratio=(0.55, 0.2),
            supplier=(3, 2), default_base=0.22
        ),
        "Small Manufacturer / Artisan": dict(
            vintage=(5, 3), revenue=(45000, 20000), upi_avg=(18, 10),
            gst_consistency=(0.65, 0.2), invoice_val=(12000, 8000),
            savings=(8000, 5000), digital_ratio=(0.45, 0.2),
            supplier=(8, 5), default_base=0.16
        ),
        "Service Business (Salon/Repair/etc)": dict(
            vintage=(4, 2.5), revenue=(30000, 15000), upi_avg=(35, 18),
            gst_consistency=(0.5, 0.25), invoice_val=(1500, 1000),
            savings=(6000, 4000), digital_ratio=(0.70, 0.18),
            supplier=(4, 3), default_base=0.14
        ),
        "Farmer / Agri-Allied": dict(
            vintage=(8, 4), revenue=(25000, 12000), upi_avg=(10, 8),
            gst_consistency=(0.1, 0.15), invoice_val=(8000, 5000),
            savings=(5000, 4000), digital_ratio=(0.25, 0.18),
            supplier=(5, 3), default_base=0.25
        ),
        "Gig Worker / Freelancer": dict(
            vintage=(1.5, 1), revenue=(22000, 12000), upi_avg=(40, 20),
            gst_consistency=(0.2, 0.2), invoice_val=(5000, 3000),
            savings=(4000, 3000), digital_ratio=(0.85, 0.12),
            supplier=(2, 2), default_base=0.24
        ),
        "Trader / Wholesaler": dict(
            vintage=(6, 3), revenue=(80000, 40000), upi_avg=(25, 15),
            gst_consistency=(0.75, 0.18), invoice_val=(25000, 15000),
            savings=(15000, 10000), digital_ratio=(0.55, 0.2),
            supplier=(15, 8), default_base=0.12
        ),
    }
    p = params[seg]

    vintage = clip(rng.normal(p["vintage"][0], p["vintage"][1], n), 0.25, 30)
    revenue = clip(rng.normal(p["revenue"][0], p["revenue"][1], n), 2000, 500000)
    upi_avg = clip(rng.normal(p["upi_avg"][0], p["upi_avg"][1], n), 0, 150)
    gst_cons = clip(rng.normal(p["gst_consistency"][0], p["gst_consistency"][1], n), 0, 1)
    invoice_val = clip(rng.normal(p["invoice_val"][0], p["invoice_val"][1], n), 500, 200000)
    savings = clip(rng.normal(p["savings"][0], p["savings"][1], n), 0, 100000)
    digital_ratio = clip(rng.normal(p["digital_ratio"][0], p["digital_ratio"][1], n), 0, 1)
    supplier_count = clip(rng.normal(p["supplier"][0], p["supplier"][1], n), 0, 50).astype(int)

    # Derived / correlated features
    upi_consistency = clip(1 - rng.exponential(0.15, n) * (1 / (upi_avg / 20 + 1)), 0, 1)
    upi_days_active = clip(upi_avg * rng.uniform(0.5, 0.9, n), 0, 30)
    gst_trend = clip(rng.normal(0.02, 0.08, n) + (vintage / 30) * 0.05, -0.3, 0.3)
    invoice_count = clip(rng.normal(p["invoice_val"][0] / 5000, 3, n), 0, 50)
    customer_conc = clip(rng.beta(2, 3, n), 0.1, 0.99)  # higher = more concentrated = worse
    rev_volatility = clip(rng.exponential(0.15, n) + (1 / (vintage + 1)) * 0.1, 0.01, 1)
    debt_to_rev = clip(rng.exponential(0.3, n), 0, 3)
    existing_emi_ratio = clip(rng.beta(1.5, 4, n), 0, 0.8)
    savings_regularity = clip(savings / (revenue * 0.5 + 1) + rng.normal(0, 0.1, n), 0, 1)
    utility_score = clip(rng.beta(5, 1.5, n) if p["digital_ratio"][0] > 0.5 else rng.beta(3, 2, n), 0, 1)
    rent_score = clip(rng.beta(4, 1.5, n), 0, 1)
    mobile_reg = clip(rng.beta(6, 1.5, n) if p["digital_ratio"][0] > 0.6 else rng.beta(3, 2, n), 0, 1)
    months_of_data = clip(rng.normal(np.minimum(vintage * 12, 24), 4, n), 1, 36).astype(int)

    # State assignment (weighted toward states with higher population density)
    states_arr = rng.choice(STATES, n, p=STATE_WEIGHTS)
    state_gap = np.array([config.STATE_CREDIT_GAP[s] for s in states_arr])
    biz_risk = config.BUSINESS_TYPE_RISK_MAP[seg]

    # Data completeness: farmers and street vendors have lower completeness
    base_completeness = 0.75 if p["gst_consistency"][0] > 0.4 else 0.55
    completeness = clip(rng.normal(base_completeness, 0.15, n), 0.2, 1.0)

    # Default label: logistic-ish combination of risk factors
    risk_score = (
        p["default_base"]
        - 0.08 * upi_consistency
        - 0.06 * gst_cons
        - 0.05 * savings_regularity
        - 0.04 * utility_score
        + 0.07 * debt_to_rev * 0.3
        + 0.05 * existing_emi_ratio
        + 0.03 * rev_volatility
        + 0.02 * state_gap
        - 0.03 * digital_ratio
        - 0.02 * (vintage / 10)
        + rng.normal(0, 0.05, n)
    )
    default_prob = clip(risk_score, 0.02, 0.95)
    loan_default = (rng.random(n) < default_prob).astype(int)

    # Gender
    genders = rng.choice(list(GENDER_SPLIT.keys()), n, p=list(GENDER_SPLIT.values()))
    ages = clip(rng.normal(38, 10, n), 21, 65).astype(int)

    # Loan request
    loan_amounts = clip(rng.lognormal(np.log(revenue * 2), 0.6, n), 10000, 2000000)
    tenure = rng.choice([12, 18, 24, 36, 48, 60], n, p=[0.1, 0.15, 0.3, 0.25, 0.12, 0.08])

    # Generate unique names (simple approach)
    first_names_m = ["Ravi", "Amit", "Suresh", "Rajesh", "Mahesh", "Dinesh", "Vijay", "Prakash", "Sanjay", "Ramesh", "Anil", "Mukesh", "Deepak", "Ajay", "Rakesh"]
    first_names_f = ["Priya", "Sunita", "Kavitha", "Meena", "Anita", "Geeta", "Rekha", "Seema", "Pooja", "Asha", "Usha", "Nisha", "Manju", "Shanti", "Lata"]
    last_names = ["Kumar", "Singh", "Sharma", "Verma", "Gupta", "Patel", "Shah", "Joshi", "Nair", "Pillai", "Reddy", "Rao", "Mishra", "Tiwari", "Yadav"]

    names = []
    for i in range(n):
        g = genders[i]
        fn_pool = first_names_m if g == "Male" else (first_names_f if g == "Female" else first_names_m + first_names_f)
        fn = rng.choice(fn_pool)
        ln = rng.choice(last_names)
        names.append(f"{fn} {ln}")

    df = pd.DataFrame({
        "app_id": [str(uuid.uuid4()) for _ in range(n)],
        "applicant_name": names,
        "gender": genders,
        "age": ages,
        "state": states_arr,
        "pin_code": [f"{rng.integers(100000, 999999)}" for _ in range(n)],
        "business_type": seg,
        "loan_amount_requested": loan_amounts.round(0),
        "loan_tenure_months": tenure,
        # 22 scoring features
        "upi_txn_monthly_avg": upi_avg.round(1),
        "upi_txn_consistency": upi_consistency.round(4),
        "upi_days_active_per_month": upi_days_active.round(1),
        "digital_payment_ratio": digital_ratio.round(4),
        "gst_filing_consistency": gst_cons.round(4),
        "gst_revenue_trend": gst_trend.round(4),
        "invoice_avg_value": invoice_val.round(0),
        "invoice_count_monthly": invoice_count.round(1),
        "business_vintage_years": vintage.round(2),
        "supplier_count": supplier_count,
        "customer_concentration": customer_conc.round(4),
        "monthly_revenue_avg": revenue.round(0),
        "revenue_volatility": rev_volatility.round(4),
        "debt_to_monthly_revenue": debt_to_rev.round(4),
        "existing_emi_ratio": existing_emi_ratio.round(4),
        "savings_balance_avg": savings.round(0),
        "savings_regularity": savings_regularity.round(4),
        "utility_payment_score": utility_score.round(4),
        "rent_payment_score": rent_score.round(4),
        "mobile_recharge_regularity": mobile_reg.round(4),
        "state_credit_gap_index": state_gap.round(4),
        "business_type_risk": biz_risk,
        # Meta
        "data_completeness_score": completeness.round(4),
        "months_of_data": months_of_data,
        # Ground truth
        "loan_default": loan_default,
    })
    return df


def generate_all():
    dfs = []
    for seg, weight in zip(SEGMENTS, SEG_WEIGHTS):
        n_seg = int(N * weight)
        print(f"Generating {n_seg} records for: {seg}")
        df = generate_segment(seg, n_seg)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    actual_n = len(combined)
    default_rate = combined["loan_default"].mean()
    print(f"\nGenerated {actual_n} total records")
    print(f"Overall default rate: {default_rate:.1%}")
    print(f"Segment breakdown:\n{combined['business_type'].value_counts()}")
    print(f"Gender breakdown:\n{combined['gender'].value_counts()}")
    print(f"Top states:\n{combined['state'].value_counts().head(8)}")

    conn = sqlite3.connect(config.DB_NAME)
    # Drop old data if re-running
    conn.execute("DELETE FROM loan_applications")
    combined.to_sql("loan_applications", conn, if_exists="append", index=False)
    conn.commit()
    conn.close()
    print(f"\nSaved {actual_n} records to loan_applications table.")


if __name__ == "__main__":
    generate_all()
