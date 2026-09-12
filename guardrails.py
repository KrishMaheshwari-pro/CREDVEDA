"""
Responsible Lending Guardrails — flags applicants where an approval
would create an unsustainable repayment burden.
"""
import config


def compute_emi(principal: float, annual_rate_pct: float, tenure_months: int) -> float:
    """Standard EMI formula."""
    if tenure_months <= 0:
        return 0.0
    if annual_rate_pct <= 0:
        return principal / tenure_months
    r = annual_rate_pct / (12 * 100)
    return principal * r * (1 + r) ** tenure_months / ((1 + r) ** tenure_months - 1)


def max_affordable_loan(monthly_income: float, existing_emis: float,
                        tenure_months: int, annual_rate_pct: float) -> float:
    """Calculate maximum loan amount where total DTI stays under 50%."""
    max_total_emi = monthly_income * config.DTI_WARNING_THRESHOLD
    available_emi = max(0.0, max_total_emi - existing_emis)
    if available_emi <= 0 or annual_rate_pct <= 0:
        return 0.0
    r = annual_rate_pct / (12 * 100)
    return available_emi * ((1 + r) ** tenure_months - 1) / (r * (1 + r) ** tenure_months)


def check_repayment_burden(
    monthly_income: float,
    loan_amount: float,
    tenure_months: int,
    annual_rate_pct: float = 18.0,
    existing_emis: float = 0.0,
) -> dict:
    """
    Check if a proposed loan creates an unsustainable repayment burden.

    Args:
        monthly_income: applicant's monthly income/revenue (INR)
        loan_amount: requested loan amount (INR)
        tenure_months: loan tenure in months
        annual_rate_pct: annual interest rate (default 18% for MSME)
        existing_emis: sum of existing monthly EMI obligations

    Returns:
        dict with flag, severity, DTI, EMI, affordable amount, and messages
    """
    if monthly_income <= 0:
        return {
            "flagged": True,
            "severity": "CRITICAL",
            "message": "Monthly income is zero or missing — cannot assess repayment capacity.",
            "proposed_emi": 0,
            "total_dti": 1.0,
            "max_affordable_loan": 0,
        }

    proposed_emi = compute_emi(loan_amount, annual_rate_pct, tenure_months)
    total_emi = proposed_emi + existing_emis
    dti = total_emi / monthly_income
    affordable = max_affordable_loan(monthly_income, existing_emis, tenure_months, annual_rate_pct)

    if dti > config.DTI_CRITICAL_THRESHOLD:
        severity = "CRITICAL"
        flagged = True
        message = (
            f"CRITICAL: This loan would require {dti:.0%} of monthly income for repayments — "
            f"far above the safe limit of 50%. Maximum affordable loan at this income: "
            f"₹{affordable:,.0f}."
        )
        hindi_message = (
            f"चेतावनी: यह ऋण मासिक आय का {dti:.0%} चुकाने में लगेगा — "
            f"सुरक्षित सीमा 50% से बहुत अधिक। अधिकतम किफायती ऋण: ₹{affordable:,.0f}।"
        )
    elif dti > config.DTI_WARNING_THRESHOLD:
        severity = "WARNING"
        flagged = True
        message = (
            f"WARNING: Repayment burden ({dti:.0%} of income) is above the recommended 50% threshold. "
            f"Consider a smaller loan of ₹{affordable:,.0f} or a longer tenure."
        )
        hindi_message = (
            f"सावधानी: भुगतान बोझ (आय का {dti:.0%}) अनुशंसित 50% सीमा से अधिक है। "
            f"₹{affordable:,.0f} का छोटा ऋण या लंबी अवधि पर विचार करें।"
        )
    else:
        severity = "CLEAR"
        flagged = False
        message = (
            f"Repayment burden ({dti:.0%} of income) is within safe limits. "
            f"Proposed EMI: ₹{proposed_emi:,.0f}/month."
        )
        hindi_message = (
            f"भुगतान बोझ (आय का {dti:.0%}) सुरक्षित सीमा में है। "
            f"प्रस्तावित EMI: ₹{proposed_emi:,.0f}/माह।"
        )

    return {
        "flagged": flagged,
        "severity": severity,
        "proposed_emi": round(proposed_emi, 2),
        "existing_emis": round(existing_emis, 2),
        "total_monthly_obligation": round(total_emi, 2),
        "total_dti": round(dti, 4),
        "total_dti_pct": f"{dti:.1%}",
        "max_affordable_loan": round(affordable, 0),
        "monthly_income": monthly_income,
        "message": message,
        "hindi_message": hindi_message,
        "severity_color": {"CLEAR": "#28a745", "WARNING": "#ffc107", "CRITICAL": "#dc3545"}.get(severity, "#6c757d"),
    }


if __name__ == "__main__":
    cases = [
        (30000, 200000, 24, 18, 0),
        (20000, 400000, 36, 18, 5000),
        (15000, 100000, 18, 18, 0),
        (25000, 50000, 12, 18, 8000),
    ]
    for inc, loan, tenure, rate, emi in cases:
        r = check_repayment_burden(inc, loan, tenure, rate, emi)
        print(f"Income ₹{inc:,} | Loan ₹{loan:,} | {tenure}mo → DTI {r['total_dti_pct']} | {r['severity']}")
    print("guardrails.py OK")
