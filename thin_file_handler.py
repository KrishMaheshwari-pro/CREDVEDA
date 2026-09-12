"""
Thin File Handler — returns confidence bands instead of false precision
when applicant data is sparse or history is short.
"""

import config


def get_confidence_band(score: int, completeness: float, months_of_data: float) -> dict:
    """
    Compute confidence band around a credit score.

    Args:
        score: point estimate credit score (0-100)
        completeness: data_completeness_score (0.0-1.0)
        months_of_data: months of transaction history available

    Returns:
        dict with score, bounds, confidence label, and explanation
    """
    # Base uncertainty from missing data (max ±20 pts when completely empty)
    missing_penalty = (1.0 - max(0.0, min(1.0, completeness))) * 20.0

    # Time penalty shrinks as history grows (asymptotes at 18 months)
    time_factor = max(0.0, 1.0 - months_of_data / 18.0)
    time_penalty = time_factor * 10.0

    uncertainty = missing_penalty + time_penalty

    lower = max(0, round(score - uncertainty))
    upper = min(100, round(score + uncertainty))

    if uncertainty <= 5:
        confidence = "HIGH"
        confidence_color = "#28a745"
        note = f"Score is highly reliable based on {months_of_data:.0f} months of complete data."
    elif uncertainty <= 12:
        confidence = "MEDIUM"
        confidence_color = "#ffc107"
        note = f"Score has moderate confidence. {_missing_data_tip(completeness, months_of_data)}"
    else:
        confidence = "LOW"
        confidence_color = "#dc3545"
        note = f"Score range is wide due to limited data. {_missing_data_tip(completeness, months_of_data)}"

    missing_fields = _identify_missing_fields(completeness)

    return {
        "score": score,
        "lower_bound": lower,
        "upper_bound": upper,
        "uncertainty": round(uncertainty, 1),
        "confidence": confidence,
        "confidence_color": confidence_color,
        "data_quality_note": note,
        "missing_fields_hint": missing_fields,
        "completeness_pct": round(completeness * 100, 1),
        "months_of_data": months_of_data,
    }


def _missing_data_tip(completeness: float, months: float) -> str:
    tips = []
    if completeness < 0.6:
        tips.append("Providing GST details, bank statements, or utility bills will narrow the range.")
    if months < 6:
        tips.append(f"Only {months:.0f} months of history available — re-apply after 6+ months.")
    if not tips:
        tips.append("Adding more financial documents will increase confidence.")
    return " ".join(tips)


def _identify_missing_fields(completeness: float) -> str:
    if completeness >= 0.9:
        return "All key data fields are present."
    elif completeness >= 0.7:
        return "Consider adding: utility bills, rent payment record."
    elif completeness >= 0.5:
        return "Missing: GST filing history, invoice records, utility payments, rent record."
    else:
        return "Many fields missing. Key additions: bank statements, GST returns, utility bills, rent receipts, supplier invoices."


def is_thin_file(completeness: float, months_of_data: float) -> bool:
    """Returns True if this applicant qualifies as a thin-file case."""
    return (
        completeness < config.THIN_FILE_COMPLETENESS_THRESHOLD
        or months_of_data < 3
    )


def first_loan_pathway(score: int, completeness: float) -> dict | None:
    """
    For very thin-file applicants (completeness < 0.30), suggest a micro-credit pathway
    instead of a flat rejection.
    """
    if completeness >= 0.30:
        return None
    return {
        "eligible": True,
        "pathway": "First Loan Pathway",
        "step1": "Qualify for a 3-month micro-credit pilot: ₹5,000–₹10,000 with weekly repayments.",
        "step2": "Successful repayment auto-qualifies you for a larger loan evaluation.",
        "step3": "Build 6 months of digital payment history to unlock standard loan products.",
        "message": (
            "We don't have enough data to give you a full score yet — "
            "but that doesn't mean no. Start with our micro-credit pathway to build your credit identity."
        ),
        "hindi_message": (
            "हमारे पास अभी पूरा स्कोर देने के लिए पर्याप्त डेटा नहीं है — "
            "लेकिन इसका मतलब 'ना' नहीं है। अपनी क्रेडिट पहचान बनाने के लिए हमारे माइक्रो-क्रेडिट पाथवे से शुरुआत करें।"
        ),
    }


if __name__ == "__main__":
    # Smoke tests
    tests = [
        (72, 0.95, 18),
        (55, 0.70, 8),
        (40, 0.40, 3),
        (30, 0.15, 1),
    ]
    for score, comp, months in tests:
        r = get_confidence_band(score, comp, months)
        print(f"Score {score} | completeness {comp:.0%} | {months}mo → [{r['lower_bound']}-{r['upper_bound']}] ({r['confidence']})")
    print("thin_file_handler.py OK")
