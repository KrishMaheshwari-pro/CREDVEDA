"""
Improvement Path Generator — shows borrowers the 2-3 concrete actions
that would most improve their credit score over the next 6 months.
"""
import numpy as np

# Actionable improvement templates per feature
IMPROVEMENT_TEMPLATES = {
    "upi_txn_consistency": {
        "action": "Make at least 15 UPI payments per month consistently for the next 6 months.",
        "hindi_action": "अगले 6 महीनों तक हर महीने कम से कम 15 UPI भुगतान नियमित रूप से करें।",
        "timeline": "3–6 months",
        "difficulty": "Easy",
        "difficulty_color": "#28a745",
    },
    "upi_txn_monthly_avg": {
        "action": "Use UPI for all business collections and daily purchases to build transaction history.",
        "hindi_action": "सभी व्यावसायिक संग्रह और दैनिक खरीद के लिए UPI का उपयोग करें।",
        "timeline": "1–3 months",
        "difficulty": "Easy",
        "difficulty_color": "#28a745",
    },
    "digital_payment_ratio": {
        "action": "Switch at least 70% of your transactions to digital channels (UPI, NEFT, card).",
        "hindi_action": "अपने कम से कम 70% लेनदेन को डिजिटल माध्यमों (UPI, NEFT, कार्ड) पर स्थानांतरित करें।",
        "timeline": "2–4 months",
        "difficulty": "Easy",
        "difficulty_color": "#28a745",
    },
    "gst_filing_consistency": {
        "action": "File GST returns on time for 3 consecutive quarters. Set a calendar reminder for the 20th of each month.",
        "hindi_action": "लगातार 3 तिमाहियों तक GST रिटर्न समय पर दाखिल करें। हर महीने की 20 तारीख के लिए कैलेंडर अनुस्मारक सेट करें।",
        "timeline": "6–9 months",
        "difficulty": "Medium",
        "difficulty_color": "#ffc107",
    },
    "savings_balance_avg": {
        "action": "Maintain a minimum savings balance of ₹5,000 consistently for 3+ months.",
        "hindi_action": "3+ महीनों तक न्यूनतम ₹5,000 की बचत बैलेंस लगातार बनाए रखें।",
        "timeline": "3 months",
        "difficulty": "Easy",
        "difficulty_color": "#28a745",
    },
    "savings_regularity": {
        "action": "Set up an automatic monthly transfer of even ₹500–₹1,000 to a savings account on salary day.",
        "hindi_action": "वेतन दिवस पर बचत खाते में ₹500–₹1,000 का स्वचालित मासिक हस्तांतरण सेट करें।",
        "timeline": "3 months",
        "difficulty": "Easy",
        "difficulty_color": "#28a745",
    },
    "utility_payment_score": {
        "action": "Set up auto-pay for all utility bills (electricity, water, phone) to ensure zero missed payments.",
        "hindi_action": "सभी उपयोगिता बिलों (बिजली, पानी, फोन) के लिए ऑटो-पे सेट करें ताकि कोई भुगतान न चूके।",
        "timeline": "3–6 months",
        "difficulty": "Easy",
        "difficulty_color": "#28a745",
    },
    "rent_payment_score": {
        "action": "Pay rent via bank transfer or UPI and request a receipt — this creates a verifiable payment record.",
        "hindi_action": "बैंक ट्रांसफर या UPI के माध्यम से किराया दें और रसीद मांगें — यह एक सत्यापन योग्य भुगतान रिकॉर्ड बनाता है।",
        "timeline": "3–6 months",
        "difficulty": "Easy",
        "difficulty_color": "#28a745",
    },
    "invoice_count_monthly": {
        "action": "Create formal invoices for all business transactions using a free app like Vyapar or Zoho Invoice.",
        "hindi_action": "Vyapar या Zoho Invoice जैसे मुफ्त ऐप का उपयोग करके सभी व्यावसायिक लेनदेन के लिए औपचारिक इनवॉइस बनाएं।",
        "timeline": "2–4 months",
        "difficulty": "Easy",
        "difficulty_color": "#28a745",
    },
    "debt_to_monthly_revenue": {
        "action": "Prioritise clearing 1–2 small existing debts before applying for this loan. Reduces burden and improves score.",
        "hindi_action": "इस ऋण के लिए आवेदन करने से पहले 1–2 छोटे मौजूदा कर्जों को चुकाने को प्राथमिकता दें।",
        "timeline": "3–6 months",
        "difficulty": "Medium",
        "difficulty_color": "#ffc107",
    },
    "revenue_volatility": {
        "action": "Diversify your income sources or add recurring service contracts to reduce monthly revenue swings.",
        "hindi_action": "मासिक राजस्व में उतार-चढ़ाव को कम करने के लिए आय स्रोतों में विविधता लाएं या नियमित सेवा अनुबंध जोड़ें।",
        "timeline": "4–8 months",
        "difficulty": "Hard",
        "difficulty_color": "#dc3545",
    },
    "customer_concentration": {
        "action": "Add at least 3 new customers in the next 3 months to reduce dependence on any single client.",
        "hindi_action": "किसी एक ग्राहक पर निर्भरता कम करने के लिए अगले 3 महीनों में कम से कम 3 नए ग्राहक जोड़ें।",
        "timeline": "3–6 months",
        "difficulty": "Medium",
        "difficulty_color": "#ffc107",
    },
    "gst_revenue_trend": {
        "action": "Focus on growing monthly revenue and ensure all income is reported in GST returns.",
        "hindi_action": "मासिक राजस्व बढ़ाने पर ध्यान दें और सुनिश्चित करें कि सभी आय GST रिटर्न में रिपोर्ट हो।",
        "timeline": "4–6 months",
        "difficulty": "Hard",
        "difficulty_color": "#dc3545",
    },
    "mobile_recharge_regularity": {
        "action": "Set up auto-recharge for your mobile plan to ensure continuous connectivity — a positive signal of stability.",
        "hindi_action": "निरंतर कनेक्टिविटी सुनिश्चित करने के लिए अपने मोबाइल प्लान के लिए ऑटो-रिचार्ज सेट करें।",
        "timeline": "1–2 months",
        "difficulty": "Easy",
        "difficulty_color": "#28a745",
    },
    "supplier_count": {
        "action": "Register with 2–3 new suppliers via formal purchase orders to diversify your supply chain.",
        "hindi_action": "अपनी आपूर्ति श्रृंखला में विविधता लाने के लिए औपचारिक खरीद आदेशों के माध्यम से 2–3 नए आपूर्तिकर्ताओं के साथ पंजीकरण करें।",
        "timeline": "3–6 months",
        "difficulty": "Medium",
        "difficulty_color": "#ffc107",
    },
}

# Non-actionable features (demographic/structural — not shown as improvements)
NON_ACTIONABLE = {"business_vintage_years", "state_credit_gap_index", "business_type_risk",
                  "months_of_data", "data_completeness_score"}


def generate_improvement_path(shap_values, feature_names, current_score: int, top_n: int = 3) -> dict:
    """
    Generate the top N actionable improvement steps for a borrower.

    Args:
        shap_values: 1D array of SHAP values for one applicant
        feature_names: list of feature names
        current_score: current credit score (0-100)
        top_n: number of improvement steps to return

    Returns:
        dict with steps list and projected score after improvements
    """
    pairs = list(zip(feature_names, shap_values))
    # Filter to negative contributions in actionable features only
    actionable_negatives = [
        (f, v) for f, v in pairs
        if v < 0 and f in IMPROVEMENT_TEMPLATES
    ]
    actionable_negatives.sort(key=lambda x: x[1])  # most negative first

    steps = []
    projected_gain = 0.0

    for feature, shap_val in actionable_negatives:
        if len(steps) >= top_n:
            break
        template = IMPROVEMENT_TEMPLATES[feature].copy()
        # Conservative estimate: recovering 50% of the SHAP loss
        gain = abs(shap_val) * 100 * 0.5
        projected_gain += gain
        steps.append({
            "step": len(steps) + 1,
            "feature": feature,
            "action": template["action"],
            "hindi_action": template["hindi_action"],
            "timeline": template["timeline"],
            "difficulty": template["difficulty"],
            "difficulty_color": template["difficulty_color"],
            "estimated_score_gain": round(gain, 1),
        })

    projected_score = min(100, round(current_score + projected_gain))

    # 6-month trajectory: linear improvement across months
    trajectory = [current_score]
    monthly_gain = projected_gain / 6
    for m in range(1, 7):
        trajectory.append(min(100, round(current_score + monthly_gain * m)))

    return {
        "current_score": current_score,
        "projected_score": projected_score,
        "total_estimated_gain": round(projected_gain, 1),
        "steps": steps,
        "trajectory_months": list(range(0, 7)),
        "trajectory_scores": trajectory,
        "summary": (
            f"Following all {len(steps)} steps could increase your score by approximately "
            f"{round(projected_gain, 0):.0f} points over 6 months, "
            f"from {current_score} to {projected_score}."
        ),
        "hindi_summary": (
            f"सभी {len(steps)} कदमों का पालन करने से 6 महीनों में आपका स्कोर "
            f"लगभग {round(projected_gain, 0):.0f} अंक बढ़ सकता है, "
            f"{current_score} से {projected_score} तक।"
        ),
    }


if __name__ == "__main__":
    import numpy as np
    np.random.seed(0)
    features = list(IMPROVEMENT_TEMPLATES.keys()) + list(NON_ACTIONABLE)
    shap_vals = np.random.randn(len(features)) * 0.08
    result = generate_improvement_path(shap_vals, features, current_score=48)
    print(f"Current: {result['current_score']} → Projected: {result['projected_score']}")
    print(f"Steps: {len(result['steps'])}")
    for s in result['steps']:
        print(f"  Step {s['step']}: [{s['difficulty']}] {s['action'][:60]}...")
    print(f"Trajectory: {result['trajectory_scores']}")
    print("improvement_path.py OK")
