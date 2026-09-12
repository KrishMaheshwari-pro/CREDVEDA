"""
Reason Code Engine — maps SHAP feature contributions to standard RC codes.
Each reason code explains WHY a credit score was reduced, in plain language.
"""
import numpy as np

# RC-01 to RC-20: each maps a feature name to its reason code
REASON_CODE_MAP = {
    "upi_txn_consistency": {
        "code": "RC-01",
        "short": "Irregular digital payment activity",
        "borrower_text": "Your UPI payment activity is irregular. Consistent digital payments over 6+ months will improve your score.",
        "hindi_text": "आपकी UPI भुगतान गतिविधि अनियमित है। 6+ महीनों तक नियमित डिजिटल भुगतान करने से आपका स्कोर बेहतर होगा।",
        "lender_note": "Low consistency in digital transaction patterns — suggests informal or cash-based operations.",
        "actionable": True,
    },
    "gst_filing_consistency": {
        "code": "RC-02",
        "short": "Gaps in GST filing history",
        "borrower_text": "You have missed GST return filings. Filing on time for 3 consecutive quarters will significantly boost your score.",
        "hindi_text": "आपने GST रिटर्न दाखिल करने में चूक की है। लगातार 3 तिमाहियों तक समय पर दाखिल करने से आपका स्कोर काफी बेहतर होगा।",
        "lender_note": "GST filing gaps indicate compliance risk and potential revenue underreporting.",
        "actionable": True,
    },
    "revenue_volatility": {
        "code": "RC-03",
        "short": "High month-to-month revenue swings",
        "borrower_text": "Your income varies significantly month to month. Building a more stable customer base will reduce this risk signal.",
        "hindi_text": "आपकी आय महीने-दर-महीने काफी बदलती रहती है। एक स्थिर ग्राहक आधार बनाने से यह जोखिम कम होगा।",
        "lender_note": "High revenue volatility increases repayment risk, especially during business downturns.",
        "actionable": True,
    },
    "debt_to_monthly_revenue": {
        "code": "RC-04",
        "short": "High existing debt relative to income",
        "borrower_text": "Your current debt burden is high relative to your monthly income. Reducing existing obligations before applying will improve eligibility.",
        "hindi_text": "आपकी मौजूदा कर्ज की जिम्मेदारी आपकी मासिक आय के मुकाबले अधिक है। आवेदन से पहले मौजूदा देनदारियां कम करने से पात्रता बेहतर होगी।",
        "lender_note": "Debt-to-revenue ratio exceeds recommended threshold, indicating potential over-indebtedness.",
        "actionable": True,
    },
    "business_vintage_years": {
        "code": "RC-05",
        "short": "Recently established business",
        "borrower_text": "Your business is relatively new. As it matures, your credit profile will strengthen. Consider a smaller starter loan first.",
        "hindi_text": "आपका व्यवसाय अपेक्षाकृत नया है। जैसे-जैसे यह परिपक्व होगा, आपकी क्रेडिट प्रोफ़ाइल मजबूत होगी।",
        "lender_note": "Business vintage below 2 years — limited track record for repayment assessment.",
        "actionable": False,
    },
    "savings_balance_avg": {
        "code": "RC-06",
        "short": "Low average savings buffer",
        "borrower_text": "Your savings balance is low. Maintaining at least ₹5,000 in savings consistently for 3 months will improve your score.",
        "hindi_text": "आपकी बचत का स्तर कम है। 3 महीने तक लगातार कम से कम ₹5,000 बचाने से आपका स्कोर बेहतर होगा।",
        "lender_note": "Low savings buffer indicates limited financial resilience to absorb income shocks.",
        "actionable": True,
    },
    "utility_payment_score": {
        "code": "RC-07",
        "short": "Missed or late utility bill payments",
        "borrower_text": "You have missed or delayed utility bill payments. Paying all bills on time for the next 6 months will positively impact your score.",
        "hindi_text": "आपने उपयोगिता बिल भुगतान में देरी की है। अगले 6 महीनों तक सभी बिल समय पर चुकाने से आपका स्कोर बेहतर होगा।",
        "lender_note": "Utility payment delinquency is a strong predictor of loan default behaviour.",
        "actionable": True,
    },
    "invoice_count_monthly": {
        "code": "RC-08",
        "short": "Low business invoice volume",
        "borrower_text": "Your business generates few invoices. Growing your customer base and creating formal invoices for all transactions will help.",
        "hindi_text": "आपका व्यवसाय कम इनवॉइस उत्पन्न करता है। अपना ग्राहक आधार बढ़ाएं और सभी लेनदेन के लिए औपचारिक इनवॉइस बनाएं।",
        "lender_note": "Low invoice activity suggests limited verifiable business activity.",
        "actionable": True,
    },
    "upi_txn_monthly_avg": {
        "code": "RC-09",
        "short": "Low digital transaction frequency",
        "borrower_text": "You make very few digital transactions. Using UPI or card payments for business and personal expenses daily will build your digital footprint.",
        "hindi_text": "आप बहुत कम डिजिटल लेनदेन करते हैं। व्यापार और व्यक्तिगत खर्चों के लिए UPI का उपयोग करने से आपका डिजिटल रिकॉर्ड बनेगा।",
        "lender_note": "Very low UPI activity suggests cash-heavy operations with limited verifiable transaction history.",
        "actionable": True,
    },
    "customer_concentration": {
        "code": "RC-10",
        "short": "Revenue concentrated in few customers",
        "borrower_text": "Most of your revenue comes from very few customers. Diversifying your customer base reduces business risk.",
        "hindi_text": "आपकी अधिकांश आय बहुत कम ग्राहकों से आती है। अपने ग्राहक आधार में विविधता लाने से व्यावसायिक जोखिम कम होगा।",
        "lender_note": "High customer concentration creates dependency risk — loss of one client could cause repayment failure.",
        "actionable": True,
    },
    "supplier_count": {
        "code": "RC-11",
        "short": "Limited supplier relationships",
        "borrower_text": "You have very few suppliers. Building relationships with multiple suppliers demonstrates business depth and stability.",
        "hindi_text": "आपके पास बहुत कम आपूर्तिकर्ता हैं। कई आपूर्तिकर्ताओं के साथ संबंध बनाने से व्यापार की गहराई और स्थिरता दिखती है।",
        "lender_note": "Low supplier diversity may indicate limited business scale or informal procurement.",
        "actionable": True,
    },
    "digital_payment_ratio": {
        "code": "RC-12",
        "short": "Low adoption of traceable digital payments",
        "borrower_text": "Most of your transactions are cash-based. Switching to digital payments (UPI, card, bank transfer) creates a verifiable credit trail.",
        "hindi_text": "आपके अधिकांश लेनदेन नकद-आधारित हैं। डिजिटल भुगतान (UPI, कार्ड, बैंक ट्रांसफर) से क्रेडिट इतिहास बनता है।",
        "lender_note": "Low digital payment ratio limits verifiability of income and business activity.",
        "actionable": True,
    },
    "gst_revenue_trend": {
        "code": "RC-13",
        "short": "Declining GST-reported revenue",
        "borrower_text": "Your GST-reported revenue has been declining recently. Stabilising or growing your revenue will improve your credit profile.",
        "hindi_text": "आपका GST-रिपोर्टेड राजस्व हाल ही में घट रहा है। राजस्व को स्थिर या बढ़ाने से आपकी क्रेडिट प्रोफ़ाइल बेहतर होगी।",
        "lender_note": "Negative GST revenue trend signals business contraction and increased repayment risk.",
        "actionable": True,
    },
    "savings_regularity": {
        "code": "RC-14",
        "short": "Inconsistent savings behaviour",
        "borrower_text": "Your savings pattern is irregular. Setting aside a fixed amount every month, even a small one, demonstrates financial discipline.",
        "hindi_text": "आपकी बचत का पैटर्न अनियमित है। हर महीने एक निश्चित राशि बचाने से वित्तीय अनुशासन दिखता है।",
        "lender_note": "Inconsistent savings indicates limited financial planning capacity.",
        "actionable": True,
    },
    "rent_payment_score": {
        "code": "RC-15",
        "short": "Late or missed rent payments",
        "borrower_text": "You have missed or delayed rent payments. Paying rent on time is one of the clearest signals of your repayment commitment.",
        "hindi_text": "आपने किराया भुगतान में देरी की है। समय पर किराया चुकाना आपकी चुकाने की प्रतिबद्धता का सबसे स्पष्ट संकेत है।",
        "lender_note": "Rent payment delinquency strongly predicts loan default.",
        "actionable": True,
    },
    "existing_emi_ratio": {
        "code": "RC-16",
        "short": "High share of income committed to EMIs",
        "borrower_text": "A large portion of your income is already committed to EMIs. Consider clearing existing loans before taking additional credit.",
        "hindi_text": "आपकी आय का एक बड़ा हिस्सा पहले से EMI में चला जाता है। अतिरिक्त ऋण लेने से पहले मौजूदा कर्ज चुकाने पर विचार करें।",
        "lender_note": "High existing EMI ratio leaves little buffer for additional debt service.",
        "actionable": True,
    },
    "data_completeness_score": {
        "code": "RC-17",
        "short": "Insufficient data — low confidence score",
        "borrower_text": "We don't have enough information to give you a precise score. Providing bank statements, GST details, or utility bills will improve confidence.",
        "hindi_text": "हमारे पास आपको सटीक स्कोर देने के लिए पर्याप्त जानकारी नहीं है। बैंक स्टेटमेंट, GST विवरण, या उपयोगिता बिल प्रदान करने से विश्वास बढ़ेगा।",
        "lender_note": "Score has low confidence due to sparse data. Consider requesting additional documentation.",
        "actionable": True,
    },
    "months_of_data": {
        "code": "RC-18",
        "short": "Limited transaction history available",
        "borrower_text": "We only have a short history of your financial activity. Continue building your digital financial footprint over the next 6-12 months.",
        "hindi_text": "हमारे पास आपकी वित्तीय गतिविधि का केवल अल्पकालिक इतिहास है। अगले 6-12 महीनों में अपना डिजिटल वित्तीय रिकॉर्ड बनाते रहें।",
        "lender_note": "Short history limits predictive accuracy. Re-evaluate after 6+ months of data.",
        "actionable": False,
    },
    "state_credit_gap_index": {
        "code": "RC-19",
        "short": "Operating in underserved credit geography",
        "borrower_text": "Your location has historically had limited access to formal credit. This is a systemic factor, not a personal failing — we score you on your own merits.",
        "hindi_text": "आपके क्षेत्र में ऐतिहासिक रूप से औपचारिक ऋण तक सीमित पहुंच रही है। यह एक व्यवस्थागत कारक है, व्यक्तिगत कमी नहीं।",
        "lender_note": "Geographic credit gap may affect data availability and baseline default rates in this region.",
        "actionable": False,
    },
    "business_type_risk": {
        "code": "RC-20",
        "short": "Sector carries elevated credit risk",
        "borrower_text": "Your business sector has historically higher credit risk. Demonstrating consistent revenue and strong payment records will overcome this factor.",
        "hindi_text": "आपके व्यावसायिक क्षेत्र में ऐतिहासिक रूप से अधिक ऋण जोखिम रहा है। नियमित राजस्व और मजबूत भुगतान रिकॉर्ड से यह कारक दूर होगा।",
        "lender_note": "Sector-level risk adjustment applied based on historical default rates for this business type.",
        "actionable": False,
    },
}


def shap_to_reason_codes(shap_values, feature_names, top_n=5):
    """
    Convert SHAP values to standard reason codes.

    Args:
        shap_values: 1D array of SHAP values for one applicant
        feature_names: list of feature names corresponding to shap_values
        top_n: maximum number of reason codes to return

    Returns:
        list of dicts with reason code details, sorted by impact (most negative first)
    """
    if len(shap_values) != len(feature_names):
        return []

    pairs = list(zip(feature_names, shap_values))
    # Only negative contributions reduce the score
    negative = [(f, v) for f, v in pairs if v < 0]
    negative.sort(key=lambda x: x[1])  # most negative first

    result = []
    for feature, shap_val in negative[:top_n]:
        if feature in REASON_CODE_MAP:
            entry = REASON_CODE_MAP[feature].copy()
            entry["feature"] = feature
            entry["shap_value"] = float(shap_val)
            entry["score_impact"] = round(abs(shap_val) * 100, 1)
            result.append(entry)

    return result


def get_positive_signals(shap_values, feature_names, top_n=3):
    """Return top features that are HELPING the score (positive SHAP contributions)."""
    pairs = list(zip(feature_names, shap_values))
    positive = [(f, v) for f, v in pairs if v > 0]
    positive.sort(key=lambda x: x[1], reverse=True)

    result = []
    for feature, shap_val in positive[:top_n]:
        desc = REASON_CODE_MAP.get(feature, {}).get("short", feature.replace("_", " ").title())
        result.append({
            "feature": feature,
            "description": desc,
            "shap_value": float(shap_val),
            "score_impact": round(shap_val * 100, 1),
        })
    return result


if __name__ == "__main__":
    # Quick smoke test
    features = list(REASON_CODE_MAP.keys())
    fake_shap = np.random.randn(len(features)) * 0.1
    codes = shap_to_reason_codes(fake_shap, features)
    print(f"Sample reason codes ({len(codes)} returned):")
    for c in codes:
        print(f"  {c['code']}: {c['short']} (impact: -{c['score_impact']} pts)")
    print("reason_codes.py OK")
