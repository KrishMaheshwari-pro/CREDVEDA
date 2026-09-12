"""
Loan Product Matcher — maps credit score bands to eligible Indian
government and NBFC loan schemes for MSMEs and individuals.
"""

LOAN_PRODUCTS = [
    {
        "id": "mudra_tarun",
        "name": "MUDRA Tarun",
        "full_name": "Pradhan Mantri MUDRA Yojana — Tarun",
        "min_score": 70,
        "max_loan_inr": 1_000_000,
        "interest_rate": "8.5%–12% p.a.",
        "tenure": "Up to 5 years",
        "collateral": "None required",
        "eligibility": "Non-corporate, non-farm micro/small businesses; 2+ years vintage",
        "best_for": ["Small Manufacturer / Artisan", "Service Business (Salon/Repair/etc)", "Trader / Wholesaler"],
        "description": "Up to ₹10 lakh for established micro-enterprises with strong credit profile.",
        "apply_via": "Any scheduled bank, MFI, or NBFC",
        "tag_color": "#28a745",
        "tag": "Best Match",
    },
    {
        "id": "mudra_kishore",
        "name": "MUDRA Kishore",
        "full_name": "Pradhan Mantri MUDRA Yojana — Kishore",
        "min_score": 50,
        "max_loan_inr": 500_000,
        "interest_rate": "9%–14% p.a.",
        "tenure": "Up to 5 years",
        "collateral": "None required",
        "eligibility": "Non-corporate micro businesses; 1+ year vintage",
        "best_for": ["Street Vendor / Micro Retail", "Service Business (Salon/Repair/etc)", "Gig Worker / Freelancer"],
        "description": "₹50,000 to ₹5 lakh for growing micro-enterprises with good standing.",
        "apply_via": "Scheduled banks, MFIs, or PM Jan Dhan linked bank",
        "tag_color": "#17a2b8",
        "tag": "Good Fit",
    },
    {
        "id": "mudra_shishu",
        "name": "MUDRA Shishu",
        "full_name": "Pradhan Mantri MUDRA Yojana — Shishu",
        "min_score": 30,
        "max_loan_inr": 50_000,
        "interest_rate": "10%–12% p.a.",
        "tenure": "Up to 3 years",
        "collateral": "None required",
        "eligibility": "Any non-farm micro-enterprise; new or existing",
        "best_for": ["Street Vendor / Micro Retail", "Farmer / Agri-Allied", "Gig Worker / Freelancer"],
        "description": "Up to ₹50,000 for early-stage micro-enterprises — the entry-level MUDRA loan.",
        "apply_via": "Any bank, NBFC, or MFI",
        "tag_color": "#ffc107",
        "tag": "Starter Loan",
    },
    {
        "id": "pm_svanidhi",
        "name": "PM SVANidhi",
        "full_name": "PM Street Vendor's AtmaNirbhar Nidhi",
        "min_score": 20,
        "max_loan_inr": 50_000,
        "interest_rate": "7% p.a. (subsidised)",
        "tenure": "1 year (rolling, up to ₹50K in 3rd cycle)",
        "collateral": "None required",
        "eligibility": "Street vendors with town vending certificate or letter of recommendation",
        "best_for": ["Street Vendor / Micro Retail"],
        "description": "Subsidised working capital loan for street vendors. Interest rebate on timely repayment.",
        "apply_via": "PM SVANidhi portal (pmsvanidhi.mohua.gov.in) or ULB",
        "tag_color": "#6f42c1",
        "tag": "Government Scheme",
    },
    {
        "id": "cgtmse",
        "name": "CGTMSE",
        "full_name": "Credit Guarantee Fund Trust for Micro and Small Enterprises",
        "min_score": 65,
        "max_loan_inr": 20_000_000,
        "interest_rate": "Market rate (collateral-free guarantee)",
        "tenure": "Up to 7 years",
        "collateral": "None (guarantee covers up to 85% of loan amount)",
        "eligibility": "Micro and Small Enterprises; new or existing",
        "best_for": ["Small Manufacturer / Artisan", "Trader / Wholesaler", "Service Business (Salon/Repair/etc)"],
        "description": "Credit guarantee scheme enabling collateral-free loans up to ₹2 crore. Lender applies on your behalf.",
        "apply_via": "Through member lending institutions (banks, NBFCs)",
        "tag_color": "#28a745",
        "tag": "High Limit",
    },
    {
        "id": "kisan_credit",
        "name": "Kisan Credit Card",
        "full_name": "Kisan Credit Card Scheme",
        "min_score": 35,
        "max_loan_inr": 300_000,
        "interest_rate": "4%–7% p.a. (with interest subvention)",
        "tenure": "Revolving credit; annual renewal",
        "collateral": "Land records (or none for <₹1.6L)",
        "eligibility": "Farmers, fishermen, animal husbandry practitioners",
        "best_for": ["Farmer / Agri-Allied"],
        "description": "Revolving credit for agricultural inputs, crop production, and allied activities. Interest subvention available.",
        "apply_via": "Cooperative banks, RRBs, scheduled commercial banks",
        "tag_color": "#28a745",
        "tag": "Agriculture",
    },
    {
        "id": "standup_india",
        "name": "Stand-Up India",
        "full_name": "Stand-Up India Scheme",
        "min_score": 60,
        "max_loan_inr": 10_000_000,
        "interest_rate": "Base rate + 3% + tenor premium",
        "tenure": "Up to 7 years (18-month moratorium)",
        "collateral": "Third-party guarantee or collateral",
        "eligibility": "SC/ST or women entrepreneurs setting up greenfield enterprises",
        "best_for": ["Small Manufacturer / Artisan", "Service Business (Salon/Repair/etc)", "Trader / Wholesaler"],
        "description": "₹10 lakh to ₹1 crore for SC/ST or women entrepreneurs in manufacturing, services, or trading.",
        "apply_via": "Scheduled commercial banks; standupmitra.in portal",
        "tag_color": "#6f42c1",
        "tag": "Women / SC-ST",
    },
]


def get_eligible_products(score: int, business_type: str = None, gender: str = None) -> list:
    """
    Return loan products eligible for this applicant, ranked by best match.

    Args:
        score: credit score (0-100)
        business_type: applicant's business segment
        gender: applicant's gender (for Stand-Up India eligibility)

    Returns:
        list of eligible products, sorted by max_loan_inr descending
    """
    eligible = []
    for product in LOAN_PRODUCTS:
        if score < product['min_score']:
            continue

        # Stand-Up India: only for women or SC/ST (we approximate by gender here)
        if product['id'] == 'standup_india' and gender not in ('Female', None):
            continue

        # PM SVANidhi: only for street vendors
        if product['id'] == 'pm_svanidhi' and business_type and 'Vendor' not in business_type:
            continue

        # Kisan Credit Card: only for farmers
        if product['id'] == 'kisan_credit' and business_type and 'Farmer' not in business_type:
            continue

        match_score = 0
        if business_type and business_type in product.get('best_for', []):
            match_score += 2
        if score >= product['min_score'] + 15:
            match_score += 1

        p = product.copy()
        p['match_score'] = match_score
        p['max_loan_display'] = f"INR {product['max_loan_inr']:,}"
        eligible.append(p)

    eligible.sort(key=lambda x: (-x['match_score'], -x['max_loan_inr']))
    return eligible


def get_score_band_summary(score: int) -> dict:
    """Return the score band label and description for a given score."""
    import config
    for band in config.SCORE_BANDS:
        if band['min'] <= score <= band['max']:
            return band
    return config.SCORE_BANDS[-1]


if __name__ == "__main__":
    import sys
    test_cases = [
        (82, "Small Manufacturer / Artisan", "Male"),
        (62, "Service Business (Salon/Repair/etc)", "Female"),
        (45, "Street Vendor / Micro Retail", "Male"),
        (28, "Farmer / Agri-Allied", "Male"),
    ]
    for score, btype, gender in test_cases:
        products = get_eligible_products(score, btype, gender)
        sys.stdout.write(f"Score {score} | {btype}: {len(products)} eligible products\n")
        sys.stdout.flush()
        for p in products[:2]:
            product_info = f"  > {p['name']}: {p['max_loan_display']} @ {p['interest_rate']}\n"
            sys.stdout.write(product_info)
            sys.stdout.flush()
    sys.stdout.write("loan_product_matcher.py OK\n")
    sys.stdout.flush()
