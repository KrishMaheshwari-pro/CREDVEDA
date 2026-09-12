"""
Seed demo data for CredVeda: a few lender accounts, a couple of borrower
listings (with the bank's own competing direct offer), and lender commitments
that carry different rates and collateral choices — so the borrower's passport
shows a real "bank offer vs marketplace" comparison and the bank's proposals
inbox has something to act on.

Idempotent: it deletes its own previously-seeded rows (by the DEMO_ tag) before
re-inserting, so it can be run repeatedly. Run:  python seed_demo.py
"""
import hashlib
import sqlite3
import uuid
from datetime import datetime, timezone

import config
import scoring

NOW = datetime.now(timezone.utc).isoformat()


def h(pw):
    return hashlib.sha256(pw.encode()).hexdigest()


# (username, password, role)
DEMO_USERS = [
    ("bank_demo",     "demo1234", "bank"),
    ("lender_ramesh", "demo1234", "lender"),
    ("lender_priya",  "demo1234", "lender"),
    ("lender_acme",   "demo1234", "lender"),
    # Small lenders, for the many-pieces listing below. A co-funded loan can be
    # split across any number of them, each at its own rate, and the borrower
    # needs to see every slice -- two lenders never showed that off properly.
    ("lender_meera",  "demo1234", "lender"),
    ("lender_arjun",  "demo1234", "lender"),
    ("lender_fatima", "demo1234", "lender"),
    ("lender_vikram", "demo1234", "lender"),
    ("lender_nitin",  "demo1234", "lender"),
    ("lender_sunita", "demo1234", "lender"),
    ("lender_kabir",  "demo1234", "lender"),
]

# Two borrowers to list. Each: (applicant_id, amount, tenure, purpose, floor,
# bank_offer_rate, bank_collateral, bank_collateral_detail, [commitments])
# commitment = (lender, amount, rate, collateral_required)
LISTINGS = [
    {
        "applicant_id": "APP-00425", "amount": 168000, "tenure": 36,
        "purpose": "Shop expansion", "score": 847,
        "entity_type": "Small Business", "business_type": "Retail / Kirana Store",
        "geography_tier": "Tier 2", "floor": 12.0,
        "bank_offer_rate": 15.5, "bank_collateral": 1,
        "bank_collateral_detail": "Property / gold worth 120% of loan",
        "commitments": [
            ("lender_ramesh", 100000, 11.9, 0),   # at floor, unsecured — instant
            ("lender_priya",   68000, 12.0, 0),    # at floor, unsecured — instant
        ],
    },
    {
        "applicant_id": "APP-01820", "amount": 142000, "tenure": 48,
        "purpose": "Equipment / machinery", "score": 866,
        "entity_type": "Small Business", "business_type": "Food & Beverage",
        "geography_tier": "Tier 1", "floor": 11.0,
        "bank_offer_rate": 14.0, "bank_collateral": 1,
        "bank_collateral_detail": "Hypothecation of kitchen equipment",
        "commitments": [
            ("lender_acme",   90000, 11.0, 0),     # floor, unsecured
            ("lender_priya",  52000, 11.5, 1),     # slightly above floor, secured -> pending
        ],
    },
    {
        "applicant_id": "APP-01981", "amount": 160000, "tenure": 36,
        "purpose": "Working capital", "score": 846,
        "entity_type": "Small Business", "business_type": "Repair & Local Services",
        "geography_tier": "Tier 2", "floor": 11.5,
        "bank_offer_rate": 15.0, "bank_collateral": 1,
        "bank_collateral_detail": "Two-wheeler / equipment hypothecation",
        "commitments": [
            ("lender_ramesh", 80000, 11.5, 0),     # floor, unsecured -> active
            ("lender_acme",   80000, 12.4, 1),     # above floor, secured -> pending (bank inbox)
        ],
    },
    {
        # The many-pieces showcase. Both options cover the full ₹2.5L, but the
        # bank does it as one loan at one rate while the marketplace does it as
        # ten slices at ten different rates -- which is exactly what the
        # borrower's funding breakdown is there to make legible. Every rate is
        # at or under the floor, so all ten land as active commitments.
        "applicant_id": "APP-00183", "amount": 250000, "tenure": 36,
        "purpose": "Inventory & working capital", "score": 810,
        "entity_type": "Small Business", "business_type": "Wholesale Trading",
        "geography_tier": "Tier 1", "floor": 13.0,
        "bank_offer_rate": 15.75, "bank_collateral": 1,
        "bank_collateral_detail": "Stock hypothecation + personal guarantee",
        "commitments": [
            ("lender_meera",  40000, 11.40, 0),
            ("lender_ramesh", 35000, 11.75, 0),
            ("lender_priya",  32000, 12.00, 0),
            ("lender_acme",   30000, 12.10, 0),
            ("lender_arjun",  28000, 12.25, 0),
            ("lender_fatima", 25000, 12.40, 0),
            ("lender_vikram", 22000, 12.55, 1),
            ("lender_nitin",  18000, 12.70, 0),
            ("lender_sunita", 12000, 12.85, 0),
            ("lender_kabir",   8000, 13.00, 1),
        ],
    },
    {
        # Fresh presentation borrower. Fully funded on the marketplace at a
        # blended rate well under the bank's own direct offer, so the borrower's
        # fundings page shows a clean "bank vs marketplace" choice with both
        # offers acceptable.
        "applicant_id": "APP-03380", "amount": 127000, "tenure": 48,
        "purpose": "Salon expansion & equipment", "score": 844,
        "entity_type": "Small Business", "business_type": "Salon & Personal Care",
        "geography_tier": "Tier 2", "floor": 11.0,
        "bank_offer_rate": 14.5, "bank_collateral": 1,
        "bank_collateral_detail": "Equipment hypothecation + personal guarantee",
        "commitments": [
            ("lender_priya",  70000, 10.8, 0),     # below floor, unsecured -> active
            ("lender_meera",  57000, 11.0, 0),     # at floor, unsecured -> active (fully funds it)
        ],
    },
]


def main():
    conn = sqlite3.connect(config.DB_NAME)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # --- users -------------------------------------------------------------
    for u, pw, role in DEMO_USERS:
        cur.execute("DELETE FROM users WHERE username=?", (u,))
        try:
            cur.execute("INSERT INTO users (username, password, role) VALUES (?,?,?)", (u, h(pw), role))
        except sqlite3.OperationalError:
            cur.execute("INSERT INTO users (username, password) VALUES (?,?)", (u, h(pw)))

    # --- clear previously-seeded demo listings + their commitments ---------
    demo_ids = [l["applicant_id"] for l in LISTINGS]
    old = cur.execute(
        "SELECT listing_id FROM marketplace_listings WHERE applicant_id IN (%s)"
        % ",".join("?" * len(demo_ids)), demo_ids
    ).fetchall()
    for row in old:
        cur.execute("DELETE FROM lender_interests WHERE listing_id=?", (row["listing_id"],))
    cur.execute("DELETE FROM marketplace_listings WHERE applicant_id IN (%s)"
                % ",".join("?" * len(demo_ids)), demo_ids)

    # --- listings + commitments -------------------------------------------
    for i, l in enumerate(LISTINGS, 1):
        listing_id = f"DEMO-{i:03d}"
        tier_label, _ = scoring.score_tier(l["score"])
        cur.execute("""
            INSERT INTO marketplace_listings
            (listing_id, applicant_id, listed_by, listed_at, amount_requested, tenure_months,
             credit_score, tier_label, entity_type, business_type, geography_tier, purpose,
             status, total_committed, interest_rate, borrower_email, borrower_phone,
             bank_offer_rate, bank_offer_collateral, bank_offer_collateral_detail)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'open',0,?,?,?,?,?,?)
        """, (listing_id, l["applicant_id"], "bank_demo", NOW,
              l["amount"], l["tenure"], l["score"], tier_label,
              l["entity_type"], l["business_type"], l["geography_tier"], l["purpose"],
              l["floor"], "borrower@example.com", "9800000000",
              l["bank_offer_rate"], l["bank_collateral"], l["bank_collateral_detail"]))

        active_total = 0.0
        active_rows = []
        for lender, amt, rate, collat in l["commitments"]:
            status = "active" if rate <= l["floor"] else "pending"
            cur.execute("""
                INSERT INTO lender_interests
                (interest_id, listing_id, lender_username, committed_amount, proposed_rate,
                 message, collateral_required, status, created_at)
                VALUES (?,?,?,?,?,?,?,?,?)
            """, (str(uuid.uuid4())[:12], listing_id, lender, amt, rate,
                  "", collat, status, NOW))
            if status == "active":
                active_total += amt
                active_rows.append((amt, rate))

        blended = (sum(a * r for a, r in active_rows) / sum(a for a, r in active_rows)
                   if active_rows else l["floor"])
        new_status = "funded" if active_total >= l["amount"] else "open"
        cur.execute("UPDATE marketplace_listings SET total_committed=?, blended_rate=?, status=? WHERE listing_id=?",
                    (active_total, round(blended, 2), new_status, listing_id))

    conn.commit()
    conn.close()

    lenders = [u for u, _, role in DEMO_USERS if role == "lender"]
    print("Seeded demo data:")
    print("  Bank login:   bank_demo / demo1234")
    print(f"  Lenders ({len(lenders)}):  {', '.join(lenders)}  (all / demo1234)")
    print("  Borrower passports (public):")
    for l in LISTINGS:
        n = len(l["commitments"])
        print(f"    /passport/{l['applicant_id']}   ({l['business_type']}, score {l['score']}, "
              f"{n} lender piece{'s' if n != 1 else ''})")


if __name__ == "__main__":
    main()
