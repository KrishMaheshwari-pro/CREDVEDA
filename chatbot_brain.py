"""
Offline intelligence for the CredVeda assistant.

The assistant used to be two disconnected halves. The widget shipped a
hardcoded keyword matcher that added a fixed +180 to a fake counter no matter
what numbers you typed, and the server had a Gemini route whose fallback
returned canned paragraphs -- a route the widget never actually called.

This module is the replacement, and it needs no external API. The platform
already owns a trained model, per-applicant reason codes, an improvement path,
FOIR rules and a scheme matcher; a grounded assistant is mostly a matter of
reading the question and calling that machinery. Three layers:

  1. extract_signals() -- pulls real numbers out of Indian-style free text
     ("2.5 lakhs", "₹80,000", "50k", "30% digital", "bureau 650")
  2. detect_intent()   -- scores every intent and picks the best, so a question
     that mentions two topics resolves by weight rather than by whichever
     `if` happened to come first
  3. answer()          -- routes to a handler that computes a real reply from
     live data, and threads a running applicant profile through the session so
     follow-ups ("what if they also had GST?") re-score and report the delta

Everything returns plain HTML fragments; the widget renders them as-is.
"""
from __future__ import annotations

import re

import config
import scoring
from loan_product_matcher import get_eligible_products

# --- number parsing -----------------------------------------------------------

# Indian magnitude words. Written as (pattern, multiplier); crore before lakh so
# the longer word wins, and "k"/"thousand" last so it cannot eat "lakh".
_MAGNITUDES = [
    (r"cr(?:ore)?s?\b", 10_000_000),
    (r"lakh?s?\b|lacs?\b|l\b", 100_000),
    (r"thousand\b|k\b", 1_000),
]

_NUM = r"(\d+(?:[.,]\d+)*)"


def _to_float(tok: str) -> float:
    """'2,50,000' and '2.5' both have to survive. Commas are Indian grouping
    separators, never decimal points, so they are simply dropped."""
    try:
        return float(tok.replace(",", ""))
    except ValueError:
        return 0.0


def _amounts(text: str) -> list[tuple[int, float]]:
    """Every rupee-ish amount as (position, value), magnitude words applied.

    Positions are kept so callers can pick the amount nearest a keyword rather
    than the largest one -- "borrow 3 lakhs, existing emi 12000" must not read
    the loan size as the EMI.
    """
    out: list[tuple[int, float]] = []
    for pat, mult in _MAGNITUDES:
        # pat carries alternations, so it must be grouped -- ungrouped, the `|`
        # splits the whole expression and the magnitude word can match on its
        # own with no number captured.
        for m in re.finditer(_NUM + r"\s*(?:" + pat + r")", text):
            out.append((m.start(), _to_float(m.group(1)) * mult))
        text = re.sub(_NUM + r"\s*(?:" + pat + r")",
                      lambda m: " " * (m.end() - m.start()), text)
    for m in re.finditer(r"(?:₹|rs\.?|inr)\s*" + _NUM, text):
        out.append((m.start(), _to_float(m.group(1))))
    text = re.sub(r"(?:₹|rs\.?|inr)\s*" + _NUM, lambda m: " " * (m.end() - m.start()), text)
    for m in re.finditer(_NUM, text):
        v = _to_float(m.group(1))
        if v >= 1000:                       # bare small integers are not amounts
            out.append((m.start(), v))
    return sorted(out)


def _amount_near(text: str, keywords: tuple[str, ...], amounts: list[tuple[int, float]],
                 window: int = 45) -> tuple[int, float] | None:
    """The amount closest to any of `keywords`, within `window` characters.

    Returns (position, value) so the caller can mark it consumed. Amounts that
    follow the keyword are slightly favoured, because people write "wants 3
    lakhs" and "EMI 8000" -- the figure comes after the label. Without that
    tie-break, "existing emi 8000, wants 3 lakhs" reads the loan as 8000, the
    8000 being one character nearer to "wants" than "3 lakhs" is.
    """
    if not amounts:
        return None
    best, best_dist = None, float(window + 1)
    for kw in keywords:
        start = 0
        while (i := text.find(kw, start)) != -1:
            kw_end = i + len(kw)
            for pos, val in amounts:
                d = abs(pos - kw_end) + (0 if pos >= i else 6)   # penalise "before"
                if d < best_dist:
                    best, best_dist = (pos, val), d
            start = i + 1
    return best


def _pct(text: str, *keywords: str) -> float | None:
    """A percentage that sits near one of `keywords`, returned as 0-1.

    Accepts either order -- "30% digital" and "digital adoption is 30%" -- and
    also a bare fraction ("digital 0.3")."""
    for kw in keywords:
        for pat in (rf"{kw}[^.]{{0,25}}?{_NUM}\s*%", rf"{_NUM}\s*%[^.]{{0,25}}?{kw}"):
            m = re.search(pat, text)
            if m:
                return min(1.0, _to_float(m.group(1)) / 100.0)
        m = re.search(rf"{kw}[^.]{{0,20}}?(0?\.\d+)", text)
        if m:
            return min(1.0, _to_float(m.group(1)))
    return None


def extract_signals(text: str) -> dict:
    """Underwriting signals mentioned in one message.

    Only keys the user actually talked about are returned, so a follow-up turn
    overlays onto the running profile instead of resetting the untouched
    fields back to defaults.
    """
    t = " " + text.lower() + " "
    sig: dict = {}

    # Each of these takes the amount nearest its own keyword, and an amount is
    # claimed by only one field. Without that, one sentence carrying a revenue,
    # an EMI and a loan size lets the same figure be read as two of them.
    amounts = _amounts(t)
    for key, words in (
        ("existing_monthly_emi", ("emi", "instal", "repaying", "obligation")),
        ("requested_loan_amount", ("needs", "wants", "asking", "requested", "loan of",
                                   "borrow", "apply for")),
        ("avg_monthly_inflow", ("revenue", "turnover", "sales", "inflow", "income",
                                "earn", "makes", "making")),
    ):
        hit = _amount_near(t, words, amounts)
        if hit is not None:
            pos, val = hit
            sig[key] = val
            amounts = [(p, v) for p, v in amounts if p != pos]

    # Ratios.
    for key, words in (
        ("digital_adoption_ratio", ("digital", "upi", "online", "cashless")),
        ("utility_ontime_ratio", ("utility", "electricity", "bill")),
        ("rent_ontime_ratio", ("rent",)),
        # Deliberately not a bare "gst": that matched the nearest percentage in
        # the sentence, so "40% digital, GST filed" read as 40% filing regularity.
        ("gst_filing_regularity", ("gst filing", "gst return", "filing regularity",
                                   "filed on time", "gst compliance")),
        ("txn_bounce_rate", ("bounce", "cheque return", "nsf")),
        ("inflow_volatility_cv", ("volatil", "fluctuat", "irregular")),
    ):
        v = _pct(t, *words)
        if v is not None:
            sig[key] = v

    # Bureau score: a bare 300-900 integer near the word.
    m = re.search(r"(?:bureau|cibil|credit score|score of)\D{0,15}(\d{3})", t)
    if m:
        raw = int(m.group(1))
        if 300 <= raw <= 900:
            sig["bureau_score_available"] = 1
            sig["bureau_score_raw"] = raw

    # Vintage, in months.
    m = re.search(_NUM + r"\s*(year|yr|month|mo)", t)
    if m and any(w in t for w in ("vintage", "operating", "running", "business for",
                                  "trading", "old", "since", "experience")):
        n = _to_float(m.group(1))
        sig["vintage_months"] = n * 12 if m.group(2).startswith(("year", "yr")) else n

    # GST registration, stated either way.
    if re.search(r"\b(no|not|isn't|without|un)\w*\s+gst|gst\s*(:|=)?\s*(no|none)", t):
        sig["gst_registered"] = 0
    elif "gst" in t:
        sig["gst_registered"] = 1

    if any(w in t for w in ("individual", "salaried", "person", "worker", "employee")):
        sig["entity_type"] = "Individual"
    elif any(w in t for w in ("business", "shop", "kirana", "store", "trader", "sme", "msme")):
        sig["entity_type"] = "Small Business"

    return sig


# --- intent routing -----------------------------------------------------------

# Weighted cues per intent. Scoring every intent and taking the max beats an
# if/elif chain: "why is the score low and how do I improve it" mentions two
# topics, and the heavier match should win rather than the earliest branch.
_INTENTS: dict[str, list[tuple[str, float]]] = {
    "score_what_if": [("what if", 3), ("suppose", 2), ("instead", 1.5), ("change", 1),
                      ("increase", 1), ("raise", 1), ("would the score", 3), ("recalculate", 2)],
    "score_now":     [("score", 2), ("assess", 2), ("underwrite", 2), ("rate this", 2),
                      ("how much", 1), ("eligible", 1.5), ("qualify", 1.5)],
    "why_score":     [("why", 3), ("reason", 2.5), ("driver", 2), ("factor", 2),
                      ("explain", 2.5), ("what drove", 3), ("because", 1)],
    "improve":       [("improve", 3), ("better", 2), ("boost", 2.5), ("raise", 2),
                      ("increase the score", 3), ("fix", 1.5), ("advice", 1.5), ("how do i", 1.5)],
    "schemes":       [("scheme", 3), ("mudra", 3), ("cgtmse", 3), ("svanidhi", 3),
                      ("government", 2), ("subsidy", 2), ("product", 1.5), ("programme", 2),
                      ("program", 1.5)],
    "emi":           [("emi", 3), ("instal", 2), ("monthly payment", 3), ("afford", 2.5),
                      ("foir", 3), ("burden", 2), ("repay", 1.5)],
    "thin_file":     [("thin", 3), ("band", 2), ("range", 2), ("confidence", 2.5),
                      ("uncertain", 2), ("sparse", 2.5), ("no history", 2.5), ("new to credit", 3)],
    "fairness":      [("fair", 3), ("bias", 3), ("discriminat", 3), ("gender", 2.5),
                      ("caste", 2.5), ("religion", 2.5), ("geography", 2)],
    "marketplace":   [("marketplace", 3), ("lender", 2), ("co-fund", 3), ("syndicat", 3),
                      ("invest", 2), ("fund this", 2.5)],
    "data_needed":   [("what data", 3), ("what do you need", 3), ("which document", 3),
                      ("upload", 2), ("statement", 1.5), ("kyc", 2)],
    "help":          [("help", 2), ("what can you", 3), ("who are you", 3), ("hello", 2),
                      ("hi ", 1.5), ("hey", 1.5), ("start", 1)],
}


def detect_intent(text: str, n_signals: int, has_profile: bool) -> str:
    t = " " + text.lower().strip() + " "
    best, best_score = "help", 0.0
    for intent, cues in _INTENTS.items():
        s = sum(w for cue, w in cues if cue in t)
        if s > best_score:
            best, best_score = intent, s

    # Describing an applicant outranks a topical cue. "revenue 2.5L, 40%
    # digital, EMI 8000" carries the word "emi" and would otherwise route to
    # the FOIR explainer, throwing away every number in the sentence.
    if n_signals >= 2:
        return "score_what_if" if has_profile else "score_now"
    if n_signals == 1 and best_score < 3:
        return "score_what_if" if has_profile else "score_now"
    if best_score == 0:
        return "score_now" if n_signals else "help"
    return best


# --- profile handling ---------------------------------------------------------

_DEFAULTS = {
    "avg_monthly_inflow": 60000.0,
    "inflow_growth_rate_6m": 0.02,
    "inflow_volatility_cv": 0.20,
    "monthly_txn_count": 30,
    "txn_bounce_rate": 0.05,
    "digital_adoption_ratio": 0.45,
    "gst_registered": 0,
    "gst_filing_regularity": 0.80,
    "overdue_invoice_ratio": 0.10,
    "utility_ontime_ratio": 0.85,
    "rent_ontime_ratio": 0.85,
    "supplier_concentration_hhi": 0.30,
    "repeat_supplier_ratio": 0.60,
    "vintage_months": 24.0,
    "existing_loan_count": 0,
    "bureau_score_available": 0,
    "bureau_score_raw": 650,
    "existing_monthly_emi": 0.0,
    "requested_loan_amount": 100000.0,
    "requested_tenure_months": 24,
    "entity_type": "Small Business",
}


# Plausible range per field. A value can be finite and still be nonsense --
# 1e308 is a valid float that overflows float32 inside the model -- so every
# number is clamped into a range the scorer was actually trained on.
_BOUNDS = {
    "avg_monthly_inflow": (0.0, 1e9),
    "monthly_expenses": (0.0, 1e9),
    "existing_monthly_emi": (0.0, 1e9),
    "requested_loan_amount": (0.0, 1e9),
    "requested_tenure_months": (1.0, 480.0),
    "vintage_months": (0.0, 1200.0),
    "bureau_score_raw": (300.0, 900.0),
    "existing_loan_count": (0.0, 50.0),
    "monthly_txn_count": (0.0, 100000.0),
    "inflow_growth_rate_6m": (-1.0, 10.0),
}
_RATIOS = {
    "inflow_volatility_cv", "txn_bounce_rate", "digital_adoption_ratio",
    "gst_filing_regularity", "overdue_invoice_ratio", "utility_ontime_ratio",
    "rent_ontime_ratio", "supplier_concentration_hhi", "repeat_supplier_ratio",
}


def _sanitise(profile) -> dict:
    """Conversation state round-trips through the browser, so it arrives as
    whatever the client chose to send. Keep only known keys holding values the
    model can actually use; anything else is dropped rather than trusted."""
    if not isinstance(profile, dict):
        return {}
    clean = {}
    for k, v in profile.items():
        if k not in _DEFAULTS:
            continue
        if k == "entity_type":
            if v in ("Small Business", "Individual"):
                clean[k] = v
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if f != f or f in (float("inf"), float("-inf")):
            continue                      # NaN and infinity are not signals
        if k == "gst_registered":
            clean[k] = 1.0 if f > 0 else 0.0
            continue
        if k == "bureau_score_available":
            clean[k] = 1.0 if f > 0 else 0.0
            continue
        lo, hi = (0.0, 1.0) if k in _RATIOS else _BOUNDS.get(k, (0.0, 1e9))
        clean[k] = min(hi, max(lo, f))
    return clean


def _build_raw(profile: dict) -> tuple[dict, str, dict]:
    """Turn a conversational profile into the raw feature dict the model wants.

    Mirrors /api/simulate so the assistant and the simulator cannot drift into
    scoring the same inputs differently.
    """
    p = {**_DEFAULTS, **profile}
    entity_type = p["entity_type"]
    is_business = entity_type == "Small Business"

    inflow = max(float(p["avg_monthly_inflow"]), 1.0)
    expenses = float(p.get("monthly_expenses") or inflow * 0.65)
    net_cashflow = max(inflow - expenses, 1500.0)
    existing_emi = float(p["existing_monthly_emi"])
    loan = float(p["requested_loan_amount"])
    tenure = max(1, int(p["requested_tenure_months"]))
    proposed_emi = scoring.emi(loan, tenure)

    gst = 1 if (is_business and float(p["gst_registered"]) > 0) else 0
    bureau = float(p["bureau_score_available"]) > 0

    raw = {
        "avg_monthly_inflow": inflow,
        "inflow_growth_rate_6m": float(p["inflow_growth_rate_6m"]),
        "inflow_volatility_cv": float(p["inflow_volatility_cv"]),
        "monthly_txn_count": float(p["monthly_txn_count"]),
        "txn_bounce_rate": float(p["txn_bounce_rate"]),
        "digital_adoption_ratio": float(p["digital_adoption_ratio"]),
        "gst_registered": gst,
        "gst_filing_regularity": float(p["gst_filing_regularity"]) if gst else None,
        "overdue_invoice_ratio": float(p["overdue_invoice_ratio"]) if gst else None,
        "utility_ontime_ratio": float(p["utility_ontime_ratio"]),
        "rent_ontime_ratio": float(p["rent_ontime_ratio"]),
        "supplier_concentration_hhi": float(p["supplier_concentration_hhi"]) if is_business else None,
        "repeat_supplier_ratio": float(p["repeat_supplier_ratio"]) if is_business else None,
        "vintage_months": float(p["vintage_months"]),
        "existing_loan_count": int(p["existing_loan_count"]),
        "bureau_score_available": 1 if bureau else 0,
        "bureau_score_norm": (float(p["bureau_score_raw"]) - 300) / 600.0 if bureau else None,
        "repayment_burden_ratio": (existing_emi + proposed_emi) / net_cashflow,
        "requested_loan_amount": loan,
    }
    meta = {"proposed_emi": proposed_emi, "net_cashflow": net_cashflow, "loan": loan,
            "tenure": tenure, "existing_emi": existing_emi}
    return raw, entity_type, meta


# --- formatting helpers -------------------------------------------------------

def _inr(n) -> str:
    """Indian digit grouping: 12,34,567."""
    n = int(round(float(n)))
    s, neg = str(abs(n)), n < 0
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        head = re.sub(r"(\d)(?=(\d\d)+$)", r"\1,", head)
        s = head + "," + tail
    return ("-₹" if neg else "₹") + s


def _tone(score: int) -> str:
    if score >= 750:
        return "var(--good)"
    if score >= config.APPROVAL_SCORE_THRESHOLD:
        return "var(--warn)"
    return "var(--bad)"


def _score_line(score: int, tier: str) -> str:
    return (f'<b>Score:</b> <span style="color:{_tone(score)};font-weight:800;font-size:1.05em;">'
            f'{score}</span> <span style="opacity:.75;">/ {config.SCORE_MAX} · {tier}</span>')


def _bullets(items: list[str]) -> str:
    if not items:
        return ""
    li = "".join(f"<li style='margin:.15rem 0;'>{x}</li>" for x in items)
    return f"<ul style='margin:.4rem 0 0;padding-left:1.1rem;'>{li}</ul>"


def _assumed_note(profile: dict) -> str:
    """Be explicit about what was assumed. An underwriting number presented
    without saying which inputs were guessed is worse than no number."""
    known = {k for k in profile if k in _DEFAULTS}
    missing = [lbl for k, lbl in (
        ("avg_monthly_inflow", "monthly revenue"),
        ("digital_adoption_ratio", "digital share"),
        ("vintage_months", "business vintage"),
        ("utility_ontime_ratio", "bill punctuality"),
    ) if k not in known]
    if not missing:
        return ""
    return (f"<div style='margin-top:.5rem;font-size:.92em;opacity:.72;'>Assumed typical values for "
            f"{', '.join(missing)} — tell me the real figures and I'll re-score.</div>")


# --- answer handlers ----------------------------------------------------------

def _score_reply(profile: dict, previous: dict | None) -> str:
    raw, entity_type, meta = _build_raw(profile)
    res = scoring.score_full(raw, entity_type=entity_type)
    res.pop("_X_scaled", None)
    score, tier = res["credit_score"], res["tier_label"]

    head = ""
    prev_score = None
    if isinstance(previous, dict):
        try:                              # the client round-trips this, so it
            prev_score = int(previous.get("credit_score"))   # may be anything
        except (TypeError, ValueError):
            prev_score = None
    if prev_score is not None:
        delta = score - prev_score
        arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "▬")
        colour = "var(--good)" if delta > 0 else ("var(--bad)" if delta < 0 else "var(--text-dim)")
        head = (f"<div style='margin-bottom:.35rem;'>Re-scored with that change: "
                f"<span style='color:{colour};font-weight:700;'>{arrow} {delta:+d} points</span></div>")

    parts = [head, _score_line(score, tier)]

    if res.get("band_first"):
        parts.append(f"<div style='margin-top:.2rem;opacity:.8;'>Likely range {res['band_low']}–"
                     f"{res['band_high']} · {res['confidence_label']} "
                     f"({res['data_completeness'] * 100:.0f}% data).</div>")

    # `approved` is score >= cut-off AND no critical guardrail, so a strong
    # score can still decline. Saying "below the cut-off" for a 701 against a
    # cut-off of 650 is simply false -- name the real blocker instead.
    critical = [g for g in res.get("guardrail_flags", []) if g.get("severity") == "critical"]
    above_cutoff = score >= config.APPROVAL_SCORE_THRESHOLD
    if res["approved"]:
        decision, dcolour, why = "Approved", "var(--good)", ""
    elif critical and above_cutoff:
        # Anything described in chat is self-declared, so the verification
        # guardrail always fires. Presenting that as a flat decline reads as a
        # credit judgement when it is really "nothing here is documented yet".
        decision, dcolour = "Indicative only", "var(--warn)"
        why = (f" — clears the {config.APPROVAL_SCORE_THRESHOLD} cut-off, but nothing you have told me "
               f"is document-verified, so it cannot be a real decision")
    elif critical:
        decision, dcolour = "Declined", "var(--bad)"
        short = (critical[0].get("label") or critical[0].get("message", "")).split("—")[0].strip()
        why = f" — below the cut-off, and: {short[:90]}"
    else:
        decision, dcolour = "Below the approval cut-off", "var(--bad)"
        why = f" — needs {config.APPROVAL_SCORE_THRESHOLD}+"
    parts.append(f"<div style='margin-top:.35rem;'><b>Decision:</b> "
                 f"<span style='color:{dcolour};font-weight:700;'>{decision}</span>"
                 f"<span style='opacity:.75;'>{why}</span></div>")

    foir = res["foir"]
    parts.append(f"<div style='margin-top:.35rem;'><b>Affordability:</b> EMI {_inr(meta['proposed_emi'])}/mo "
                 f"on {_inr(meta['loan'])} over {meta['tenure']}m · FOIR {foir['foir']:.0%} "
                 f"{'✅' if foir['passes'] else '⚠️'}</div>")

    codes = [rc["label"] for rc in res.get("reason_codes", [])[:3]]
    if codes:
        parts.append("<div style='margin-top:.5rem;'><b>What drove it</b></div>" + _bullets(codes))

    parts.append(_assumed_note(profile))
    return "".join(p for p in parts if p), res


def _applicant_why(ctx: dict) -> str:
    pos = [rc["label"] for rc in ctx["reason_codes"] if rc.get("impact") == "positive"][:3]
    neg = [rc["label"] for rc in ctx["reason_codes"] if rc.get("impact") == "negative"][:3]
    out = [f"<b>{ctx['applicant_id']}</b> scores ",
           f"<span style='color:{_tone(ctx['credit_score'])};font-weight:800;'>{ctx['credit_score']}</span>",
           f" ({ctx['tier_label']}).<br>"]
    if pos:
        out.append("<div style='margin-top:.4rem;'><b style='color:var(--good);'>Helping</b></div>" + _bullets(pos))
    if neg:
        out.append("<div style='margin-top:.4rem;'><b style='color:var(--bad);'>Holding it back</b></div>" + _bullets(neg))
    if not pos and not neg:
        out.append("Reason codes have not been computed for this applicant yet — open the score report to generate them.")
    out.append("<div style='margin-top:.5rem;opacity:.78;'>Protected attributes (gender, geography, "
               "business type) are never model inputs.</div>")
    return "".join(out)


def _applicant_improve(ctx: dict) -> str:
    steps = ctx.get("improvement_path") or []
    if not steps:
        return ("No improvement path has been computed for this applicant yet. In general the fastest "
                "gains come from on-time utility and rent payments, a higher share of receipts through "
                "UPI, and regular GST filing.")
    items = []
    for s in steps[:3]:
        label = s.get("label") or s.get("feature", "")
        gain = s.get("expected_gain") or s.get("points") or s.get("delta")
        items.append(f"{label}" + (f" <span style='color:var(--good);font-weight:700;'>+{int(gain)} pts</span>"
                                   if gain else ""))
    total = sum(int(s.get("expected_gain") or s.get("points") or s.get("delta") or 0) for s in steps[:3])
    tail = (f"<div style='margin-top:.45rem;'>Together that is roughly "
            f"<b style='color:var(--good);'>+{total} points</b>, which would move "
            f"{ctx['applicant_id']} to about <b>{ctx['credit_score'] + total}</b>.</div>") if total else ""
    return (f"<b>Fastest gains for {ctx['applicant_id']}</b>" + _bullets(items) + tail)


def _applicant_emi(ctx: dict) -> str:
    foir = ctx.get("foir") or {}
    if not foir:
        return "No affordability assessment is available for this applicant yet."
    verdict = ("comfortably within" if foir.get("passes") else "above")
    colour = "var(--good)" if foir.get("passes") else "var(--bad)"
    return (f"<b>Affordability for {ctx['applicant_id']}</b><br>"
            f"FOIR is <span style='color:{colour};font-weight:700;'>{foir.get('foir', 0):.0%}</span>, "
            f"{verdict} the {config.FOIR_CAUTION:.0%} caution line "
            f"(hard stop {config.FOIR_HARD_STOP:.0%}).<br>"
            f"<span style='opacity:.78;'>FOIR is the share of net monthly cash flow already going to "
            f"loan repayments — it caps lending regardless of how strong the score is.</span>")


def _applicant_schemes(ctx: dict) -> str:
    score_norm = int((ctx["credit_score"] - config.SCORE_MIN) /
                     (config.SCORE_MAX - config.SCORE_MIN) * 100)
    prods = get_eligible_products(score_norm, ctx.get("business_type") or "", ctx.get("gender") or "")
    if not prods:
        return ("No government scheme matches this profile at the current score. Raising the score past "
                f"{config.APPROVAL_SCORE_THRESHOLD} opens the MUDRA ladder.")
    items = [f"<b>{p['name']}</b> — up to {_inr(p['max_loan_inr'])} at {p['interest_rate']}"
             for p in prods[:3]]
    return f"<b>Schemes {ctx['applicant_id']} may qualify for</b>" + _bullets(items)


_STATIC = {
    "thin_file": (
        "A range instead of one number means the file is thin — the model is honest about "
        "uncertainty rather than inventing precision. Six months of bank statements, GST returns "
        "or utility bills usually narrows the band and lifts the midpoint, because each verified "
        "source raises data completeness."),
    "fairness": (
        "Gender, geography and business type are never scoring inputs — they are withheld from the "
        "model entirely and used only afterwards in the Fairness Audit to check outcomes. If a group "
        "shows a lower approval rate, the audit flags it and traces whether it is a thin-file effect "
        "or genuine proxy bias."),
    "marketplace": (
        "The Credit Marketplace lets registered lenders co-fund a scored applicant. One loan can be "
        "split across many lenders, each at their own rate, and the borrower sees the blended rate "
        "plus every individual piece. Competition between lenders is what pushes the rate below the "
        "bank's own direct offer."),
    "data_needed": (
        "Most useful first: 6 months of bank or UPI statements, GST returns if registered, and "
        "utility or rent payment records. Those three drive cash-flow consistency, formality and "
        "payment reliability — the heaviest factors in the score. No credit bureau file is required."),
}

_HELP = (
    "I'm the CredVeda underwriting assistant, and I run the real scoring model — not a canned demo.<br>"
    "<div style='margin-top:.45rem;'><b>Try me with</b></div>"
    + _bullets([
        "Describe an applicant in plain words: <i>“kirana shop, revenue 2.5 lakhs a month, 40% digital, GST filed, EMI 8000”</i>",
        "Then change one thing: <i>“what if their bureau score was 700?”</i> — I re-score and show the delta",
        "On an applicant page: <i>“why this score?”</i>, <i>“how do they improve?”</i>, <i>“can they afford it?”</i>",
    ])
)


def answer(message: str, applicant_ctx: dict | None = None,
           profile: dict | None = None, previous: dict | None = None) -> dict:
    """Produce a reply plus the updated conversation state.

    Returns {reply, profile, last_result, intent}. The caller (the browser)
    holds the state and sends it back, so the server stays stateless.
    """
    message = (message or "").strip()
    if not message:
        return {"reply": _HELP, "profile": _sanitise(profile), "last_result": None, "intent": "help"}

    profile = _sanitise(profile)
    if not isinstance(previous, dict):
        previous = None
    signals = extract_signals(message)
    # entity_type alone is a weak cue ("business" appears in many questions),
    # so it does not count towards treating the message as a profile.
    n_signals = len([k for k in signals if k != "entity_type"])
    intent = detect_intent(message, n_signals, bool(profile))

    # A question about the applicant on screen beats the generic explainer.
    if applicant_ctx and not signals:
        if intent in ("why_score", "score_now"):
            return {"reply": _applicant_why(applicant_ctx), "profile": profile,
                    "last_result": None, "intent": "why_score"}
        if intent == "improve":
            return {"reply": _applicant_improve(applicant_ctx), "profile": profile,
                    "last_result": None, "intent": "improve"}
        if intent == "emi":
            return {"reply": _applicant_emi(applicant_ctx), "profile": profile,
                    "last_result": None, "intent": "emi"}
        if intent == "schemes":
            return {"reply": _applicant_schemes(applicant_ctx), "profile": profile,
                    "last_result": None, "intent": "schemes"}

    if intent in ("score_now", "score_what_if"):
        if not signals and not profile:
            return {"reply": ("Tell me about the applicant and I'll run the real model. For example: "
                              "<i>“kirana shop, revenue 2 lakhs a month, 35% digital, GST registered, "
                              "existing EMI 6000, wants 3 lakhs”</i>."),
                    "profile": profile, "last_result": None, "intent": "score_now"}
        profile.update(signals)
        reply, res = _score_reply(profile, previous if intent == "score_what_if" else None)
        return {"reply": reply, "profile": profile,
                "last_result": {"credit_score": res["credit_score"]}, "intent": intent}

    if intent == "improve":
        if profile:
            raw, entity_type, _ = _build_raw(profile)
            res = scoring.score_full(raw, entity_type=entity_type)
            res.pop("_X_scaled", None)
            ctx = {"applicant_id": "this profile", "credit_score": res["credit_score"],
                   "improvement_path": res.get("improvement_path")}
            return {"reply": _applicant_improve(ctx), "profile": profile,
                    "last_result": {"credit_score": res["credit_score"]}, "intent": "improve"}
        return {"reply": ("The three that move the needle fastest: auto-pay every utility and rent bill, "
                          "route more receipts through UPI so income is verifiable, and file GST on time "
                          "if registered. Describe an applicant and I'll compute their specific top three "
                          "with point estimates."),
                "profile": profile, "last_result": None, "intent": "improve"}

    if intent in _STATIC:
        return {"reply": _STATIC[intent], "profile": profile, "last_result": None, "intent": intent}

    if intent == "schemes":
        if profile:
            raw, entity_type, _ = _build_raw(profile)
            res = scoring.score_full(raw, entity_type=entity_type)
            res.pop("_X_scaled", None)
            ctx = {"applicant_id": "this profile", "credit_score": res["credit_score"],
                   "business_type": "", "gender": ""}
            return {"reply": _applicant_schemes(ctx), "profile": profile,
                    "last_result": {"credit_score": res["credit_score"]}, "intent": "schemes"}
        return {"reply": ("Eligibility depends on the score, so give me the applicant's numbers and I'll "
                          "match them. The ladder runs MUDRA Shishu (up to ₹50,000), Kishore (₹5 lakh) "
                          "and Tarun (₹10 lakh), with CGTMSE for collateral-free cover and PM SVANidhi "
                          "for street vendors."),
                "profile": profile, "last_result": None, "intent": "schemes"}

    if intent == "emi":
        return {"reply": ("FOIR is the share of net monthly cash flow already committed to repayments. "
                          f"Past {config.FOIR_CAUTION:.0%} it is a caution, past {config.FOIR_HARD_STOP:.0%} "
                          "a hard stop — regardless of score. Give me revenue and existing EMI and I'll "
                          "work out what is actually affordable."),
                "profile": profile, "last_result": None, "intent": "emi"}

    return {"reply": _HELP, "profile": profile, "last_result": None, "intent": "help"}
