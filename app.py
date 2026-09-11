"""
CredVeda -- Alternative Credit Scoring for Individuals and Small Businesses.

Flask web app covering the full loop: score an applicant from alternative
data, explain the decision to both the underwriter and the borrower, fund
the loan, track repayment, and feed that behaviour back into the next
assessment -- plus the fairness audit and model report card over the top.

# Updated on 2026-02-18
"""
import hashlib
import json
import os
import sqlite3
from functools import wraps

import joblib
import pandas as pd
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

import config
import fairness
import lifecycle
import scoring
from chatbot import get_chatbot_assets

app = Flask(__name__)
# A stable key keeps sessions alive across the dev-server's auto-reloads --
# with a random key every code edit silently signs everyone out, which is
# maddening mid-demo. Override via the environment for any real deployment.
app.secret_key = os.environ.get("CREDVEDA_SECRET_KEY", "credveda-local-demo-key-not-for-production")

PAGE_SIZE = 20


@app.context_processor
def inject_chatbot_assets():
    return {"chatbot_assets": get_chatbot_assets()}


# --- Database / auth helpers -----------------------------------------------
def get_db_connection():
    try:
        return sqlite3.connect(config.DB_NAME)
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None


def init_user_db():
    conn = get_db_connection()
    if conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(stored_password, provided_password):
    return stored_password == hash_password(provided_password)


def add_user(username, password):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        row = conn.execute("SELECT password FROM users WHERE username = ?", (username,)).fetchone()
        return bool(row) and verify_password(row[0], password)
    finally:
        conn.close()


def login_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated


# --- Portfolio data (loaded once at startup) --------------------------------
portfolio_df = pd.DataFrame()


def load_portfolio():
    global portfolio_df
    conn = get_db_connection()
    if not conn:
        portfolio_df = pd.DataFrame()
        return
    try:
        applicants = pd.read_sql_query("SELECT * FROM applicants", conn)
        scores = pd.read_sql_query("SELECT * FROM credit_scores", conn)
        portfolio_df = applicants.merge(scores, on="applicant_id", how="inner")
        print(f"Loaded portfolio: {len(portfolio_df)} scored applicants.")
    except Exception as e:
        print(f"WARNING: could not load portfolio ({e}). Run database.py, data_ingestion.py, "
              f"model_training.py, then scoring.py first.")
        portfolio_df = pd.DataFrame()
    finally:
        conn.close()


# --- Auth routes -------------------------------------------------------------
@app.route("/login", methods=["GET", "POST"])
def login():
    if session.get("logged_in"):
        return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if not username or not password:
            error = "Username and password cannot be empty."
        elif authenticate_user(username, password):
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("home"))
        else:
            error = "Invalid Username or Password."
    return render_template("login.html", error=error)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("logged_in"):
        return redirect(url_for("home"))
    error = None
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if not username or not password:
            error = "Username and password cannot be empty."
        elif add_user(username, password):
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("home"))
        else:
            error = "Username already exists. Please choose a different one."
    return render_template("signup.html", error=error)


@app.route("/logout")
@login_required
def logout():
    session.pop("logged_in", None)
    session.pop("username", None)
    return redirect(url_for("login"))


# --- Marketing / info routes -------------------------------------------------
@app.route("/")
@login_required
def home():
    stats = {}
    if not portfolio_df.empty:
        stats = {
            "n": len(portfolio_df),
            "approval_rate": round((portfolio_df["credit_score"] >= config.APPROVAL_SCORE_THRESHOLD).mean() * 100, 1),
            "thin_file_pct": round((portfolio_df["data_completeness"] < 0.75).mean() * 100, 1),
            "n_business": int((portfolio_df["entity_type"] == "Small Business").sum()),
            "n_individual": int((portfolio_df["entity_type"] == "Individual").sum()),
        }
    return render_template("home.html", username=session.get("username"), stats=stats)


@app.route("/ai-features")
def ai_features_page():
    return render_template("ai_features.html")


@app.route("/about")
def about_page():
    return render_template("about.html")


@app.route("/contact")
@login_required
def contact_page():
    return render_template("contact.html")


@app.route("/privacy")
@login_required
def privacy_page():
    return render_template("privacy.html")


@app.route("/terms")
@login_required
def terms_page():
    return render_template("terms.html")


@app.route("/set_theme", methods=["POST"])
def set_theme():
    data = request.get_json(silent=True) or {}
    theme = data.get("theme")
    if theme in ["light", "dark"]:
        session["theme"] = theme
        return jsonify({"status": "success", "theme": theme})
    return jsonify({"status": "error", "message": "Invalid theme"}), 400


# --- Dashboard: portfolio browser --------------------------------------------
@app.route("/dashboard")
@login_required
def dashboard_page():
    if portfolio_df.empty:
        return render_template("error.html", message=(
            "No scored applicants found. Run database.py, then data_ingestion.py, "
            "then model_training.py, then scoring.py, and restart the app."
        ))

    df = portfolio_df.copy()

    entity_type = request.args.get("entity_type", "")
    business_type = request.args.get("business_type", "")
    geography_tier = request.args.get("geography_tier", "")
    gender = request.args.get("gender", "")
    confidence = request.args.get("confidence", "")
    search = request.args.get("search", "").strip()
    page = max(1, request.args.get("page", 1, type=int))

    if entity_type:
        df = df[df["entity_type"] == entity_type]
    if business_type:
        df = df[df["business_type"] == business_type]
    if geography_tier:
        df = df[df["geography_tier"] == geography_tier]
    if gender:
        df = df[df["gender"] == gender]
    if confidence:
        df = df[df["confidence_label"] == confidence]
    if search:
        df = df[df["applicant_id"].str.contains(search, case=False, na=False)]

    total = len(df)
    total_pages = max(1, (total + PAGE_SIZE - 1) // PAGE_SIZE)
    page = min(page, total_pages)
    page_df = df.sort_values("credit_score", ascending=False).iloc[(page - 1) * PAGE_SIZE: page * PAGE_SIZE]

    rows = []
    for record in page_df.to_dict(orient="records"):
        tier_label, tier_tone = scoring.score_tier(record["credit_score"])
        record["tier_label"] = tier_label
        record["tier_tone"] = tier_tone
        record["band_first"] = record["data_completeness"] < config.BAND_FIRST_COMPLETENESS
        rows.append(record)

    stats = {
        "n": len(portfolio_df),
        "filtered_n": total,
        "approval_rate": round((portfolio_df["credit_score"] >= config.APPROVAL_SCORE_THRESHOLD).mean() * 100, 1),
        "avg_score": round(portfolio_df["credit_score"].mean(), 0),
        "thin_file_pct": round((portfolio_df["data_completeness"] < 0.75).mean() * 100, 1),
        "guardrail_pct": round(portfolio_df["guardrail_flags"].str.contains('"critical"', na=False).mean() * 100, 1),
    }

    filter_options = {
        "entity_types": sorted(portfolio_df["entity_type"].unique().tolist()),
        "business_types": sorted(portfolio_df["business_type"].unique().tolist()),
        "geography_tiers": config.GEOGRAPHY_TIERS,
        "genders": config.GENDERS,
        "confidence_labels": [b[1] for b in config.CONFIDENCE_BANDS],
    }

    scores_list = portfolio_df[['credit_score', 'guardrail_flags']].copy()
    scores_list['guardrail_blocked'] = scores_list['guardrail_flags'].str.contains('"critical"', na=False)
    scores_list = [{"score": row["credit_score"], "guardrail_blocked": row["guardrail_blocked"]} for _, row in scores_list.iterrows()]

    return render_template(
        "dashboard.html",
        applicants=rows,
        stats=stats,
        filters=filter_options,
        selected={
            "entity_type": entity_type, "business_type": business_type,
            "geography_tier": geography_tier, "gender": gender,
            "confidence": confidence, "search": search,
        },
        page=page, total_pages=total_pages,
        score_min=config.SCORE_MIN, score_max=config.SCORE_MAX,
        approval_threshold=config.APPROVAL_SCORE_THRESHOLD,
        scores_list=scores_list
    )


@app.route("/applicant/<applicant_id>")
@login_required
def applicant_detail(applicant_id):
    if portfolio_df.empty or applicant_id not in set(portfolio_df["applicant_id"]):
        return render_template("error.html", message=f"Applicant {applicant_id} not found."), 404

    raw = portfolio_df.loc[portfolio_df["applicant_id"] == applicant_id].iloc[0].to_dict()
    result = scoring.get_or_compute_full(applicant_id, raw, raw["entity_type"])
    # Layer any on-platform repayment track record over the model score.
    result = lifecycle.apply_repayment_history(applicant_id, result)

    return render_template(
        "applicant_detail.html",
        applicant=raw,
        result=result,
        loans=lifecycle.loans_for(applicant_id),
        active_loan=lifecycle.active_loan_for(applicant_id),
        score_min=config.SCORE_MIN, score_max=config.SCORE_MAX,
        approval_threshold=config.APPROVAL_SCORE_THRESHOLD,
    )


# --- Loan lifecycle -----------------------------------------------------------
@app.route("/portfolio")
@login_required
def loan_portfolio():
    """The lending book: what we funded, what came back, and whether the
    scores we gave at funding time turned out to be right."""
    return render_template(
        "portfolio.html",
        stats=lifecycle.portfolio_stats(),
        loans=lifecycle.portfolio_loans(),
        alerts=lifecycle.alerts(),
        buckets=lifecycle.backtest_buckets(),
        approval_threshold=config.APPROVAL_SCORE_THRESHOLD,
    )


@app.route("/applicant/<applicant_id>/fund", methods=["POST"])
@login_required
def fund_applicant(applicant_id):
    if portfolio_df.empty or applicant_id not in set(portfolio_df["applicant_id"]):
        return render_template("error.html", message=f"Applicant {applicant_id} not found."), 404
    if lifecycle.active_loan_for(applicant_id):
        return redirect(url_for("applicant_detail", applicant_id=applicant_id), code=303)

    raw = portfolio_df.loc[portfolio_df["applicant_id"] == applicant_id].iloc[0].to_dict()
    result = scoring.get_or_compute_full(applicant_id, raw, raw["entity_type"])

    # The guardrail is binding, not advisory: a critical affordability flag
    # blocks disbursal here as well as in the UI.
    if any(g["severity"] == "critical" for g in result.get("guardrail_flags", [])):
        return redirect(url_for("applicant_detail", applicant_id=applicant_id), code=303)

    principal = float(request.form.get("principal") or raw["requested_loan_amount"])
    tenure = int(float(request.form.get("tenure") or raw["requested_tenure_months"]))
    loan_id = lifecycle.disburse(applicant_id, principal, tenure,
                                 result["credit_score"], result["probability_good"])
    return redirect(url_for("loan_detail", loan_id=loan_id), code=303)


@app.route("/loan/<loan_id>")
@login_required
def loan_detail(loan_id):
    loan = lifecycle.get_loan(loan_id)
    if loan is None:
        return render_template("error.html", message=f"Loan {loan_id} not found."), 404
    applicant = None
    if not portfolio_df.empty:
        match = portfolio_df.loc[portfolio_df["applicant_id"] == loan["applicant_id"]]
        if not match.empty:
            applicant = match.iloc[0].to_dict()
    return render_template(
        "loan_detail.html",
        loan=loan,
        applicant=applicant,
        schedule=lifecycle.schedule_for(loan_id),
        history=lifecycle.repayment_history(loan["applicant_id"]),
        adjustment=lifecycle.repayment_adjustment(loan["applicant_id"]),
    )


@app.route("/loan/<loan_id>/advance", methods=["POST"])
@login_required
def advance_single_loan(loan_id):
    lifecycle.advance_loan(loan_id)
    return redirect(url_for("loan_detail", loan_id=loan_id), code=303)


@app.route("/simulate/advance", methods=["POST"])
@login_required
def simulate_advance():
    """Fast-forwards the whole book. A live portfolio can't be demonstrated by
    waiting real months, so this is the clock."""
    months = max(1, min(int(request.form.get("months", 1)), 12))
    summary = lifecycle.advance_all(months)
    session["last_simulation"] = summary
    return redirect(url_for("loan_portfolio"), code=303)


@app.route("/applicant/<applicant_id>/explanation")
@login_required
def applicant_explanation(applicant_id):
    """The applicant-facing view of the same decision: no model jargon, no
    SHAP values, no reason-code numbers -- just what was decided, the reasons
    in second person, and what to do next. This is the artefact a lender would
    actually hand the borrower to justify a rejection."""
    if portfolio_df.empty or applicant_id not in set(portfolio_df["applicant_id"]):
        return render_template("error.html", message=f"Applicant {applicant_id} not found."), 404

    raw = portfolio_df.loc[portfolio_df["applicant_id"] == applicant_id].iloc[0].to_dict()
    result = scoring.get_or_compute_full(applicant_id, raw, raw["entity_type"])
    result = lifecycle.apply_repayment_history(applicant_id, result)
    active = lifecycle.active_loan_for(applicant_id)
    return render_template(
        "borrower_view.html",
        applicant=raw,
        result=result,
        active_loan=active,
        schedule=lifecycle.schedule_for(active["loan_id"]) if active else None,
        score_min=config.SCORE_MIN, score_max=config.SCORE_MAX,
        approval_threshold=config.APPROVAL_SCORE_THRESHOLD,
    )


@app.route("/api/applicant/<applicant_id>/explain")
@login_required
def api_applicant_explain(applicant_id):
    if portfolio_df.empty or applicant_id not in set(portfolio_df["applicant_id"]):
        return jsonify({"error": "Applicant not found"}), 404
    raw = portfolio_df.loc[portfolio_df["applicant_id"] == applicant_id].iloc[0].to_dict()
    result = scoring.get_or_compute_full(applicant_id, raw, raw["entity_type"])
    return jsonify(result)


# --- Live scoring: New Assessment form ---------------------------------------
BUSINESS_TYPE_CHOICES = config.BUSINESS_TYPES_BY_ENTITY


@app.route("/apply", methods=["GET", "POST"])
@login_required
def apply_page():
    result = None
    form_values = {}

    if request.method == "POST":
        f = request.form
        form_values = f.to_dict()

        def num(name, default=0.0):
            try:
                return float(f.get(name, default) or default)
            except ValueError:
                return default

        entity_type = f.get("entity_type", "Small Business")
        is_business = entity_type == "Small Business"

        avg_monthly_inflow = max(num("avg_monthly_inflow", 20000), 1.0)
        monthly_expenses = num("monthly_expenses", avg_monthly_inflow * 0.65)
        net_cashflow = max(avg_monthly_inflow - monthly_expenses, 1500)

        existing_monthly_emi = num("existing_monthly_emi", 0)
        requested_loan_amount = num("requested_loan_amount", 100000)
        requested_tenure_months = int(num("requested_tenure_months", 24))
        proposed_emi = scoring.emi(requested_loan_amount, requested_tenure_months)
        repayment_burden_ratio = (existing_monthly_emi + proposed_emi) / net_cashflow

        gst_registered = 1 if (is_business and f.get("gst_registered") == "on") else 0
        bureau_available = f.get("bureau_score_available") == "on"

        raw = {
            "avg_monthly_inflow": avg_monthly_inflow,
            "inflow_growth_rate_6m": num("inflow_growth_rate_6m", 0) / 100.0,
            "inflow_volatility_cv": num("inflow_volatility_cv", 20) / 100.0,
            "monthly_txn_count": num("monthly_txn_count", 40),
            "txn_bounce_rate": num("txn_bounce_rate", 5) / 100.0,
            "digital_adoption_ratio": num("digital_adoption_ratio", 60) / 100.0,
            "gst_registered": gst_registered,
            "gst_filing_regularity": (num("gst_filing_regularity", 80) / 100.0) if gst_registered else None,
            "overdue_invoice_ratio": (num("overdue_invoice_ratio", 10) / 100.0) if gst_registered else None,
            "utility_ontime_ratio": num("utility_ontime_ratio", 85) / 100.0,
            "rent_ontime_ratio": num("rent_ontime_ratio", 85) / 100.0,
            "supplier_concentration_hhi": (num("supplier_concentration_hhi", 30) / 100.0) if is_business else None,
            "repeat_supplier_ratio": (num("repeat_supplier_ratio", 60) / 100.0) if is_business else None,
            "vintage_months": num("vintage_months", 24),
            "existing_loan_count": int(num("existing_loan_count", 0)),
            "bureau_score_available": 1 if bureau_available else 0,
            "bureau_score_norm": ((num("bureau_score_raw", 650) - 300) / 600.0) if bureau_available else None,
            "repayment_burden_ratio": repayment_burden_ratio,
            "requested_loan_amount": requested_loan_amount,
        }
        result = scoring.score_full(raw, entity_type=entity_type)
        result["computed_proposed_emi"] = round(proposed_emi)
        result["computed_net_cashflow"] = round(net_cashflow)

    return render_template(
        "apply.html",
        result=result,
        form_values=form_values,
        business_type_choices=BUSINESS_TYPE_CHOICES,
        score_min=config.SCORE_MIN, score_max=config.SCORE_MAX,
        approval_threshold=config.APPROVAL_SCORE_THRESHOLD,
    )


# --- Fairness / Responsible AI report -----------------------------------------
@app.route("/fairness")
@login_required
def fairness_page():
    report = fairness.run_fairness_audit()
    return render_template("fairness.html", report=report,
                            threshold=config.FAIRNESS_ADVERSE_IMPACT_THRESHOLD)


@app.route("/models")
@login_required
def models_page():
    """The model report card: what we compared, which metrics we judged on,
    which one is actually in production and why."""
    try:
        card = joblib.load("model_leaderboard.joblib")
    except FileNotFoundError:
        return render_template("error.html", message="Run model_training.py first to build the model report card.")
    return render_template(
        "models.html",
        card=card,
        buckets=lifecycle.backtest_buckets(),
        anomaly_rate=config.ANOMALY_CONTAMINATION,
    )


@app.route("/api/fairness_data")
@login_required
def api_fairness_data():
    return jsonify(fairness.run_fairness_audit())


# --- Setup and Run ------------------------------------------------------------
if __name__ == "__main__":
    init_user_db()
    scoring.load_components()
    load_portfolio()

    print("\n--- Starting CredVeda: Alternative Credit Scoring (Flask) ---")
    print("Sign up for a new account, then open http://127.0.0.1:5000/login")
    print("---------------------------------------------------------------")
    app.run(debug=True)
# Updated on 2026-02-18
