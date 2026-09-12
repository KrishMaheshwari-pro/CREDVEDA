"""
CredVeda -- Alternative Credit Scoring for Individuals and Small Businesses.

Full-stack platform: score from alternative data → explain → fund → track →
fairness audit → credit marketplace (multi-lender syndication) → AI underwriter.
"""
import csv as _csv
import hashlib
import io
import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from functools import wraps

import joblib
import pandas as pd
import requests as _requests
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for

import config
import fairness
import guardrails
import lifecycle
import scoring
from chatbot import get_chatbot_assets
from loan_product_matcher import get_eligible_products, get_score_band_summary
from improvement_path import generate_improvement_path
from thin_file_handler import first_loan_pathway, get_confidence_band, is_thin_file


def _risk_based_rate(score) -> float:
    """Risk-based annual interest rate (%) for a 300-900 credit score.

    Single home for the pricing curve: the same expression was inlined in
    seven places, so any change to the floor, ceiling or slope had to be made
    seven times to stay consistent.
    """
    # The 42.0 intercept quoted an 850-score borrower 19.6% p.a.; the pricing
    # table this curve is specified by wants 9.6% there (720 -> 13.1,
    # 620 -> 16.0, 480 -> 19.4), all of which 32.0 reproduces.
    return round(max(8.5, min(28.0, 32.0 - float(score) / 38.0)), 2)

load_dotenv()

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
        conn = sqlite3.connect(config.DB_NAME)
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        print(f"Database connection error: {e}")
        return None


def init_user_db():
    conn = get_db_connection()
    if not conn:
        return
    try:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL
            )
        """)
        # Non-destructive migrations for marketplace tables
        for stmt in [
            "ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'bank'",
            "ALTER TABLE marketplace_listings ADD COLUMN interest_rate REAL DEFAULT 12.0",
            "ALTER TABLE marketplace_listings ADD COLUMN blended_rate REAL",
            "ALTER TABLE marketplace_listings ADD COLUMN borrower_email TEXT",
            "ALTER TABLE marketplace_listings ADD COLUMN borrower_phone TEXT",
            "ALTER TABLE lender_interests ADD COLUMN proposed_rate REAL",
            "ALTER TABLE lender_interests ADD COLUMN message TEXT",
            "ALTER TABLE lender_interests ADD COLUMN collateral_required INTEGER DEFAULT 0",
            "ALTER TABLE marketplace_listings ADD COLUMN bank_offer_rate REAL",
            "ALTER TABLE marketplace_listings ADD COLUMN bank_offer_collateral INTEGER DEFAULT 1",
            "ALTER TABLE marketplace_listings ADD COLUMN bank_offer_collateral_detail TEXT",
            "ALTER TABLE marketplace_listings ADD COLUMN chosen_option TEXT",
            "ALTER TABLE marketplace_listings ADD COLUMN rejected_options TEXT",
        ]:
            try:
                conn.execute(stmt)
            except Exception:
                pass  # Column already exists — safe to ignore
        conn.commit()
    finally:
        conn.close()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def verify_password(stored_password, provided_password):
    return stored_password == hash_password(provided_password)


# Columns added to the marketplace tables after their original schema shipped.
# Kept here (rather than only in database.py) because database.py DROPs
# credit_scores, so it can't be re-run against a live DB just to pick up a
# new column without throwing away every cached score.
_LISTING_ADDED_COLUMNS = (
    ("interest_rate", "REAL DEFAULT 12.0"),
    ("borrower_email", "TEXT"),
    ("borrower_phone", "TEXT"),
    ("blended_rate", "REAL"),
)
_INTEREST_ADDED_COLUMNS = (
    ("proposed_rate", "REAL"),
    ("message", "TEXT"),
)


def _migrate_db():
    """Bring an existing database up to the schema the app expects.

    Idempotent and non-destructive: every statement is CREATE ... IF NOT
    EXISTS or an ALTER whose "duplicate column" error is swallowed, so this
    is safe to run on every boot.
    """
    conn = get_db_connection()
    if not conn:
        return
    try:
        try:
            conn.execute("ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'bank'")
        except sqlite3.OperationalError:
            pass  # column already exists — safe to ignore

        # The marketplace tables live in database.py, which is a manual
        # one-shot script. A DB created before the marketplace shipped simply
        # doesn't have them, and every marketplace/passport/applicant page
        # then 500s on "no such table". Create them here instead.
        conn.execute("""
            CREATE TABLE IF NOT EXISTS marketplace_listings (
                listing_id       TEXT PRIMARY KEY,
                applicant_id     TEXT NOT NULL,
                listed_by        TEXT NOT NULL,
                listed_at        TEXT NOT NULL,
                amount_requested REAL NOT NULL,
                tenure_months    INTEGER NOT NULL,
                credit_score     INTEGER NOT NULL,
                tier_label       TEXT,
                entity_type      TEXT,
                business_type    TEXT,
                geography_tier   TEXT,
                purpose          TEXT,
                status           TEXT DEFAULT 'open',
                total_committed  REAL DEFAULT 0.0,
                fully_funded_at  TEXT,
                interest_rate    REAL DEFAULT 12.0,
                borrower_email   TEXT,
                borrower_phone   TEXT,
                blended_rate     REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS lender_interests (
                interest_id       TEXT PRIMARY KEY,
                listing_id        TEXT NOT NULL,
                lender_username   TEXT NOT NULL,
                committed_amount  REAL NOT NULL,
                status            TEXT DEFAULT 'active',
                created_at        TEXT NOT NULL,
                proposed_rate     REAL,
                message           TEXT
            )
        """)

        # Tables that predate the rate-negotiation feature need the new columns.
        for table, columns in (("marketplace_listings", _LISTING_ADDED_COLUMNS),
                               ("lender_interests", _INTEREST_ADDED_COLUMNS)):
            for column, decl in columns:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl}")
                except sqlite3.OperationalError:
                    pass  # column already exists — safe to ignore

        conn.execute("CREATE INDEX IF NOT EXISTS idx_li_listing ON lender_interests(listing_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ml_status ON marketplace_listings(status)")
        conn.commit()
    finally:
        conn.close()


def add_user(username, password, role="bank"):
    conn = get_db_connection()
    if not conn:
        return False
    try:
        conn.execute(
            "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
            (username, hash_password(password), role)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def authenticate_user(username, password):
    conn = get_db_connection()
    if not conn:
        return None
    try:
        row = conn.execute("SELECT password, role FROM users WHERE username = ?", (username,)).fetchone()
        if row and verify_password(row[0], password):
            return row[1] or "bank"  # return role string
        return None
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
        role = session.get("role", "bank")
        return redirect(url_for("marketplace") if role == "lender" else url_for("home"))
    error = None
    portal = request.args.get("portal", request.form.get("portal", "bank"))
    if portal not in ("bank", "lender"):
        portal = "bank"
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        if not username or not password:
            error = "Username and password cannot be empty."
        else:
            role = authenticate_user(username, password)
            if role and role != portal:
                # The portal tab used to be decoration: the role comes from the
                # account, so signing in under the Lender tab with a bank
                # account silently landed on the bank home page and looked like
                # both portals led to the same screen. Say so instead.
                other = "Bank / NBFC" if role == "bank" else "Lender / Investor"
                error = (f"'{username}' is a {other} account. "
                         f"Switch to the {other} tab to sign in, "
                         f"or create a separate account for this portal.")
            elif role:
                session["logged_in"] = True
                session["username"] = username
                session["role"] = role
                return redirect(url_for("marketplace") if role == "lender" else url_for("home"))
            else:
                error = "Invalid username or password."
    return render_template("login.html", error=error, portal=portal)


@app.route("/signup", methods=["GET", "POST"])
def signup():
    if session.get("logged_in"):
        return redirect(url_for("home"))
    error = None
    portal = request.args.get("portal", request.form.get("portal", "bank"))
    if portal not in ("bank", "lender"):
        portal = "bank"
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        role = request.form.get("role", portal)
        if role not in ("bank", "lender"):
            role = "bank"
        if not username or not password:
            error = "Username and password cannot be empty."
        elif add_user(username, password, role):
            session["logged_in"] = True
            session["username"] = username
            session["role"] = role
            return redirect(url_for("marketplace") if role == "lender" else url_for("home"))
        else:
            error = "Username already exists. Please choose a different one."
    return render_template("signup.html", error=error, portal=portal)


@app.route("/logout")
@login_required
def logout():
    session.pop("logged_in", None)
    session.pop("username", None)
    session.pop("role", None)
    return redirect(url_for("login"))


# --- Marketing / info routes -------------------------------------------------
@app.route("/")
def home():
    stats = {}
    if session.get("logged_in") and not portfolio_df.empty:
        stats = {
            "n": len(portfolio_df),
            "approval_rate": round((portfolio_df["credit_score"] >= config.APPROVAL_SCORE_THRESHOLD).mean() * 100, 1),
            "thin_file_pct": round((portfolio_df["data_completeness"] < 0.75).mean() * 100, 1),
            "n_business": int((portfolio_df["entity_type"] == "Small Business").sum()),
            "n_individual": int((portfolio_df["entity_type"] == "Individual").sum()),
        }
    return render_template("home.html", username=session.get("username"), stats=stats)


@app.route("/borrower-portal", methods=["GET", "POST"])
def borrower_portal():
    """Public page for borrowers to look up their credit passport by applicant ID."""
    error = None
    if request.method == "POST":
        app_id = (request.form.get("applicant_id") or "").strip().upper()
        goto = request.form.get("goto", "passport")   # 'passport' | 'fundings'
        if not app_id:
            error = "Please enter your Applicant ID."
        else:
            conn = get_db_connection()
            try:
                row = conn.execute(
                    "SELECT applicant_id FROM credit_scores WHERE applicant_id=?", (app_id,)
                ).fetchone()
            finally:
                conn.close()
            found = bool(row) or (not portfolio_df.empty and app_id in set(portfolio_df["applicant_id"]))
            if found:
                dest = "borrower_fundings" if goto == "fundings" else "credit_passport"
                return redirect(url_for(dest, applicant_id=app_id))
            error = f"No application found for ID '{app_id}'. Check with your loan officer."
    return render_template("borrower_portal.html", error=error)


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

    # Geography breakdown: avg score + approval rate by tier
    geo_grp = (portfolio_df.groupby("geography_tier")
               .agg(avg_score=("credit_score", "mean"),
                    count=("credit_score", "count"),
                    approval_rate=("credit_score", lambda x: (x >= config.APPROVAL_SCORE_THRESHOLD).mean()))
               .reset_index().sort_values("avg_score"))
    geo_data = [{"label": r["geography_tier"],
                 "avg_score": round(r["avg_score"], 1),
                 "count": int(r["count"]),
                 "approval_rate": round(r["approval_rate"] * 100, 1)}
                for _, r in geo_grp.iterrows()]

    # Sector breakdown: top 8 by count, approval rate
    sector_grp = (portfolio_df.groupby("business_type")
                  .agg(count=("credit_score", "count"),
                       avg_score=("credit_score", "mean"),
                       approval_rate=("credit_score", lambda x: (x >= config.APPROVAL_SCORE_THRESHOLD).mean()))
                  .reset_index().nlargest(8, "count"))
    sector_data = [{"label": r["business_type"],
                    "count": int(r["count"]),
                    "avg_score": round(r["avg_score"], 1),
                    "approval_rate": round(r["approval_rate"] * 100, 1)}
                   for _, r in sector_grp.iterrows()]

    # Thin-file by geography (credit desert signal)
    thin_grp = (portfolio_df.groupby("geography_tier")
                .agg(thin_rate=("data_completeness", lambda x: (x < 0.5).mean()))
                .reset_index())
    thin_by_geo = {r["geography_tier"]: round(r["thin_rate"] * 100, 1) for _, r in thin_grp.iterrows()}

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
        scores_list=scores_list,
        geo_data=geo_data,
        sector_data=sector_data,
        thin_by_geo=thin_by_geo,
    )


@app.route("/applicant/<applicant_id>")
@login_required
def applicant_detail(applicant_id):
    if portfolio_df.empty or applicant_id not in set(portfolio_df["applicant_id"]):
        return render_template("error.html", message=f"Applicant {applicant_id} not found."), 404

    raw = portfolio_df.loc[portfolio_df["applicant_id"] == applicant_id].iloc[0].to_dict()
    result = scoring.get_or_compute_full(applicant_id, raw, raw["entity_type"])
    result = lifecycle.apply_repayment_history(applicant_id, result)

    score_norm = int((result["credit_score"] - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN) * 100)
    matched_products = get_eligible_products(score_norm, raw.get("business_type", ""), raw.get("gender", ""))

    critical_flags = [g for g in result.get("guardrail_flags", []) if g["severity"] == "critical"]
    is_listable = (result["credit_score"] >= config.MARKETPLACE_MIN_SCORE and not critical_flags)

    conn = get_db_connection()
    existing_listing = None
    if conn:
        try:
            row = conn.execute(
                "SELECT listing_id FROM marketplace_listings WHERE applicant_id=? AND status='open'",
                (applicant_id,)
            ).fetchone()
            existing_listing = dict(row) if row else None
        finally:
            conn.close()

    return render_template(
        "applicant_detail.html",
        applicant=raw,
        result=result,
        loans=lifecycle.loans_for(applicant_id),
        active_loan=lifecycle.active_loan_for(applicant_id),
        matched_products=matched_products[:4],
        is_listable=is_listable,
        existing_listing=existing_listing,
        score_min=config.SCORE_MIN, score_max=config.SCORE_MAX,
        approval_threshold=config.APPROVAL_SCORE_THRESHOLD,
        marketplace_min_score=config.MARKETPLACE_MIN_SCORE,
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
    score_norm = int((result["credit_score"] - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN) * 100)
    matched_products = get_eligible_products(score_norm, raw.get("business_type", ""), raw.get("gender", ""))
    band = get_score_band_summary(score_norm)
    thin_pathway = first_loan_pathway(result["credit_score"], result["data_completeness"])
    return render_template(
        "borrower_view.html",
        applicant=raw,
        result=result,
        active_loan=active,
        schedule=lifecycle.schedule_for(active["loan_id"]) if active else None,
        matched_products=matched_products[:4],
        band=band,
        thin_pathway=thin_pathway,
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

        # Income is split into verifiable digital/bank inflow and (corroborable)
        # cash inflow. The model scores against the TOTAL; the DVI weighs the two
        # by how much of each there is and how well each is evidenced.
        digital_inflow = max(num("avg_monthly_inflow", 20000), 0.0)
        cash_inflow = max(num("cash_monthly_inflow", 0), 0.0)
        avg_monthly_inflow = max(digital_inflow + cash_inflow, 1.0)
        income_confidence = scoring.blend_income_confidence(
            digital_inflow, f.get("src_income_digital", "verified"),
            cash_inflow, f.get("src_income_cash", "self_declared"),
        )
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
        # Derive a display tag for the blended income confidence so the DVI
        # card can show a 🟢/🟡/🔴 chip for income like the other components.
        if income_confidence >= 85:
            income_tag = "verified"
        elif income_confidence >= 55:
            income_tag = "estimated"
        else:
            income_tag = "self_declared"
        field_sources = {
            "src_income":  income_tag,
            "src_emi":     f.get("src_emi",      "self_declared"),
            "src_gst":     f.get("src_gst",      "self_declared"),
            "src_utility": f.get("src_utility",  "self_declared"),
        }
        verif_pen = scoring.verification_penalty(
            digital_amt=digital_inflow, digital_src=f.get("src_income_digital", "verified"),
            cash_amt=cash_inflow, cash_src=f.get("src_income_cash", "self_declared"),
            emi_amt=existing_monthly_emi, emi_src=f.get("src_emi", "self_declared"),
            gst_registered=bool(gst_registered), gst_src=f.get("src_gst", "self_declared"),
            utility_src=f.get("src_utility", "self_declared"), entity_type=entity_type,
        )
        result = scoring.score_full(raw, entity_type=entity_type, field_sources=field_sources,
                                    income_confidence=income_confidence, verif_penalty=verif_pen)
        result["computed_proposed_emi"] = round(proposed_emi)
        result["computed_net_cashflow"] = round(net_cashflow)

    floor_rate = None
    is_listable = False
    score_gap = None
    matched_products = []
    draft_id = None
    can_verify = False
    if result:
        base_score = result["credit_score"]            # what a fully-verified file would score
        adj_score = result.get("adjusted_score", base_score)  # after the unverified-data penalty
        penalty = result.get("dvi_penalty", 0)
        dvi = result.get("dvi")
        floor_rate = _risk_based_rate(adj_score)
        MIN = config.MARKETPLACE_MIN_SCORE

        critical_flags = [g for g in result.get("guardrail_flags", []) if g["severity"] == "critical"]
        foir_critical_flags = [g for g in critical_flags if "Data Verification Index" not in g.get("message", "")]
        foir_critical = bool(foir_critical_flags)

        # There is unverified must-prove data that a field visit / documents could
        # upgrade (that's exactly what created the penalty).
        can_verify = penalty > 0

        # 1) Affordability breach → hard decline, verification cannot fix it.
        # 2) Verified enough & clears the bar → listable.
        # 3) Would clear the bar if verified, but unproven data dropped it → ops queue.
        # 4) Below the bar even fully verified → improvement path (score_gap).
        is_listable = (not foir_critical) and adj_score >= MIN and (dvi is None or dvi >= 40)
        verify_to_unlock = (not foir_critical) and (not is_listable) and base_score >= MIN and penalty > 0
        score_gap = max(0, MIN - base_score)   # genuine shortfall, present even when verified
        score = adj_score
        score_norm = int((score - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN) * 100)
        matched_products = get_eligible_products(
            score_norm,
            form_values.get("business_type", ""),
            form_values.get("gender", "")
        )
        draft_id = "DRAFT-" + str(uuid.uuid4())[:8].upper()

    return render_template(
        "apply.html",
        result=result,
        form_values=form_values,
        business_type_choices=BUSINESS_TYPE_CHOICES,
        score_min=config.SCORE_MIN, score_max=config.SCORE_MAX,
        approval_threshold=config.APPROVAL_SCORE_THRESHOLD,
        floor_rate=floor_rate,
        is_listable=is_listable,
        verify_to_unlock=verify_to_unlock if result else False,
        can_verify=can_verify,
        foir_critical=(foir_critical if result else False),
        score_gap=score_gap,
        marketplace_min_score=config.MARKETPLACE_MIN_SCORE,
        matched_products=matched_products[:3],
        draft_id=draft_id,
    )


def _pct(v):
    """DB stores several fields as 0–1 ratios; the form shows them as %."""
    try:
        return round(float(v) * 100, 1)
    except (TypeError, ValueError):
        return None


@app.route("/api/lookup-applicant")
@login_required
def lookup_applicant():
    """Type-ahead search for a returning borrower already on the platform.
    Returns id/type/business/last-score plus a form-ready `prefill` so the bank
    can reuse an already-verified profile instead of re-entering and
    re-verifying everything. Because the profile was vetted before, the caller
    marks its data sources as verified, which lifts the DVI."""
    q = (request.args.get("q", "") or "").strip().upper()
    if len(q) < 2:
        return jsonify({"matches": []})

    conn = get_db_connection()
    if not conn:
        return jsonify({"matches": []})
    try:
        rows = conn.execute(
            "SELECT * FROM applicants WHERE UPPER(applicant_id) LIKE ? ORDER BY applicant_id LIMIT 8",
            (f"%{q}%",)
        ).fetchall()
        matches = []
        for r in rows:
            a = dict(r)
            sc = conn.execute(
                "SELECT credit_score, data_completeness, scored_at FROM credit_scores WHERE applicant_id=?",
                (a["applicant_id"],)
            ).fetchone()
            prefill = {
                "entity_type": a.get("entity_type"),
                "business_type": a.get("business_type"),
                "geography_tier": a.get("geography_tier"),
                "gender": a.get("gender"),
                "vintage_months": a.get("vintage_months"),
                "avg_monthly_inflow": a.get("avg_monthly_inflow"),
                "monthly_txn_count": a.get("monthly_txn_count"),
                "txn_bounce_rate": _pct(a.get("txn_bounce_rate")),
                "digital_adoption_ratio": _pct(a.get("digital_adoption_ratio")),
                "inflow_growth_rate_6m": _pct(a.get("inflow_growth_rate_6m")),
                "inflow_volatility_cv": _pct(a.get("inflow_volatility_cv")),
                "gst_registered": bool(a.get("gst_registered")),
                "gst_filing_regularity": _pct(a.get("gst_filing_regularity")),
                "overdue_invoice_ratio": _pct(a.get("overdue_invoice_ratio")),
                "utility_ontime_ratio": _pct(a.get("utility_ontime_ratio")),
                "rent_ontime_ratio": _pct(a.get("rent_ontime_ratio")),
                "supplier_concentration_hhi": _pct(a.get("supplier_concentration_hhi")),
                "repeat_supplier_ratio": _pct(a.get("repeat_supplier_ratio")),
                "existing_loan_count": a.get("existing_loan_count"),
                "existing_monthly_emi": a.get("existing_monthly_emi"),
                "bureau_score_available": bool(a.get("bureau_score_available")),
                "bureau_score_raw": a.get("bureau_score_raw"),
                "requested_loan_amount": a.get("requested_loan_amount"),
                "requested_tenure_months": a.get("requested_tenure_months"),
            }
            matches.append({
                "applicant_id": a["applicant_id"],
                "entity_type": a.get("entity_type", ""),
                "business_type": a.get("business_type", ""),
                "geography_tier": a.get("geography_tier", ""),
                "credit_score": sc["credit_score"] if sc else None,
                "data_completeness": sc["data_completeness"] if sc else None,
                "scored_at": (sc["scored_at"] or "")[:10] if sc else None,
                "prefill": prefill,
            })
    finally:
        conn.close()
    return jsonify({"matches": matches})


def _ensure_verification_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS verification_queue (
            vq_id            TEXT PRIMARY KEY,
            applicant_id     TEXT,
            entity_type      TEXT,
            business_type    TEXT,
            requested_amount REAL,
            declared_score   INTEGER,
            effective_score  INTEGER,
            penalty          INTEGER,
            dvi              INTEGER,
            reason           TEXT,
            unverified       TEXT,
            status           TEXT DEFAULT 'pending',
            created_at       TEXT
        )
    """)


@app.route("/send-to-verification", methods=["POST"])
@login_required
def send_to_verification():
    """Records an applicant whose score is dragged down by unverified must-prove
    data into the field-verification queue. This is the bank officer's action;
    it presumes the borrower agrees to the verification (call / documents /
    visit). Once verified, the officer re-assesses with 🟢 sources and the
    penalty is removed."""
    f = request.form

    def num(name, default=0.0):
        try:
            return float(f.get(name, default) or default)
        except ValueError:
            return default

    entity_type = f.get("entity_type", "Small Business")
    is_business = entity_type == "Small Business"
    digital_inflow = max(num("avg_monthly_inflow", 20000), 0.0)
    cash_inflow = max(num("cash_monthly_inflow", 0), 0.0)
    avg_monthly_inflow = max(digital_inflow + cash_inflow, 1.0)
    income_confidence = scoring.blend_income_confidence(
        digital_inflow, f.get("src_income_digital", "verified"),
        cash_inflow, f.get("src_income_cash", "self_declared"),
    )
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
    verif_pen = scoring.verification_penalty(
        digital_amt=digital_inflow, digital_src=f.get("src_income_digital", "verified"),
        cash_amt=cash_inflow, cash_src=f.get("src_income_cash", "self_declared"),
        emi_amt=existing_monthly_emi, emi_src=f.get("src_emi", "self_declared"),
        gst_registered=bool(gst_registered), gst_src=f.get("src_gst", "self_declared"),
        utility_src=f.get("src_utility", "self_declared"), entity_type=entity_type,
    )
    result = scoring.score_full(raw, entity_type=entity_type, income_confidence=income_confidence,
                                verif_penalty=verif_pen)

    # Which must-prove inputs are unverified (the officer's checklist)
    unverified = []
    if f.get("src_income_digital", "verified") != "verified" and digital_inflow > 0:
        unverified.append("Bank/digital income")
    if f.get("src_income_cash", "self_declared") != "verified" and cash_inflow > 0:
        unverified.append("Cash income")
    if f.get("src_emi", "self_declared") != "verified" and existing_monthly_emi > 0:
        unverified.append("Existing EMIs")
    if gst_registered and f.get("src_gst", "self_declared") != "verified":
        unverified.append("GST filings")

    applicant_id = f.get("returning_applicant_id") or ("VERIFY-" + str(uuid.uuid4())[:8].upper())
    conn = get_db_connection()
    try:
        _ensure_verification_table(conn)
        conn.execute("""
            INSERT INTO verification_queue
            (vq_id, applicant_id, entity_type, business_type, requested_amount,
             declared_score, effective_score, penalty, dvi, reason, unverified, status, created_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,'pending',?)
        """, (str(uuid.uuid4())[:12], applicant_id, entity_type, f.get("business_type", ""),
              requested_loan_amount, result["credit_score"], result["adjusted_score"],
              verif_pen, result.get("dvi"), "Unverified must-prove data",
              ", ".join(unverified), datetime.now(timezone.utc).isoformat()))
        conn.commit()
    finally:
        conn.close()
    return redirect(url_for("lender_ops", flagged=applicant_id))


@app.route("/save-and-list", methods=["POST"])
@login_required
def save_and_list():
    """Score form data, save to DB, and create a marketplace listing in one step."""
    f = request.form

    def num(name, default=0.0):
        try:
            return float(f.get(name, default) or default)
        except ValueError:
            return default

    entity_type = f.get("entity_type", "Small Business")
    is_business = entity_type == "Small Business"
    digital_inflow = max(num("avg_monthly_inflow", 20000), 0.0)
    cash_inflow = max(num("cash_monthly_inflow", 0), 0.0)
    avg_monthly_inflow = max(digital_inflow + cash_inflow, 1.0)
    income_confidence = scoring.blend_income_confidence(
        digital_inflow, f.get("src_income_digital", "verified"),
        cash_inflow, f.get("src_income_cash", "self_declared"),
    )
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
    if income_confidence >= 85:
        income_tag = "verified"
    elif income_confidence >= 55:
        income_tag = "estimated"
    else:
        income_tag = "self_declared"
    field_sources = {
        "src_income":  income_tag,
        "src_emi":     f.get("src_emi",      "self_declared"),
        "src_gst":     f.get("src_gst",      "self_declared"),
        "src_utility": f.get("src_utility",  "self_declared"),
    }
    verif_pen = scoring.verification_penalty(
        digital_amt=digital_inflow, digital_src=f.get("src_income_digital", "verified"),
        cash_amt=cash_inflow, cash_src=f.get("src_income_cash", "self_declared"),
        emi_amt=existing_monthly_emi, emi_src=f.get("src_emi", "self_declared"),
        gst_registered=bool(gst_registered), gst_src=f.get("src_gst", "self_declared"),
        utility_src=f.get("src_utility", "self_declared"), entity_type=entity_type,
    )
    result = scoring.score_full(raw, entity_type=entity_type, field_sources=field_sources,
                                income_confidence=income_confidence, verif_penalty=verif_pen)

    # Eligibility uses the DVI-adjusted score. A real affordability (FOIR)
    # critical blocks listing; a DVI critical should have been routed to the
    # verification queue rather than reaching this endpoint, so block it too.
    eff_score = result.get("adjusted_score", result["credit_score"])
    critical_flags = [g for g in result.get("guardrail_flags", []) if g["severity"] == "critical"]
    if eff_score < config.MARKETPLACE_MIN_SCORE or critical_flags:
        return redirect(url_for("apply_page"))

    applicant_id = "NEW-" + str(uuid.uuid4())[:8].upper()
    # Rates are fixed, auto-computed server-side — not taken from the form, so
    # they can't be edited before listing. Lenders bargain on the marketplace.
    floor_rate = _risk_based_rate(eff_score)
    interest_rate = floor_rate
    purpose = f.get("purpose", "Working capital")
    borrower_email = (f.get("borrower_email") or "").strip()[:120]
    borrower_phone = (f.get("borrower_phone") or "").strip()[:20]
    geography_tier = f.get("geography_tier", "Tier 2")
    business_type = f.get("business_type", "")

    # Bank's own competing direct offer + collateral policy. The rate is a fixed
    # auto-computed spread over the floor — not editable before listing.
    bank_direct = f.get("bank_direct_offer") == "on"
    bank_offer_rate = round(min(32.0, floor_rate + 3.5), 2) if bank_direct else None
    bank_offer_collateral = 1 if (bank_direct and f.get("bank_offer_collateral", "1") == "1") else 0
    bank_offer_collateral_detail = (f.get("bank_offer_collateral_detail") or "").strip()[:200]

    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT OR REPLACE INTO credit_scores
            (applicant_id, credit_score, probability_good, confidence_label, band_low, band_high,
             data_completeness, reason_codes, guardrail_flags, improvement_path, shap_factors, scored_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            applicant_id, result["credit_score"],
            result.get("probability_good", 0), result.get("confidence_label", ""),
            result.get("band_low", result["credit_score"]), result.get("band_high", result["credit_score"]),
            result.get("data_completeness", 1.0),
            json.dumps(result.get("reason_codes", [])),
            json.dumps(result.get("guardrail_flags", [])),
            json.dumps(result.get("improvement_path", [])),
            json.dumps(result.get("shap_factors", [])),
            datetime.now(timezone.utc).isoformat()
        ))

        listing_id = _next_listing_id(conn)
        tier_label, _ = scoring.score_tier(result["credit_score"])
        conn.execute("""
            INSERT INTO marketplace_listings
            (listing_id, applicant_id, listed_by, listed_at, amount_requested, tenure_months,
             credit_score, tier_label, entity_type, business_type, geography_tier, purpose,
             status, total_committed, interest_rate, borrower_email, borrower_phone,
             bank_offer_rate, bank_offer_collateral, bank_offer_collateral_detail)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'open',0,?,?,?,?,?,?)
        """, (listing_id, applicant_id, session.get("username"),
              datetime.now(timezone.utc).isoformat(),
              requested_loan_amount, requested_tenure_months,
              result["credit_score"], tier_label,
              entity_type, business_type, geography_tier,
              purpose, interest_rate, borrower_email, borrower_phone,
              bank_offer_rate, bank_offer_collateral, bank_offer_collateral_detail))
        conn.commit()
    finally:
        conn.close()

    return redirect(url_for("marketplace_listing_detail", listing_id=listing_id))


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
        feature_importance=card.get("feature_importance", {}),
    )


@app.route("/api/fairness_data")
@login_required
def api_fairness_data():
    return jsonify(fairness.run_fairness_audit())


@app.route("/api/feature_importance")
@login_required
def api_feature_importance():
    try:
        card = joblib.load("model_leaderboard.joblib")
        return jsonify(card.get("feature_importance", {}))
    except FileNotFoundError:
        return jsonify({}), 404


# --- Score Simulator ----------------------------------------------------------
@app.route("/simulator")
@login_required
def simulator_page():
    return render_template("score_simulator.html",
                           score_min=config.SCORE_MIN, score_max=config.SCORE_MAX,
                           feature_cols=config.FEATURE_COLS)


@app.route("/api/simulate", methods=["POST"])
@login_required
def api_simulate():
    """Real-time scoring for the interactive simulator — calls the actual model."""
    data = request.get_json(silent=True) or {}

    def num(k, default=0.0):
        try:
            return float(data.get(k, default) or default)
        except (ValueError, TypeError):
            return default

    entity_type = data.get("entity_type", "Small Business")
    is_business = entity_type == "Small Business"
    avg_inflow = max(num("avg_monthly_inflow", 30000), 1.0)
    monthly_expenses = num("monthly_expenses", avg_inflow * 0.65)
    net_cashflow = max(avg_inflow - monthly_expenses, 1500)
    existing_emi = num("existing_monthly_emi", 0)
    loan_amount = num("requested_loan_amount", 100000)
    tenure = max(1, int(num("requested_tenure_months", 24)))
    proposed_emi = scoring.emi(loan_amount, tenure)
    repayment_burden = (existing_emi + proposed_emi) / net_cashflow

    gst = 1 if (is_business and num("gst_registered", 0) > 0) else 0
    bureau = num("bureau_score_available", 0) > 0

    raw = {
        "avg_monthly_inflow": avg_inflow,
        "inflow_growth_rate_6m": num("inflow_growth_rate_6m", 0.0),
        "inflow_volatility_cv": num("inflow_volatility_cv", 0.20),
        "monthly_txn_count": num("monthly_txn_count", 30),
        "txn_bounce_rate": num("txn_bounce_rate", 0.05),
        "digital_adoption_ratio": num("digital_adoption_ratio", 0.50),
        "gst_registered": gst,
        "gst_filing_regularity": num("gst_filing_regularity", 0.80) if gst else None,
        "overdue_invoice_ratio": num("overdue_invoice_ratio", 0.10) if gst else None,
        "utility_ontime_ratio": num("utility_ontime_ratio", 0.85),
        "rent_ontime_ratio": num("rent_ontime_ratio", 0.85),
        "supplier_concentration_hhi": num("supplier_concentration_hhi", 0.30) if is_business else None,
        "repeat_supplier_ratio": num("repeat_supplier_ratio", 0.60) if is_business else None,
        "vintage_months": num("vintage_months", 24),
        "existing_loan_count": int(num("existing_loan_count", 0)),
        "bureau_score_available": 1 if bureau else 0,
        "bureau_score_norm": (num("bureau_score_raw", 650) - 300) / 600.0 if bureau else None,
        "repayment_burden_ratio": repayment_burden,
        "requested_loan_amount": loan_amount,
    }
    result = scoring.score_full(raw, entity_type=entity_type)
    result.pop("_X_scaled", None)

    score_norm = int((result["credit_score"] - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN) * 100)
    matched = get_eligible_products(score_norm, data.get("business_type", ""), data.get("gender", ""))

    return jsonify({
        "credit_score": result["credit_score"],
        "tier_label": result["tier_label"],
        "tier_tone": result["tier_tone"],
        "band_low": result["band_low"],
        "band_high": result["band_high"],
        "confidence_label": result["confidence_label"],
        "band_first": result["band_first"],
        "probability_good": result["probability_good"],
        "data_completeness": result["data_completeness"],
        "approved": result["approved"],
        "foir": result["foir"],
        "reason_codes": result["reason_codes"],
        "improvement_path": result["improvement_path"],
        "guardrail_flags": result["guardrail_flags"],
        "matched_schemes": [{"name": p["name"], "max_loan_inr": p["max_loan_inr"],
                              "interest_rate": p["interest_rate"], "tag": p["tag"],
                              "tag_color": p["tag_color"]} for p in matched[:3]],
        "proposed_emi": round(proposed_emi),
        "net_cashflow": round(net_cashflow),
    })


# --- Lender Operations Queue --------------------------------------------------
@app.route("/lender-ops")
@login_required
def lender_ops():
    if portfolio_df.empty:
        return render_template("error.html", message=(
            "No portfolio data. Run database.py → data_ingestion.py → model_training.py → scoring.py first."))

    queue = []
    for _, row in portfolio_df.iterrows():
        try:
            flags = json.loads(row.get("guardrail_flags", "[]") or "[]")
        except Exception:
            flags = []
        severity = ("CRITICAL" if any(g["severity"] == "critical" for g in flags)
                    else "WARNING" if any(g["severity"] == "warning" for g in flags)
                    else "CLEAR")
        tier_label, tier_tone = scoring.score_tier(int(row["credit_score"]))
        score_color = {"good": "#22c55e", "warn": "#f59e0b", "bad": "#ef4444"}.get(tier_tone, "#4f8cff")
        queue.append({
            "applicant_id": row["applicant_id"],
            "entity_type": row.get("entity_type", ""),
            "business_type": row.get("business_type", ""),
            "geography_tier": row.get("geography_tier", ""),
            "geography_state": row.get("geography_state", ""),
            "credit_score": int(row["credit_score"]),
            "tier_label": tier_label,
            "tier_tone": tier_tone,
            "score_color": score_color,
            "data_completeness": float(row.get("data_completeness", 0)),
            "confidence_label": row.get("confidence_label", ""),
            "guardrail_severity": severity,
            "guardrail_flags": flags,
            "requested_loan_amount": float(row.get("requested_loan_amount", 0) or 0),
            "band_low": int(row.get("band_low", 0)),
            "band_high": int(row.get("band_high", 0)),
            "approved": (int(row["credit_score"]) >= config.APPROVAL_SCORE_THRESHOLD
                         and severity != "CRITICAL"),
            "listable": (int(row["credit_score"]) >= config.MARKETPLACE_MIN_SCORE
                         and severity != "CRITICAL"),
        })
    queue.sort(key=lambda x: (
        {"CRITICAL": 0, "WARNING": 1, "CLEAR": 2}[x["guardrail_severity"]],
        -x["credit_score"]
    ))

    ops_stats = {
        "total": len(queue),
        "approved": sum(1 for q in queue if q["approved"]),
        "critical": sum(1 for q in queue if q["guardrail_severity"] == "CRITICAL"),
        "warning": sum(1 for q in queue if q["guardrail_severity"] == "WARNING"),
        "thin_file": sum(1 for q in queue if q["data_completeness"] < 0.5),
        "avg_score": round(sum(q["credit_score"] for q in queue) / len(queue)) if queue else 0,
        "listable": sum(1 for q in queue if q["listable"]),
    }

    # Cases explicitly flagged for field verification (from the assess page).
    verify_items = []
    conn = get_db_connection()
    if conn:
        try:
            _ensure_verification_table(conn)
            verify_items = [dict(r) for r in conn.execute(
                "SELECT * FROM verification_queue WHERE status='pending' ORDER BY created_at DESC"
            ).fetchall()]
        finally:
            conn.close()

    return render_template("lender_ops.html", queue=queue, stats=ops_stats,
                           verify_items=verify_items,
                           flagged=request.args.get("flagged"),
                           approval_threshold=config.APPROVAL_SCORE_THRESHOLD,
                           min_marketplace_score=config.MARKETPLACE_MIN_SCORE,
                           score_min=config.SCORE_MIN, score_max=config.SCORE_MAX)


# --- Score Report (printable) -------------------------------------------------
def _app_data(applicant_id, raw):
    """Header fields for the printable report templates.

    They read app_data.business_type / .state / .monthly_revenue_avg as well
    as .applicant_name; passing only the name left the rest Undefined, which
    blew up on the "{:,.0f}".format(...) of the revenue line.
    """
    return {
        "applicant_name": applicant_id,
        "applicant_id": applicant_id,
        "business_type": raw.get("business_type") or raw.get("entity_type") or "—",
        "state": raw.get("geography_state") or raw.get("geography_tier") or "—",
        "monthly_revenue_avg": float(raw.get("avg_monthly_inflow", 0) or 0),
    }


def _build_improvement_steps(result):
    """Build the improvement block both score_report.html and
    improvement_portal.html expect: the three actionable steps plus the
    6-month trajectory used to draw the projection chart.

    scoring.score_full() already produces an `improvement_path` list, but the
    templates also need `projected_score` / `trajectory_scores` /
    `trajectory_months`, which only improvement_path.generate_improvement_path()
    computes. Both are combined here, keyed under every field-name alias the
    two templates use between them.
    """
    factors = result.get("shap_factors") or []
    score_norm = int((result["credit_score"] - config.SCORE_MIN)
                     / (config.SCORE_MAX - config.SCORE_MIN) * 100)

    traj = {}
    if factors:
        try:
            traj = generate_improvement_path(
                [float(f.get("contribution", 0) or 0) for f in factors],
                [f.get("feature", "") for f in factors],
                current_score=score_norm,
            )
        except Exception:
            traj = {}

    # Prefer scoring's richer narrative list; fall back to the trajectory
    # module's own steps when the forest produced no actionable negatives.
    path = result.get("improvement_path") or []
    if path:
        steps = [{
            "feature": step.get("feature", ""),
            "title": step.get("label", ""),
            "action": step.get("narrative", ""),
            "detail": step.get("narrative", ""),
            "description": step.get("narrative", ""),
            "difficulty": step.get("difficulty", "Moderate"),
            "timeline": step.get("timeline", "~6 months"),
            "score_gain": step.get("estimated_point_gain", 0),
            "estimated_score_gain": step.get("estimated_point_gain", 0),
        } for step in path]
    else:
        steps = [{
            "feature": step.get("feature", ""),
            "title": step.get("action", ""),
            "action": step.get("action", ""),
            "detail": step.get("action", ""),
            "description": step.get("action", ""),
            "difficulty": step.get("difficulty", "Moderate"),
            "timeline": step.get("timeline", "~6 months"),
            "score_gain": step.get("estimated_score_gain", 0),
            "estimated_score_gain": step.get("estimated_score_gain", 0),
        } for step in traj.get("steps", [])]

    total_gain = sum(float(st.get("score_gain", 0) or 0) for st in steps)
    block = {
        "steps": steps,
        "current_score": result["credit_score"],
        "total_estimated_gain": round(total_gain, 1),
        "projected_score": min(config.SCORE_MAX, int(result["credit_score"] + total_gain)),
    }
    if traj.get("trajectory_scores"):
        # Trajectory comes back on the 0-100 scale; the charts are drawn in
        # 300-900, so rescale before handing it to the template.
        span = config.SCORE_MAX - config.SCORE_MIN
        block["trajectory_scores"] = [
            int(config.SCORE_MIN + (v / 100) * span) for v in traj["trajectory_scores"]
        ]
        block["trajectory_months"] = traj.get("trajectory_months", list(range(7)))
        block["summary"] = traj.get("summary", "")
        block["hindi_summary"] = traj.get("hindi_summary", "")
    return block


def _build_score_data(applicant_id, raw, result, matched):
    """Adapt the scoring result to the contract score_report.html was written
    against. The template reads a `score_data` object whose field names never
    matched scoring.score_full()'s output (score vs credit_score, confidence vs
    confidence_label, ...), so the page raised UndefinedError on every request.
    Mapping here keeps the print template untouched."""
    flags = result.get("guardrail_flags") or []
    severity = ("CRITICAL" if any(g.get("severity") == "critical" for g in flags)
                else "WARNING" if any(g.get("severity") == "warning" for g in flags)
                else "CLEAR")

    foir = result.get("foir") or {}
    amount = float(raw.get("requested_loan_amount", 0) or 0)
    tenure = int(raw.get("requested_tenure_months", 24) or 24)
    rate = _risk_based_rate(result["credit_score"])
    proposed_emi = guardrails.compute_emi(amount, rate, tenure) if amount else 0.0

    # Percentile against the scored book, so "peer" means this lender's own
    # portfolio rather than an abstract national curve.
    percentile = None
    if not portfolio_df.empty:
        percentile = int(round(
            (portfolio_df["credit_score"] < result["credit_score"]).mean() * 100))

    tier = result.get("tier_label", "")
    summary_en = (
        f"Your score of {result['credit_score']} places you in the {tier} band. "
        f"This was calculated from {int(round(result.get('data_completeness', 0) * 100))}% "
        f"complete alternative data — payments, GST, utility and rent behaviour — "
        f"with no reliance on a bureau file."
    )
    summary_hi = (
        f"आपका स्कोर {result['credit_score']} है, जो {tier} श्रेणी में आता है। "
        f"यह आपके भुगतान, जीएसटी, बिजली-पानी और किराए के व्यवहार से निकाला गया है — "
        f"किसी ब्यूरो रिकॉर्ड की ज़रूरत नहीं।"
    )

    return {
        "score": result["credit_score"],
        "score_lower": result.get("band_low"),
        "score_upper": result.get("band_high"),
        "confidence": result.get("confidence_label", "Moderate confidence"),
        "peer_percentile": percentile,
        "guardrail_severity": severity,
        "guardrail_flag": flags[0].get("message") if flags else None,
        "proposed_emi": proposed_emi,
        "total_dti": (foir.get("foir") or 0) * 100,
        "plain_language_summary": summary_en,
        "hindi_summary": summary_hi,
        # score_report.html renders rc.impact numerically (`rc.impact > 0`,
        # `| round(1)`, "pts"), but scoring emits impact as the string
        # "positive"/"negative" and keeps the magnitude in `contribution`.
        # Convert the contribution to score points on the 300-900 scale.
        "reason_codes": [{
            "code": rc.get("code", ""),
            "short_text": rc.get("label", ""),
            "borrower_text": rc.get("text", ""),
            "hindi_text": rc.get("hindi_text", ""),
            "impact": round(float(rc.get("contribution", 0) or 0)
                            * (config.SCORE_MAX - config.SCORE_MIN), 1),
            "direction": rc.get("impact", "neutral"),
        } for rc in (result.get("reason_codes") or [])],
        "improvement_steps": _build_improvement_steps(result),
        "loan_products": [{
            "name": prod.get("name", ""),
            "description": prod.get("description", ""),
            "max_amount": float(prod.get("max_loan_inr", 0) or 0),
            "max_amount_display": prod.get("max_loan_display", ""),
            "rate": prod.get("interest_rate", ""),
            "tenure": prod.get("tenure", ""),
        } for prod in matched],
    }


@app.route("/score-report/<applicant_id>")
@login_required
def score_report(applicant_id):
    if portfolio_df.empty or applicant_id not in set(portfolio_df["applicant_id"]):
        return render_template("error.html", message=f"Applicant {applicant_id} not found."), 404
    raw = portfolio_df.loc[portfolio_df["applicant_id"] == applicant_id].iloc[0].to_dict()
    result = scoring.get_or_compute_full(applicant_id, raw, raw["entity_type"])
    result = lifecycle.apply_repayment_history(applicant_id, result)
    score_norm = int((result["credit_score"] - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN) * 100)
    matched = get_eligible_products(score_norm, raw.get("business_type", ""), raw.get("gender", ""))
    band = get_score_band_summary(score_norm)
    thin_pathway = first_loan_pathway(result["credit_score"], result["data_completeness"])
    return render_template("score_report.html", applicant=raw, result=result,
                           matched_products=matched[:4], band=band, thin_pathway=thin_pathway,
                           score_data=_build_score_data(applicant_id, raw, result, matched[:4]),
                           app_data=_app_data(applicant_id, raw),
                           score_min=config.SCORE_MIN, score_max=config.SCORE_MAX,
                           approval_threshold=config.APPROVAL_SCORE_THRESHOLD)


# --- Improvement Portal -------------------------------------------------------
@app.route("/improvement/<applicant_id>")
@login_required
def improvement_portal(applicant_id):
    if portfolio_df.empty or applicant_id not in set(portfolio_df["applicant_id"]):
        return render_template("error.html", message=f"Applicant {applicant_id} not found."), 404
    raw = portfolio_df.loc[portfolio_df["applicant_id"] == applicant_id].iloc[0].to_dict()
    result = scoring.get_or_compute_full(applicant_id, raw, raw["entity_type"])
    result = lifecycle.apply_repayment_history(applicant_id, result)
    score_norm = int((result["credit_score"] - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN) * 100)
    matched = get_eligible_products(score_norm, raw.get("business_type", ""), raw.get("gender", ""))
    return render_template("improvement_portal.html", applicant=raw, result=result,
                           band=get_score_band_summary(score_norm),
                           score_data=_build_score_data(applicant_id, raw, result, matched[:4]),
                           matched_products=matched[:4],
                           thin_pathway=first_loan_pathway(
                               result["credit_score"], result["data_completeness"]),
                           app_data=_app_data(applicant_id, raw),
                           approval_threshold=config.APPROVAL_SCORE_THRESHOLD,
                           score_min=config.SCORE_MIN, score_max=config.SCORE_MAX)


# --- Credit Marketplace -------------------------------------------------------
def _next_listing_id(conn):
    n = conn.execute("SELECT COUNT(*) FROM marketplace_listings").fetchone()[0]
    return f"MKT-{n + 1:04d}"


def _listing_total_committed(conn, listing_id):
    row = conn.execute(
        "SELECT COALESCE(SUM(committed_amount),0) FROM lender_interests "
        "WHERE listing_id=? AND status='active'", (listing_id,)
    ).fetchone()
    return float(row[0])


@app.route("/marketplace")
@login_required
def marketplace():
    conn = get_db_connection()
    if not conn:
        return render_template("error.html", message="Database unavailable"), 500
    try:
        listings = conn.execute("""
            SELECT ml.*,
                COALESCE((SELECT SUM(li.committed_amount) FROM lender_interests li
                           WHERE li.listing_id = ml.listing_id AND li.status='active'), 0) AS total_committed,
                (SELECT COUNT(*) FROM lender_interests li
                 WHERE li.listing_id = ml.listing_id AND li.status='active') AS n_lenders
            FROM marketplace_listings ml
            WHERE ml.status = 'open'
            ORDER BY ml.credit_score DESC, ml.listed_at DESC
            LIMIT ?
        """, (config.MARKETPLACE_MAX_LISTINGS,)).fetchall()
        listings = [dict(r) for r in listings]

        stats_row = conn.execute("""
            SELECT COUNT(*) AS total,
                SUM(CASE WHEN status='open'   THEN 1 ELSE 0 END) AS open_count,
                SUM(CASE WHEN status='funded' THEN 1 ELSE 0 END) AS funded_count,
                SUM(CASE WHEN status='open'   THEN amount_requested ELSE 0 END) AS open_amount,
                AVG(CASE WHEN status='open'   THEN credit_score END) AS avg_score
            FROM marketplace_listings
        """).fetchone()
        stats = dict(stats_row) if stats_row else {}

        my_ids = {r[0] for r in conn.execute(
            "SELECT listing_id FROM lender_interests WHERE lender_username=? AND status='active'",
            (session.get("username"),)
        ).fetchall()}

        # Enrich with pct_funded
        for lst in listings:
            amt = lst["amount_requested"] or 1
            lst["pct_funded"] = min(100, round(lst["total_committed"] / amt * 100))
            tier_label, tier_tone = scoring.score_tier(lst["credit_score"])
            lst["tier_tone"] = tier_tone

    finally:
        conn.close()

    return render_template("marketplace.html", listings=listings, stats=stats,
                           my_listing_ids=my_ids,
                           min_score=config.MARKETPLACE_MIN_SCORE,
                           approval_threshold=config.APPROVAL_SCORE_THRESHOLD,
                           score_min=config.SCORE_MIN, score_max=config.SCORE_MAX)


@app.route("/marketplace/list/<applicant_id>", methods=["POST"])
@login_required
def marketplace_list_applicant(applicant_id):
    if portfolio_df.empty or applicant_id not in set(portfolio_df["applicant_id"]):
        return render_template("error.html", message="Applicant not found"), 404

    raw = portfolio_df.loc[portfolio_df["applicant_id"] == applicant_id].iloc[0].to_dict()
    result = scoring.get_or_compute_full(applicant_id, raw, raw["entity_type"])

    if (result["credit_score"] < config.MARKETPLACE_MIN_SCORE
            or any(g["severity"] == "critical" for g in result.get("guardrail_flags", []))):
        return redirect(url_for("applicant_detail", applicant_id=applicant_id))

    conn = get_db_connection()
    try:
        existing = conn.execute(
            "SELECT listing_id FROM marketplace_listings WHERE applicant_id=? AND status='open'",
            (applicant_id,)
        ).fetchone()
        if existing:
            return redirect(url_for("marketplace_listing_detail",
                                    listing_id=existing["listing_id"]))

        listing_id = _next_listing_id(conn)
        purpose = request.form.get("purpose", "Working capital")
        tier_label, _ = scoring.score_tier(result["credit_score"])
        score_norm = int((result["credit_score"] - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN) * 100)
        floor_rate = _risk_based_rate(result["credit_score"])
        interest_rate = float(request.form.get("interest_rate") or floor_rate)
        borrower_email = (request.form.get("borrower_email") or "").strip()[:120]
        borrower_phone = (request.form.get("borrower_phone") or "").strip()[:20]

        conn.execute("""
            INSERT INTO marketplace_listings
            (listing_id, applicant_id, listed_by, listed_at, amount_requested, tenure_months,
             credit_score, tier_label, entity_type, business_type, geography_tier, purpose,
             status, total_committed, interest_rate, borrower_email, borrower_phone)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,'open',0,?,?,?)
        """, (listing_id, applicant_id, session.get("username"),
              datetime.now(timezone.utc).isoformat(),
              float(raw.get("requested_loan_amount", 0) or 0),
              int(raw.get("requested_tenure_months", 24) or 24),
              result["credit_score"], tier_label,
              raw.get("entity_type"), raw.get("business_type"),
              raw.get("geography_tier"), purpose,
              interest_rate, borrower_email, borrower_phone))
        conn.commit()
    finally:
        conn.close()

    return redirect(url_for("marketplace_listing_detail", listing_id=listing_id))


@app.route("/marketplace/<listing_id>")
@login_required
def marketplace_listing_detail(listing_id):
    conn = get_db_connection()
    try:
        row = conn.execute(
            "SELECT * FROM marketplace_listings WHERE listing_id=?", (listing_id,)
        ).fetchone()
        if not row:
            return render_template("error.html", message="Listing not found"), 404
        listing = dict(row)

        commitments = [dict(r) for r in conn.execute(
            "SELECT interest_id, lender_username, committed_amount, proposed_rate, message, "
            "collateral_required, created_at "
            "FROM lender_interests WHERE listing_id=? AND status='active' ORDER BY created_at",
            (listing_id,)
        ).fetchall()]

        pending_proposals = [dict(r) for r in conn.execute(
            "SELECT interest_id, lender_username, committed_amount, proposed_rate, message, "
            "collateral_required, created_at "
            "FROM lender_interests WHERE listing_id=? AND status='pending' ORDER BY created_at",
            (listing_id,)
        ).fetchall()]
    finally:
        conn.close()

    total_committed = sum(c["committed_amount"] for c in commitments)
    pct_funded = min(100, round(total_committed / (listing["amount_requested"] or 1) * 100))
    already_committed = any(c["lender_username"] == session.get("username") for c in commitments)
    has_pending = any(p["lender_username"] == session.get("username") for p in pending_proposals)
    is_owner = listing["listed_by"] == session.get("username")

    blended_rate = listing.get("blended_rate") or listing.get("interest_rate")
    emi_monthly = None
    if blended_rate and listing.get("amount_requested"):
        emi_monthly = round(scoring.emi(listing["amount_requested"],
                                        listing["tenure_months"] or 24,
                                        float(blended_rate) / 100))

    result, raw, matched_products = {}, {}, []
    if not portfolio_df.empty and listing["applicant_id"] in set(portfolio_df["applicant_id"]):
        raw = portfolio_df.loc[portfolio_df["applicant_id"] == listing["applicant_id"]].iloc[0].to_dict()
        result = scoring.get_or_compute_full(listing["applicant_id"], raw, listing["entity_type"] or "Small Business")
        score_norm = int((result["credit_score"] - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN) * 100)
        matched_products = get_eligible_products(score_norm, listing["business_type"], raw.get("gender", ""))

    tier_label, tier_tone = scoring.score_tier(listing["credit_score"])
    listing["tier_label"] = tier_label
    listing["tier_tone"] = tier_tone
    floor_rate = _risk_based_rate(listing["credit_score"])

    return render_template("marketplace_listing.html",
                           listing=listing, commitments=commitments,
                           pending_proposals=pending_proposals,
                           total_committed=total_committed, pct_funded=pct_funded,
                           already_committed=already_committed, has_pending=has_pending,
                           is_owner=is_owner, blended_rate=blended_rate,
                           emi_monthly=emi_monthly, floor_rate=floor_rate,
                           floor=floor_rate,
                           matched_products=matched_products[:4],
                           result=result, raw=raw,
                           score_min=config.SCORE_MIN, score_max=config.SCORE_MAX,
                           approval_threshold=config.APPROVAL_SCORE_THRESHOLD)


@app.route("/marketplace/<listing_id>/commit", methods=["POST"])
@login_required
def marketplace_commit(listing_id):
    conn = get_db_connection()
    try:
        listing = conn.execute(
            "SELECT * FROM marketplace_listings WHERE listing_id=? AND status='open'",
            (listing_id,)
        ).fetchone()
        if not listing:
            return redirect(url_for("marketplace"))
        listing = dict(listing)

        amount = float(request.form.get("amount", 0) or 0)
        proposed_rate = request.form.get("proposed_rate", "")
        message = (request.form.get("message") or "").strip()[:300]
        collateral_required = 1 if request.form.get("collateral_required") == "1" else 0
        if amount <= 0:
            return redirect(url_for("marketplace_listing_detail", listing_id=listing_id))

        existing = conn.execute(
            "SELECT interest_id FROM lender_interests WHERE listing_id=? AND lender_username=? AND status IN ('active','pending')",
            (listing_id, session.get("username"))
        ).fetchone()
        if existing:
            return redirect(url_for("marketplace_listing_detail", listing_id=listing_id))

        floor_rate = float(listing.get("interest_rate") or 12.0)
        try:
            p_rate = float(proposed_rate) if proposed_rate else floor_rate
        except ValueError:
            p_rate = floor_rate

        # Auto-accept if proposed rate ≤ floor; otherwise pending (listing bank must approve)
        commit_status = "active" if p_rate <= floor_rate else "pending"

        interest_id = str(uuid.uuid4())[:12]
        conn.execute("""
            INSERT INTO lender_interests
            (interest_id, listing_id, lender_username, committed_amount, proposed_rate, message,
             collateral_required, status, created_at)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (interest_id, listing_id, session.get("username"), amount, p_rate, message,
              collateral_required, commit_status, datetime.now(timezone.utc).isoformat()))

        # Only count active (accepted) commitments toward total
        active_total = float(conn.execute(
            "SELECT COALESCE(SUM(committed_amount),0) FROM lender_interests WHERE listing_id=? AND status='active'",
            (listing_id,)
        ).fetchone()[0])
        if commit_status == "active":
            active_total += amount

        # Recalculate blended rate from all active commitments
        active_rows = conn.execute(
            "SELECT committed_amount, proposed_rate FROM lender_interests WHERE listing_id=? AND status='active'",
            (listing_id,)
        ).fetchall()
        if commit_status == "active":
            active_rows = list(active_rows) + [(amount, p_rate)]
        blended = _compute_blended_rate(active_rows, floor_rate)

        new_status = "funded" if active_total >= listing["amount_requested"] else "open"
        funded_at = datetime.now(timezone.utc).isoformat() if new_status == "funded" else None
        conn.execute(
            "UPDATE marketplace_listings SET total_committed=?, status=?, fully_funded_at=?, blended_rate=? WHERE listing_id=?",
            (active_total, new_status, funded_at, blended, listing_id)
        )
        conn.commit()

        if new_status == "funded" and listing["applicant_id"] and not lifecycle.active_loan_for(listing["applicant_id"]):
            if not portfolio_df.empty and listing["applicant_id"] in set(portfolio_df["applicant_id"]):
                raw = portfolio_df.loc[portfolio_df["applicant_id"] == listing["applicant_id"]].iloc[0].to_dict()
                res = scoring.get_or_compute_full(listing["applicant_id"], raw, listing["entity_type"] or "Small Business")
                lifecycle.disburse(listing["applicant_id"], listing["amount_requested"],
                                   listing["tenure_months"] or 24,
                                   res["credit_score"], res["probability_good"])
    finally:
        conn.close()

    return redirect(url_for("marketplace_listing_detail", listing_id=listing_id))


def _compute_blended_rate(active_rows, floor_rate):
    """Weighted average interest rate across all active commitments."""
    total_amt = sum(float(r[0]) for r in active_rows)
    if not total_amt:
        return floor_rate
    weighted = sum(float(r[0]) * float(r[1] if r[1] else floor_rate) for r in active_rows)
    return round(weighted / total_amt, 2)


@app.route("/marketplace/<listing_id>/respond/<interest_id>", methods=["POST"])
@login_required
def marketplace_respond(listing_id, interest_id):
    """Listing bank accepts or rejects a pending proposal."""
    action = request.form.get("action")  # 'accept' or 'reject'
    conn = get_db_connection()
    try:
        listing = conn.execute(
            "SELECT * FROM marketplace_listings WHERE listing_id=? AND listed_by=? AND status='open'",
            (listing_id, session.get("username"))
        ).fetchone()
        if not listing:
            return redirect(url_for("marketplace_listing_detail", listing_id=listing_id))
        listing = dict(listing)

        interest = conn.execute(
            "SELECT * FROM lender_interests WHERE interest_id=? AND listing_id=? AND status='pending'",
            (interest_id, listing_id)
        ).fetchone()
        if not interest:
            return redirect(url_for("marketplace_listing_detail", listing_id=listing_id))
        interest = dict(interest)

        new_status = "active" if action == "accept" else "rejected"
        conn.execute("UPDATE lender_interests SET status=? WHERE interest_id=?", (new_status, interest_id))

        # Recalculate total and blended rate
        active_rows = conn.execute(
            "SELECT committed_amount, proposed_rate FROM lender_interests WHERE listing_id=? AND status='active'",
            (listing_id,)
        ).fetchall()
        active_total = sum(float(r[0]) for r in active_rows)
        blended = _compute_blended_rate(active_rows, float(listing.get("interest_rate") or 12.0))

        new_lst_status = "funded" if active_total >= listing["amount_requested"] else "open"
        funded_at = datetime.now(timezone.utc).isoformat() if new_lst_status == "funded" else None
        conn.execute(
            "UPDATE marketplace_listings SET total_committed=?, status=?, fully_funded_at=?, blended_rate=? WHERE listing_id=?",
            (active_total, new_lst_status, funded_at, blended, listing_id)
        )
        conn.commit()

        if new_lst_status == "funded" and listing["applicant_id"] and not lifecycle.active_loan_for(listing["applicant_id"]):
            if not portfolio_df.empty and listing["applicant_id"] in set(portfolio_df["applicant_id"]):
                raw = portfolio_df.loc[portfolio_df["applicant_id"] == listing["applicant_id"]].iloc[0].to_dict()
                res = scoring.get_or_compute_full(listing["applicant_id"], raw, listing["entity_type"] or "Small Business")
                lifecycle.disburse(listing["applicant_id"], listing["amount_requested"],
                                   listing["tenure_months"] or 24,
                                   res["credit_score"], res["probability_good"])
    finally:
        conn.close()
    return redirect(url_for("marketplace_listing_detail", listing_id=listing_id))


@app.route("/marketplace/<listing_id>/cancel", methods=["POST"])
@login_required
def marketplace_cancel(listing_id):
    conn = get_db_connection()
    try:
        listing = conn.execute(
            "SELECT * FROM marketplace_listings WHERE listing_id=? AND listed_by=?",
            (listing_id, session.get("username"))
        ).fetchone()
        if listing and listing["status"] == "open":
            conn.execute("UPDATE marketplace_listings SET status='cancelled' WHERE listing_id=?",
                         (listing_id,))
            conn.commit()
    finally:
        conn.close()
    return redirect(url_for("marketplace"))


@app.route("/bank/proposals")
@login_required
def bank_proposals():
    """Central inbox: all pending lender proposals across the bank's listings."""
    if session.get("role") != "bank":
        return redirect(url_for("marketplace"))
    conn = get_db_connection()
    try:
        proposals = [dict(r) for r in conn.execute("""
            SELECT li.interest_id, li.listing_id, li.lender_username,
                   li.committed_amount, li.proposed_rate, li.message, li.created_at,
                   li.collateral_required,
                   ml.entity_type, ml.business_type, ml.applicant_id,
                   ml.interest_rate AS floor_rate, ml.amount_requested,
                   cs.credit_score
            FROM lender_interests li
            JOIN marketplace_listings ml ON ml.listing_id = li.listing_id
            LEFT JOIN credit_scores cs ON cs.applicant_id = ml.applicant_id
            WHERE ml.listed_by = ? AND li.status = 'pending'
            ORDER BY li.created_at DESC
        """, (session.get("username"),)).fetchall()]
    finally:
        conn.close()
    return render_template("bank_proposals.html", proposals=proposals)


# --- AI Underwriter Chatbot backend ------------------------------------------
def _chatbot_fallback(msg: str) -> str:
    msg = msg.lower()
    if any(w in msg for w in ["score", "why", "reason", "factor", "drove"]):
        return ("Your score is driven by payment reliability (utility/rent on-time rates), "
                "cash-flow consistency, and digital transaction footprint. The Reason Codes "
                "panel shows exactly which factors moved it and in which direction.")
    if any(w in msg for w in ["improve", "better", "increase", "raise", "boost"]):
        return ("The three fastest improvements are: (1) set up auto-pay for all utility bills, "
                "(2) route more purchases through UPI to build a verifiable digital trail, and "
                "(3) file GST returns on time if you run a business. The Improvement Plan tab "
                "shows your personalised top-3 actions with estimated point gains.")
    if any(w in msg for w in ["loan", "scheme", "mudra", "eligible", "product", "apply"]):
        return ("Based on your score, the Loan Schemes section on your report shows eligible "
                "government programmes — MUDRA Shishu/Kishore/Tarun, CGTMSE, and PM SVANidhi. "
                "Each lists the maximum amount, rate, and where to apply.")
    if any(w in msg for w in ["thin", "file", "band", "range", "confidence", "sparse"]):
        return ("A range instead of a single number means limited data. Sharing 6 months of "
                "bank statements, GST returns, or utility bills will usually narrow the band "
                "and lift the midpoint. The Confidence section explains what's missing.")
    if any(w in msg for w in ["fair", "bias", "gender", "geography", "discriminat"]):
        return ("Gender, geography, and business type are never model inputs — they're "
                "withheld entirely and used only in the Fairness Audit to check outcomes. "
                "If any group shows a lower approval rate, the audit flags it and traces "
                "whether it's a thin-file effect or a proxy bias.")
    if any(w in msg for w in ["market", "lender", "invest", "fund", "syndic"]):
        return ("The Credit Marketplace lets any registered lender browse and co-fund "
                "verified, scored applicants. Multiple lenders can share a single loan, "
                "spreading risk and driving down rates for good borrowers.")
    return ("I'm here to help with credit assessments, improvement steps, and loan products. "
            "Ask me why a score was given, how to raise it, or what schemes you qualify for.")


@app.route("/api/chatbot", methods=["POST"])
@login_required
def api_chatbot():
    data = request.get_json(silent=True) or {}
    user_message = (data.get("message") or "").strip()[:600]
    applicant_id = (data.get("applicant_id") or "").strip()

    if not user_message:
        return jsonify({"error": "No message"}), 400

    # Build applicant context block if on a specific applicant page
    context_block = ""
    if applicant_id and not portfolio_df.empty and applicant_id in set(portfolio_df["applicant_id"]):
        try:
            raw = portfolio_df.loc[portfolio_df["applicant_id"] == applicant_id].iloc[0].to_dict()
            result = scoring.get_or_compute_full(applicant_id, raw, raw["entity_type"])
            rc_labels = ", ".join(rc["label"] for rc in result.get("reason_codes", [])[:3])
            ip_labels = ", ".join(ip.get("label", ip.get("feature", "")) for ip in result.get("improvement_path", [])[:3])
            context_block = f"""
Applicant on screen: {applicant_id}
- Type: {raw.get('entity_type')} / {raw.get('business_type')}
- Location: {raw.get('geography_tier')}, {raw.get('geography_state')}
- Score: {result['credit_score']} ({result['tier_label']}) | {result['confidence_label']}
- Data completeness: {result['data_completeness'] * 100:.0f}%
- FOIR: {result['foir']['foir']:.0%} ({"PASS" if result['foir']['passes'] else "CAUTION"})
- Decision: {"APPROVED" if result['approved'] else "DECLINED"}
- Top reason codes: {rc_labels or "none computed yet"}
- Improvement steps: {ip_labels or "none computed yet"}
- Active guardrails: {len(result.get('guardrail_flags', []))} flag(s)
"""
        except Exception:
            pass

    system_prompt = f"""You are the CredVeda AI Underwriter — a specialist assistant embedded in the CredVeda alternative credit scoring platform for Indian individuals and small businesses (PS #4 solution).

CredVeda scores applicants from alternative data: UPI/digital payment flows, GST filing history, utility and rent payments, supplier relationships, and business vintage. Protected attributes (gender, geography, business type) are NEVER scoring inputs — only used in fairness audits.

{context_block}

Your role:
1. Explain WHY a specific score was given and what each reason code means in plain English
2. Tell borrowers the 2-3 concrete actions that will most improve their score in 6 months
3. Explain what government schemes (MUDRA, CGTMSE, PM SVANidhi, Kisan Credit Card, Stand-Up India) they may qualify for
4. Clarify confidence bands and thin-file handling
5. Explain responsible lending guardrails (FOIR, stress tests)
6. Describe the Credit Marketplace — how lenders co-fund vetted applicants

Rules: Answer in 3-5 clear sentences. Use ₹ for amounts. No ML jargon. If asked about the current applicant, use the context above. Be warm but precise."""

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key.startswith("YOUR_"):
        return jsonify({"reply": _chatbot_fallback(user_message), "source": "fallback"})

    try:
        resp = _requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}",
            json={
                "system_instruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_message}]}],
                "generationConfig": {"maxOutputTokens": config.CHATBOT_MAX_TOKENS, "temperature": 0.6},
            },
            timeout=20,
        )
        resp.raise_for_status()
        reply = resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        return jsonify({"reply": reply, "source": "gemini"})
    except Exception as exc:
        return jsonify({"reply": _chatbot_fallback(user_message), "source": "fallback",
                        "debug": str(exc)})


def _financing_offers(listing, commitments, blended_rate):
    """Builds the borrower's competing financing options: the bank's own direct
    loan (usually pricier, collateral-backed) vs the marketplace of lenders
    (competing on rate, collateral optional). Same score, same loan — the
    borrower picks the terms. Shared by the passport and the fundings page."""
    offers = []
    if not listing:
        return offers
    rejected = set((listing.get("rejected_options") or "").split(",")) - {""}
    amt = float(listing.get("amount_requested") or 0)
    tenure = listing.get("tenure_months") or 24
    if listing.get("bank_offer_rate"):
        br = float(listing["bank_offer_rate"])
        offers.append({
            "type": "bank", "name": "Bank direct loan", "funder": "Listing bank",
            "rate": round(br, 2),
            "collateral": bool(listing.get("bank_offer_collateral", 1)),
            "collateral_detail": listing.get("bank_offer_collateral_detail") or "Standard security / hypothecation",
            "emi": round(scoring.emi(amt, tenure, br / 100)) if amt else None,
            "amount": amt, "n_funders": 1, "available": True,
            "rejected": "bank" in rejected,
        })
    if commitments:
        mr = float(blended_rate or listing.get("interest_rate") or 12.0)
        any_collateral = any(c.get("collateral_required") for c in commitments)
        committed = float(listing.get("total_committed") or 0)
        offers.append({
            "type": "marketplace", "name": "Marketplace co-funding",
            "funder": f"{len(commitments)} lender" + ("s" if len(commitments) != 1 else ""),
            "rate": round(mr, 2),
            "collateral": any_collateral,
            "collateral_detail": "Some lenders require collateral" if any_collateral else "Unsecured — no collateral",
            "emi": round(scoring.emi(amt, tenure, mr / 100)) if amt else None,
            "amount": committed, "n_funders": len(commitments),
            "available": committed >= amt and amt > 0,
            "rejected": "marketplace" in rejected,
        })
    avail = [o for o in offers if o["available"] and not o["rejected"]]
    if avail:
        min(avail, key=lambda o: o["rate"])["recommended"] = True
    return offers


def _borrower_listing_offers(applicant_id):
    """Loads the borrower's active listing, its active commitments and the
    computed offers — the data the fundings page needs."""
    conn = get_db_connection()
    listing, commitments, blended_rate = None, [], None
    if conn:
        try:
            row = conn.execute(
                "SELECT * FROM marketplace_listings WHERE applicant_id=? ORDER BY listed_at DESC LIMIT 1",
                (applicant_id,)
            ).fetchone()
            if row:
                listing = dict(row)
                commitments = [dict(r) for r in conn.execute(
                    "SELECT lender_username, committed_amount, proposed_rate, collateral_required, created_at "
                    "FROM lender_interests WHERE listing_id=? AND status='active' ORDER BY created_at",
                    (listing["listing_id"],)
                ).fetchall()]
                blended_rate = listing.get("blended_rate") or listing.get("interest_rate")
        finally:
            conn.close()
    return listing, commitments, blended_rate, _financing_offers(listing, commitments, blended_rate)


@app.route("/fundings/<applicant_id>")
def borrower_fundings(applicant_id):
    """Borrower-facing page listing every financing offer for their loan, with
    accept / reject controls. Public — reached from the borrower portal."""
    listing, commitments, blended_rate, offers = _borrower_listing_offers(applicant_id)
    if not listing:
        return render_template("borrower_fundings.html", applicant_id=applicant_id,
                               listing=None, offers=[], chosen_option=None)
    return render_template("borrower_fundings.html",
                           applicant_id=applicant_id, listing=listing, offers=offers,
                           chosen_option=listing.get("chosen_option"))


# --- Credit Passport (public — no login) -------------------------------------
@app.route("/passport/<applicant_id>")
def credit_passport(applicant_id):
    """Public-facing credit passport. Borrower gets this URL from their lender.
    No login required — contains no raw personal data, only the score & terms."""
    if portfolio_df.empty or applicant_id not in set(portfolio_df["applicant_id"]):
        return render_template("error.html", message="Passport not found."), 404

    raw = portfolio_df.loc[portfolio_df["applicant_id"] == applicant_id].iloc[0].to_dict()
    result = scoring.get_or_compute_full(applicant_id, raw, raw["entity_type"])
    result = lifecycle.apply_repayment_history(applicant_id, result)

    score_norm = int((result["credit_score"] - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN) * 100)
    matched_products = get_eligible_products(score_norm, raw.get("business_type", ""), raw.get("gender", ""))
    thin_pathway = first_loan_pathway(result["credit_score"], result["data_completeness"])

    # Funding info from marketplace
    listing, commitments, blended_rate = None, [], None
    pending_commitments, declined_commitments = [], []
    conn = get_db_connection()
    if conn:
        try:
            row = conn.execute(
                "SELECT * FROM marketplace_listings WHERE applicant_id=? ORDER BY listed_at DESC LIMIT 1",
                (applicant_id,)
            ).fetchone()
            if row:
                listing = dict(row)
                commitments = [dict(r) for r in conn.execute(
                    "SELECT lender_username, committed_amount, proposed_rate, collateral_required, created_at "
                    "FROM lender_interests WHERE listing_id=? AND status='active' ORDER BY created_at",
                    (listing["listing_id"],)
                ).fetchall()]
                # Offers still waiting on the bank, and ones it turned down.
                # The borrower could previously only see accepted money, so a
                # lender asking for a higher rate — and the bank's answer —
                # was invisible to the person whose loan it is.
                pending_commitments = [dict(r) for r in conn.execute(
                    "SELECT lender_username, committed_amount, proposed_rate, created_at, message "
                    "FROM lender_interests WHERE listing_id=? AND status='pending' ORDER BY created_at",
                    (listing["listing_id"],)
                ).fetchall()]
                declined_commitments = [dict(r) for r in conn.execute(
                    "SELECT lender_username, committed_amount, proposed_rate, created_at "
                    "FROM lender_interests WHERE listing_id=? AND status='rejected' ORDER BY created_at",
                    (listing["listing_id"],)
                ).fetchall()]
                blended_rate = listing.get("blended_rate") or listing.get("interest_rate")
        finally:
            conn.close()

    emi_monthly = None
    if listing and blended_rate:
        emi_monthly = round(scoring.emi(listing["amount_requested"], listing["tenure_months"] or 24,
                                        blended_rate / 100))

    helped = [r for r in result.get("reason_codes", []) if r.get("impact") == "positive"][:3]
    # The passport showed only what helped. Half an explanation invites the
    # borrower to repeat whatever cost them points, so surface the negative
    # reason codes too -- the plain-language borrower text, not the model
    # internals (no SHAP chart, no anomaly flag, no "recommend decline").
    held_back = [r for r in result.get("reason_codes", []) if r.get("impact") == "negative"][:3]
    improvement = result.get("improvement_path", [])[:3]

    # Funding maths the borrower actually asks about: how much is still open.
    funding = None
    if listing:
        asked = float(listing.get("amount_requested") or 0)
        committed = float(listing.get("total_committed") or 0)
        pending_total = sum(float(c.get("committed_amount") or 0) for c in pending_commitments)
        funding = {
            "asked": asked,
            "committed": committed,
            "remaining": max(0.0, asked - committed),
            "pending_total": pending_total,
            "pct": min(100, round(committed / asked * 100)) if asked else 0,
            "n_lenders": len(commitments),
            "n_pending": len(pending_commitments),
        }

    offers = _financing_offers(listing, commitments, blended_rate)

    return render_template("credit_passport.html",
                           applicant_id=applicant_id,
                           result=result,
                           raw=raw,
                           listing=listing,
                           commitments=commitments,
                           offers=offers,
                           chosen_option=(listing.get("chosen_option") if listing else None),
                           blended_rate=blended_rate,
                           emi_monthly=emi_monthly,
                           matched_products=matched_products[:3],
                           thin_pathway=thin_pathway,
                           helped=helped,
                           held_back=held_back,
                           improvement=improvement,
                           pending_commitments=pending_commitments,
                           declined_commitments=declined_commitments,
                           funding=funding,
                           score_min=config.SCORE_MIN, score_max=config.SCORE_MAX,
                           approval_threshold=config.APPROVAL_SCORE_THRESHOLD)


@app.route("/fundings/<applicant_id>/action", methods=["POST"])
def borrower_offer_action(applicant_id):
    """Borrower accepts or rejects one competing financing option. Accepting one
    finalises the choice; rejecting just removes that option so the other can
    still be accepted. No login — reached from the fundings page."""
    action = request.form.get("action")     # 'accept' | 'reject'
    option = request.form.get("option")     # 'bank' | 'marketplace'
    if option not in ("bank", "marketplace") or action not in ("accept", "reject"):
        return redirect(url_for("borrower_fundings", applicant_id=applicant_id))
    conn = get_db_connection()
    if conn:
        try:
            row = conn.execute(
                "SELECT rejected_options FROM marketplace_listings WHERE applicant_id=?",
                (applicant_id,)
            ).fetchone()
            if row is not None:
                if action == "accept":
                    conn.execute(
                        "UPDATE marketplace_listings SET chosen_option=? WHERE applicant_id=?",
                        (option, applicant_id))
                else:  # reject: add to the rejected set
                    rejected = set((row["rejected_options"] or "").split(",")) - {""}
                    rejected.add(option)
                    conn.execute(
                        "UPDATE marketplace_listings SET rejected_options=? WHERE applicant_id=?",
                        (",".join(sorted(rejected)), applicant_id))
                conn.commit()
        finally:
            conn.close()
    return redirect(url_for("borrower_fundings", applicant_id=applicant_id))


# --- Bank Statement Parser (frontend-only) -----------------------------------
@app.route("/parse-statement")
@login_required
def parse_statement_page():
    return render_template("parse_statement.html")


# --- Bulk Batch Assessment ----------------------------------------------------
BULK_REQUIRED = [
    "applicant_id", "entity_type", "avg_monthly_inflow", "monthly_txn_count",
    "txn_bounce_rate", "digital_adoption_ratio", "utility_ontime_ratio",
    "rent_ontime_ratio", "vintage_months", "requested_loan_amount", "requested_tenure_months",
]

BULK_OPTIONAL = [
    "inflow_growth_rate_6m", "inflow_volatility_cv", "gst_registered",
    "gst_filing_regularity", "overdue_invoice_ratio", "supplier_concentration_hhi",
    "repeat_supplier_ratio", "existing_loan_count", "existing_monthly_emi",
    "bureau_score_available", "bureau_score_norm", "repayment_burden_ratio",
    "business_type", "geography_tier", "geography_state", "gender",
]


@app.route("/bulk-assess", methods=["GET", "POST"])
@login_required
def bulk_assess():
    if request.method == "GET":
        sample_fields = BULK_REQUIRED + BULK_OPTIONAL[:4]
        return render_template("bulk_assess.html", sample_fields=sample_fields,
                               required_fields=BULK_REQUIRED, optional_fields=BULK_OPTIONAL)

    # POST — process uploaded CSV
    file = request.files.get("csv_file")
    if not file or not file.filename.endswith(".csv"):
        return render_template("bulk_assess.html",
                               error="Please upload a .csv file.",
                               required_fields=BULK_REQUIRED, optional_fields=BULK_OPTIONAL)

    stream = io.TextIOWrapper(file.stream, encoding="utf-8-sig", errors="replace")
    try:
        reader = _csv.DictReader(stream)
        rows = list(reader)
    except Exception as e:
        return render_template("bulk_assess.html",
                               error=f"Could not parse CSV: {e}",
                               required_fields=BULK_REQUIRED, optional_fields=BULK_OPTIONAL)

    if not rows:
        return render_template("bulk_assess.html",
                               error="CSV is empty.",
                               required_fields=BULK_REQUIRED, optional_fields=BULK_OPTIONAL)

    results = []
    errors = []

    for i, row in enumerate(rows[:500]):  # cap at 500 rows per upload
        try:
            def _f(k, default=0.0):
                v = row.get(k, "")
                if v in (None, "", "NA", "N/A", "null", "nan"):
                    return None
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return None

            app_id = (row.get("applicant_id") or f"BULK-{i+1:04d}").strip()[:32]
            entity_type = (row.get("entity_type") or "Small Business").strip()

            raw = {
                "applicant_id": app_id,
                "entity_type": entity_type,
                "business_type": row.get("business_type", ""),
                "geography_tier": row.get("geography_tier", ""),
                "geography_state": row.get("geography_state", ""),
                "gender": row.get("gender", ""),
                "avg_monthly_inflow": _f("avg_monthly_inflow") or 20000,
                "inflow_growth_rate_6m": _f("inflow_growth_rate_6m") or 0.0,
                "inflow_volatility_cv": _f("inflow_volatility_cv") or 0.25,
                "monthly_txn_count": _f("monthly_txn_count") or 25,
                "txn_bounce_rate": _f("txn_bounce_rate") or 0.08,
                "digital_adoption_ratio": _f("digital_adoption_ratio") or 0.45,
                "gst_registered": int(bool(_f("gst_registered"))),
                "gst_filing_regularity": _f("gst_filing_regularity"),
                "overdue_invoice_ratio": _f("overdue_invoice_ratio"),
                "utility_ontime_ratio": _f("utility_ontime_ratio") or 0.75,
                "rent_ontime_ratio": _f("rent_ontime_ratio") or 0.75,
                "supplier_concentration_hhi": _f("supplier_concentration_hhi"),
                "repeat_supplier_ratio": _f("repeat_supplier_ratio"),
                "vintage_months": _f("vintage_months") or 12,
                "existing_loan_count": int(_f("existing_loan_count") or 0),
                "existing_monthly_emi": _f("existing_monthly_emi") or 0,
                "bureau_score_available": int(bool(_f("bureau_score_available"))),
                "bureau_score_norm": _f("bureau_score_norm"),
                "requested_loan_amount": _f("requested_loan_amount") or 100000,
                "requested_tenure_months": int(_f("requested_tenure_months") or 24),
            }
            # Compute repayment burden if not provided
            if _f("repayment_burden_ratio") is not None:
                raw["repayment_burden_ratio"] = _f("repayment_burden_ratio")

            res = scoring.score_full(raw, entity_type=entity_type)

            score_norm = int((res["credit_score"] - config.SCORE_MIN) / (config.SCORE_MAX - config.SCORE_MIN) * 100)
            matched = get_eligible_products(score_norm, raw.get("business_type", ""), raw.get("gender", ""))

            # Risk-based pricing
            score = res["credit_score"]
            recommended_rate = _risk_based_rate(score)

            critical_flags = [g for g in res.get("guardrail_flags", []) if g["severity"] == "critical"]

            results.append({
                "applicant_id": app_id,
                "entity_type": entity_type,
                "credit_score": res["credit_score"],
                "tier_label": res["tier_label"],
                "tier_tone": res["tier_tone"],
                "band_low": res["band_low"],
                "band_high": res["band_high"],
                "confidence_label": res["confidence_label"],
                "probability_good": round(res["probability_good"] * 100, 1),
                "approved": res["approved"],
                "foir_pct": round(res["foir"]["foir"] * 100, 1),
                "data_completeness_pct": round(res["data_completeness"] * 100, 1),
                "guardrail_severity": ("CRITICAL" if critical_flags else
                                       ("WARNING" if res.get("guardrail_flags") else "CLEAR")),
                "recommended_rate_pct": recommended_rate,
                "top_reason": (res["reason_codes"][0]["label"] if res.get("reason_codes") else ""),
                "top_scheme": (matched[0]["name"] if matched else "—"),
                "is_thin_file": res["data_completeness"] < 0.5,
            })
        except Exception as exc:
            errors.append({"row": i + 2, "applicant_id": row.get("applicant_id", "?"), "error": str(exc)})

    # Download CSV if requested
    if request.form.get("action") == "download":
        out = io.StringIO()
        if results:
            writer = _csv.DictWriter(out, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        from flask import Response
        return Response(
            out.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment;filename=credveda_bulk_results.csv"}
        )

    return render_template("bulk_assess.html",
                           results=results, errors=errors,
                           required_fields=BULK_REQUIRED, optional_fields=BULK_OPTIONAL,
                           n_total=len(results),
                           n_approved=sum(1 for r in results if r["approved"]),
                           n_critical=sum(1 for r in results if r["guardrail_severity"] == "CRITICAL"),
                           n_thin=sum(1 for r in results if r["is_thin_file"]),
                           avg_score=round(sum(r["credit_score"] for r in results) / len(results)) if results else 0)


# --- Risk-Based Pricing API ---------------------------------------------------
@app.route("/api/pricing/<applicant_id>")
@login_required
def api_pricing(applicant_id):
    """Returns recommended interest rate band based on credit score."""
    if portfolio_df.empty or applicant_id not in set(portfolio_df["applicant_id"]):
        return jsonify({"error": "not found"}), 404
    raw = portfolio_df.loc[portfolio_df["applicant_id"] == applicant_id].iloc[0].to_dict()
    result = scoring.get_or_compute_full(applicant_id, raw, raw["entity_type"])
    score = result["credit_score"]
    rate_mid = _risk_based_rate(score)
    rate_low = round(rate_mid - 0.75, 2)
    rate_high = round(rate_mid + 0.75, 2)
    ltv_cap = min(80, max(40, int((score - 300) / 6)))
    return jsonify({
        "credit_score": score,
        "tier_label": result["tier_label"],
        "rate_mid": rate_mid,
        "rate_low": max(8.5, rate_low),
        "rate_high": min(30.0, rate_high),
        "ltv_cap_pct": ltv_cap,
        "max_tenure_months": 60 if score >= 700 else (36 if score >= 600 else 24),
        "collateral_free": score >= 650,
        "cgtmse_eligible": score >= 600,
    })


# --- Setup and Run ------------------------------------------------------------
if __name__ == "__main__":
    init_user_db()
    _migrate_db()
    scoring.load_components()
    load_portfolio()

    print("\n--- Starting CredVeda: Alternative Credit Scoring (Flask) ---")
    print("Sign up for a new account, then open http://127.0.0.1:5000/login")
    print("---------------------------------------------------------------")
    app.run(debug=True)
# Updated on 2026-02-18
