# ⚡ CredVeda — Alternative Credit Scoring for Individuals & Small Businesses

> Built for Hackathon PS #4: *Alternative Credit Scoring for Individuals and Small Businesses*.
> A large share of Indian borrowers and micro-enterprises have little or no formal credit history.
> CredVeda scores them anyway — from the data they already generate every day — and explains every score
> in plain language.

---

## The problem this solves

A bureau score either doesn't exist for these borrowers or is meaningless, so lenders either reject them
outright or price risk crudely. CredVeda's scope is exactly the PS's scope: **the scoring model and its
explainability**, not a full document-driven underwriting workflow.

## What CredVeda actually does

| PS #4 requirement | Where it lives |
|---|---|
| Alternative feature engineering (digital payments, GST/invoices, utility/rent, suppliers, vintage) | `data_ingestion.py`, feature schema in `config.FEATURE_COLS` |
| Explainable modelling — score + top contributing factors in plain language | `scoring.py` (`shap_factors`, `map_reason_codes`), applicant detail page |
| …"so a rejection can be justified **to the applicant**" | **Applicant view** at `/applicant/<id>/explanation` — same decision, zero model jargon, second person, printable |
| Reason codes usable by a lending operations team | `config.REASON_CODES` (RC01–RC14) |
| Fairness testing across geography, gender, business type | `fairness.py`, **Fairness Report** page |
| Thin-file handling — confidence band **instead of** false precision | `scoring.py` (`compute_completeness`, `confidence_band`); below 75% completeness the UI drops the point score entirely and leads with the band (`band_first`) |
| Responsible lending guardrails — unsustainable repayment burden | `scoring.py` (`guardrail_checks`) |
| Bonus: borrower improvement path, next ~6 months | `scoring.py` (`improvement_path`) — a real what-if simulation against the trained model, not a canned tip |

---

## ⚠️ Important: this uses synthetic data

Real UPI/Account Aggregator transaction feeds, the GST portal, utility/rent aggregators, and credit
bureaus all require regulated business partnerships that aren't accessible for a hackathon build. Instead,
`data_ingestion.py` generates a large (4,000-row), realistic **synthetic** population of individuals and
small businesses, with deliberate, non-random correlations to business type and geography (e.g. rural
applicants genuinely show lower digital footprints) so the fairness and thin-file logic have something
real to detect — not injected bias, just realistic data sparsity.

The rest of the pipeline (feature schema → model → scoring → fairness → guardrails) is written to the
feature schema in `config.py`, not to the synthetic generator. **Swapping in real data connectors later is
a data-layer change** (replace `data_ingestion.py`), not a rebuild.

---

## 🏗️ Architecture

```
config.py            Feature schema, reason codes, fairness/guardrail thresholds
data_ingestion.py     Synthetic applicant + alt-data generator -> 'applicants' table
database.py           SQLite schema setup (users, credit_scores tables)
model_training.py     Trains a RandomForestClassifier on the alt-data features, saves model/scaler/percentiles
scoring.py             Core scoring library: score, confidence band, reason codes, guardrails, improvement path
fairness.py            Group fairness audit (four-fifths adverse-impact rule)
evaluate_models.py     Cross-validated model evaluation + fairness summary ("model card")
generate_report_graphs.py  PNG charts for a deck: confusion matrix, score distribution, feature importance, fairness
app.py                 Flask web app: auth, dashboard, applicant detail, live "New Assessment" form, fairness report
chatbot.py             Rule-based assistant widget (HTML/CSS/JS), injected into every page via a context processor
static/css/credveda.css  The design system — tokens, components, light/dark themes, responsive rules
templates/             Jinja2 templates (Chart.js for charts, inline SVG for the score gauge)
```

**Frontend:** one stylesheet driven by CSS custom properties, no UI framework. Light/dark themes swap by
setting `data-theme` on `<html>` **server-side from the session**, so there's no flash of the wrong theme on
load. Layout is responsive down to phone width.

**Explainability without SHAP:** this environment's `numpy` is newer than `shap`'s `numba` dependency
supports. Rather than downgrade `numpy` globally (which could break other projects on the machine),
`scoring.py` implements the same additive tree-contribution decomposition SHAP's `TreeExplainer` uses for
tree ensembles directly (`forest_explain`), with zero extra dependencies — bias + per-feature contributions
reproduces `model.predict_proba` exactly.

**Performance note:** full explanations (reason codes, improvement path) walk every tree in the forest and
take ~0.3–0.5s per applicant. The whole 4,000-applicant portfolio is scored with one vectorised
`predict_proba` call (`scoring.generate_scores`, a few seconds total); full explanations are computed lazily,
the first time an applicant's detail page is opened, and cached back into the database.

---

## 🚀 Getting started

```bash
python -m venv venv
venv\Scripts\activate          # Windows; source venv/bin/activate on Mac/Linux
pip install -r requirements.txt

python database.py             # create tables
python data_ingestion.py       # generate the synthetic applicant population
python model_training.py       # train the model
python scoring.py              # batch-score the whole portfolio

python app.py                  # http://127.0.0.1:5000
```

Sign up for a new account (any username/password — it's a local demo, no email required) and log in.

Optional, after the above:
- `python evaluate_models.py` — cross-validated ROC-AUC/accuracy + fairness summary printed to console.
- `python generate_report_graphs.py` — saves PNG charts to `output/` for a pitch deck.

---

## 📊 Using the website

- **Home** — overview + live portfolio stats.
- **Dashboard** — browse all 4,000 scored applicants; filter by entity type, business type, geography tier,
  gender, or confidence level; click any applicant for their full explanation.
- **Applicant detail page** — score gauge, confidence band, guardrail flags, reason codes, a feature-contribution
  chart, the improvement path, and the raw alternative-data snapshot. This is the **lender/underwriter** view.
- **Applicant view** (`/applicant/<id>/explanation`) — the same decision rewritten for the borrower: no model
  terms, no reason-code numbers, what helped, what hurt, and the three things to do next. Printable.
- **New Assessment** (`/apply`) — type in an alternative-data profile yourself (no applicant lookup) and get
  an instant score, reason codes, guardrails and improvement path, computed live by the trained model.
  Three one-click preset scenarios (*established kirana store*, *thin-file rural worker*, *over-leveraged
  applicant*) demonstrate each behaviour — strong approval, band-first thin-file handling, and the
  affordability guardrail overriding the score.
- **Fairness Report** (`/fairness`) — approval rate and average score by gender, geography tier, business
  type and entity type, with the four-fifths rule applied and any disparity flagged and explained.
- **AI Features / About** — how the scoring, explainability, fairness and thin-file logic actually work.

---

## 🛠️ Tech Stack

- **Backend:** Python, Flask, SQLite
- **ML:** scikit-learn (RandomForestClassifier), a hand-written SHAP-style tree explainer, pandas, numpy
- **Frontend:** Jinja2 templates, Tailwind (CDN), Chart.js

---

## 🔮 Extending this toward production

- Replace `data_ingestion.py` with real connectors (Account Aggregator for UPI/bank data, GSTN API for
  filings, a utility/rent BBPS aggregator) — the feature schema in `config.py` is the contract, and nothing
  downstream needs to change.
- Persist real applications (currently `/apply` is stateless-by-design, matching the PS's scope of "the
  scoring model and its explainability, not a full underwriting workflow").
- Swap the illustrative `ASSUMED_ANNUAL_INTEREST_RATE` and `APPROVAL_SCORE_THRESHOLD` in `config.py` for a
  real lender's actual pricing/policy.
- Add authentication scoped to loan officers vs. applicants, and an audit trail for score overrides.

---

## 📜 License

MIT License.
