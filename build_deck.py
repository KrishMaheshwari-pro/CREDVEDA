"""
Assemble the CredVeda systemic-risk pitch deck by filling the
TURING HACKX template, adding two themed slides, and embedding visuals.
Output -> 'CredVeda - TURING HACKX Presentation.pptx' (original template untouched).
"""
import copy
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.opc.constants import RELATIONSHIP_TYPE as RT
from pptx.oxml.ns import qn

SRC = "TURING HACKX Presentation.pptx"
OUT = "CredVeda - TURING HACKX Presentation.pptx"
A = "ppt_assets"

# palette (hex, no '#')
WHITE="FFFFFF"; LAV="E9D5FF"; MUTE="C4B5E0"; ACC="E879F9"; CYAN="34D7EE"
VIOLET="C084FC"; AMBER="FBBF24"; ROSE="FB4E7D"; GREEN="4ADE80"
TITLEF="Anton"; HEADF="Canva Sans Bold"; BODYF="Canva Sans"

prs = Presentation(SRC)
R_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"

# ---------------- helpers ----------------
def shp(slide, name):
    for s in slide.shapes:
        if s.name == name:
            return s
    return None

def prep_tf(tf):
    tf.word_wrap = True
    try:
        tf.auto_size = MSO_AUTO_SIZE.NONE
    except Exception:
        pass
    # keep first paragraph, drop the rest, clear runs of first
    for p in tf.paragraphs[1:]:
        p._p.getparent().remove(p._p)
    p0 = tf.paragraphs[0]
    for r in list(p0.runs):
        r._r.getparent().remove(r._r)
    return p0

def para(tf, text, size, color=WHITE, bold=False, font=BODYF, first=False,
         before=2, after=8, align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(before)
    p.space_after = Pt(after)
    run = p.add_run()
    run.text = text
    f = run.font
    f.size = Pt(size)
    f.bold = bold
    f.name = font
    f.color.rgb = RGBColor.from_string(color)
    return p

def set_title(slide, text, size=None):
    tb = shp(slide, "TextBox 11")
    if tb is None:
        return
    # normalize title geometry so long titles never wrap / overlap the body
    tb.left = Inches(0.9); tb.top = Inches(2.68)
    tb.width = Inches(18.3); tb.height = Inches(1.25)
    tb.text_frame.word_wrap = False
    p = tb.text_frame.paragraphs[0]
    if p.runs:
        p.runs[0].text = text
        for r in p.runs[1:]:
            r._r.getparent().remove(r._r)
    else:
        r = p.add_run(); r.text = text
    if size:
        p.runs[0].font.size = Pt(size)

def move(shape, left=None, top=None, width=None, height=None):
    if left is not None: shape.left = Inches(left)
    if top is not None: shape.top = Inches(top)
    if width is not None: shape.width = Inches(width)
    if height is not None: shape.height = Inches(height)

def add_text(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tb.text_frame.word_wrap = True
    try:
        tb.text_frame.auto_size = MSO_AUTO_SIZE.NONE
    except Exception:
        pass
    return tb

def add_img(slide, path, left, top, width):
    return slide.shapes.add_picture(f"{A}/{path}", Inches(left), Inches(top), width=Inches(width))

def duplicate(prs, index):
    src = prs.slides[index]
    new = prs.slides.add_slide(src.slide_layout)
    for s in list(new.shapes):
        s._element.getparent().remove(s._element)
    for s in src.shapes:
        new.shapes._spTree.append(copy.deepcopy(s._element))
    rmap = {}
    for rel in src.part.rels.values():
        if rel.reltype == RT.SLIDE_LAYOUT:
            continue
        if rel.is_external:
            nid = new.part.relate_to(rel.target_ref, rel.reltype, is_external=True)
        else:
            nid = new.part.relate_to(rel.target_part, rel.reltype)
        rmap[rel.rId] = nid
    for el in new.shapes._spTree.iter():
        for attr, val in list(el.attrib.items()):
            if attr.startswith("{"+R_NS+"}") and val in rmap:
                el.set(attr, rmap[val])
    # copy the per-slide background (<p:bg>) so the slide isn't white
    src_csld = src._element.find(qn("p:cSld"))
    new_csld = new._element.find(qn("p:cSld"))
    src_bg = src_csld.find(qn("p:bg"))
    if src_bg is not None and new_csld.find(qn("p:bg")) is None:
        new_csld.insert(0, copy.deepcopy(src_bg))
    return new

def reorder(prs, order):
    lst = prs.slides._sldIdLst
    ids = list(lst)
    for e in ids:
        lst.remove(e)
    for i in order:
        lst.append(ids[i])

BULLET = "▸ "

# ============================================================
# SLIDE 1 — TITLE
# ============================================================
s1 = prs.slides[0]
tb = shp(s1, "TextBox 13")
move(tb, left=2.72, top=3.5, width=9.4, height=6.4)
tf = tb.text_frame
prep_tf(tf)
para(tf, "THEME", 17, LAV, bold=True, font=HEADF, first=True, after=2)
para(tf, "Financial Market Shock Propagation & Systemic Risk", 27, WHITE, bold=True, font=HEADF, after=1)
para(tf, "FinTech  ·  Problem Statement 02", 15, MUTE, font=BODYF, after=16)
para(tf, "IDEA", 17, LAV, bold=True, font=HEADF, after=2)
para(tf, "CredVeda — Systemic Risk & Shock-Propagation Engine", 23, WHITE, bold=True, font=HEADF, after=16)
para(tf, "TEAM", 17, LAV, bold=True, font=HEADF, after=2)
para(tf, "Bilseri", 23, WHITE, bold=True, font=HEADF)

# ============================================================
# SLIDE 2 — PROPOSED SOLUTION
# ============================================================
s2 = prs.slides[1]
set_title(s2, "PROPOSED SOLUTION")
c = shp(s2, "TextBox 12")
move(c, left=0.99, top=4.15, width=13.7, height=3.4)
tf = c.text_frame
prep_tf(tf)
para(tf, "The gap: markets are deeply interconnected, yet risk tools score each entity in isolation — so systemic vulnerability stays invisible until it is too late.", 16, WHITE, bold=True, font=BODYF, first=True, after=6)
para(tf, BULLET+"CredVeda turns the market into a living network and simulates how a shock to one entity cascades across the whole system.", 15, LAV, after=4)
para(tf, BULLET+"3-layer engine:  1) Node Intelligence — AI fragility score per entity (ML + SHAP)   2) Interconnection Graph — data-driven edges (correlation, sector, supply-chain, macro-beta)   3) Shock Propagation — cascade along strongest pathways.", 15, LAV, after=4)
para(tf, BULLET+"Pick any shock scenario → watch it ripple → see who breaks, the exact contagion pathways, and a single Systemic Risk Index.", 15, LAV, after=4)
para(tf, BULLET+"Uniqueness: Explainable AI + network contagion (DebtRank-style) + live NLP shock detection + cross-border (US + India) spillover — one interactive platform.", 15, ACC, bold=True)
add_img(s2, "before_after.png", left=1.7, top=7.2, width=10.8)

# ============================================================
# SLIDE 3 (existing TECHNICAL) -> keep as Technical Approach
# ============================================================
s_tech = prs.slides[2]
set_title(s_tech, "TECHNICAL APPROACH")
c = shp(s_tech, "TextBox 12")
move(c, left=0.99, top=4.2, width=7.5, height=6.4)
tf = c.text_frame
prep_tf(tf)
rows = [
    ("Data & Ingestion  (built)", "Python · yfinance · FRED · NewsAPI · API-Ninjas · SQLite"),
    ("Graph & Contagion", "NetworkX · NumPy/SciPy · DebtRank / threshold-cascade engine"),
    ("ML & Explainability", "scikit-learn (RandomForest / XGBoost) · SHAP"),
    ("NLP Shock Detection", "VADER sentiment + event-keyword engine"),
    ("AI Copilot", "Gemini / OpenAI LLM — explains scenarios in plain English"),
    ("Visualization", "Plotly / Cytoscape.js network · Chart.js · animated cascade"),
    ("Application  (built)", "Flask · TailwindCSS · session auth"),
]
first = True
for head, body in rows:
    para(tf, BULLET+head, 16, ACC, bold=True, font=HEADF, first=first, after=0)
    para(tf, "    "+body, 13.5, LAV, after=7)
    first = False
add_img(s_tech, "architecture.png", left=8.2, top=4.35, width=6.9)

# ============================================================
# SLIDE 4 (existing FEASIBILITY)
# ============================================================
s_feas = prs.slides[3]
set_title(s_feas, "FEASIBILITY & VIABILITY")
c = shp(s_feas, "TextBox 12")
move(c, left=0.99, top=4.2, width=13.7, height=6.4)
tf = c.text_frame
prep_tf(tf)
para(tf, "Head start (de-risked): a working data pipeline for ~250 US + India entities, trained ML models, SHAP explainability, and a full Flask app already exist — the systemic-risk layer plugs on top.", 17, WHITE, bold=True, first=True, after=7)
para(tf, BULLET+"Data: all sources are free / public APIs already integrated (yfinance, FRED, NewsAPI, API-Ninjas).", 16, LAV, after=6)
para(tf, "Challenges → Mitigation", 16, ACC, bold=True, font=HEADF, after=3)
para(tf, "    • Spurious edges → rolling-window correlation + sector priors + significance thresholds", 15, LAV, after=3)
para(tf, "    • Propagation realism → grounded in DebtRank / threshold models, calibrated to the 2020 crash", 15, LAV, after=3)
para(tf, "    • Performance → sparsified graph, caching, nightly pre-compute", 15, LAV, after=3)
para(tf, "    • Demo reliability → deterministic user-defined scenarios (live news detection as a bonus)", 15, LAV, after=8)
para(tf, BULLET+"Viability: directly valuable to regulators, banks, asset managers & fintechs for stress-testing and early warning.", 16, GREEN, bold=True)

# ============================================================
# SLIDE 5 (existing IMPACT)
# ============================================================
s_imp = prs.slides[4]
set_title(s_imp, "IMPACT & BENEFITS")
c = shp(s_imp, "TextBox 12")
move(c, left=0.99, top=4.2, width=13.7, height=2.9)
tf = c.text_frame
prep_tf(tf)
para(tf, BULLET+"Regulators / central banks: spot 'too-connected-to-fail' nodes and systemic weak points before a crisis.", 16, LAV, first=True, after=5)
para(tf, BULLET+"Banks / asset managers: stress-test portfolios and quantify hidden contagion exposure.", 16, LAV, after=5)
para(tf, BULLET+"Fintechs / investors: an understandable view of how one shock moves the whole market.", 16, LAV, after=5)
para(tf, BULLET+"Big picture: shifts the market from isolated indicators → interconnected early warning.", 16, ACC, bold=True)
add_img(s_imp, "risk_index.png", left=0.99, top=7.3, width=12.2)

# ============================================================
# SLIDE 6 (existing REFERENCES)
# ============================================================
s_ref = prs.slides[5]
set_title(s_ref, "RESEARCH & REFERENCES")
c = shp(s_ref, "TextBox 12")
move(c, left=0.99, top=4.2, width=13.7, height=6.4)
tf = c.text_frame
prep_tf(tf)
para(tf, "Systemic-risk & contagion models", 16, ACC, bold=True, font=HEADF, first=True, after=3)
para(tf, "    • DebtRank — Battiston et al. (2012)   • Financial contagion in networks — Gai & Kapadia (2010)   • Eisenberg–Noe clearing (2001)", 14.5, LAV, after=7)
para(tf, "Network science", 16, ACC, bold=True, font=HEADF, after=3)
para(tf, "    • Cascade / threshold models — Watts (2002)   • Centrality measures (eigenvector, betweenness)", 14.5, LAV, after=7)
para(tf, "Explainable AI & NLP", 16, ACC, bold=True, font=HEADF, after=3)
para(tf, "    • SHAP — Lundberg & Lee (2017)   • VADER sentiment — Hutto & Gilbert (2014)", 14.5, LAV, after=7)
para(tf, "Data sources", 16, ACC, bold=True, font=HEADF, after=3)
para(tf, "    • Yahoo Finance (yfinance)   • FRED / Federal Reserve (DGS10)   • NewsAPI   • API-Ninjas earnings transcripts", 14.5, LAV)

# ============================================================
# NEW SLIDE A — HOW IT WORKS (dup of slide 1 / Solution structure)
# ============================================================
howto = duplicate(prs, 1)
# reset any leftover picture from the duplicated Solution slide (before_after)
for pic in [s for s in howto.shapes if s.shape_type == 13]:
    pic._element.getparent().remove(pic._element)
set_title(howto, "SHOCK PROPAGATION IN ACTION")
c = shp(howto, "TextBox 12")
tf = c.text_frame; prep_tf(tf)
move(c, left=8.9, top=7.5, width=6.4, height=3.4)
para(tf, "Live simulation", 16, ACC, bold=True, font=HEADF, first=True, after=6)
para(tf, BULLET+"Shock originates at one entity (e.g. a systemic bank)", 14.5, LAV, after=5)
para(tf, BULLET+"Contagion spreads along the strongest links", 14.5, LAV, after=5)
para(tf, BULLET+"Fragile, highly-connected nodes fail first", 14.5, LAV, after=5)
para(tf, BULLET+"Result: 93% of the network impacted — mapped & explained", 14.5, AMBER, bold=True)
add_img(howto, "pipeline.png", left=0.99, top=3.95, width=12.8)
add_img(howto, "cascade.gif", left=0.99, top=7.35, width=6.9)

# ============================================================
# NEW SLIDE B — FEATURE SUITE (dup of Solution structure)
# ============================================================
feat = duplicate(prs, 1)
for pic in [s for s in feat.shapes if s.shape_type == 13]:
    pic._element.getparent().remove(pic._element)
set_title(feat, "FEATURE SUITE")
# left column reuses TextBox 12
c = shp(feat, "TextBox 12")
move(c, left=0.99, top=4.15, width=7.4, height=6.6)
tf = c.text_frame; prep_tf(tf)
para(tf, "CORE ENGINE", 18, CYAN, bold=True, font=HEADF, first=True, after=6)
core = [
    "Interconnection graph of financial entities",
    "Dependency & exposure modeling",
    "Multi-scenario shock simulator",
    "Impact-pathway detection",
    "Vulnerable-entity ranking",
    "Systemic-impact estimation",
    "Interactive network visualization",
    "Scenario comparison (side-by-side)",
    "Explainability panel (why edges, why failures)",
]
for i, it in enumerate(core):
    para(tf, BULLET+it, 15, LAV, after=4)
# right column new textbox
rc = add_text(feat, 8.5, 4.15, 6.2, 6.6)
tf2 = rc.text_frame
para(tf2, "ADVANCED  —  THE WOW", 18, ACC, bold=True, font=HEADF, first=True, after=6)
adv = [
    "Reverse stress testing ('what breaks 30% of the market?')",
    "Monte-Carlo systemic simulation",
    "SIFI 'too-connected-to-fail' detector (centrality)",
    "Early-warning Systemic Risk Index (a 'VIX for contagion')",
    "AI Copilot — explains scenarios in plain English",
    "Temporal graph — how interconnection tightens before crises",
    "Cross-border US ↔ India contagion",
    "Portfolio systemic-exposure upload",
]
for it in adv:
    para(tf2, BULLET+it, 15, LAV, after=4)

# ============================================================
# NEW SLIDE C — ALGORITHMS & ML MODELS (dup of Solution structure)
# ============================================================
mlslide = duplicate(prs, 1)
for pic in [s for s in mlslide.shapes if s.shape_type == 13]:
    pic._element.getparent().remove(pic._element)
set_title(mlslide, "ALGORITHMS & ML MODELS")
c = shp(mlslide, "TextBox 12")
tf = c.text_frame; prep_tf(tf)
move(c, left=0.99, top=3.98, width=14.0, height=0.55)
para(tf, "Every model powering CredVeda's fragility score — feeding the systemic-propagation algorithms.", 14, LAV, first=True, font=BODYF)
add_img(mlslide, "ml_algos.png", left=0.99, top=4.55, width=13.3)

# ============================================================
# REORDER  (current indices: 0 title,1 sol,2 tech,3 feas,4 imp,5 ref,
#           6 howto, 7 feat, 8 mlslide)
# final: Title, Solution, HowItWorks, Technical, Algorithms&ML, Feature, Feasibility, Impact, References
# ============================================================
reorder(prs, [0, 1, 6, 2, 8, 7, 3, 4, 5])

prs.save(OUT)
print("SAVED:", OUT)
print("slides:", len(prs.slides))
