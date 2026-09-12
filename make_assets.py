"""
Generate custom, on-theme visuals for the CredVeda / TURING HACKX deck.
Palette matches the template's deep-purple/violet look. No stock photos.
Outputs -> ppt_assets/
"""
import os
import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Wedge
from matplotlib.lines import Line2D
from PIL import Image

OUT = "ppt_assets"
os.makedirs(OUT, exist_ok=True)

# ---- Theme palette (matches TURING HACKX: black bg, red/orange accents) ----
BG     = "#000000"   # pure black — matches the slide background exactly
PANEL  = "#141418"   # subtle panel
PANEL2 = "#1f1f27"
EDGE   = "#3a3a46"   # dim grey edges
SAFE   = "#2dd4bf"   # teal (healthy)
WARN   = "#ff9f1c"   # orange (stressed)
FAIL   = "#ff453a"   # red (distressed) = theme accent
IMP    = "#ff6b6b"   # coral
TEXT   = "#ffffff"
MUTE   = "#9aa0a6"
ACCENT = "#ff453a"   # red accent
GOOD   = "#2dd4bf"   # teal
CORAL  = "#ff6b6b"
AMBER  = "#ffd166"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "text.color": TEXT,
    "axes.edgecolor": EDGE,
})

def hx(c):
    c = c.lstrip("#")
    return np.array([int(c[i:i+2], 16) for i in (0, 2, 4)]) / 255.0

def lerp(a, b, t):
    return a + (b - a) * t

def stress_color(s):
    """0 -> cyan, 0.5 -> amber, 1 -> rose."""
    s = max(0.0, min(1.0, s))
    if s <= 0.5:
        t = s / 0.5
        rgb = lerp(hx(SAFE), hx(WARN), t)
    else:
        t = (s - 0.5) / 0.5
        rgb = lerp(hx(WARN), hx(FAIL), t)
    return tuple(rgb)

# ============================================================
# NETWORK DEFINITION (curated for legibility)
# ============================================================
NODES = {
    # name: (x, y, sector, fragility 0..1, importance 0..1)
    "AAPL":     (1.9, 7.3, "Tech",   0.30, 0.9),
    "MSFT":     (3.2, 8.0, "Tech",   0.28, 0.85),
    "NVDA":     (2.6, 6.1, "Tech",   0.45, 0.8),
    "AMZN":     (1.7, 4.4, "Retail", 0.40, 0.75),
    "WMT":      (3.1, 3.3, "Retail", 0.25, 0.6),
    "GS":       (6.2, 6.4, "Banks",  0.55, 0.7),
    "JPM":      (7.4, 7.6, "Banks",  0.50, 1.0),
    "BAC":      (8.6, 6.6, "Banks",  0.70, 0.7),
    "RATES↑": (9.6, 8.5, "Macro", 0.20, 1.0),
    "TSLA":     (7.7, 3.6, "Auto",   0.65, 0.75),
    "F":        (6.3, 2.6, "Auto",   0.85, 0.5),
    "XOM":      (11.7, 7.1, "Energy", 0.35, 0.7),
    "CVX":      (12.9, 6.0, "Energy", 0.33, 0.6),
    "RELIANCE": (12.2, 3.7, "India",  0.45, 0.8),
    "TCS":      (13.3, 4.7, "India",  0.30, 0.6),
    "HDFCBK":   (11.5, 2.7, "India",  0.55, 0.65),
}
EDGES = [
    ("AAPL", "MSFT", .9), ("AAPL", "NVDA", .8), ("MSFT", "NVDA", .8),
    ("NVDA", "TSLA", .6), ("AMZN", "WMT", .7), ("AAPL", "AMZN", .5),
    ("GS", "JPM", .9), ("JPM", "BAC", .9), ("GS", "BAC", .7),
    ("RATES↑", "JPM", .8), ("RATES↑", "GS", .6), ("RATES↑", "BAC", .6),
    ("RATES↑", "XOM", .5), ("RATES↑", "RELIANCE", .4),
    ("JPM", "TSLA", .5), ("BAC", "F", .6), ("TSLA", "F", .7),
    ("XOM", "CVX", .9), ("XOM", "TSLA", .4), ("JPM", "XOM", .5),
    ("JPM", "AAPL", .4), ("JPM", "RELIANCE", .35),
    ("RELIANCE", "TCS", .8), ("RELIANCE", "HDFCBK", .7), ("TCS", "HDFCBK", .6),
    ("BAC", "HDFCBK", .3), ("AMZN", "TSLA", .35),
]

def neighbors(n):
    out = []
    for a, b, w in EDGES:
        if a == n:
            out.append((b, w))
        elif b == n:
            out.append((a, w))
    return out

def propagate(origin, rounds=5, decay=0.6):
    stress = {n: 0.0 for n in NODES}
    stress[origin] = 1.0
    seq = [dict(stress)]
    for _ in range(rounds):
        nxt = dict(stress)
        for n in NODES:
            if n == origin:
                nxt[n] = 1.0
                continue
            inflow = sum(w * stress[m] for m, w in neighbors(n))
            frag = NODES[n][3]
            nxt[n] = min(1.0, stress[n] + decay * inflow * (0.45 + 0.7 * frag))
        stress = nxt
        seq.append(dict(stress))
    return seq

ORIGIN = "JPM"
SEQ = propagate(ORIGIN, rounds=5)

def draw_network(ax, stress, highlight_edges=True, show_labels=True, pathways=None):
    ax.set_facecolor(BG)
    ax.set_xlim(0.5, 14.2)
    ax.set_ylim(1.5, 9.4)
    ax.axis("off")
    # edges
    for a, b, w in EDGES:
        xa, ya = NODES[a][0], NODES[a][1]
        xb, yb = NODES[b][0], NODES[b][1]
        active = highlight_edges and stress[a] > 0.12 and stress[b] > 0.12
        col = ACCENT if active else EDGE
        alpha = 0.85 if active else 0.28
        lw = 1.2 + 3.2 * w if active else 0.7 + 1.6 * w
        ax.plot([xa, xb], [ya, yb], color=col, alpha=alpha, lw=lw, zorder=1,
                solid_capstyle="round")
    # pathways (thick glowing magenta)
    if pathways:
        for path in pathways:
            xs = [NODES[p][0] for p in path]
            ys = [NODES[p][1] for p in path]
            for gl, a in [(11, .10), (7, .18), (3.2, .95)]:
                ax.plot(xs, ys, color=ACCENT, lw=gl, alpha=a, zorder=2,
                        solid_capstyle="round")
    # nodes
    for n, (x, y, sec, frag, imp) in NODES.items():
        s = stress[n]
        size = 260 + 900 * imp
        # glow for stressed
        if s > 0.05:
            ax.scatter([x], [y], s=size * (2.6 + 3 * s), color=stress_color(s),
                       alpha=0.16 + 0.20 * s, zorder=3, edgecolors="none")
        ax.scatter([x], [y], s=size, color=stress_color(s), zorder=4,
                   edgecolors="white", linewidths=1.6)
        if show_labels:
            ax.text(x, y - 0.02, n, ha="center", va="center", fontsize=7.6,
                    fontweight="bold", color="#0a0416", zorder=5)

def render_frame(stress, caption, sub, pathways=None):
    fig, ax = plt.subplots(figsize=(11.2, 6.3), dpi=170)
    fig.patch.set_facecolor(BG)
    draw_network(ax, stress, pathways=pathways)
    # title band
    ax.text(0.5, 9.15, "MARKET CONTAGION SIMULATION", fontsize=15,
            fontweight="bold", color=TEXT, ha="left")
    ax.text(0.5, 8.72, caption, fontsize=11.5, color=ACCENT, ha="left",
            fontweight="bold")
    # stats box
    stressed = sum(1 for v in stress.values() if v > 0.35)
    total = len(NODES)
    pct = int(100 * stressed / total)
    ax.text(13.9, 9.15, f"Systemic impact", fontsize=10, color=MUTE, ha="right")
    ax.text(13.9, 8.6, f"{pct}%", fontsize=26, color=FAIL if pct > 40 else WARN,
            ha="right", fontweight="bold")
    ax.text(13.9, 8.2, f"{stressed}/{total} entities stressed", fontsize=8.5,
            color=MUTE, ha="right")
    ax.text(0.5, 1.75, sub, fontsize=9, color=MUTE, ha="left", style="italic")
    # legend
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SAFE,
               markersize=10, label="Healthy", markeredgecolor="white"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=WARN,
               markersize=10, label="Stressed", markeredgecolor="white"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=FAIL,
               markersize=10, label="Distressed", markeredgecolor="white"),
    ]
    leg = ax.legend(handles=handles, loc="lower right", frameon=False,
                    fontsize=8.5, labelcolor=TEXT, ncol=3,
                    bbox_to_anchor=(1.0, -0.02))
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor=BG, dpi=170)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")

def bfs_path(origin, target):
    from collections import deque
    q = deque([[origin]])
    seen = {origin}
    while q:
        path = q.popleft()
        if path[-1] == target:
            return path
        for m, w in neighbors(path[-1]):
            if m not in seen:
                seen.add(m)
                q.append(path + [m])
    return None

# ---- Animated cascade GIF ----
captions = [
    ("Round 0  —  Shock originates at JPM (systemically important bank)",
     "A single institution is hit — in isolation, this looks contained."),
    ("Round 1  —  Distress spreads to directly connected banks & rates",
     "Correlation, exposure and macro links transmit the shock outward."),
    ("Round 2  —  Contagion crosses into autos, tech and energy",
     "Second-order neighbours inherit stress along the strongest pathways."),
    ("Round 3  —  Fragile, highly-connected nodes begin to fail",
     "Low-buffer entities (high fragility score) tip into distress first."),
    ("Round 4  —  Cross-border spillover reaches Indian market",
     "Systemic risk is now visible — far beyond the original shock."),
    ("Round 5  —  Full systemic impact mapped",
     "CredVeda quantifies who breaks, the pathways, and total impact."),
]
frames = []
for i, st in enumerate(SEQ):
    cap, sub = captions[min(i, len(captions) - 1)]
    frames.append(render_frame(st, cap, sub))
# hold last frame longer
final_paths = [p for p in [bfs_path(ORIGIN, t) for t in ["F", "HDFCBK", "CVX", "TSLA"]] if p]
hero_frame = render_frame(SEQ[-1], captions[-1][0], captions[-1][1], pathways=final_paths)
frames += [hero_frame] * 3
durations = [1100] * len(SEQ) + [1400] * 3
frames[0].save(os.path.join(OUT, "cascade.gif"), save_all=True,
               append_images=frames[1:], duration=durations, loop=0, dispose=2)
hero_frame.save(os.path.join(OUT, "network_hero.png"))
print("saved cascade.gif + network_hero.png")

# ============================================================
# PIPELINE (5 stages)
# ============================================================
def rounded(ax, x, y, w, h, fc, ec, lw=2, rad=0.04):
    box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.01,rounding_size={rad}",
                         linewidth=lw, edgecolor=ec, facecolor=fc, zorder=3)
    ax.add_patch(box)

def pipeline():
    fig, ax = plt.subplots(figsize=(13.5, 3.5), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 13.5); ax.set_ylim(0, 3.5); ax.axis("off")
    stages = [
        ("1", "DATA INGESTION", "Prices, fundamentals,\nmacro, news & transcripts", SAFE),
        ("2", "NODE FRAGILITY", "AI risk score per entity\n(ML + SHAP explainability)", WARN),
        ("3", "INTERCONNECTION\nGRAPH", "Edges from correlation,\nsector, supply-chain, beta", CORAL),
        ("4", "SHOCK\nPROPAGATION", "Cascade simulation\n(DebtRank / threshold)", AMBER),
        ("5", "SYSTEMIC RISK\nDASHBOARD", "Pathways, vulnerable nodes,\nSystemic Risk Index", FAIL),
    ]
    n = len(stages)
    w = 2.15; gap = (13.5 - n * w) / (n + 1)
    y = 0.75; h = 2.0
    for i, (num, title, desc, col) in enumerate(stages):
        x = gap + i * (w + gap)
        rounded(ax, x, y, w, h, PANEL, col, lw=2.4, rad=0.05)
        ax.add_patch(Circle((x + 0.32, y + h - 0.34), 0.2, color=col, zorder=4))
        ax.text(x + 0.32, y + h - 0.34, num, ha="center", va="center",
                fontsize=12, fontweight="bold", color="#0a0416", zorder=5)
        ax.text(x + w / 2, y + h - 0.62, title, ha="center", va="top",
                fontsize=10.2, fontweight="bold", color=col)
        ax.text(x + w / 2, y + 0.62, desc, ha="center", va="center",
                fontsize=7.8, color=TEXT)
        if i < n - 1:
            ar = FancyArrowPatch((x + w + 0.06, y + h / 2),
                                 (x + w + gap - 0.06, y + h / 2),
                                 arrowstyle="-|>", mutation_scale=18,
                                 color=ACCENT, lw=2.4, zorder=4)
            ax.add_patch(ar)
    ax.text(gap, 3.2, "HOW CREDVEDA WORKS", fontsize=13, fontweight="bold", color=TEXT)
    ax.text(13.5 - gap, 0.28,
            "Existing CredVeda foundation ▸ new systemic-risk engine",
            fontsize=8.5, color=MUTE, ha="right", style="italic")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "pipeline.png"), facecolor=BG, dpi=180)
    plt.close(fig)
    print("saved pipeline.png")

pipeline()

# ============================================================
# ARCHITECTURE
# ============================================================
def architecture():
    fig, ax = plt.subplots(figsize=(11.5, 6.4), dpi=175)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 6.6); ax.axis("off")

    def layer(y, h, title, items, col, tag):
        rounded(ax, 0.4, y, 10.7, h, PANEL, col, lw=2.2, rad=0.03)
        ax.text(0.7, y + h - 0.28, title, fontsize=11, fontweight="bold", color=col)
        ax.text(11.2, y + h - 0.28, tag, fontsize=7.5, color=MUTE, ha="right",
                style="italic")
        bx = 0.7; bw = 2.35; bg = 0.14
        for it in items:
            rounded(ax, bx, y + 0.18, bw, h - 0.7, PANEL2, col, lw=1.2, rad=0.04)
            ax.text(bx + bw / 2, y + 0.18 + (h - 0.7) / 2, it, ha="center",
                    va="center", fontsize=7.6, color=TEXT, wrap=True)
            bx += bw + bg

    ax.text(0.4, 6.35, "SYSTEM ARCHITECTURE", fontsize=13, fontweight="bold", color=TEXT)
    layer(4.9, 1.25, "FRONTEND", ["Flask + Tailwind UI", "Interactive network\n(Plotly / Cytoscape.js)",
          "Chart.js trends", "AI Copilot chat"], SAFE, "existing + extended")
    layer(3.15, 1.5, "INTELLIGENCE ENGINE",
          ["Graph model\n(NetworkX)", "Contagion sim\n(DebtRank / cascade)",
           "ML fragility\n(scikit-learn + SHAP)", "NLP shock detect\n(VADER)"], FAIL, "NEW core")
    layer(1.5, 1.4, "DATA & STORAGE",
          ["yfinance\nprices+fundamentals", "FRED\nmacro (DGS10)",
           "NewsAPI + API-Ninjas\nnews & transcripts", "SQLite\nfeature store"], WARN, "existing pipeline")
    # arrows between layers
    for yy in [(4.9, 4.65), (3.15, 2.9)]:
        ax.add_patch(FancyArrowPatch((5.75, yy[0]), (5.75, yy[0] - 0.24),
                     arrowstyle="<|-|>", mutation_scale=15, color=ACCENT, lw=2))
    ax.text(0.4, 0.55, "Teal = already built in CredVeda   •   Red = new systemic-risk layer",
            fontsize=8.3, color=MUTE, style="italic")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "architecture.png"), facecolor=BG, dpi=175)
    plt.close(fig)
    print("saved architecture.png")

architecture()

# ============================================================
# SYSTEMIC RISK INDEX gauge + trend
# ============================================================
def risk_index():
    fig, ax = plt.subplots(figsize=(11.5, 3.4), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 3.4)
    # gauge (left)
    cx, cy, r = 2.4, 1.2, 1.35
    segs = [(180, 132, GOOD), (132, 84, WARN), (84, 0, FAIL)]
    for a2, a1, col in segs:
        ax.add_patch(Wedge((cx, cy), r, a1, a2, width=0.42, facecolor=col,
                     edgecolor=BG, lw=2))
    val = 72
    ang = 180 - (val / 100) * 180
    ax.add_patch(FancyArrowPatch((cx, cy), (cx + 0.98 * r * np.cos(np.radians(ang)),
                 cy + 0.98 * r * np.sin(np.radians(ang))), arrowstyle="-|>",
                 mutation_scale=16, color=TEXT, lw=2.6))
    ax.add_patch(Circle((cx, cy), 0.08, color=TEXT, zorder=5))
    ax.text(cx, cy - 0.55, f"{val}", fontsize=30, fontweight="bold", color=FAIL, ha="center")
    ax.text(cx, cy - 1.02, "SYSTEMIC RISK INDEX", fontsize=9.5, color=TEXT,
            ha="center", fontweight="bold")
    ax.text(cx, cy - 1.34, "HIGH — contagion likely to spread", fontsize=8,
            color=MUTE, ha="center", style="italic")
    # trend (right)
    x = np.linspace(0, 10, 60)
    base = 35 + 8 * np.sin(x / 1.5)
    spike = base + np.clip((x - 6) * 9, 0, None)
    ox, oy, ow, oh = 4.7, 0.55, 6.4, 2.3
    rounded(ax, ox, oy, ow, oh, PANEL, EDGE, lw=1.6, rad=0.03)
    ib, ih = oy + 0.35, oh - 0.95            # inner bottom / height
    ilo, ihi = spike.min(), spike.max()
    norm = (spike - ilo) / (ihi - ilo)
    xp = ox + 0.5 + (x / 10) * (ow - 1.0)
    yp = ib + (0.08 + 0.84 * norm) * ih
    ax.fill_between(xp, ib, yp, color=ACCENT, alpha=0.16, zorder=4)
    ax.plot(xp, yp, color=ACCENT, lw=3.0, zorder=5)
    # danger zone marker where the shock spikes
    si = int(np.argmax(x >= 6))
    ax.plot([xp[si], xp[si]], [ib, yp[si]], color=FAIL, lw=1.2, ls="--",
            alpha=0.7, zorder=5)
    ax.scatter([xp[-1]], [yp[-1]], s=90, color=FAIL, edgecolors="white",
               lw=1.2, zorder=6)
    ax.text(ox + 0.5, oy + oh - 0.3, "Early-warning: systemic risk rising before the crash",
            fontsize=9.5, color=TEXT, fontweight="bold", zorder=6)
    ax.text(xp[-1] - 0.05, yp[-1] + 0.14, "▲ shock", fontsize=8.5, color=FAIL,
            ha="right", zorder=6, fontweight="bold")
    ax.text(xp[si] + 0.1, ib + 0.12, "risk builds", fontsize=7.5, color=MUTE,
            ha="left", zorder=6, style="italic")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "risk_index.png"), facecolor=BG, dpi=180)
    plt.close(fig)
    print("saved risk_index.png")

risk_index()

# ============================================================
# BEFORE / AFTER
# ============================================================
def before_after():
    fig, ax = plt.subplots(figsize=(11.5, 4.2), dpi=180)
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG); ax.axis("off")
    ax.set_xlim(0, 11.5); ax.set_ylim(0, 4.4)
    rng = np.random.default_rng(7)
    # BEFORE: isolated dots
    rounded(ax, 0.3, 0.4, 5.2, 3.4, PANEL, MUTE, lw=1.6, rad=0.02)
    ax.text(0.55, 3.55, "TODAY  —  isolated scores", fontsize=11,
            fontweight="bold", color=MUTE)
    for _ in range(16):
        x = 0.7 + rng.random() * 4.4
        y = 0.7 + rng.random() * 2.5
        ax.scatter([x], [y], s=260, color=SAFE, edgecolors="white", lw=1,
                   alpha=0.9, zorder=4)
    ax.text(2.9, 0.62, "no relationships — systemic risk invisible", fontsize=8,
            color=MUTE, ha="center", style="italic", zorder=5)
    # arrow
    ax.add_patch(FancyArrowPatch((5.65, 2.1), (6.05, 2.1), arrowstyle="-|>",
                 mutation_scale=22, color=ACCENT, lw=3))
    # AFTER: connected
    rounded(ax, 6.0, 0.4, 5.2, 3.4, PANEL, ACCENT, lw=2, rad=0.02)
    ax.text(6.25, 3.55, "CREDVEDA  —  interconnected network", fontsize=11,
            fontweight="bold", color=ACCENT)
    pts = {}
    for i in range(11):
        pts[i] = (6.4 + rng.random() * 4.4, 0.75 + rng.random() * 2.4)
    keys = list(pts)
    for i in keys:
        for j in keys:
            if i < j and rng.random() < 0.28:
                ax.plot([pts[i][0], pts[j][0]], [pts[i][1], pts[j][1]],
                        color=EDGE, alpha=0.5, lw=1.1, zorder=1)
    for i, (x, y) in pts.items():
        col = FAIL if i == 0 else (WARN if i in (1, 4, 7) else SAFE)
        ax.scatter([x], [y], s=300, color=col, edgecolors="white", lw=1.2, zorder=3)
    ax.text(8.6, 0.62, "shock at one node ripples across the system", fontsize=8,
            color=MUTE, ha="center", style="italic")
    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "before_after.png"), facecolor=BG, dpi=180)
    plt.close(fig)
    print("saved before_after.png")

before_after()

# ============================================================
# ML MODELS + ALGORITHMS FLOW
# ============================================================
def ml_algos():
    fig, ax = plt.subplots(figsize=(13.6, 6.6), dpi=175)
    fig.patch.set_facecolor(BG); ax.set_facecolor(BG)
    ax.set_xlim(0, 13.6); ax.set_ylim(0, 6.6); ax.axis("off")

    def box(x, y, w, h, title, sub, col, tsize=10.0, ssize=7.6):
        rounded(ax, x, y, w, h, PANEL, col, lw=2.2, rad=0.05)
        ax.text(x + w / 2, y + h - 0.26, title, ha="center", va="top",
                fontsize=tsize, fontweight="bold", color=col)
        if sub:
            ax.text(x + w / 2, y + 0.26, sub, ha="center", va="bottom",
                    fontsize=ssize, color=TEXT)

    def arrow(x1, y1, x2, y2, col=ACCENT, cs=None, lw=2.4):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=16, color=col, lw=lw, connectionstyle=cs, zorder=6))

    ax.text(0.3, 6.35, "1   NODE INTELLIGENCE  —  per-entity fragility  (Machine Learning)",
            fontsize=11.5, fontweight="bold", color=SAFE)
    ax.text(0.3, 2.5, "2   SYSTEMIC PROPAGATION  —  network contagion  (Graph Algorithms)",
            fontsize=11.5, fontweight="bold", color=FAIL)

    # ---- Row 1: ML pipeline ----
    box(0.3, 4.05, 2.2, 1.6, "FEATURE\nENGINEERING",
        "19 features:\ntechnical · fundamental\nmacro · NLP", SAFE, ssize=7.2)
    mx, my, mw, mh = 2.9, 4.05, 5.0, 1.6
    rounded(ax, mx, my, mw, mh, PANEL, WARN, lw=2.4, rad=0.04)
    ax.text(mx + mw / 2, my + mh - 0.22, "ML MODELS   (CredVeda engine)", ha="center",
            va="top", fontsize=10.2, fontweight="bold", color=WARN)
    models = ["Random Forest", "XGBoost", "Decision Tree", "Neural Net (MLP)", "KNN", "KMeans"]
    cw, ch, gx, gy = 1.48, 0.42, 0.09, 0.12
    sx = mx + (mw - (3 * cw + 2 * gx)) / 2
    sy = my + 0.18
    for idx, mdl in enumerate(models):
        r, c = divmod(idx, 3)
        cx = sx + c * (cw + gx)
        cy = sy + (1 - r) * (ch + gy)
        rounded(ax, cx, cy, cw, ch, PANEL2, CORAL, lw=1.2, rad=0.08)
        ax.text(cx + cw / 2, cy + ch / 2, mdl, ha="center", va="center",
                fontsize=7.6, color=TEXT, fontweight="bold")
    box(8.3, 4.05, 2.4, 1.6, "FRAGILITY /\nCREDIT SCORE", "0–100 per entity\n(prob. of distress)", CORAL, ssize=7.4)
    box(11.1, 4.05, 2.2, 1.6, "SHAP\nEXPLAINABILITY", "why each score\nmoved (XAI)", AMBER, ssize=7.4)
    arrow(2.52, 4.85, 2.86, 4.85)
    arrow(7.92, 4.85, 8.26, 4.85)
    arrow(10.72, 4.85, 11.06, 4.85)

    # ---- Row 2: graph / contagion algorithms ----
    box(0.3, 0.5, 2.9, 1.5, "CORRELATION\n& EXPOSURE", "return corr · sector · beta\n→ graph (NetworkX)", SAFE, tsize=9.5, ssize=7.2)
    box(3.7, 0.5, 2.9, 1.5, "DEBTRANK /\nTHRESHOLD CASCADE", "propagate shock\nround-by-round", WARN, tsize=9.5, ssize=7.4)
    box(7.1, 0.5, 2.7, 1.5, "CENTRALITY →\nSIFI DETECTION", "'too-connected-\nto-fail' nodes", CORAL, tsize=9.5, ssize=7.4)
    box(10.3, 0.5, 3.0, 1.5, "SYSTEMIC RISK\nINDEX", "impact %, pathways,\nvulnerable entities", FAIL, tsize=10, ssize=7.4)
    arrow(3.22, 1.25, 3.66, 1.25)
    arrow(6.62, 1.25, 7.06, 1.25)
    arrow(9.82, 1.25, 10.26, 1.25)

    # ---- link: fragility feeds the node buffer in propagation ----
    arrow(9.5, 4.05, 5.15, 2.02, col=CORAL, cs="arc3,rad=-0.25", lw=2.6)
    ax.text(8.2, 3.05, "fragility = node buffer", fontsize=8.6, color=CORAL,
            ha="center", style="italic", fontweight="bold")

    fig.tight_layout(pad=0.3)
    fig.savefig(os.path.join(OUT, "ml_algos.png"), facecolor=BG, dpi=175)
    plt.close(fig)
    print("saved ml_algos.png")

ml_algos()

print("\nALL ASSETS GENERATED in", OUT)
for f in sorted(os.listdir(OUT)):
    print("  -", f, round(os.path.getsize(os.path.join(OUT, f)) / 1024), "KB")
