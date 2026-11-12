"""Figures for the paper: paper/generated/<name>.pdf, with PNG previews in reports/figures/.

Print style for ESANN: vector PDF at the final size (12.2 cm wide, the text block), Computer Modern like the text
(matplotlib's cmr10 and its mathtext 'cm'), 8 pt type, TrueType fonts embedded (pdf.fonttype 42), no figure titles
(the caption is the paper's), black and grays only: identity is carried by marker shape, fill, line style and hatch,
never by color alone. Every plotted value is also in a CSV of reports/analysis/ (the table view); nothing is typed:
axis ranges follow the data.

fig1_interaction_v2  the first author's Fig. 1 (analysis/reproduce_paper.py draws it from the v1 runs): I_E relative
                     to the additive prediction E_P + E_Q, the ratio of mean errors over layers (Q1,
                     reports/analysis/q1_layer_interaction.csv, ratio_mean_pct), for the three core models and the six
                     E1 cells, bars for joint (light gray), P->Q (dark gray) and Q->P (white, hatched); negative =
                     sub-additive. Small white circles: the same ratio with one shared Hessian per layer (E2 aggregate,
                     reports/analysis/e2_cells.csv). Data: reports/analysis/fig1_interaction_v2.csv.
fig_order_scale      Q3/E6: the order spread 100 (max - min) / min of WikiText-2 ppl over the three orders against the
                     number of compressed weights (log axis), one panel per cell (25%+4, 50%+4, 50%+3), one marker,
                     fill and line style per family, models labelled by size. Data: reports/analysis/fig_order_scale.csv.
"""
import logging
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator, NullLocator  # noqa: E402

from .common import FAMILY_ORDER, MQ, PAPER_ORDERS, cell_label, finite, short, write_csv  # noqa: E402

logging.getLogger("fontTools").setLevel(logging.ERROR)   # cmr10.ttf has a 1904 timestamp; subsetting warns about it
CM = 1 / 2.54
WIDTH_CM = 12.2                 # ESANN text block (21 cm - 2 x 4.4 cm)
FIG1_HEIGHT_CM = 0.30 * WIDTH_CM   # the coordinator's 0.30 \textwidth (the author's original: 5.1 cm)
SCALE_HEIGHT_CM = 4.0
PT = 8
PAPER_RC = {
    "font.family": "serif", "font.serif": ["cmr10"], "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False, "font.size": PT, "axes.labelsize": PT, "axes.titlesize": PT, "xtick.labelsize": PT,
    "ytick.labelsize": PT, "legend.fontsize": PT, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.linewidth": 0.5,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "lines.linewidth": 0.8, "hatch.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "legend.frameon": False, "legend.handlelength": 1.4, "legend.columnspacing": 1.4,
    "legend.borderaxespad": 0.2, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black",
    "xtick.labelcolor": "black", "ytick.labelcolor": "black", "axes.titlecolor": "black", "text.color": "black", "axes.titlepad": 2.5, "axes.labelpad": 2.0, "xtick.major.pad": 2.0, "ytick.major.pad": 2.0,
    "figure.constrained_layout.h_pad": 0.02, "figure.constrained_layout.w_pad": 0.02,
    "figure.constrained_layout.hspace": 0.0, "figure.constrained_layout.wspace": 0.02,
}
ARROW = r"$\rightarrow$"
BAR_STYLE = (("joint", "Joint", "0.75", ""), ("p_then_q", f"P{ARROW}Q", "0.45", ""), ("q_then_p", f"Q{ARROW}P", "white", "////"))
FIG1_CELLS = ((0.25, 4), (round(1 / 3, 4), 4), (0.5, 4), (0.25, 3), (round(1 / 3, 4), 3), (0.5, 3))
FAMILY_STYLE = {"SmolLM2": {"marker": "o", "color": "black", "mfc": "black", "ls": "-"},
                "Qwen2.5": {"marker": "s", "color": "black", "mfc": "white", "ls": "--"},
                "OLMo-2": {"marker": "^", "color": "0.4", "mfc": "0.4", "ls": ":"}}
OTHER_STYLE = {"marker": "D", "color": "0.6", "mfc": "0.6", "ls": "-."}
SCALE_TITLES = {"25%+4": "25% + 4-bit", "50%+4": "50% + 4-bit", "50%+3": "50% + 3-bit"}


def _style():
    """Matplotlib's defaults (figures.py changes the global rcParams at import), then the print style."""
    from contextlib import ExitStack
    stack = ExitStack()
    stack.enter_context(plt.style.context("default"))
    stack.enter_context(plt.rc_context(PAPER_RC))
    return stack


def _save(fig, paths, name):
    os.makedirs(paths.generated, exist_ok=True)
    os.makedirs(paths.figures, exist_ok=True)
    pdf = os.path.join(paths.generated, f"{name}.pdf")
    png = os.path.join(paths.figures, f"{name}.png")
    fig.savefig(pdf, metadata={"CreationDate": None, "Creator": "scripts/make_tables.py"})   # final size: no tight bbox
    fig.savefig(png, dpi=300)
    plt.close(fig)
    return [pdf, png]


def _pending(ax, text="pending"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", style="italic", color="0.4")
    ax.set_xticks([])
    ax.set_yticks([])


def _nice_limits(vals, step=5.0, pad=1.0):
    lo, hi = min(vals + [0.0]), max(vals + [0.0])
    return step * math.floor((lo - pad) / step), step * math.ceil((hi + pad) / step)


# ----------------------------------------------------------------------------- Fig. 1, layer-level interaction
def fig1_rows(q1rows, e2cells, models):
    e2 = {(c["model"], c["cell"]): c for c in e2cells if c.get("n_layers")}
    q1 = {(r["model"], r["cell"], r["order"]): r for r in q1rows if r["seed"] == 0}
    rows = []
    for m in models:
        for s, b in FIG1_CELLS:
            cell = cell_label(s, b)
            for o in PAPER_ORDERS:
                r, c = q1.get((m, cell, o)), e2.get((m, cell))
                rows.append({"model": m, "cell": cell, "order": o, "ratio_mean_pct": r["ratio_mean_pct"] if r else None,
                             "n_layers": r["n_layers"] if r else None,
                             "e2_ratio_pct": c.get(f"ratio_{o}_pct") if c else None, "e2_n_layers": c.get("n_layers") if c else None})
    return rows


def fig1_interaction(rows, models, paths, e2=True, height_cm=FIG1_HEIGHT_CM):
    with _style():
        fig, axs = plt.subplots(1, len(models), figsize=(WIDTH_CM * CM, height_cm * CM), sharey=True, layout="constrained",
                                squeeze=False)
        axs = axs[0]
        x, wd = np.arange(len(FIG1_CELLS)), 0.27
        val = {(r["model"], r["cell"], r["order"]): r for r in rows}
        allv = [r[k] for r in rows for k in ("ratio_mean_pct", "e2_ratio_pct") if finite(r[k])]
        n_bars = n_e2 = 0
        for ax, m in zip(axs, models):
            have = False
            for i, (o, lab, col, hatch) in enumerate(BAR_STYLE):
                pts = [(x[j] + (i - 1) * wd, val.get((m, cell_label(s, b), o), {})) for j, (s, b) in enumerate(FIG1_CELLS)]
                bars = [(px, r["ratio_mean_pct"]) for px, r in pts if finite(r.get("ratio_mean_pct"))]
                if bars:
                    have = True
                    n_bars += len(bars)
                    ax.bar([p for p, _ in bars], [v for _, v in bars], wd, color=col, edgecolor="black", linewidth=0.4,
                           hatch=hatch, label=lab, zorder=2)
                e2pts = [(px, r["e2_ratio_pct"]) for px, r in pts if finite(r.get("e2_ratio_pct"))]
                if e2 and e2pts:
                    n_e2 += len(e2pts)
                    ax.plot([p for p, _ in e2pts], [v for _, v in e2pts], ls="none", marker="o", ms=2.4, mfc="white",
                            mec="black", mew=0.5, zorder=3, label="one shared Hessian (E2)")
            ax.set_title(m)
            if not have:
                _pending(ax)
                continue
            ax.axhline(0, color="black", lw=0.6, zorder=2.5)
            ax.axvline(2.5, color="0.6", lw=0.5, ls=":", zorder=1)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{round(s * 100)}" for s, _ in FIG1_CELLS])
            ax.tick_params(axis="x", length=0, pad=1.5)
            ax.set_xlim(-0.6, len(FIG1_CELLS) - 0.4)
            for xx, txt in ((1.0, "4-bit"), (4.0, "3-bit")):
                ax.annotate(txt, (xx, 0), xycoords=("data", "axes fraction"), xytext=(0, -2 - 1.2 * PT),
                            textcoords="offset points", ha="center", va="top")
        if allv:
            axs[0].set_ylim(*_nice_limits(allv))
            axs[0].yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10], integer=True))
        axs[0].set_ylabel(r"$I_E/(E_P+E_Q)$ (%)")
        handles = [Patch(facecolor=c, edgecolor="black", linewidth=0.4, hatch=h, label=lab) for _, lab, c, h in BAR_STYLE]
        if e2 and n_e2:
            handles.append(Line2D([], [], ls="none", marker="o", ms=2.4, mfc="white", mec="black", mew=0.5,
                                  label="one shared Hessian (E2)"))
        fig.legend(handles=handles, ncol=len(handles), loc="outside upper center")
        files = _save(fig, paths, "fig1_interaction_v2")
    cap = (f"Fig. 1 (v2, fp32, seed 0): layer-level interaction I_E/(E_P+E_Q), ratio of mean errors over layers, of the"
           f" three core models in the six E1 cells (sparsity %, 4-bit | 3-bit); joint, P->Q, Q->P bars ({n_bars}); white"
           f" circles: the same with one shared Hessian per layer (E2, {n_e2}); negative = sub-additive. Data:"
           " reports/analysis/fig1_interaction_v2.csv.")
    return files, cap


# ----------------------------------------------------------------------------- Q3/E6: spread vs compressed weights
def _size_ticks(ax, lo, hi):
    cand = [1e8, 3e8, 1e9, 3e9, 1e10]
    ticks = [t for t in cand if lo <= t <= hi]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e9:g}B"))


LABEL_GAP = 2.5          # points between a marker's edge and its label
LABEL_PT = PT - 0.5
MARK_PT = 3.8            # marker size (points)


def _place_labels(fig, ax, pts):
    """Label (x, y, text) points without overlaps, greedily in the order of x: the first candidate box that overlaps
    no marker and no label already placed and stays inside the panel: above, below, right or left of the marker, else
    raised above it (one to five label heights, shifted sideways if needed) with a thin leader line. Positions are
    computed on the final layout, in display units."""
    from matplotlib.transforms import Bbox
    if not pts:
        return []
    fig.canvas.draw()
    rend, px = fig.canvas.get_renderer(), fig.dpi / 72.0
    frame = ax.get_window_extent(rend)
    half = (MARK_PT / 2 + 0.8) * px
    disp = [ax.transData.transform((x, y)) for x, y, _ in pts]
    marks = [Bbox.from_extents(dx - half, dy - half, dx + half, dy + half) for dx, dy in disp]
    placed, notes = [], []
    for (x, y, text), (dx, dy) in sorted(zip(pts, disp), key=lambda p: p[0][0]):
        probe = ax.text(0, 0, text, fontsize=LABEL_PT)
        ext = probe.get_window_extent(rend)
        probe.remove()
        w, h, g = ext.width + 3 * px, ext.height + 1 * px, LABEL_GAP * px   # padded: neighbouring labels never touch
        cands = [(0, half + g + h / 2, False), (0, -(half + g + h / 2), False), (half + g + w / 2, 0, False),
                 (-(half + g + w / 2), 0, False)]
        cands += [(sx * 0.7 * w, half + g + h / 2 + k * (h + g), True) for k in range(1, 6) for sx in (0, 1, -1, 2, -2)]
        choice = cands[0]
        for ox, oy, lead in cands:
            b = Bbox.from_extents(dx + ox - w / 2, dy + oy - h / 2, dx + ox + w / 2, dy + oy + h / 2)
            if b.x0 < frame.x0 or b.x1 > frame.x1 or b.y0 < frame.y0 or b.y1 > frame.y1:
                continue
            if any(b.overlaps(o) for o in marks + placed):
                continue
            choice = (ox, oy, lead)
            break
        ox, oy, lead = choice
        placed.append(Bbox.from_extents(dx + ox - w / 2, dy + oy - h / 2, dx + ox + w / 2, dy + oy + h / 2))
        a = ax.annotate(text, (x, y), xytext=(ox / px, oy / px), textcoords="offset points", ha="center", va="center",
                        fontsize=LABEL_PT, arrowprops={"arrowstyle": "-", "lw": 0.4, "color": "0.45", "shrinkA": 0.0,
                                                      "shrinkB": MARK_PT / 2 + 0.5} if lead else None)
        a.set_in_layout(False)
        notes.append(a)
    return notes


def fig_order_scale(rows, paths, height_cm=SCALE_HEIGHT_CM):
    cells = list(SCALE_TITLES)
    with _style():
        fig, axs = plt.subplots(1, len(cells), figsize=(WIDTH_CM * CM, height_cm * CM), layout="constrained", squeeze=False)
        axs = axs[0]
        xs = [r["compressed_weights"] for r in rows if finite(r["compressed_weights"])]
        lo, hi = (min(xs) / 1.6, max(xs) * 1.6) if xs else (7e7, 4e9)
        fams, n_pts, pending_labels = [], 0, []
        for ax, cell in zip(axs, cells):
            ax.set_title(SCALE_TITLES[cell])
            R = [r for r in rows if r["cell"] == cell and finite(r["spread_pct"]) and finite(r["compressed_weights"])]
            if not R:
                _pending(ax)
                continue
            labels = []
            for fam in FAMILY_ORDER + tuple(sorted({r["family"] for r in R} - set(FAMILY_ORDER))):
                F = sorted([r for r in R if r["family"] == fam], key=lambda r: r["compressed_weights"])
                if not F:
                    continue
                st = FAMILY_STYLE.get(fam, OTHER_STYLE)
                ax.plot([r["compressed_weights"] for r in F], [r["spread_pct"] for r in F], ls=st["ls"], color=st["color"],
                        marker=st["marker"], mfc=st["mfc"], mec=st["color"], ms=MARK_PT, mew=0.7, lw=0.8, clip_on=False, zorder=3)
                n_pts += len(F)
                if fam not in fams:
                    fams.append(fam)
                labels += [(r["compressed_weights"], r["spread_pct"], r["size"]) for r in F]
            ax.set_xscale("log")
            ax.set_xlim(lo, hi)
            _size_ticks(ax, lo, hi)
            # panels of the same bit-width (25%+4 and 50%+4) share the y range, so the two 4-bit cells compare directly
            top = max(r["spread_pct"] for r in rows if finite(r["spread_pct"]) and r["cell"] in cells
                      and r["cell"].split("+")[1] == cell.split("+")[1])
            ax.set_ylim(0, top * 1.3 if top > 0 else 1)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
            pending_labels.append((ax, labels))
        axs[0].set_ylabel("order spread (%)")
        fig.supxlabel("compressed weights (log scale)", fontsize=PT)
        handles = [Line2D([], [], ls=FAMILY_STYLE[f]["ls"], color=FAMILY_STYLE[f]["color"], marker=FAMILY_STYLE[f]["marker"],
                          mfc=FAMILY_STYLE[f]["mfc"], mec=FAMILY_STYLE[f]["color"], ms=MARK_PT, mew=0.7, lw=0.8, label=f)
                   for f in FAMILY_ORDER if f in fams]
        if handles:
            fig.legend(handles=handles, ncol=len(handles), loc="outside upper center")
        fig.canvas.draw()
        fig.set_layout_engine("none")      # freeze the layout: the labels below must not move the axes
        for ax, labels in pending_labels:
            _place_labels(fig, ax, labels)
        files = _save(fig, paths, "fig_order_scale")
    cap = (f"Order spread 100 (max - min)/min of WikiText-2 perplexity over joint, P->Q and Q->P (seed 0) against the"
           f" number of compressed weights, per family ({', '.join(fams) or 'no data yet'}; {n_pts} points), in 25%+4,"
           " 50%+4 and 50%+3; labels: model size. Data: reports/analysis/fig_order_scale.csv.")
    return files, cap


def write_all(store, qres, e2res, paths):
    """Both paper figures; returns [(files, caption)]. The data CSVs are written by write_data (always)."""
    core = [short(m) for m in MQ.CORE]
    out = [fig1_interaction(fig1_rows(qres["q1"], e2res["cells"], core), core, paths)]
    out.append(fig_order_scale(qres["fig_scale"], paths))
    return out


def write_data(qres, e2res, paths):
    core = [short(m) for m in MQ.CORE]
    write_csv(os.path.join(paths.reports, "fig1_interaction_v2.csv"), fig1_rows(qres["q1"], e2res["cells"], core))
