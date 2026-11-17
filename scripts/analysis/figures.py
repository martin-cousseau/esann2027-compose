"""Figures (PNG and PDF) for reports/figures/, with one-line captions in reports/figures/README.md.

Colors: the first three categorical slots of the dataviz reference palette (validated all-pairs on a white
surface: CVD separation and normal-vision floors pass), always paired with distinct marker shapes so that
identity never rests on color alone (print, grayscale, color-vision deficiency). Every plotted value is
also in a CSV of reports/analysis/ or a .dat file of paper/generated/ (the table view).
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402

from .common import ORDER_TXT, PAPER_ORDERS, cell_label, finite, write_text  # noqa: E402

INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")
MARKERS = ("o", "s", "^")
ORDER_STYLE = {o: (SERIES[i], MARKERS[i]) for i, o in enumerate(PAPER_ORDERS)}
LABEL = {o: ORDER_TXT[o].replace("->", "→") for o in ORDER_TXT}
RATIO = r"$I_E/(E_P+E_Q)$"


def _plain_log(ax):
    ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0, 1.5, 2.0, 3.0, 5.0, 7.0)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 7.5, "axes.titlesize": 7.5, "axes.labelsize": 7.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK, "axes.linewidth": 0.6,
    "xtick.color": AXIS, "ytick.color": AXIS, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "grid.linestyle": "-", "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 6.5,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "lines.linewidth": 1.2,
    "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
})


def _save(fig, paths, name):
    os.makedirs(paths.figures, exist_ok=True)
    out = []
    for ext in ("png", "pdf"):
        p = os.path.join(paths.figures, f"{name}.{ext}")
        fig.savefig(p, dpi=200 if ext == "png" else None, bbox_inches="tight")
        out.append(p)
    plt.close(fig)
    return out


def _empty(ax, text="no data yet"):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


# ----------------------------------------------------------------------------- frontier per model
def frontier(fr, store, paths, model):
    pts = [p for p in fr["points"] if p["model"] == model and finite(p["wt_ppl"])]
    quant = sorted([p for p in pts if p["kind"] == "quant"], key=lambda p: -p["bits"])
    sparse = [p for p in pts if p["kind"] == "sparse+quant" and p["order"] in PAPER_ORDERS]
    best = [p for p in sparse if p.get("best_in_cell")]
    others = [p for p in sparse if not p.get("best_in_cell")]
    dense = [p for p in pts if p["kind"] == "dense"]
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.7), sharey=True, layout="constrained")
    for ax, key, title in ((axs[0], "bits_tcr", "bits per weight as TCr counts them (no mask)"),
                           (axs[1], "bits_bitmap", "bits per weight with a 1-bit mask")):
        ax.set_title(title, loc="left", pad=3)
        ax.set_xlabel("bits per weight (scales and zero points excluded)")
        if not quant and not sparse:
            _empty(ax)
            continue
        if dense:
            ax.axhline(dense[0]["wt_ppl"], color=MUTED, lw=0.6)
        if others:
            ax.scatter([p[key] for p in others], [p["wt_ppl"] for p in others], s=10, color=GRID, edgecolor=MUTED,
                       linewidth=0.4, marker="o", zorder=2, label="other orders, same cell")
        if quant:
            ax.plot([p[key] for p in quant], [p["wt_ppl"] for p in quant], color=SERIES[0], marker=MARKERS[0], ms=4.5,
                    mec=SURFACE, mew=0.8, zorder=3, label="quantization only")
        if best:
            ax.scatter([p[key] for p in best], [p["wt_ppl"] for p in best], s=26, color=SERIES[1], marker=MARKERS[1],
                       edgecolor=SURFACE, linewidth=0.8, zorder=4, label="pruning + quantization, best order")
            for p in best:
                ax.annotate(cell_label(p["sparsity"], p["bits"]).replace("%+", "/"), (p[key], p["wt_ppl"]),
                            xytext=(3, 3), textcoords="offset points", fontsize=5.5, color=INK2)
        for p in quant:
            ax.annotate(f"{p['bits']}-bit", (p[key], p["wt_ppl"]), xytext=(3, -8), textcoords="offset points",
                        fontsize=5.5, color=INK2)
        ax.set_yscale("log")
        _plain_log(ax)
    axs[0].set_ylabel("WikiText-2 perplexity (log scale)")
    if dense and (quant or sparse):
        axs[1].annotate("dense", (1.0, dense[0]["wt_ppl"]), xycoords=("axes fraction", "data"), xytext=(-2, 2),
                        textcoords="offset points", ha="right", fontsize=5.5, color=MUTED)
    h, lab = axs[1].get_legend_handles_labels() if (quant or sparse) else ([], [])
    if h:
        fig.legend(h, lab, ncol=3, loc="outside lower center")
    fig.suptitle(model, x=0.0, ha="left", fontsize=8, color=INK)
    files = _save(fig, paths, f"frontier_{model}")
    n = len(quant) + len(best)
    return files, (f"{model}: WikiText-2 perplexity against bits per weight, TCr accounting (left) and with the "
                   f"1-bit mask counted (right); quantization only, best order per sparse cell, other orders in gray"
                   f" ({n} points; data: paper/generated/fig_frontier_*_{model}.dat, reports/analysis/frontier_points.csv).")


# ----------------------------------------------------------------------------- order comparison per cell
def orders_per_cell(q2rows, paths, models):
    metrics = (("wt_ppl", "WikiText-2 ppl"), ("wt_kl", "KL to dense"), ("flips9", "flips vs dense"))
    models = [m for m in models if any(r["model"] == m for r in q2rows)] or models[:3]
    fig, axs = plt.subplots(len(metrics), len(models), figsize=(2.2 * len(models), 1.5 * len(metrics)), squeeze=False,
                            layout="constrained")
    n_pts = 0
    for j, m in enumerate(models):
        R = [r for r in q2rows if r["model"] == m]
        cells = [r["cell"] for r in R]
        for i, (key, label) in enumerate(metrics):
            ax = axs[i][j]
            if i == 0:
                ax.set_title(m, loc="left", pad=3)
            have = False
            for k, o in enumerate(PAPER_ORDERS):
                xs, ys = [], []
                for x, r in enumerate(R):
                    vals = [r.get(f"{key}_{oo}") for oo in PAPER_ORDERS]
                    v = r.get(f"{key}_{o}")
                    if all(finite(z) for z in vals) and min(vals) > 0 and finite(v):
                        xs.append(x + (k - 1) * 0.18)
                        ys.append(100 * (v / min(vals) - 1))
                if xs:
                    have = True
                    n_pts += len(xs)
                    col, mk = ORDER_STYLE[o]
                    ax.scatter(xs, ys, s=22, color=col, marker=mk, edgecolor=SURFACE, linewidth=0.6, zorder=3,
                               label=LABEL[o])
            if not have:
                _empty(ax)
                continue
            ax.set_xticks(range(len(cells)))
            ax.set_xticklabels([c.replace("%+", "/") for c in cells], fontsize=5.5)
            ax.axhline(0, color=AXIS, lw=0.6)
            if j == 0:
                ax.set_ylabel(f"{label}\n% above best order")
            if i == len(metrics) - 1:
                ax.set_xlabel("cell (sparsity % / bits)", fontsize=6)
    h, lab = [], []
    for row in axs:
        for ax in row:
            hh, ll = ax.get_legend_handles_labels()
            if hh:
                h, lab = hh, ll
                break
        if h:
            break
    if h:
        fig.legend(h, lab, ncol=3, loc="outside upper center")
    files = _save(fig, paths, "orders_per_cell")
    return files, ("Order comparison per cell (seed 0): WikiText-2 perplexity, KL to the dense model and pooled flips"
                   f" of each order relative to the best order of the cell ({n_pts} points; data: "
                   "reports/analysis/q2_order_ranking.csv).")


# ----------------------------------------------------------------------------- E2 scatter
def e2_scatter(e2res, paths):
    L = e2res["layers"]
    models = [c["model"] for c in e2res["coverage"] if c["n_lines"]]
    rho = {(r["model"], r["cell"]): r for r in e2res["spearman"] if r["arm"] == "p_then_q"}
    bits = sorted({x["cell"].split("+")[1] for x in L}, key=lambda b: -int(b)) or ["4", "3"]
    cols = models or ["E2"]
    fig, axs = plt.subplots(len(bits), len(cols), figsize=(2.3 * len(cols), 1.9 * len(bits)), squeeze=False,
                            layout="constrained")
    n_pts = 0
    for j, m in enumerate(cols):
        for i, b in enumerate(bits):
            ax = axs[i][j]
            if i == 0:
                ax.set_title(m, loc="left", pad=3)
            pts = [x for x in L if x["model"] == m and x["cell"].endswith(f"+{b}")]
            spars = sorted({x["cell"].split("%")[0] for x in pts}, key=int)
            if not pts:
                _empty(ax)
                continue
            for k, s in enumerate(spars[:3]):
                P_ = [x for x in pts if x["cell"].startswith(f"{s}%") and finite(x.get("h_corr_index"))
                      and finite(x.get("ratio_p_then_q_pct"))]
                rr = rho.get((m, f"{s}%+{b}"))
                lab = f"{s}% + {b}-bit" + (f", Spearman ρ {rr['rho']:+.2f}" if rr and finite(rr.get("rho")) else "")
                ax.scatter([x["h_corr_index"] for x in P_], [x["ratio_p_then_q_pct"] for x in P_], s=12, color=SERIES[k],
                           marker=MARKERS[k], edgecolor=SURFACE, linewidth=0.4, alpha=0.9, label=lab, zorder=3)
                n_pts += len(P_)
            ax.axhline(0, color=AXIS, lw=0.6)
            ax.legend(loc="best", handletextpad=0.2, borderaxespad=0.2, fontsize=5.5, frameon=True, framealpha=0.85,
                      edgecolor="none", facecolor=SURFACE)
            if j == 0:
                ax.set_ylabel(f"P→Q layer {RATIO} (%)\n{b}-bit cells")
            if i == len(bits) - 1:
                ax.set_xlabel("Hessian off-diagonal correlation index")
    files = _save(fig, paths, "e2_interaction_vs_hcorr")
    return files, ("E2, same Hessian per layer: each layer's P->Q interaction I_E/(E_P+E_Q) against the off-diagonal "
                   f"correlation index of its Hessian, by cell; Spearman rank correlation in the legend ({n_pts} points;"
                   " data: reports/analysis/e2_layers.csv, e2_spearman.csv).")


def write_all(store, fr, qres, e2res, paths):
    from . import paper_figures
    captions = []
    for m in store.models():
        files, cap = frontier(fr, store, paths, m)
        captions.append((files, cap))
    captions.append(orders_per_cell(qres["q2"], paths, store.models()))
    captions.append(e2_scatter(e2res, paths))
    paper = paper_figures.write_all(store, qres, e2res, paths)
    lines = ["# Figures (generated by scripts/make_tables.py; do not edit)", "",
             "Each figure exists as PNG and PDF. One line per figure:", ""]
    for files, cap in captions:
        names = ", ".join(os.path.basename(f) for f in files)
        lines.append(f"- `{names}`: {cap}")
    lines += ["", "Paper figures (ESANN print style, 12.2 cm wide; the PDF is in paper/generated/, the PNG here is its"
              " preview):", ""]
    for files, cap in paper:
        lines.append(f"- `{', '.join(paths.rel(f) if f.startswith(paths.root) else f for f in files)}`: {cap}")
    write_text(os.path.join(paths.figures, "README.md"), "\n".join(lines) + "\n")
    return captions + paper
