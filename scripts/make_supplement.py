#!/usr/bin/env python3
"""Supplementary tables of the paper: every model state of the study (seed 0) with all its measurements.

Reads ONLY the analysis tables that scripts/make_tables.py writes (reports/analysis/*.csv), so the supplement cannot
disagree with the paper's numbers, and writes into --out (default paper/supplement/):
  complete_results.csv  one row per (model, state) at seed 0: TCr, WikiText-2 and C4 perplexity, KL to the dense
                        model, mean normalized layer error E, the layer interaction I_E (% of E_P + E_Q) of combined
                        states, semantic retention on the paper's three tasks (Sr3) and on nine tasks (Sr9) with 95%
                        bootstrap intervals, and the model-level interaction I_Sr (Sr3) of combined states
  task_scores.csv       one row per (model, state, task): score, dense score and retention (score / dense score)
  seeds.csv             WikiText-2 perplexity per calibration seed, mean and sd, for the configurations with seeds
  layer_ablation.csv    E2, one dense Hessian shared by all orders, per (model, setting): the layer interaction of the
                        three orders and the split of the Q->P / P->Q error gap into mask, update order and grid
  no_compensation.csv   E9, the same layers without error compensation, per (model, setting): interaction, the exact
                        terms of the paper's decomposition, the zeros of rounding and their overlap with the mask
  weaker_first.csv      per (model, setting): the single-method perplexities, the order a weaker-first rule recommends,
                        and the best and worst of the three orders by perplexity, KL and nine-task accuracy
  supplement.tex / .pdf the same, typeset (landscape A4), one table per model
  README.md             column definitions and how to regenerate
Usage: python scripts/make_tables.py && python scripts/make_supplement.py [--out DIR] [--no-pdf]
"""
import argparse
import csv
import os
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = os.path.join(ROOT, "reports", "analysis")
ORDERS = ["dense", "quant", "prune", "joint", "p_then_q", "q_then_p"]
ORDER_TXT = {"dense": "dense", "quant": "quant. only", "prune": "pruning only", "joint": "joint", "p_then_q": "P->Q", "q_then_p": "Q->P"}
ORDER_TEX = {"dense": "dense", "quant": "quant.\\ only", "prune": "pruning only", "joint": "joint", "p_then_q": "P$\\to$Q", "q_then_p": "Q$\\to$P"}
TASKS = ["arc_easy", "piqa", "hellaswag", "arc_challenge", "winogrande", "openbookqa", "boolq", "sciq", "lambada_openai"]
COLS = ["model", "family", "order", "sparsity_pct", "bits", "tcr", "wikitext2_ppl", "c4_ppl", "wikitext2_kl", "c4_kl",
        "layer_error", "I_E_pct", "sr3", "sr3_lo", "sr3_hi", "sr9", "sr9_lo", "sr9_hi", "I_Sr3", "I_Sr3_lo", "I_Sr3_hi",
        "I_Sr3_cn", "I_Sr3_cn_lo", "I_Sr3_cn_hi"]


def rows(name):
    path = os.path.join(R, name)
    if not os.path.exists(path):
        sys.exit(f"make_supplement: {path} is missing; run scripts/make_tables.py first")
    with open(path) as f:
        return list(csv.DictReader(f))


def fl(x):
    try:
        v = float(x)
        return v if v == v else None
    except (TypeError, ValueError):
        return None


def tcr(order, s, b):
    if order == "dense":
        return 0.0
    if order == "prune":
        return s
    return 1 - b / 16 * (1 - s)


def key(r):
    return (round(float(r["sparsity"]), 4), int(float(r["bits"])))


def build(out):
    states = [r for r in rows("e1_states.csv") if int(float(r.get("seed") or 0)) == 0 and r["order"] in ORDERS]
    q1 = {(r["model"], r["order"], round(float(r["sparsity"]), 4), int(r["bits"])): fl(r["ratio_mean_pct"])
          for r in rows("q1_layer_interaction.csv") if int(float(r["seed"])) == 0}
    isr = {(r["model"], r["state_id"]): r for r in rows("e1_isr.csv") if r["set"] == "sr3_paper"}
    isr_cn = {(r["model"], r["state_id"]): r for r in rows("e1_isr.csv") if r["set"] == "sr3_paper_cn"}
    tasks = defaultdict(dict)
    for r in rows("e1_tasks.csv"):   # (metric of Sr3 / Table 2 where the task is one of its three, else of Sr9, score), (metric of Sr9, score)
        first = (r["paper_metric"], fl(r["paper"])) if r.get("paper_metric") else (r["v2_metric"], fl(r["v2"]))
        tasks[(r["model"], r["state_id"])][r["task"]] = (first, (r["v2_metric"], fl(r["v2"])))
    models = list(dict.fromkeys(r["model"] for r in states))
    seen, res = set(), []
    for r in sorted(states, key=lambda r: (models.index(r["model"]), ORDERS.index(r["order"]) if r["order"] in ("dense", "quant", "prune") else 3,
                                         -int(float(r["bits"])), float(r["sparsity"]), ORDERS.index(r["order"]))):
        if (r["model"], r["state_id"]) in seen:      # E6B re-runs repeat a state with benchmarks: keep the first line
            continue
        seen.add((r["model"], r["state_id"]))
        s, b = key(r)
        i = isr.get((r["model"], r["state_id"]), {})
        res.append({"model": r["model"], "family": r["family"], "order": r["order"], "sparsity_pct": round(100 * s, 1), "bits": b,
                    "tcr": round(tcr(r["order"], s, b), 4), "wikitext2_ppl": fl(r["wt_ppl"]), "c4_ppl": fl(r["c4_ppl"]),
                    "wikitext2_kl": fl(r["wt_kl"]), "c4_kl": fl(r.get("c4_kl")), "layer_error": fl(r["mean_rel_err"]),
                    "I_E_pct": q1.get((r["model"], r["order"], s, b)) if r["order"] in ("joint", "p_then_q", "q_then_p") else None,
                    "sr3": fl(r["sr3_paper"]), "sr3_lo": fl(r["sr3_paper_lo"]), "sr3_hi": fl(r["sr3_paper_hi"]),
                    "sr9": fl(r["sr9_v2"]), "sr9_lo": fl(r["sr9_v2_lo"]), "sr9_hi": fl(r["sr9_v2_hi"]),
                    "I_Sr3": fl(i.get("value")), "I_Sr3_lo": fl(i.get("lo")), "I_Sr3_hi": fl(i.get("hi")),
                    "I_Sr3_cn": fl(isr_cn.get((r["model"], r["state_id"]), {}).get("value")),
                    "I_Sr3_cn_lo": fl(isr_cn.get((r["model"], r["state_id"]), {}).get("lo")),
                    "I_Sr3_cn_hi": fl(isr_cn.get((r["model"], r["state_id"]), {}).get("hi")), "_sid": r["state_id"]})
    dense_sid = {x["model"]: x["_sid"] for x in res if x["order"] == "dense"}
    trows = []
    for x in res:
        t, d = tasks.get((x["model"], x["_sid"]), {}), tasks.get((x["model"], dense_sid.get(x["model"])), {})
        for name in TASKS:
            if name in t and t[name][0][1] is not None:
                (pm, ps), (vm, vs) = t[name]
                (_, pd), (_, vd) = d.get(name, ((None, None), (None, None)))
                trows.append({"model": x["model"], "order": x["order"], "sparsity_pct": x["sparsity_pct"], "bits": x["bits"], "task": name,
                              "metric": pm, "score": ps, "dense_score": pd, "retention": ps / pd if pd else None,
                              "sr9_metric": vm, "sr9_score": vs, "sr9_dense_score": vd})
    seeds = []
    for r in rows("seed_ppl_orders.csv"):
        v = [fl(r.get(f"wt_ppl_seed{k}")) for k in (0, 1, 2)]
        seeds.append({"model": r["model"], "order": r["order"], "sparsity_pct": round(100 * float(r["sparsity"]), 1), "bits": int(float(r["bits"])),
                      "ppl_seed0": v[0], "ppl_seed1": v[1], "ppl_seed2": v[2], "mean": fl(r["wt_ppl_mean"]), "sd": fl(r["wt_ppl_sd"])})
    os.makedirs(out, exist_ok=True)
    for name, data, cols in (("complete_results.csv", res, COLS), ("task_scores.csv", trows, list(trows[0]) if trows else []),
                             ("seeds.csv", seeds, list(seeds[0]) if seeds else [])):
        with open(os.path.join(out, name), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for d in data:
                w.writerow({k: ("" if d.get(k) is None else d[k]) for k in cols})
        print(f"make_supplement: {name}: {len(data)} rows")
    return res, trows, seeds, models


GRID = [(25.0, 4), (33.3, 4), (50.0, 4), (25.0, 3), (33.3, 3), (50.0, 3)]
ABL_COLS = ["model", "sparsity_pct", "bits", "n_layers", "I_E_joint_pct", "I_E_p_then_q_pct", "I_E_q_then_p_pct",
            "gap_pct", "mask_pct", "order_pct", "grid_pct"]
NOCOMP_COLS = ["model", "sparsity_pct", "bits", "n_layers", "I_E_mask_first_pct", "I_E_p_then_q_pct", "I_E_q_then_p_pct",
               "absorption_pct", "cross_pct", "gap_pct", "mask_pct", "grid_pct", "zeros_of_rounding_pct",
               "pruned_weights_among_them_pct", "E_P_comp_over_nocomp", "E_Q_comp_over_nocomp"]
WF_COLS = ["model", "sparsity_pct", "bits", "ppl_pruning_only", "ppl_quant_only", "rule_recommends", "ppl_joint", "ppl_p_then_q",
           "ppl_q_then_p", "best_ppl", "worst_ppl", "best_kl", "worst_kl", "best_sr9", "worst_sr9"]


def _cell(r):
    return (round(100 * float(r["sparsity"]), 1), int(float(r["bits"])))


def build_layer_tables(out, models):
    """E2 (one shared Hessian), E9 (no compensation) and the weaker-first test, per (model, setting) of the grid."""
    order = lambda r: (models.index(r["model"]) if r["model"] in models else 99, GRID.index((r["sparsity_pct"], r["bits"])))  # noqa: E731
    abl = []
    for c in rows("e2_cells.csv"):
        if _cell(c) not in GRID or not fl(c.get("n_layers")):
            continue
        add = fl(c["E_P_sum"]) + fl(c["E_Q_sum"])
        need = ("p_then_q", "q_then_p", "q_then_p_pmask", "p_then_q_stale")
        E = {a: add * (1 + fl(c[f"ratio_{a}_pct"]) / 100) for a in need if fl(c.get(f"ratio_{a}_pct")) is not None}
        r = {"model": c["model"], "sparsity_pct": _cell(c)[0], "bits": _cell(c)[1], "n_layers": int(float(c["n_layers"])),
             "I_E_joint_pct": fl(c.get("ratio_joint_pct")), "I_E_p_then_q_pct": fl(c.get("ratio_p_then_q_pct")),
             "I_E_q_then_p_pct": fl(c.get("ratio_q_then_p_pct"))}
        if all(a in E for a in need):
            pq = E["p_then_q"]
            r.update(gap_pct=100 * (E["q_then_p"] - pq) / pq, mask_pct=100 * (E["q_then_p"] - E["q_then_p_pmask"]) / pq,
                     order_pct=100 * (E["q_then_p_pmask"] - E["p_then_q_stale"]) / pq, grid_pct=100 * (E["p_then_q_stale"] - pq) / pq)
        abl.append(r)
    noc = []
    for c in rows("e9_cells.csv"):
        if _cell(c) not in GRID or not fl(c.get("n_layers")):
            continue
        noc.append({"model": c["model"], "sparsity_pct": _cell(c)[0], "bits": _cell(c)[1], "n_layers": int(float(c["n_layers"])),
                    "I_E_mask_first_pct": fl(c["ratio_p_then_q_stale_pct"]), "I_E_p_then_q_pct": fl(c["ratio_p_then_q_pct"]),
                    "I_E_q_then_p_pct": fl(c["ratio_q_then_p_pct"]), "absorption_pct": fl(c["absorption_pct"]),
                    "cross_pct": fl(c["cross_pct"]), "gap_pct": fl(c["gap_pct"]), "mask_pct": fl(c["mask_pct"]), "grid_pct": fl(c["grid_pct"]),
                    "zeros_of_rounding_pct": 100 * fl(c["nocomp_zeros"]), "pruned_weights_among_them_pct": 100 * fl(c["nocomp_overlap_mask_zeros"]),
                    "E_P_comp_over_nocomp": fl(c["comp_over_nocomp_P"]), "E_Q_comp_over_nocomp": fl(c["comp_over_nocomp_Q"])})
    q2 = {(r["model"], _cell(r)): r for r in rows("q2_order_ranking.csv") if int(float(r["seed"])) == 0}
    wf = []
    for k in rows("e7_kim.csv"):
        if k["metric"] != "wt_ppl" or int(float(k["seed"])) != 0 or k["in_counts"] not in ("True", "true", "1") or _cell(k) not in GRID:
            continue
        q = q2.get((k["model"], _cell(k)), {})
        wf.append({"model": k["model"], "sparsity_pct": _cell(k)[0], "bits": _cell(k)[1], "ppl_pruning_only": fl(k["prune_only"]),
                   "ppl_quant_only": fl(k["quant_only"]), "rule_recommends": k["prediction"], "ppl_joint": fl(q.get("wt_ppl_joint")),
                   "ppl_p_then_q": fl(q.get("wt_ppl_p_then_q")), "ppl_q_then_p": fl(q.get("wt_ppl_q_then_p")),
                   "best_ppl": q.get("wt_ppl_best") or None, "worst_ppl": q.get("wt_ppl_worst") or None, "best_kl": q.get("wt_kl_best") or None,
                   "worst_kl": q.get("wt_kl_worst") or None, "best_sr9": q.get("sr9_v2_best") or None, "worst_sr9": q.get("sr9_v2_worst") or None})
    for name, data, cols in (("layer_ablation.csv", abl, ABL_COLS), ("no_compensation.csv", noc, NOCOMP_COLS), ("weaker_first.csv", wf, WF_COLS)):
        data.sort(key=order)
        with open(os.path.join(out, name), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for d in data:
                w.writerow({k: ("" if d.get(k) is None else d[k]) for k in cols})
        print(f"make_supplement: {name}: {len(data)} rows")
    return abl, noc, wf


def f(v, nd):
    return "" if v is None else f"{v:.{nd}f}"


def sg(v, nd=1):
    return "" if v is None else f"{v:+.{nd}f}"


def tex_layer_tables(abl, noc, wf):
    """The three per-setting tables, typeset."""
    L = [r"\clearpage\section*{Layer-level controls and the weaker-first test (per model and setting, seed 0)}",
         r"Sums over the linear layers of every second or fourth decoder block. "
         r"\textbf{One shared Hessian} (\texttt{layer\_ablation.csv}): every order is calibrated with the dense model's Hessian. "
         r"$I_E$ in \% of $E_P+E_Q$; gap $=100\,(E_{QP}-E_{PQ})/E_{PQ}$, split exactly into three parts, each obtained by exchanging "
         r"one ingredient between Q$\to$P and P$\to$Q: the mask (chosen on rounded weights instead of the pruning-only mask), "
         r"the order of the compensated updates (same mask and grid, second sweep started from quantized instead of pruned weights) "
         r"and the grid (fitted before pruning instead of on the kept weights).",
         r"\begin{longtable}{lrrrrrrrrrr}",
         r"\toprule model & $s$ (\%) & $b$ & layers & $I_E$ joint & $I_E$ P$\to$Q & $I_E$ Q$\to$P & gap & mask & update order & grid \\ \midrule \endhead"]
    for r in abl:
        L.append(" & ".join([r["model"], f(r["sparsity_pct"], 1), str(r["bits"]), str(r["n_layers"]), sg(r["I_E_joint_pct"]), sg(r["I_E_p_then_q_pct"]),
                             sg(r["I_E_q_then_p_pct"]), sg(r.get("gap_pct")), sg(r.get("mask_pct")), sg(r.get("order_pct")), sg(r.get("grid_pct"), 2)]) + r" \\")
    L += [r"\bottomrule\end{longtable}",
          r"\textbf{Without compensation} (\texttt{no\_compensation.csv}): the same layers and Hessian, sweeps run on the diagonal of "
          r"the Hessian (no error is propagated), errors measured with the full Hessian. Mask-first: mask of pruning alone, grid of "
          r"quantization alone (the setting of the paper's decomposition); absorption and cross terms in \% of $E_P+E_Q$ (they sum to "
          r"the mask-first $I_E$); gap, mask and grid as above (the update-order part is zero by construction); zeros: share of the "
          r"weights that rounding alone sets to zero, and share of the pruned weights among them.",
          r"\begin{longtable}{lrrrrrrrrrrrr}",
          r"\toprule model & $s$ (\%) & $b$ & $I_E$ mask-first & $I_E$ P$\to$Q & $I_E$ Q$\to$P & absorption & cross terms & gap & mask & grid & zeros (\%) & pruned among zeros (\%) \\ \midrule \endhead"]
    for r in noc:
        L.append(" & ".join([r["model"], f(r["sparsity_pct"], 1), str(r["bits"]), sg(r["I_E_mask_first_pct"]), sg(r["I_E_p_then_q_pct"]),
                             sg(r["I_E_q_then_p_pct"]), sg(r["absorption_pct"]), sg(r["cross_pct"], 3), sg(r["gap_pct"]), sg(r["mask_pct"]),
                             sg(r["grid_pct"], 3), f(r["zeros_of_rounding_pct"], 1), f(r["pruned_weights_among_them_pct"], 1)]) + r" \\")
    L += [r"\bottomrule\end{longtable}",
          r"\textbf{Weaker-first test} (\texttt{weaker\_first.csv}): WikiText-2 perplexity of each method alone and of the three orders; "
          r"the rule puts first the method with the lower perplexity alone. Best and worst order by perplexity, by KL and by nine-task "
          r"accuracy (blank where not measured).",
          r"\begin{longtable}{lrrrrlrrrllllll}",
          r"\toprule model & $s$ (\%) & $b$ & pruning only & quant.\ only & rule: first & joint & P$\to$Q & Q$\to$P & best & worst & best (KL) & worst (KL) & best (Sr9) & worst (Sr9) \\ \midrule \endhead"]
    first = {"q_then_p": "quantize", "p_then_q": "prune"}
    for r in wf:
        o = lambda k: ORDER_TEX.get(r.get(k) or "", "")  # noqa: E731
        L.append(" & ".join([r["model"], f(r["sparsity_pct"], 1), str(r["bits"]), f(r["ppl_pruning_only"], 2), f(r["ppl_quant_only"], 2),
                             first.get(r["rule_recommends"], r["rule_recommends"]), f(r["ppl_joint"], 2), f(r["ppl_p_then_q"], 2), f(r["ppl_q_then_p"], 2),
                             o("best_ppl"), o("worst_ppl"), o("best_kl"), o("worst_kl"), o("best_sr9"), o("worst_sr9")]) + r" \\")
    L += [r"\bottomrule\end{longtable}"]
    return L


def tex_seed_table(seeds):
    """WikiText-2 perplexity over calibration seeds 0-2 per model and setting: mean +- sample sd of each order, and
    the worst order of each seed (J joint, P P->Q, Q Q->P)."""
    by = defaultdict(dict)
    for r in seeds:
        if r["order"] in ("joint", "p_then_q", "q_then_p"):
            by[(r["model"], r["sparsity_pct"], r["bits"])][r["order"]] = r
    letter = {"joint": "J", "p_then_q": "P", "q_then_p": "Q"}
    L = [r"\clearpage\section*{Calibration seeds (WikiText-2 perplexity, seeds 0, 1 and 2)}",
         r"Mean $\pm$ sample standard deviation over the three calibration seeds of each order, and the worst order of each seed "
         r"(J: joint, P: P$\to$Q, Q: Q$\to$P; seed 0 first). Per-seed values: \texttt{seeds.csv}.",
         r"\begin{longtable}{lrrrrrl}",
         r"\toprule model & $s$ (\%) & $b$ & joint & P$\to$Q & Q$\to$P & worst (seeds 0, 1, 2) \\ \midrule \endhead"]
    for (m, s, b), d in sorted(by.items(), key=lambda kv: (kv[0][0], -kv[0][2], kv[0][1])):
        if len(d) < 3:
            continue
        cell = []
        for o in ("joint", "p_then_q", "q_then_p"):
            r = d[o]
            cell.append("" if r["mean"] is None else f"{r['mean']:.2f} $\\pm$ {r['sd']:.2f}" if r["sd"] is not None else f"{r['mean']:.2f}")
        worst = []
        for k in ("ppl_seed0", "ppl_seed1", "ppl_seed2"):
            v = {o: d[o][k] for o in d if d[o][k] is not None}
            worst.append(letter[max(v, key=v.get)] if len(v) == 3 else "-")
        L.append(" & ".join([m, f(s, 1), str(b)] + cell + [" ".join(worst)]) + r" \\")
    L.append(r"\bottomrule\end{longtable}")
    return L


def tex(res, trows, models, out, layer_tables=()):
    best = {}
    for x in res:
        if x["order"] in ("joint", "p_then_q", "q_then_p") and x["wikitext2_ppl"] is not None:
            k = (x["model"], x["sparsity_pct"], x["bits"])
            if k not in best or x["wikitext2_ppl"] < best[k]:
                best[k] = x["wikitext2_ppl"]
    L = [r"\documentclass[9pt]{extarticle}" if False else r"\documentclass{article}",
         r"\usepackage[a4paper,landscape,margin=1.5cm]{geometry}\usepackage{booktabs,longtable}\usepackage[T1]{fontenc}",
         r"\begin{document}\footnotesize", r"\section*{Supplementary tables}",
         "Every number below is produced by \\texttt{scripts/make\\_supplement.py} from the analysis tables of "
         "\\texttt{scripts/make\\_tables.py} (\\texttt{reports/analysis/*.csv}), which also generate every number of the paper. "
         "Seed 0, 32-bit. TCr: theoretical compression rate $1-\\frac{b}{16}(1-s)$. ppl: perplexity; KL: divergence to the "
         "dense model on WikiText-2; $E$: mean normalized layer error; $I_E$: layer interaction in \\% of $E_P+E_Q$ (mean over "
         "layers); Sr3: semantic retention on ARC-Easy, PIQA and HellaSwag; Sr9: on nine tasks (also ARC-Challenge, WinoGrande, "
         "OpenBookQA, BoolQ, SciQ, LAMBADA); $I_{Sr}$: Sr3 of the combination minus the product of the single methods' Sr3, "
         "with its 95\\% paired-bootstrap interval. Bold: best of the three orders by WikiText-2 perplexity. "
         "CSV files with every column: \\texttt{complete\\_results.csv}, \\texttt{task\\_scores.csv}, \\texttt{seeds.csv}; "
         "the layer-level controls and the weaker-first test follow the per-model tables."]
    for m in models:
        R_ = [x for x in res if x["model"] == m]
        L += [r"\begin{longtable}{llrrrrrrrrrrl}", r"\caption*{\textbf{" + m + r"}: all states (from e1\_states.csv, q1\_layer\_interaction.csv, e1\_isr.csv)}\\",
              r"\toprule order & $s$ (\%) & $b$ & TCr & WikiText-2 ppl & C4 ppl & KL & $E$ & $I_E$ (\%) & Sr3 & Sr9 & $I_{Sr}$ [95\%] \\ \midrule \endhead"]
        for x in R_:
            ppl = f(x["wikitext2_ppl"], 2)
            if x["order"] in ("joint", "p_then_q", "q_then_p") and best.get((m, x["sparsity_pct"], x["bits"])) == x["wikitext2_ppl"]:
                ppl = r"\textbf{" + ppl + "}"
            isr = "" if x["I_Sr3"] is None else f"{x['I_Sr3']:+.3f} [{x['I_Sr3_lo']:+.3f}, {x['I_Sr3_hi']:+.3f}]"
            L.append(" & ".join([ORDER_TEX[x["order"]], f(x["sparsity_pct"], 1) if x["order"] != "dense" else "", str(x["bits"]),
                                 f(x["tcr"], 3), ppl, f(x["c4_ppl"], 2), f(x["wikitext2_kl"], 4), f(x["layer_error"], 5),
                                 "" if x["I_E_pct"] is None else f"{x['I_E_pct']:+.1f}", f(x["sr3"], 3), f(x["sr9"], 3), isr]) + r" \\")
        L += [r"\bottomrule\end{longtable}"]
        T = [t for t in trows if t["model"] == m]
        if T:
            by = defaultdict(dict)
            for t in T:
                by[(t["order"], t["sparsity_pct"], t["bits"])][t["task"]] = t
            L += [r"\begin{longtable}{lrr" + "r" * len(TASKS) + "}",
                  r"\caption*{" + m + r": score / retention per task (retention = score / dense score; from e1\_tasks.csv). Metric: that of Sr3 and Table 2 for the paper's three tasks (acc for ARC-Easy and PIQA, acc\_norm for HellaSwag), that of Sr9 for the other six (acc\_norm for ARC-Challenge and OpenBookQA, acc for WinoGrande, BoolQ, SciQ, LAMBADA); task\_scores.csv also gives the Sr9 metric of every task}\\",
                  r"\toprule order & $s$ & $b$ & " + " & ".join(t.replace("_", r"\_") for t in TASKS) + r" \\ \midrule \endhead"]
            for (o, s, b), d in by.items():
                cells = [f"{d[t]['score']:.3f}/{d[t]['retention']:.3f}" if t in d and d[t]["retention"] is not None else "" for t in TASKS]
                L.append(" & ".join([ORDER_TEX[o], f(s, 1) if o != "dense" else "", str(b)] + cells) + r" \\")
            L += [r"\bottomrule\end{longtable}"]
        L.append(r"\clearpage")
    L += list(layer_tables)
    L.append(r"\end{document}")
    path = os.path.join(out, "supplement.tex")
    with open(path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "paper", "supplement"))
    ap.add_argument("--no-pdf", action="store_true")
    a = ap.parse_args()
    res, trows, seeds, models = build(a.out)
    abl, noc, wf = build_layer_tables(a.out, models)
    path = tex(res, trows, models, a.out, tex_layer_tables(abl, noc, wf) + tex_seed_table(seeds))
    if not a.no_pdf:
        for _ in range(2):
            subprocess.run(["pdflatex", "-interaction=nonstopmode", "supplement.tex"], cwd=a.out, stdout=subprocess.DEVNULL, check=False)
        for ext in (".aux", ".log", ".out"):
            p = os.path.join(a.out, "supplement" + ext)
            if os.path.exists(p):
                os.remove(p)
        print(f"make_supplement: {os.path.join(a.out, 'supplement.pdf')}")
    print(f"make_supplement: {path}")


if __name__ == "__main__":
    main()
