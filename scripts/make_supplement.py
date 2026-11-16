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
        "layer_error", "I_E_pct", "sr3", "sr3_lo", "sr3_hi", "sr9", "sr9_lo", "sr9_hi", "I_Sr3", "I_Sr3_lo", "I_Sr3_hi"]


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
    tasks = defaultdict(dict)
    for r in rows("e1_tasks.csv"):
        tasks[(r["model"], r["state_id"])][r["task"]] = (r["v2_metric"], fl(r["v2"]))
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
                    "I_Sr3": fl(i.get("value")), "I_Sr3_lo": fl(i.get("lo")), "I_Sr3_hi": fl(i.get("hi")), "_sid": r["state_id"]})
    dense_sid = {x["model"]: x["_sid"] for x in res if x["order"] == "dense"}
    trows = []
    for x in res:
        t, d = tasks.get((x["model"], x["_sid"]), {}), tasks.get((x["model"], dense_sid.get(x["model"])), {})
        for name in TASKS:
            if name in t and t[name][1] is not None:
                ds = d.get(name, (None, None))[1]
                trows.append({"model": x["model"], "order": x["order"], "sparsity_pct": x["sparsity_pct"], "bits": x["bits"], "task": name,
                              "metric": t[name][0], "score": t[name][1], "dense_score": ds,
                              "retention": t[name][1] / ds if ds else None})
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


def f(v, nd):
    return "" if v is None else f"{v:.{nd}f}"


def tex(res, trows, models, out):
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
         "CSV files with every column: \\texttt{complete\\_results.csv}, \\texttt{task\\_scores.csv}, \\texttt{seeds.csv}."]
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
                  r"\caption*{" + m + r": score / retention per task (from e1\_tasks.csv; retention = score / dense score)}\\",
                  r"\toprule order & $s$ & $b$ & " + " & ".join(t.replace("_", r"\_") for t in TASKS) + r" \\ \midrule \endhead"]
            for (o, s, b), d in by.items():
                cells = [f"{d[t]['score']:.3f}/{d[t]['retention']:.3f}" if t in d and d[t]["retention"] is not None else "" for t in TASKS]
                L.append(" & ".join([ORDER_TEX[o], f(s, 1) if o != "dense" else "", str(b)] + cells) + r" \\")
            L += [r"\bottomrule\end{longtable}"]
        L.append(r"\clearpage")
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
    path = tex(res, trows, models, a.out)
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
