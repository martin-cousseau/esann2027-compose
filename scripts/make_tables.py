#!/usr/bin/env python3
"""Regenerate every analysis output of the v2 study from the result files (CPU only).

Safe to re-run while results are still arriving in results/v2/: whatever does not exist yet prints as --
and is listed in reports/analysis/missing.md; nothing is ever filled in.

    python scripts/make_tables.py                 # everything, 10,000 bootstrap resamples
    python scripts/make_tables.py --boot 1000     # quicker look (the outputs record the number used)
    python scripts/make_tables.py --no-figures --no-fetch

Reads   results/v2/**, results/results_full.jsonl, results/results_fix.jsonl, results/merged_results.json,
        analysis/reproduce_paper.py (AFFECTED list, parsed), scripts/make_queues.py (the plan),
        scripts/run_e2.py and colab/chain_c.sh (E2 cells and blocks), and once from the Hugging Face Hub the
        BoolQ validation labels (cached in reports/analysis/boolq_chance.json).
Writes  paper/generated/   table_main_rows.tex, table_kl_flips.tex, table_mechanism_rows.tex, macros.tex,
                           MACROS.md, fig_frontier_quant_<model>.dat, fig_frontier_sparse_<model>.dat
        reports/figures/   frontier_<model>, orders_per_cell, e2_interaction_vs_hcorr (PNG + PDF), README.md
        reports/analysis/  one CSV/JSON per analysis, summary.md, missing.md
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.analysis import boolq_chance, det, e1, e2, frontier, latex, questions, repro, summary  # noqa: E402
from scripts.analysis import stats as S  # noqa: E402
from scripts.analysis.common import Missing, Paths, Store  # noqa: E402


def run(paths=None, B=S.BOOT_B, figures=True, fetch=True, log=print):
    paths = paths or Paths()
    t0 = time.time()
    for d in (paths.generated, paths.figures, paths.reports):
        os.makedirs(d, exist_ok=True)
    missing = Missing()
    store = Store(paths)
    log(f"[make_tables] {len(store.states)} usable states, {len(store.det)} DET lines, {len(store.failures)} failure records")
    boolq = boolq_chance.load_or_fetch(paths, allow_fetch=fetch)
    e1res = e1.analyze(store, boolq[0], missing, B=B)
    e1.write(e1res, paths)
    qres = questions.analyze(store, e1res, paths, missing)
    questions.write(qres, paths)
    detres = det.analyze(store, missing)
    det.write(detres, paths)
    fr = frontier.analyze(store, missing)
    frontier.write(fr, paths)
    e2res = e2.analyze(paths, missing)
    e2.write(e2res, paths)
    rep = repro.analyze(paths, missing)
    repro.write(rep, paths)
    tex = latex.write_all(store, e1res, qres, detres, fr, e2res, boolq[:2], paths, missing, repro=rep)
    figs = []
    if figures:
        from scripts.analysis import figures as F
        figs = F.write_all(store, fr, qres, e2res, paths)
    summary.write(store, e1res, qres, detres, fr, e2res, boolq, tex, missing, paths, B, figs, rep)
    log(f"[make_tables] {tex['macros']} macros ({tex['macros_missing']} still --), {len(figs)} figures, "
        f"{len(missing.items)} missing entries; {time.time() - t0:.1f}s")
    return {"store": store, "e1": e1res, "q": qres, "det": detres, "frontier": fr, "e2": e2res, "tex": tex, "repro": rep,
            "missing": missing, "figures": figs}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=S.BOOT_B, help="bootstrap resamples (default 10,000)")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-fetch", action="store_true", help="never download the BoolQ labels")
    a = ap.parse_args()
    run(B=a.boot, figures=not a.no_figures, fetch=not a.no_fetch)
    return 0


if __name__ == "__main__":
    sys.exit(main())
