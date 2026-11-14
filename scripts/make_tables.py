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
                           MACROS.md, fig_frontier_quant_<model>.dat, fig_frontier_sparse_<model>.dat,
                           fig1_interaction_v2.pdf, fig_order_scale.pdf (paper figures, ESANN print style)
        reports/figures/   frontier_<model>, orders_per_cell, e2_interaction_vs_hcorr (PNG + PDF), PNG previews of
                           the paper figures, README.md
        reports/analysis/  one CSV/JSON per analysis (E7 order rules: e7_kim.csv, e7_harma.csv, e7_summary.json;
                           E8 layer error vs KL: e8_layer_error_vs_kl*.csv/json, e8_layer_error_by_order.csv;
                           E5 seeds: seed_overview.csv, seed_ppl_orders.csv, seed_order_*.csv, seed_best_order.csv;
                           E6/Q3 family and scale: scale_family_cells.csv, e6_scale_spread.csv, scale_matched.csv,
                           fig_order_scale.csv; I_Sr as Table 2 defines it: isr_table2*.csv, isr_ijcnn.csv;
                           BATCH check: batch_*.csv, batch_summary.json; fig1_interaction_v2.csv),
                           summary.md, missing.md
Also reads results/v2/BATCH (the batch-size re-runs, never study states) and the PUB scores of
analysis/reproduce_paper.py (parsed with ast).
"""
import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.analysis import (batch, boolq_chance, det, e1, e2, e7, e8, frontier, isr, latex, paper_figures, questions,  # noqa: E402
                              repro, summary)
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
    if store.excluded_models:
        log(f"[make_tables] INTERIM: models left out ({len(store.excluded)} lines): {', '.join(store.excluded_models)}")
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
    e7res = e7.analyze(store, e1res, missing)
    e7.write(e7res, paths)
    e8res = e8.analyze(store, missing)
    e8.write(e8res, paths)
    isres = isr.analyze(store, e1res, paths, missing)
    isr.write(isres, paths)
    bres = batch.analyze(store, paths, missing)
    batch.write(bres, paths)
    paper_figures.write_data(qres, e2res, paths)
    tex = latex.write_all(store, e1res, qres, detres, fr, e2res, boolq[:2], paths, missing, repro=rep, e7res=e7res, e8res=e8res,
                          isr=isres, batch=bres)
    figs = []
    if figures:
        from scripts.analysis import figures as F
        figs = F.write_all(store, fr, qres, e2res, paths)
    summary.write(store, e1res, qres, detres, fr, e2res, boolq, tex, missing, paths, B, figs, rep, e7res, e8res,
                  isr=isres, batch=bres)
    log(f"[make_tables] {tex['macros']} macros ({tex['macros_missing']} still --, {tex['aliases']} aliases), {len(figs)} figures, "
        f"{len(missing.items)} missing entries; {time.time() - t0:.1f}s")
    return {"store": store, "e1": e1res, "q": qres, "det": detres, "frontier": fr, "e2": e2res, "tex": tex, "repro": rep, "e7": e7res, "e8": e8res,
            "isr": isres, "batch": bres, "missing": missing, "figures": figs}


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
