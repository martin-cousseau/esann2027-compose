"""reports/analysis/summary.md and missing.md, generated from the analysis results (never typed by hand)."""
import datetime as dt
import os

from . import stats as S
from .common import MECH_ARMS, ORDER_TXT, P, PAPER_ORDERS, finite, write_text


def f(v, nd=4, pct=False, signed=False):
    if isinstance(v, bool):
        return "yes" if v else "no"
    if not finite(v):
        return "--" if v is None or isinstance(v, (int, float)) else str(v)
    if pct:
        v = 100 * v
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def ci(t, nd=3, signed=False):
    if not t:
        return "--"
    return f"{f(t[0], nd, signed=signed)} [{f(t[1], nd, signed=signed)}, {f(t[2], nd, signed=signed)}]"


def table(header, rows):
    if not rows:
        return "_none yet_\n"
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows]
    return "\n".join(out) + "\n\n"


def seeds_md(q):
    L = ["\n## E5: orders over calibration seeds (4- and 3-bit cells, every multi-seed model)\n\n"
         "Sources: reports/analysis/seed_overview.csv, seed_order_stability.csv, seed_order_pairs.csv, seed_best_order.csv,"
         " seed_ppl_orders.csv (mean and sd over seeds per order and cell), e1_seeds.csv. A cell counts when every seed has"
         " the three orders (complete); (cell, seed) pairs only in complete cells. Log ratio = mean over seeds of"
         " log(ppl B / ppl A) (> 0: A better); paired t-test over seeds on log ppl.\n\n"]
    rows = []
    for r in q.get("seed_overview", []):
        bt = r["bits"] if r["bits"] == "both" else f"{r['bits']}-bit"
        rows.append([r["model"], bt, f"{r['states']}/{r['states_planned']}", r["cells"],
                     f"{r['best_stable']}/{r['cells']}" if r["cells"] else "--",
                     f"{r['qp_worst_all_seeds']}/{r['cells']}" if r["cells"] else "--",
                     f"{r['runs_qp_worst']}/{r['runs']}" if r["runs"] else "--",
                     f"{r['runs_joint_best']}/{r['runs']}" if r["runs"] else "--",
                     f"{r['pair_joint_p_then_q_a_all_seeds']}/{r['pair_joint_p_then_q_cells']}" if r["pair_joint_p_then_q_cells"] else "--",
                     f"{r['pair_joint_p_then_q_sig']}/{r['pair_joint_p_then_q_cells']}" if r["pair_joint_p_then_q_cells"] else "--",
                     f(r["sd_max_single_pct"], 2), f(r["sd_max_combined_pct"], 2)])
    L.append(table(["model", "bits", "E5 states", "complete cells", "same best order in every seed", "Q->P worst in every seed",
                    "(cell, seed): Q->P worst", "(cell, seed): joint best", "joint < P->Q in every seed", "joint vs P->Q: p < 0.05",
                    "max sd/mean single %", "max sd/mean combined %"], rows))
    L.append("Per cell (stability over seeds):\n\n")
    L.append(table(["model", "cell", "seeds present", "best order per seed", "worst order per seed", "complete", "same best",
                    "Q->P worst in every seed"],
                   [[r["model"], r["cell"], r["seeds_present"] or "--", r["best_by_seed"] or "--", r["worst_by_seed"] or "--",
                     f(r["complete"]), f(r["stable"]), f(r["qp_worst_all_seeds"])] for r in q.get("seed_stability", []) if r["e5_cell"]]))
    L.append("Paired comparisons (cells with at least two seeds):\n\n")
    L.append(table(["model", "cell", "A", "B", "seeds", "mean log ratio", "sd", "A better in every seed", "p (paired t)"],
                   [[r["model"], r["cell"], ORDER_TXT[r["order_a"]], ORDER_TXT[r["order_b"]], r["n_seeds"],
                     f(r.get("mean_log_ratio_b_over_a"), 4, signed=True), f(r.get("sd_log_ratio"), 4), f(r["a_better_all_seeds"]),
                     f(r.get("p_paired_t"), 3)] for r in q["seed_pairs"] if r["n_seeds"] >= 2]))
    S_ = [r for r in q.get("seed_ppl", []) if r["wt_ppl_n"] >= 2]
    L.append("WikiText-2 ppl over seeds, mean +- sd (configurations with at least two seeds):\n\n")
    L.append(table(["model", "cell", "order", "seeds", "mean", "sd", "sd/mean %"],
                   [[r["model"], r["cell"], ORDER_TXT.get(r["order"], r["order"]), r["seeds_present"], f(r["wt_ppl_mean"], 3),
                     f(r["wt_ppl_sd"], 3), f(r["wt_ppl_sd_pct"], 2)] for r in S_]))
    return "".join(L)


def scale_md(q):
    L = ["\n## E6 and Q3: order spread and Q->P penalty against scale, per family\n\n"
         "Sources: reports/analysis/scale_family_cells.csv (per model and cell), e6_scale_spread.csv (per family and cell:"
         " least-squares slope in points per decade of compressed weights; families never mixed), scale_matched.csv (families"
         " at matched scale), fig_order_scale.csv (the figure). Spread = 100 (max - min)/min of WikiText-2 ppl over the three"
         " orders; Q->P penalty = 100 (ppl Q->P / best - 1); seed 0.\n\n"]
    C = [r for r in q["scale_cells"] if r["complete"]]
    L.append(table(["model", "family", "compressed weights", "cell", "spread %", "Q->P penalty %", "joint vs P->Q gap %", "best", "worst"],
                   [[r["model"], r["family"], r["compressed_weights"] or "--", r["cell"], f(r["spread_pct"], 2), f(r["qp_penalty_pct"], 2),
                     f(r["gap_mask_first_pct"], 2), ORDER_TXT[r["best"]], ORDER_TXT[r["worst"]]] for r in C]))
    n_inc = sum(not r["complete"] for r in q["scale_cells"])
    L.append(f"{n_inc} planned (model, cell) pairs do not have the three orders yet.\n\nPer family:\n\n")
    L.append(table(["family", "cell", "models", "compressed weights", "spread %", "slope spread/decade", "spread falls with size",
                    "Q->P penalty %", "slope penalty/decade", "joint vs P->Q gap %", "slope gap/decade"],
                   [[r["family"], r["cell"], r["models"], r["compressed_weights"], r["spread_pct"],
                     f(r.get("slope_spread_per_decade"), 2, signed=True), f(r.get("spread_shrinks_monotonically")), r["qp_penalty_pct"],
                     f(r.get("slope_qp_penalty_per_decade"), 2, signed=True), r["gap_mask_first_pct"],
                     f(r.get("slope_gap_mask_first_per_decade"), 2, signed=True)] for r in q["scale"]]))
    L.append("Families at matched scale (min/max over the group's models; -- until every model of the group has the cell):\n\n")
    L.append(table(["group", "cell", "models (with the cell)", "spread %", "largest - smallest (points)", "largest / smallest",
                    "Q->P penalty %", "size ratio"],
                   [[r["label"], r["cell"], f"{r['n_models']}/{r['n_planned']}", r["spread_pct"] or "--",
                     f(r.get("spread_gap_pp"), 2) if r["complete"] else "--", f(r.get("spread_ratio"), 2) if r["complete"] else "--",
                     r["qp_penalty_pct"] or "--", f(r.get("weights_ratio"), 2)] for r in q["scale_matched"]]))
    return "".join(L)


def batch_md(b):
    sm = b["summary"]
    L = ["\n## BATCH check: lm-eval batch sizes\n\n"
         "Sources: results/v2/BATCH/*.jsonl and results/v2/items/<state_id>__BATCH.json.gz against the E1 line and items of the"
         " same state_id (scripts/analysis/batch.py; reports/analysis/batch_states.csv, batch_items.csv, batch_flips.csv,"
         " batch_flip_rankings.csv). Items differ = correctness under the v2 main metric differs between the two runs.\n\n"]
    L.append(table(["state", "available", "WT2 NLL identical", "C4 NLL identical", "layers identical", "E1 lm-eval batch",
                    "BATCH lm-eval batch", "items differ / compared", "share %", "acc or acc_norm differ"],
                   [[f"{r['model']} {r['label']}", f(r["available"]), f(r.get("wt_nll_identical")), f(r.get("c4_nll_identical")),
                     f(r.get("layers_identical")), r.get("e1_eval_batch", "--"), r.get("batch_eval_batch", "--"),
                     f"{r['items_differ']}/{r['items_n']}" if r.get("items_n") else "--", f(r.get("items_share"), 3, pct=True),
                     r.get("items_differ_any", "--")] for r in b["states"]]))
    L.append(f"Pooled over the compared states: {sm['items_differ']} of {sm['items_n']} items differ"
             f" ({f(sm['items_share'], 3, pct=True)}%); WikiText-2 NLL identical in {sm['nll_identical']} of {sm['nll_compared']}"
             " states.\n\nPer task, pooled over the states:\n\n")
    L.append(table(["task", "items differ", "items compared"],
                   [[t, sm[f"task_{t}_differ"], sm[f"task_{t}_n"]] for t in P.ZERO_SHOT_TASKS if sm[f"task_{t}_n"]]))
    L.append(f"Flips of the compressed E1/E4 states ({sm['flips_states']}; models: {sm['flips_models'] or 'none yet'}) against the"
             f" BATCH dense reference instead of E1's: largest |change| {f(sm['flips_max_change_pp'], 3)} points"
             f" ({sm['flips_max_change_where'] or '--'}); order ranking by flips changes in {sm['ranking_changes']} of"
             f" {sm['ranking_cells']} cells. With the re-run items also for the re-run compressed states: largest |change|"
             f" {f(sm['flips_max_change_both_pp'], 3)} points, ranking changes in {sm['ranking_changes_both']} cells.\n\n")
    L.append(table(["model", "cell", "ranking by flips (E1 dense)", "with the BATCH dense", "changes"],
                   [[c["model"], c["cell"], c["ranking_e1"], c["ranking_batch"], f(c["changes"])] for c in b["cells"]]))
    return "".join(L)


def isr_md(isr):
    L = ["\n## I_Sr as in Table 2 (Sr over ARC-E acc, PIQA acc, HellaSwag acc_norm; a ratio of sums)\n\n"
         "Sources: reports/analysis/isr_table2.csv (scripts/analysis/isr.py; e1.py's I_Sr with Sr set sr3_paper, paired"
         " bootstrap), isr_table2_v1_check.csv, isr_ijcnn.csv. Definition check: stats.sr(sr3_paper) = sum over the three"
         " tasks of the state's accuracy (paper metric) / the same sum for the dense model, which is how"
         " analysis/reproduce_paper.py computed the published Table 2; recomputed from the v1 runs (500-item subsets):\n\n"]
    L.append(table(["model", "cell", "I_Sr, ratio of sums (definition)", "I_Sr, mean of per-task ratios (other reading)"],
                   [[r["model"], r["cell"], f(r.get("isr_ratio_of_sums"), 3, signed=True), f(r.get("isr_mean_of_ratios"), 3, signed=True)]
                    for r in isr["v1"]]))
    L.append("v2 (fp32, full splits), the Table 2 cells (P->Q) and every other combined state of the core models:\n\n")
    L.append(table(["model", "cell", "order", "I_Sr [95% CI]", "interval excludes 0", "Table 2 cell", "I_Sr (Sr-extended, 9 tasks)"],
                   [[r["model"], r["cell"], ORDER_TXT[r["order"]], ci((r["isr"], r["lo"], r["hi"]), signed=True) if finite(r["isr"]) else "--",
                     f(r["excludes_zero"]), f(r["table2_cell"]), f(r["isr_sr9_v2"], 3, signed=True)] for r in isr["v2"]]))
    L.append("IJCNN 2025 7-8B rows (published scores, analysis/reproduce_paper.py PUB, read with ast):\n\n")
    L.append(table(["model", "cell", "tasks", "I_Sr"],
                   [[r["model"], r["cell"], r["tasks"], f(r["isr"], 3, signed=True)] for r in isr["ijcnn"]]))
    return "".join(L)


def summary_md(store, e1, q, det, fr, e2, boolq, tex_info, missing, paths, B, figures, rep=(), e7=None, e8=None, isr=None,
               batch=None):
    rate, rec, note = boolq
    L = ["# v2 analysis: current state\n\n",
         f"Generated by `python scripts/make_tables.py` at {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}"
         " (UTC). Every number below is computed by `scripts/analysis/` from the files named in each section; nothing"
         f" is typed by hand. What is missing: `reports/analysis/missing.md` ({len(missing.items)} entries).\n\n"]
    # ------------------------------------------------------------------ inputs
    L.append("## Inputs\n\n")
    fam = {}
    for sid, r in store.states.items():
        k = store.source[sid]
        fam[k] = fam.get(k, 0) + 1
    rows = [[k, v] for k, v in sorted(fam.items())]
    L.append(table(["result file", "usable states (status ok)"], rows))
    failed = {}
    for x in store.failures:
        failed.setdefault(x["state_id"], []).append(x)
    planned_ok = sum(1 for sid in store.plan if sid in store.states)
    L.append(f"- Planned model states (scripts/make_queues.py): {len(store.plan)}; available: {planned_ok}; usable states in"
             f" total: {len(store.states)}.\n- Failure records: {len(store.failures)} for {len(failed)} states"
             f" (results/v2/*/failures__*.jsonl), {len(store.archived_failures)} archived; lines not ok:"
             f" {len(store.not_ok)}; files outside the study families ignored: {len(store.ignored_files)}"
             + (f" ({', '.join(store.ignored_files)})" if store.ignored_files else "") + "; duplicate lines ignored:"
             f" {len(store.duplicates)}; other dtypes ignored: {len(store.other_dtype)}.\n"
             f"- Determinism lines (results/v2/DET): {len(store.det)}. E2 lines: "
             + ", ".join(f"{c['model']} {c['n_lines']}" for c in e2["coverage"]) + ".\n"
             f"- Per-item files usable: {sum(1 for sid in store.states if store.items(sid) is not None)}; problems:"
             f" {len(store.item_problems)}.\n"
             f"- BoolQ chance level: {f(rate, 4)} ({note}" + (f"; {rec['dataset']} {rec['config']} {rec['split']}, revision"
                                                            f" {rec['revision']}, {rec['n']} labels, counts {rec['counts']}" if rec else "") + ").\n"
             f"- Bootstrap: {B} resamples, seed {S.BOOT_SEED}, percentile 95% intervals, items resampled within each task.\n")
    # ------------------------------------------------------------------ DET
    L.append("## Determinism (DET re-run on a second L4)\n\n")
    L.append("Source: results/v2/DET/*.jsonl against the same state_id in the study families (scripts/analysis/det.py).\n\n")
    L.append(table(["state", "status", "DET ppl", "E1 ppl", "|nll_sum diff|", "max |rel_err diff|", "layers", "DET GPU", "E1 GPU"],
                   [[d["state_id"], d.get("status"), f(d.get("det_ppl"), 6), f(d.get("e1_ppl"), 6), f(d.get("nll_sum_abs_diff"), 6),
                     f(d.get("rel_err_max_abs_diff"), 9), d.get("n_layers_compared", "--"), d.get("det_gpu") or "--",
                     d.get("e1_gpu") or "--"] for d in det]))
    # ------------------------------------------------------------------ E1 states
    L.append("\n## E1 (and later families): states\n\n")
    L.append("Sources: results/v2/<family>/<model>.jsonl (perplexity, KL, top-1, storage) and results/v2/items/<state_id>.json.gz"
             " (accuracy, Sr, flips); scripts/analysis/e1.py. Full tables: reports/analysis/e1_*.csv. KL in nats/token;"
             " flips pooled over the 9 zero-shot tasks (v2 metric); Sr with paired 95% interval.\n\n")
    for model in store.models():
        rows = []
        for st in [x for x in e1["states"] if x["model"] == model]:
            b = e1["by_sid"].get(st["state_id"], {})
            fl = (b.get("flips", {}).get("pooled9") or {}).get("flips")
            rows.append([st["label"] + (f" (seed {st['seed']})" if st["seed"] else ""), f(st["wt_ppl"], 3), f(st["c4_ppl"], 3),
                         f(st["wt_kl"], 4), f(st["wt_top1"], 4), f(fl, 2, pct=True), ci(b.get("sr", {}).get("sr3_v2")),
                         f((b.get("sr", {}).get("sr3_paper") or [None])[0], 3), f((b.get("sr", {}).get("sr9_v2") or [None])[0], 3),
                         f((b.get("sr", {}).get("sr9_cn") or [None])[0], 3), st["source"]])
        L.append(f"### {model}\n\n")
        L.append(table(["state", "WT2 ppl", "C4 ppl", "KL WT2", "top-1 WT2", "flips %", "Sr3 v2 [95% CI]", "Sr3 paper", "Sr9", "Sr9 cn", "source"], rows))
    # ------------------------------------------------------------------ I_Sr
    L.append("\n## I_Sr = Sr_PQ - Sr_P Sr_Q\n\n")
    L.append("Source: reports/analysis/e1_isr.csv (P = prune-only at the same s, Q = quant-only at the same b, same seed).\n\n")
    rows = []
    for x in e1["isr"]:
        rows.append([store.label(x["state_id"]) + f" ({x['model']})", x["set"], ci((x["value"], x["lo"], x["hi"]), signed=True),
                     f(x["sr_pq"], 3), f(x["sr_p"], 3), f(x["sr_q"], 3)])
    L.append(table(["state", "Sr set", "I_Sr [95% CI]", "Sr_PQ", "Sr_P", "Sr_Q"], rows))
    # ------------------------------------------------------------------ orders within cells
    L.append("\n## Orders within a cell: exact McNemar and paired accuracy differences\n\n")
    L.append("Sources: reports/analysis/e1_mcnemar.csv, e1_acc_orders.csv (pooled over the 9 tasks; mean accuracy difference"
             " a - b over the 9 tasks, v2 metric).\n\n")
    acc = {(x["model"], x["cell"], x["seed"], x["order_a"], x["order_b"]): x for x in e1["acc_orders"] if x["task"] == "mean9_v2"}
    rows = []
    for x in e1["mcnemar"]:
        if x["task"] != "pooled9":
            continue
        a = acc.get((x["model"], x["cell"], x["seed"], x["order_a"], x["order_b"]))
        rows.append([x["model"], x["cell"], x["seed"], f"{ORDER_TXT[x['order_a']]} vs {ORDER_TXT[x['order_b']]}", x["a_only"],
                     x["b_only"], f(x["p"], 4), ci((a["diff"], a["lo"], a["hi"]), 4, True) if a else "--"])
    L.append(table(["model", "cell", "seed", "pair", "right in a only", "right in b only", "McNemar p", "mean acc diff [95% CI]"], rows))
    # ------------------------------------------------------------------ Q1
    s1 = q["q1_summary"]
    L.append("\n## Q1: layer-level interaction from the per-layer diagnostics\n\n")
    L.append("Source: reports/analysis/q1_layer_interaction.csv (scripts/analysis/questions.py). Ratio = I_E/(E_P+E_Q) from"
             " mean errors over layers (paper's Fig. 1); share = sub-additive layers matched layer by layer.\n\n")
    L.append(f"- Mask-first (joint, P->Q) combinations sub-additive: {s1['mask_first_subadditive']} of {s1['mask_first_total']}.\n")
    for o in PAPER_ORDERS:
        for b in (4, 3):
            t = f"{o}_{b}bit"
            if s1[f"{t}_n"]:
                L.append(f"- {ORDER_TXT[o]}, {b}-bit: {s1[f'{t}_n']} cells, ratio {f(s1[f'{t}_min_pct'], 1, signed=True)}% to"
                         f" {f(s1[f'{t}_max_pct'], 1, signed=True)}%, sub-additive {s1[f'{t}_n_subadditive']}, super-additive"
                         f" {s1[f'{t}_n_superadditive']}; sub-additive layer records {f(s1[f'{t}_layer_share_subadditive'], 1, pct=True)}%;"
                         f" block 0 alone super-additive in {s1[f'{t}_cells_block0_superadditive']} of {s1[f'{t}_cells_block0_vs_rest_n']} cells.\n")
    rows = [[r["model"], r["seed"], r["cell"], ORDER_TXT[r["order"]], r["n_layers"], f(r["ratio_mean_pct"], 2, signed=True),
             f(r["share_subadditive"], 1, pct=True), f(r["ratio_block0_pct"], 2, signed=True), f(r["ratio_blocks1plus_pct"], 2, signed=True)]
            for r in q["q1"]]
    L.append("\n" + table(["model", "seed", "cell", "order", "layers", "ratio %", "sub-additive layers %", "block 0 %", "blocks 1+ %"], rows))
    # ------------------------------------------------------------------ Q2 / Q3
    s2 = q["q2_summary"]
    L.append("\n## Q2: order ranking per cell (seed 0)\n\n")
    L.append("Source: reports/analysis/q2_order_ranking.csv. Rankings best to worst.\n\n")
    for key in ("wt_ppl", "c4_ppl", "wt_kl", "flips9", "sr3_v2"):
        L.append(f"- {key}: {s2[f'{key}_cells']} complete cells; Q->P strictly worst in {s2[f'{key}_qp_worst']}; best order counts "
                 + ", ".join(f"{ORDER_TXT[o]} {s2[f'{key}_best_{o}']}" for o in PAPER_ORDERS) + ".\n")
    L.append(f"- Largest worst/best WikiText-2 ppl ratio: {f(s2.get('wt_ppl_max_worst_best_ratio'), 3)}"
             f" ({s2.get('wt_ppl_max_worst_best_where') or '--'}); C4: {f(s2.get('c4_ppl_max_worst_best_ratio'), 3)}.\n"
             f"- Identical ranking under WT2 ppl, C4 ppl, KL, flips and Sr: {s2['agreement_identical']} of {s2['agreement_cells']} cells"
             f" with all five available.\n")
    keys = ("wt_ppl", "c4_ppl", "wt_kl", "flips9", "sr3_v2")
    shown = [r for r in q["q2"] if any(r.get(f"{k}_ranking") for k in keys)]
    rows = [[r["model"], r["cell"]] + [r.get(f"{k}_ranking", "--") for k in keys] + [f(r.get("wt_ppl_worst_best_ratio"), 3)]
            for r in shown]
    L.append("\n" + table(["model", "cell", "WT2 ppl", "C4 ppl", "KL", "flips", "Sr3 v2", "worst/best WT2"], rows))
    L.append(f"{len(q['q2']) - len(shown)} of {len(q['q2'])} cells have no complete ranking yet (missing.md).\n\n")
    L.append("\n## Q3: order spread (max - min) / min of WikiText-2 ppl\n\nSource: reports/analysis/q3_order_spread.csv.\n\n\n")
    L.append(table(["model", "bits", "metric", "cells", "planned", "min %", "max %", "grid cells (25/33/50%)", "grid min %",
                    "grid max %"],
                   [[r["model"], r["bits"], r["metric"], r["n_cells"], r["n_cells_planned"], f(r["spread_min_pct"], 2),
                     f(r["spread_max_pct"], 2), f"{r['grid_n_cells']}/{r['grid_n_planned']}",
                     f(r["grid_spread_min_pct"], 2) if r["grid_complete"] else "--",
                     f(r["grid_spread_max_pct"], 2) if r["grid_complete"] else "--"] for r in q["q3"] if r["n_cells"]]))
    L.append("Grid min/max (macros \\resQThreeSpread{Min,Max}<model>{BFour,BThree}Pct): -- until every planned grid cell of"
             " the model has the three orders; min/max over every cell: E4's 60%+4 included.\n\n")
    # ------------------------------------------------------------------ fp16 vs fp32
    sf = q["fp_summary"]
    L.append("\n## fp16 (published) vs fp32 (v2): WikiText-2 perplexity, seed 0\n\n")
    L.append("Sources: " + ", ".join(q.get("v1_files") or []) + "; the paper's merge (AFFECTED re-runs in 32-bit) checked against"
             f" results/merged_results.json: {q.get('v1_check')}.\n\n")
    L.append(f"- Compared: {sf['n_compared']} of {sf['n_published']}; largest |relative difference| "
             f"{f(sf.get('max_abs_rel_delta'), 3, pct=True)}% ({sf.get('max_where', '--')}).\n\n")
    rows = [[r["model"], r["config"], "" if r["order"] in ("dense", "prune", "quant") else ORDER_TXT.get(r["order"], r["order"]),
             f(r["published_ppl"], 4), r["published_precision"],
             f(r["v2_ppl"], 4), f(r["rel_delta"], 3, pct=True, signed=True), f(r["fix_fp32_ppl"], 4)] for r in q["fp"] if r["v2_ppl"] is not None]
    L.append(table(["model", "config", "order", "published ppl", "precision", "v2 fp32 ppl", "delta %", "results_fix fp32 ppl"], rows))
    L.append("\n## Hardware and precision reproduction (REPRO16, REPRO32)\n\n")
    L.append("Source: reports/analysis/repro_hardware.csv (results/v2/REPRO*/ against results/results_full.jsonl for 16-bit and"
             " results/results_fix.jsonl for fp32; planned units from colab/chain_*.sh).\n\n")
    L.append(table(["family", "state", "ppl", "reference", "reference ppl", "delta %", "GPU"],
                   [[r["family"], r["state_id"], f(r["ppl"], 4), r["reference"], f(r["ref_ppl"], 4),
                     f(r["rel_delta_pct"], 3, signed=True), r.get("gpu") or "--"] for r in rep]))
    L.append("Equal storage with the mask counted (best order of the cell against b-bit alone; frontier_equal_storage.csv):\n\n")
    L.append(table(["model", "cell", "vs", "best order", "best ppl", "b-bit ppl", "bits/w (mask) cell / b-bit", "gain %"],
                   [[r["model"], r["cell"], f"{r['quant_bits']}-bit", ORDER_TXT.get(r["best_order"], "--") if r["best_order"] else "--",
                     f(r["best_ppl"], 3), f(r["quant_ppl"], 3), f"{f(r['best_bits_mask'], 2)} / {f(r['quant_bits_mask'], 2)}",
                     f(r["gain_pct"], 2, signed=True)] for r in fr["equal_storage"] if r["available"]]))
    # ------------------------------------------------------------------ storage
    L.append("\n## Storage frontier (E4) and free zeros\n\n")
    L.append("Source: reports/analysis/frontier_points.csv (storage block of each line); pgfplots data in paper/generated/"
             f"fig_frontier_*.dat ({sum(tex_info['dat'].values())} rows in {len(tex_info['dat'])} files).\n\n")
    rows = [[p["model"], store.label(p["state_id"]), f(p["bits_tcr"], 3), f(p["bits_bitmap"], 3), f(p["bits_entropy_actual"], 3),
             f(p["bits_symbol_entropy"], 3), f(p["zero_frac_actual"], 1, pct=True), f(p["wt_ppl"], 3), f(p["wt_kl"], 4),
             "best" if p.get("best_in_cell") else ""] for p in fr["points"]]
    L.append(table(["model", "state", "bits TCr", "bits 1-bit mask", "bits entropy", "symbol entropy", "zeros %", "WT2 ppl", "KL", ""], rows))
    L.append("\nQ->P 25%+3 against 3-bit alone (identical perplexity and every per-layer rel_err?):\n\n")
    L.append(table(["model", "status", "ppl Q->P", "ppl 3-bit", "max |rel_err diff|", "layers"],
                   [[r["model"], r["status"], f(r.get("wt_ppl_qp"), 6), f(r.get("wt_ppl_q3"), 6), f(r.get("rel_err_max_abs_diff"), 9),
                     r.get("n_layers", "--")] for r in fr["identity"]]))
    # ------------------------------------------------------------------ E2
    L.append("\n## E2: same-Hessian layer ablation\n\n")
    L.append("Sources: " + (", ".join(f"{m}: {s}" for m, s in e2["sources"].items()) or "results/v2/E2 (no lines yet)")
             + f"; block plan {e2['plan']} (colab/chain_c.sh); cells from scripts/run_e2.py. Tables: reports/analysis/e2_*.csv;"
             " the review's own summarize() printout: reports/analysis/e2_review_summary.txt.\n\n")
    L.append(table(["model", "lines", "planned", "missing", "blocks present"],
                   [[c["model"], c["n_lines"], c.get("n_expected", "--"), c.get("n_missing", "--"), c["blocks_present"] or "--"]
                    for c in e2["coverage"]]))
    rho = {(r["model"], r["cell"], r["arm"]): r for r in e2["spearman"]}
    rows = []
    for c in e2["cells"]:
        if not c.get("n_layers"):
            continue
        r = rho.get((c["model"], c["cell"], "p_then_q"), {})
        rows.append([c["model"], c["cell"], c["n_layers"], f(c.get("qp_penalty_pct_of_pq"), 1, signed=True)]
                    + [f(c.get(f"ratio_{a}_pct"), 1, signed=True) for a in PAPER_ORDERS]
                    + [f(c.get(f"share_sub_{a}"), 0, pct=True) for a in PAPER_ORDERS]
                    + [f(c.get(f"penalty_removed_{a}_pct"), 0, signed=True) for a in MECH_ARMS]
                    + [f(c.get("stale_grid_on_pq_pct"), 1, signed=True), f(c.get("mask_quality_median"), 2), f(c.get("mask_jaccard_median"), 2),
                       f(r.get("rho"), 2, signed=True) + f" (p {f(r.get('p'), 3)})"])
    L.append("\n" + table(["model", "cell", "layers", "Q->P penalty % of P->Q", "joint %", "P->Q %", "Q->P %", "joint sub %", "P->Q sub %", "Q->P sub %",
                           "refit removes %", "P-mask removes %", "both remove %", "stale on P->Q %", "mask quality", "Jaccard",
                           "Spearman P->Q vs h_corr"], rows))
    rows = []
    for c in e2["cells"]:
        if c.get("n_layers"):
            rows.append([c["model"], c["cell"]] + [f(c.get(f"ratio_block0_{a}_pct"), 1, signed=True) + " / "
                                                   + f(c.get(f"ratio_deeper_{a}_pct"), 1, signed=True) for a in PAPER_ORDERS])
    L.append("\nBlock 0 / deeper blocks, I_E/(E_P+E_Q) %:\n\n" + table(["model", "cell"] + [ORDER_TXT[a] for a in PAPER_ORDERS], rows))
    # ------------------------------------------------------------------ E3
    L.append("\n## E3: share of the Q->P perplexity penalty removed in full runs\n\nSource: reports/analysis/e3_mechanism.csv.\n\n\n")
    rows = [[r["model"], r["cell"], f(r.get("wt_ppl_q_then_p"), 3), f(r.get("wt_ppl_p_then_q"), 3)]
            + [f(r.get(f"wt_ppl_removed_{a}_pct"), 0, signed=True) for a in MECH_ARMS] + [f(r.get("wt_ppl_stale_on_pq_pct"), 1, signed=True)]
            for r in q["e3"] if finite(r.get("wt_ppl_q_then_p")) or finite(r.get("wt_ppl_p_then_q"))]
    L.append(table(["model", "cell", "Q->P ppl", "P->Q ppl", "refit removes %", "P-mask removes %", "both remove %", "stale on P->Q %"], rows))
    # ------------------------------------------------------------------ E5 seeds
    if q.get("seed_pairs") or q.get("seed_overview"):
        L.append(seeds_md(q))
    # ------------------------------------------------------------------ E6 scale and family
    if q.get("scale_cells"):
        L.append(scale_md(q))
    # ------------------------------------------------------------------ E7
    if e7 is not None:
        sm = e7["summary"]
        L.append("\n## E7: order rules (Kim et al.'s weaker-first rule with CER; Harma et al.'s threshold)\n\n"
                 "Sources: reports/analysis/e7_kim.csv, e7_harma.csv, e7_summary.json (scripts/analysis/e7.py). Counts over"
                 f" the {' and '.join(sm['scope'])} cells, seed 0. Kim: the weaker method (lower CER on the quantization curve,"
                 " C = 16/b) should come first; outcome = the better of P->Q and Q->P. rho = Spearman of the P->Q advantage"
                 " against CER(Q) - CER(P) (Kim's Hypothesis 1 expects rho > 0).\n\n")
        rows = []
        for m in sm["models"] + ["all"]:
            row = [m]
            for name in ("wt_ppl", "c4_ppl", "wt_kl", "sr9_v2", "sr3_paper"):
                k = sm["kim"].get(f"{name}|{m}") or {}
                row.append(f"{k.get('correct', '--')}/{k.get('decided', '--')}" if k.get("decided") else "--")
            k = sm["kim"].get(f"wt_ppl|{m}") or {}
            row.append(f"{k.get('predicted_qp_correct', '--')}/{k.get('predicted_qp', '--')}" if k.get("decided") else "--")
            row.append(f(k.get("rho"), 2, signed=True) + (f" (n {k.get('rho_n')})" if k.get("rho_n") else ""))
            for name in ("sr9_v2", "sr3_paper"):
                ks = sm["kim"].get(f"{name}|{m}") or {}
                row.append(f"{ks.get('significant_correct', '--')}/{ks.get('significant', '--')}" if ks.get("decided") else "--")
            rows.append(row)
        L.append(table(["model", "correct: WikiText-2 ppl", "C4 ppl", "KL", "Sr-extended", "Sr (paper, 3 tasks)",
                        "Q->P predictions correct (ppl)", "rho (ppl)", "correct where the Sr-extended interval excludes 0",
                        "correct where the paper-Sr interval excludes 0"], rows))
        rows = []
        for m in sm["models"] + ["all"]:
            ho = sm["harma_order"].get(f"wt_ppl|{m}") or {}
            row = [m, f"{ho.get('pq_better', '--')}/{ho.get('cells', '--')}" if ho.get("cells") else "--",
                   f(ho.get("min_gap_pct"), 1, signed=True) + " to " + f(ho.get("max_gap_pct"), 1, signed=True) if ho.get("cells") else "--"]
            for o in PAPER_ORDERS:
                t = sm["harma_threshold"].get(f"wt_ppl|{o}|{m}") or {}
                row.append(f"{t.get('exceeds_additive', '--')} / {t.get('exceeds_multiplicative', '--')} of {t.get('n', '--')}" if t.get("n") else "--")
            rows.append(row)
        L.append("Harma et al.: sparsity first (P->Q better than Q->P), and how many combined states exceed the additive"
                 " orthogonality threshold T_add = ppl(dense) + dPPL(P) + dPPL(Q) (their Def. 3.11) / the multiplicative"
                 " null T_mult = ppl(dense) x ppl(P)/ppl(dense) x ppl(Q)/ppl(dense) (additive loss); WikiText-2 ppl:\n\n")
        L.append(table(["model", "P->Q better", "Q->P / P->Q - 1 (%)"] + [f"{ORDER_TXT[o]} exceeds T_add / T_mult" for o in PAPER_ORDERS], rows))
    # ------------------------------------------------------------------ E8
    if e8 is not None:
        L.append("\n## E8 (analysis part): does a smaller layer error predict a smaller KL?\n\n"
                 "Sources: reports/analysis/e8_layer_error_vs_kl*.csv/json (scripts/analysis/e8.py); seed 0; mean_rel_err of the"
                 " result lines (the mean of the per-layer relative errors).\n\n")
        L.append(table(["model", "states", "rho vs KL WikiText-2", "rho vs KL C4", "rho, combined states only", "cells: same best order",
                        "cells: same ranking"],
                       [[m["model"], m["n_states"], f(m.get("rho_wt_kl"), 2, signed=True), f(m.get("rho_c4_kl"), 2, signed=True),
                         f(m.get("rho_wt_kl_combined"), 2, signed=True), f"{m['cells_same_best']}/{m['cells']}",
                         f"{m['cells_same_ranking']}/{m['cells']}"] for m in e8["models"]]))
        L.append("Cells where the joint sweep has strictly the smallest mean layer error of the three orders"
                 " (e8_layer_error_by_order.csv):\n\n")
        L.append(table(["scope", "bits", "joint smallest / cells", "other cells"],
                       [[r["scope"], r["bits"], f"{r['joint_smallest']}/{r['cells']}", r["not_joint"] or "--"]
                        for r in e8.get("joint_smallest", []) if r["cells"]]))
    if isr is not None:
        L.append(isr_md(isr))
    if batch is not None:
        L.append(batch_md(batch))
    # ------------------------------------------------------------------ checks and outputs
    ck = e1["checks"]
    L.append("\n## Checks\n\n"
             f"- Per-item means against lm-eval's aggregates: max |difference| {f(ck['item_vs_aggregate_max_abs'], 12)} over"
             f" {ck['item_vs_aggregate_n']} (state, task, metric) values.\n"
             f"- Pooled flips against extra_orders.flips: max |difference| {f(ck['flips_vs_extra_orders_max_abs'], 12)}.\n"
             f"- States whose item set differs from the common one: {len(ck['doc_id_notes'])}"
             + (": " + "; ".join(ck["doc_id_notes"][:10]) if ck["doc_id_notes"] else "") + ".\n"
             "- Dense accuracy at or below the task's chance level (v2 metric; that task has no headroom in the"
             " chance-normalized Sr): " + ("; ".join(f"{x['model']} {x['task']} {f(x['dense_v2'], 4)} vs chance {f(x['chance'], 4)}"
                                                    for x in ck.get("dense_at_or_below_chance", [])) or "none") + ".\n")
    L.append("\n## Outputs\n\n"
             f"- paper/generated/: table_main_rows.tex, table_kl_flips.tex, table_mechanism_rows.tex, macros.tex ({tex_info['macros']}"
             f" macros, {tex_info['macros_missing']} still --, {tex_info.get('aliases', 0)} of them aliases under the derived model"
             " tags), MACROS.md, fig_frontier_*.dat, fig_order_scale.pdf, fig1_interaction_v2.pdf.\n"
             f"- reports/figures/: {len(figures)} figures (PNG + PDF; the two paper figures as PNG previews), captions in README.md.\n"
             "- reports/analysis/: the CSV/JSON files named above, summary.md, missing.md.\n")
    L.append(missing_overview(store, missing))
    return "".join(L)


def missing_md(store, missing):
    L = ["# Missing results and analyses\n",
         "Generated by `python scripts/make_tables.py`; nothing here is filled in elsewhere. Planned states come from"
         " scripts/make_queues.py.\n"]
    L.append("\n## Planned model states not available\n\n\n")
    failed = {}
    for x in store.failures:
        failed.setdefault(x["state_id"], []).append(x)
    groups = {}
    for sid, info in store.plan.items():
        if sid not in store.states:
            groups.setdefault((info["family"], info["short"]), []).append(sid)
    total = {}
    for sid, info in store.plan.items():
        total[(info["family"], info["short"])] = total.get((info["family"], info["short"]), 0) + 1
    if not groups:
        L.append("_none_\n")
    for (fam, m), sids in sorted(groups.items()):
        L.append(f"- **{fam} {m}**: {len(sids)} of {total[(fam, m)]} missing: " + ", ".join(
            store.label(s) + (f" seed {store.plan[s]['seed']}" if store.plan[s]["seed"] else "") for s in sids) + "\n")
    if failed:
        L.append("\nFailure records (the runner retries a unit once; two failures and it is skipped):\n\n")
        for sid, xs in sorted(failed.items()):
            L.append(f"- `{sid}`: {len(xs)} failure(s), last at {xs[-1].get('time')}: {str(xs[-1].get('error'))[:200]}"
                     f" ({xs[-1]['_file']})\n")
    if store.archived_failures:
        L.append("\nArchived failure records (files renamed by the orchestrator; informational, the unit may since have run):\n\n")
        for x in store.archived_failures:
            L.append(f"- `{x.get('state_id')}` at {x.get('time')}: {str(x.get('error'))[:160]} ({x['_file']})"
                     + (" -- now available" if x.get("state_id") in store.states else "") + "\n")
    if store.not_ok:
        L.append("\nLines present but not used (status not ok):\n\n")
        for x in store.not_ok:
            L.append(f"- `{x['state_id']}` ({x['file']}): status {x['status']}, checks {x.get('checks')}\n")
    if store.item_problems:
        L.append("\nPer-item files:\n\n")
        for sid, why in sorted(store.item_problems.items()):
            L.append(f"- `{sid}`: {why}\n")
    L.append("\n## Analyses and table entries not computable yet (grouped by reason)\n\n")
    for sec, groups in _grouped(missing):
        L.append(f"\n### {sec}\n\n")
        for reason, whats in groups:
            L.append(f"- {reason} ({len(whats)}): " + "; ".join(whats) + "\n")
    return "".join(L)


def _grouped(missing):
    by = {}
    for x in missing.items:
        by.setdefault(x["section"], {}).setdefault(x["reason"], []).append(x["what"])
    return [(sec, sorted(g.items(), key=lambda kv: -len(kv[1]))) for sec, g in sorted(by.items())]


def missing_overview(store, missing):
    """Counts for summary.md; the full lists are in missing.md."""
    rows = []
    tot, miss = {}, {}
    for sid, info in store.plan.items():
        k = (info["family"], info["short"])
        tot[k] = tot.get(k, 0) + 1
        if sid not in store.states:
            miss[k] = miss.get(k, 0) + 1
    for k in sorted(tot):
        rows.append([k[0], k[1], tot[k], tot[k] - miss.get(k, 0), miss.get(k, 0)])
    out = "\n## What is missing (overview; full lists in reports/analysis/missing.md)\n\n"
    out += table(["family", "model", "planned states", "available", "missing"], rows)
    out += "\n" + table(["analysis / output", "entries not computable yet"],
                        [[sec, sum(len(w) for _, w in groups)] for sec, groups in _grouped(missing)])
    return out


def write(store, e1, q, det, fr, e2, boolq, tex_info, missing, paths, B, figures, rep=(), e7=None, e8=None, isr=None, batch=None):
    write_text(os.path.join(paths.reports, "summary.md"),
               summary_md(store, e1, q, det, fr, e2, boolq, tex_info, missing, paths, B, figures, rep, e7, e8, isr, batch))
    write_text(os.path.join(paths.reports, "missing.md"), missing_md(store, missing))
