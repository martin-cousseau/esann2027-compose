"""Tests of the 2026-11-12 analyses (CPU, no network): seeds at both bit-widths, family vs scale, BATCH check, I_Sr
as Table 2 defines it, E8 joint smallest, E7 with the paper's Sr, Q3 grid ranges, model-tag aliases, paper figures.

    python -m pytest -q tests/test_analysis_extended.py

Synthetic data only, in pytest's temporary directories (nothing touches results/ or the real outputs).
"""
import gzip
import hashlib
import json
import math
import os
import re
import statistics as st
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from scripts.analysis import batch, e1 as E1, isr, latex, paper_figures, questions  # noqa: E402
from scripts.analysis import stats as S  # noqa: E402
from scripts.analysis.common import Missing, P, Paths, Store  # noqa: E402
from scripts.run_grid import state_id  # noqa: E402
from test_analysis import make_fake_repo  # noqa: E402

FULL = {"SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M", "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M",
        "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B", "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B", "Qwen2.5-1.5B": "Qwen/Qwen2.5-1.5B",
        "Qwen2.5-3B": "Qwen/Qwen2.5-3B", "OLMo-2-0425-1B": "allenai/OLMo-2-0425-1B"}


def _line(model, order, s, b, seed, ppl, family="E1", cw=None, err=None):
    m = FULL[model]
    u = {"order": order, "sparsity": s, "bits": b, "seed": seed}
    return {"state_id": state_id(m, u, "float32"), "family": family, "model": m, "order": order, "sparsity": s, "bits": b,
            "seed": seed, "dtype": "float32", "status": "ok", "wikitext2": {"ppl": ppl, "nll_sum": 100 * math.log(ppl)},
            "compressed_weights": cw, "mean_rel_err": err, "diags": []}


def _write(root, family, model, lines):
    d = os.path.join(root, "results", "v2", family)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, f"{model}.jsonl"), "a") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")


def _vals(root):
    with open(os.path.join(root, "paper", "generated", "macros.tex")) as f:
        return dict(re.findall(r"^\\newcommand\{\\([^}]*)\}\{(.*)\}$", f.read(), flags=re.M))


# ----------------------------------------------------------------------------- E5: both bit-widths, every model
SEED_PPL = {  # (model, cell, seed): (joint, P->Q, Q->P)
    ("SmolLM2-360M", (0.25, 4), 0): (13.20, 13.34, 14.38), ("SmolLM2-360M", (0.25, 4), 1): (13.25, 13.31, 14.20),
    ("SmolLM2-360M", (0.25, 4), 2): (13.10, 13.40, 14.50),
    ("SmolLM2-360M", (0.25, 3), 0): (20.0, 20.4, 22.0), ("SmolLM2-360M", (0.25, 3), 1): (20.6, 20.3, 22.5),
    ("SmolLM2-360M", (0.25, 3), 2): (23.0, 20.1, 21.0),          # seed 2: joint worst, P->Q best
    ("Qwen2.5-0.5B", (0.25, 4), 0): (14.6, 14.8, 15.1), ("Qwen2.5-0.5B", (0.25, 4), 1): (14.7, 14.6, 15.0),  # seed 2 missing
    ("Qwen2.5-0.5B", (0.25, 3), 0): (21.5, 21.9, 21.8), ("Qwen2.5-0.5B", (0.25, 3), 1): (21.4, 22.0, 21.7),
    ("Qwen2.5-0.5B", (0.25, 3), 2): (21.6, 21.8, 21.9)}


def _seed_repo(root):
    by = {}
    for (m, (s, b), sd), v in SEED_PPL.items():
        for o, p in zip(("joint", "p_then_q", "q_then_p"), v):
            by.setdefault(m, []).append(_line(m, o, s, b, sd, p, family="E1" if sd == 0 else "E5"))
        by[m].append(_line(m, "quant", 0.0, b, sd, min(v) * 0.9, family="E1" if sd == 0 else "E5"))
    for m, lines in by.items():
        seen = {}
        for r in lines:
            seen[r["state_id"]] = r
        _write(root, "E1", m, [r for r in seen.values() if r["seed"] == 0])
        _write(root, "E5", m, [r for r in seen.values() if r["seed"] > 0])
    return Store(Paths(root=root))


def test_seeds_both_bitwidths_every_model(tmp_path):
    store = _seed_repo(str(tmp_path))
    stab = questions.seed_stability(store)
    pairs, best = questions.seed_pairs(store)
    e1seeds = E1.seed_summary(store)
    ov = {(r["model"], r["bits"]): r for r in questions.seed_overview(store, stab, pairs, best, e1seeds)}
    s3 = [r for r in stab if r["model"] == "SmolLM2-360M" and r["cell"] == "25%+3"][0]
    assert s3["complete"] and not s3["stable"] and not s3["qp_worst_all_seeds"]
    assert s3["best_by_seed"] == "0:joint 1:P->Q 2:P->Q" and s3["worst_by_seed"] == "0:Q->P 1:Q->P 2:joint"
    q4 = [r for r in stab if r["model"] == "Qwen2.5-0.5B" and r["cell"] == "25%+4"][0]
    assert not q4["complete"]                                       # seed 2 missing: never counted
    r = ov[("SmolLM2-360M", 3)]
    assert (r["cells"], r["best_stable"], r["qp_worst_all_seeds"], r["runs"], r["runs_qp_worst"]) == (1, 0, 0, 3, 2)
    assert r["runs_joint_best"] == 1 and r["runs_joint_beats_pq"] == 1
    r = ov[("SmolLM2-360M", 4)]
    assert (r["cells"], r["best_stable"], r["qp_worst_all_seeds"], r["pair_joint_p_then_q_a_all_seeds"]) == (1, 1, 1, 1)
    assert ov[("Qwen2.5-0.5B", 4)]["cells"] == 0 and ov[("Qwen2.5-0.5B", 3)]["cells"] == 1
    assert ov[("Qwen2.5-0.5B", 3)]["runs_qp_worst"] == 1               # only seed 2 has Q->P worst
    assert ov[("all", 4)]["cells"] == 1 and ov[("all", 3)]["cells"] == 2 and ov[("all", "both")]["cells"] == 3
    # sd over seeds: the largest 100 sd/mean of a combined configuration with three seeds
    want = max(100 * st.stdev(v) / st.mean(v) for v in
               ([SEED_PPL[("SmolLM2-360M", (0.25, 3), sd)][k] for sd in range(3)] for k in range(3)))
    assert ov[("SmolLM2-360M", 3)]["sd_max_combined_pct"] == pytest.approx(want)
    # mean and sd per order and cell (CSV rows)
    rows = {(r["model"], r["cell"], r["order"]): r for r in questions.seed_ppl_orders(store)}
    v = [SEED_PPL[("Qwen2.5-0.5B", (0.25, 3), sd)][2] for sd in range(3)]
    q = rows[("Qwen2.5-0.5B", "25%+3", "q_then_p")]
    assert q["wt_ppl_mean"] == pytest.approx(st.mean(v)) and q["wt_ppl_sd"] == pytest.approx(st.stdev(v)) and q["complete"]
    assert rows[("Qwen2.5-0.5B", "25%+4", "joint")]["wt_ppl_n"] == 2 and not rows[("Qwen2.5-0.5B", "25%+4", "joint")]["complete"]
    # macros: legacy names = SmolLM2-360M 4-bit; new names per scope and bit-width
    M = latex.Macros()
    qres = {"seed_stability": stab, "seed_pairs": pairs, "seed_best": best,
            "seed_overview": questions.seed_overview(store, stab, pairs, best, e1seeds)}
    latex.macros_e5(M, store, qres)
    vals = {r["name"]: r["value"] for r in M.rows}
    assert vals["resSeedPairJointPQAllSeeds"] == vals["resSeedPairJointPQAllSeedsSmolMediumBFour"] == "1"
    assert vals["resSeedBestStableSmolMediumBThree"] == "0" and vals["resSeedRunsQPWorstSmolMediumBThreePct"] == "67"
    assert vals["resSeedBestCellsQwenBFour"] == "--" and vals["resSeedBestCellsAllBoth"] == "3"
    assert vals["resSeedQPWorstAllSeedsAllBFour"] == "1"
    planned = sum(1 for x in store.plan.values() if x["short"] == "Qwen2.5-0.5B" and x["seed"] > 0 and x["bits"] == 4)
    assert vals["resSeedStatesQwenBFour"] == "4" and vals["resSeedStatesPlannedQwenBFour"] == str(planned)


# ----------------------------------------------------------------------------- family vs scale
SCALE = {  # model: (compressed weights, {cell: (joint, P->Q, Q->P)})
    "SmolLM2-135M": (106e6, {(0.25, 4): (23.5, 22.0, 24.4), (0.5, 4): (44.5, 41.2, 47.3)}),
    "SmolLM2-360M": (315e6, {(0.25, 4): (13.2, 13.3, 14.4), (0.5, 4): (20.5, 21.3, 23.6)}),
    "SmolLM2-1.7B": (1611e6, {(0.25, 4): (9.00, 9.06, 9.19), (0.5, 4): (14.1, 14.1, 15.0)}),
    "Qwen2.5-0.5B": (358e6, {(0.25, 4): (14.6, 14.8, 15.1), (0.5, 4): (21.9, 22.2, 22.4)}),
    "Qwen2.5-1.5B": (1310e6, {(0.25, 4): (10.0, 10.1, 10.2)}),            # 50%+4 not there yet
}


def _scale_repo(root):
    for m, (cw, cells) in SCALE.items():
        fam = "E1" if m in ("SmolLM2-135M", "SmolLM2-360M", "Qwen2.5-0.5B") else "E6"
        lines = [_line(m, "dense", 0.0, 16, 0, 10.0, fam, cw)]
        for (s, b), v in cells.items():
            lines += [_line(m, o, s, b, 0, p, fam, cw) for o, p in zip(("joint", "p_then_q", "q_then_p"), v)]
        _write(root, fam, m, lines)
    return Store(Paths(root=root))


def test_family_scale_fits_never_mix_families(tmp_path):
    store = _scale_repo(str(tmp_path))
    q2 = questions.q2(store, {"by_sid": {}}, Missing())
    cells = questions.scale_cells(store, q2)
    fits = {(r["family"], r["cell"]): r for r in questions.scale_fit(cells)}
    smol = [(SCALE[m][0], SCALE[m][1][(0.25, 4)]) for m in ("SmolLM2-135M", "SmolLM2-360M", "SmolLM2-1.7B")]
    spread = [100 * (max(v) - min(v)) / min(v) for _, v in smol]
    pen = [100 * (v[2] / min(v) - 1) for _, v in smol]
    want = np.polyfit([math.log10(n) for n, _ in smol], spread, 1)[0]
    r = fits[("SmolLM2", "25%+4")]
    assert r["n_models"] == 3 and r["slope_spread_per_decade"] == pytest.approx(want)
    assert r["slope_qp_penalty_per_decade"] == pytest.approx(np.polyfit([math.log10(n) for n, _ in smol], pen, 1)[0])
    assert r["spread_shrinks_monotonically"] is True
    q = fits[("Qwen2.5", "25%+4")]                                       # only the two Qwen models, never SmolLM2
    assert q["models"] == "Qwen2.5-0.5B Qwen2.5-1.5B" and q["n_models"] == 2
    assert fits[("Qwen2.5", "50%+4")]["n_models"] == 1 and "slope_spread_per_decade" not in fits[("Qwen2.5", "50%+4")]
    c = {(x["model"], x["cell"]): x for x in cells}
    x = c[("SmolLM2-360M", "50%+4")]
    assert x["qp_penalty_pct"] == pytest.approx(100 * (23.6 / 20.5 - 1)) and x["best"] == "joint"
    assert x["gap_mask_first_pct"] == pytest.approx(100 * 0.8 / 20.5)
    mt = {(x["group"], x["cell"]): x for x in questions.matched_scale(cells)}
    a = mt[("Small", "25%+4")]
    s360, s05 = c[("SmolLM2-360M", "25%+4")]["spread_pct"], c[("Qwen2.5-0.5B", "25%+4")]["spread_pct"]
    assert a["complete"] and a["spread_gap_pp"] == pytest.approx(abs(s360 - s05)) and a["largest_spread"] == "SmolLM2-360M"
    assert a["weights_ratio"] == pytest.approx(358 / 315)
    assert not mt[("Billion", "25%+4")]["complete"] and mt[("Billion", "25%+4")]["n_models"] == 2   # OLMo-2 not there
    M = latex.Macros()
    latex.macros_e6(M, store, {"scale_cells": cells, "scale": list(fits.values()), "scale_matched": list(mt.values())})
    vals = {r["name"]: r["value"] for r in M.rows}
    assert vals["resScaleSpreadSmolLargeSTwentyFiveBFourPct"] == f"{spread[2]:.1f}"
    assert vals["resScaleSpreadQwenMidSFiftyBFourPct"] == "--"               # planned, not there
    assert vals["resScaleMatchedSmallSTwentyFiveBFourGapPct"] == f"{abs(s360 - s05):.1f}"
    assert vals["resScaleMatchedBillionSTwentyFiveBFourGapPct"] == "--" and vals["resScaleMatchedBillionSTwentyFiveBFourModels"] == "2"
    assert vals["resScaleSlopeSmolSTwentyFiveBFour"] == "\\ensuremath{" + f"{want:+.1f}" + "}"
    assert vals["resNWeightsSmolSmallM"] == "106" and vals["resNWeightsQwenMidB"] == "1.31" and vals["resNWeightsOlmoM"] == "--"
    assert not any(n.startswith("resScaleSlopeOlmo") for n in vals)          # one OLMo-2 model: no fit
    rows = questions.fig_scale_rows(cells)
    assert {r["cell"] for r in rows} == {"25%+4", "50%+4"} and len(rows) == 9


def test_q3_grid_range_leaves_out_sixty_percent(tmp_path):
    root = str(tmp_path)
    v = {(0.25, 4): (10.0, 10.5, 11.0), (1 / 3, 4): (12.0, 12.2, 13.0), (0.5, 4): (20.0, 21.0, 22.0), (0.6, 4): (40.0, 45.0, 60.0)}
    lines = []
    for (s, b), p in v.items():
        lines += [_line("SmolLM2-135M", o, s, b, 0, x, "E1" if s != 0.6 else "E4") for o, x in zip(("joint", "p_then_q", "q_then_p"), p)]
    _write(root, "E1", "SmolLM2-135M", [r for r in lines if r["family"] == "E1"])
    _write(root, "E4", "SmolLM2-135M", [r for r in lines if r["family"] == "E4"])
    store = Store(Paths(root=root))
    q2 = questions.q2(store, {"by_sid": {}}, Missing())
    r = [x for x in questions.q3(q2) if x["model"] == "SmolLM2-135M" and x["bits"] == 4 and x["metric"] == "wt_ppl"][0]
    spreads = [100 * (max(p) - min(p)) / min(p) for (s, b), p in v.items() if s != 0.6]
    assert r["grid_complete"] and r["grid_spread_min_pct"] == pytest.approx(min(spreads))
    assert r["grid_spread_max_pct"] == pytest.approx(max(spreads)) and r["spread_max_pct"] == pytest.approx(50.0)
    M = latex.Macros()
    q = {"q1_summary": questions.q1_summary([]), "q2": q2, "q2_summary": questions.q2_summary(q2), "q3": questions.q3(q2),
         "q2_per_model": questions.q2_per_model(q2), "fp_summary": {"n_compared": 0}, "e3": []}
    latex.macros_questions(M, store, q)
    vals = {x["name"]: x["value"] for x in M.rows}
    assert vals["resQThreeSpreadMaxSmolSmallBFourPct"] == f"{max(spreads):.1f}" and vals["resQThreeSpreadMaxSmolSmallPct"] == "50.0"
    assert vals["resQThreeSpreadMinSmolMediumBFourPct"] == "--" and vals["resQThreeSpreadMinSmolSmallBThreePct"] == "--"
    assert vals["resQTwoCellsPplSmolSmall"] == "3" and vals["resQTwoQPWorstPplSmolSmall"] == "3"


# ----------------------------------------------------------------------------- BATCH check
def _flip_first(items, task, metric, k):
    out = json.loads(json.dumps(items))
    out[task][metric] = [1.0 - x if i < k else x for i, x in enumerate(out[task][metric])]
    return out


def test_batch_check_items_nll_and_flips(tmp_path):
    root = str(tmp_path)
    lines = make_fake_repo(root)
    dense = [r for r in lines if r["order"] == "dense"][0]
    v2 = os.path.join(root, "results", "v2")
    with gzip.open(os.path.join(v2, "items", f"{dense['state_id']}.json.gz"), "rt") as f:
        items = json.load(f)
    k = 5
    new_items = _flip_first(items, "arc_easy", "acc_norm", k)            # arc_easy's v2 metric is acc_norm
    name = f"{dense['state_id']}__BATCH.json.gz"
    with gzip.open(os.path.join(v2, "items", name), "wt") as f:
        json.dump(new_items, f)
    with open(os.path.join(v2, "items", name), "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    b = dict(dense, family="BATCH", items_file=f"results/v2/items/{name}", items_sha256=sha)
    os.makedirs(os.path.join(v2, "BATCH"))
    with open(os.path.join(v2, "BATCH", "SmolLM2-135M.jsonl"), "w") as f:
        f.write(json.dumps(b) + "\n")
    paths = Paths(root=root)
    store = Store(paths)
    assert dense["state_id"] in store.states and len(store.states) == 8      # BATCH is never a study state
    res = batch.analyze(store, paths, Missing())
    d = [r for r in res["states"] if r["state_id"] == dense["state_id"]][0]
    n = sum(len(v["doc_id"]) for v in items.values())
    assert d["wt_nll_identical"] is True and d["items_differ"] == k and d["items_n"] == n
    assert sum(r["available"] for r in res["states"]) == 1 and len(res["states"]) == 4
    dbin = {t: {i: int(round(x)) for i, x in zip(v["doc_id"], v["acc_norm" if t in P.ACC_NORM_TASKS else "acc"])}
            for t, v in new_items.items()}
    for r in res["flips"]:
        want = S.flip_counts(store.binary(r["state_id"]), dbin, P.ZERO_SHOT_TASKS)["pooled"]["flips"]
        assert r["flips_batch_dense"] == pytest.approx(want)
        assert r["change_pp"] == pytest.approx(100 * (want - r["flips_e1_dense"]))
    assert len(res["flips"]) == 7 and dense["state_id"] not in {r["state_id"] for r in res["flips"]}
    sm = res["summary"]
    assert sm["items_differ"] == k and sm["task_arc_easy_differ"] == k and sm["ranking_cells"] == 1
    assert sm["flips_max_change_pp"] == pytest.approx(max(abs(r["change_pp"]) for r in res["flips"]))
    M = latex.Macros()
    latex.macros_batch(M, store, res)
    vals = {r["name"]: r["value"] for r in M.rows}
    assert vals["resBatchItemsDiffer"] == str(k) and vals["resBatchItemsTotal"] == str(n) and vals["resBatchStates"] == "1"
    assert vals["resBatchNllIdenticalSmolSmallDense"] == "yes" and vals["resBatchItemsDifferQwenDense"] == "--"
    assert vals["resBatchFlipsMaxChangePct"] == f"{sm['flips_max_change_pp']:.2f}" and vals["resBatchStatesPlanned"] == "4"


# ----------------------------------------------------------------------------- I_Sr (Table 2), aliases, E7, E8
PUB = ('PUB = {"Mistral-7B-v0.3": dict(dense=(30.0, 60.0, 10.0), p25=(29.0, 57.0, 9.0), p33=(28.0, 54.0, 8.0),'
       ' q4=(28.5, 56.0, 9.5), q3=(23.0, 43.0, 6.0), j25_4=(27.0, 53.0, 8.0), j33_3=(20.0, 40.0, 5.0))}\n')


def test_isr_table2_aliases_e7_e8_on_fake_results(tmp_path):
    root = str(tmp_path)
    lines = make_fake_repo(root)
    with open(os.path.join(root, "analysis", "reproduce_paper.py"), "a") as f:
        f.write(PUB)
    from scripts.make_tables import run
    out = run(paths=Paths(root=root), B=200, figures=False, fetch=False, log=lambda *a: None)
    vals = _vals(root)
    pq = [r for r in lines if r["order"] == "p_then_q"][0]
    v = out["e1"]["by_sid"][pq["state_id"]]["isr"]["sr3_paper"]
    for sfx, i in (("", 0), ("Lo", 1), ("Hi", 2)):
        assert vals[f"resISrSmolSmallSTwentyFiveBFourPQ{sfx}"] == "\\ensuremath{" + f"{v[i]:+.3f}" + "}"
    assert vals["resISrSmolSmallSTwentyFiveBFourPQ"] == vals["resISrPaperSmolSmallSTwentyFiveBFourPQ"]
    assert vals["resISrSmolMediumSTwentyFiveBFourPQ"] == "--"
    d, p, q, j = (30.0, 60.0, 10.0), (29.0, 57.0, 9.0), (28.5, 56.0, 9.5), (27.0, 53.0, 8.0)
    want = sum(j) / sum(d) - (sum(p) / sum(d)) * (sum(q) / sum(d))
    assert vals["resIjcnnISrMistralSTwentyFiveBFourAll"] == "\\ensuremath{" + f"{want:+.3f}" + "}"
    w2 = sum(j[:2]) / sum(d[:2]) - (sum(p[:2]) / sum(d[:2])) * (sum(q[:2]) / sum(d[:2]))
    assert vals["resIjcnnISrMistralSTwentyFiveBFourNoMath"] == "\\ensuremath{" + f"{w2:+.3f}" + "}"
    # new model tags, and the derived tags kept as aliases with the same value
    assert vals["resPplQwenMidDense"] == vals["resPplQwenTwoFiveOneFiveBDense"] == "--"
    assert vals["resQThreeSpreadMinOLMoTwoZeroFourTwoFiveOneBBFourPct"] == vals["resQThreeSpreadMinOlmoBFourPct"]
    assert "resPplQwenLargeDense" in vals and "resPplQwenTwoFiveThreeBDense" in vals
    assert not any("QwenTwoFive" in n or "OLMoTwo" in n for n in vals if vals[n].startswith("\\res"))   # copies, not references
    assert out["tex"]["aliases"] == sum(1 for n in vals if "QwenTwoFiveOneFiveB" in n or "QwenTwoFiveThreeB" in n
                                         or "OLMoTwoZeroFourTwoFiveOneB" in n)
    assert vals["resPplSmolSmallDense"] == f"{lines[0]['wikitext2']['ppl']:.2f}"          # existing tags unchanged
    # E7 with the paper's Sr, E8 joint smallest
    assert vals["resESevenKimCellsSrPaperSmolSmall"] == "1" and vals["resESevenKimCellsSrPaperAll"] == "1"
    err = {r["order"]: r["mean_rel_err"] for r in lines if r["sparsity"] == 0.25 and r["bits"] == 4}
    joint_small = err["joint"] < min(err["p_then_q"], err["q_then_p"])
    assert vals["resEEightCellsSmolSmallBFour"] == "1" and vals["resEEightJointSmallestSmolSmallBFour"] == str(int(joint_small))
    assert vals["resEEightJointSmallestBThree"] == "--"
    with open(os.path.join(root, "paper", "generated", "MACROS.md")) as f:
        md = f.read()
    from test_analysis import documented_names
    assert documented_names(md) == set(vals)


def test_isr_definition_reproduces_v1_ratio_of_sums(tmp_path):
    """Table 2's Sr is a ratio of sums: isr.v1_check computes it (and the mean of per-task ratios) from v1 lines."""
    root = str(tmp_path)
    res = os.path.join(root, "results")
    os.makedirs(res)
    main = {"dense": (0.60, 0.70, 0.40), "prune": (0.58, 0.69, 0.39), "quant": (0.55, 0.66, 0.36), "p_then_q": (0.56, 0.67, 0.38)}
    with open(os.path.join(res, "results_full.jsonl"), "w") as f:
        for o, (s, b) in (("dense", (0.0, 16)), ("prune", (0.25, 16)), ("quant", (0.0, 4)), ("p_then_q", (0.25, 4))):
            f.write(json.dumps({"model": "HuggingFaceTB/SmolLM2-135M", "order": o, "sparsity": s, "bits": b, "seed": 0, "ppl": 20.0,
                                "main": dict(zip(P.PAPER_TASKS, main[o]))}) + "\n")
    open(os.path.join(res, "results_fix.jsonl"), "w").close()
    os.makedirs(os.path.join(root, "analysis"))
    with open(os.path.join(root, "analysis", "reproduce_paper.py"), "w") as f:
        f.write("AFFECTED = []\n")
    rows = isr.v1_check(Paths(root=root))
    r = [x for x in rows if x["model"] == "SmolLM2-135M" and x["cell"] == "25%+4"][0]
    D = sum(main["dense"])
    want = sum(main["p_then_q"]) / D - (sum(main["prune"]) / D) * (sum(main["quant"]) / D)
    ratio = [lambda k, i=i: main[k][i] / main["dense"][i] for i in range(3)]
    mor = {k: sum(f(k) for f in ratio) / 3 for k in ("p_then_q", "prune", "quant")}
    assert r["isr_ratio_of_sums"] == pytest.approx(want)
    assert r["isr_mean_of_ratios"] == pytest.approx(mor["p_then_q"] - mor["prune"] * mor["quant"])
    assert [x for x in rows if x["cell"] == "33%+3"][0].get("isr_ratio_of_sums") is None   # no 33%+3 lines


# ----------------------------------------------------------------------------- paper figures
def _pdf_info(path):
    with open(path, "rb") as f:
        blob = f.read()
    box = re.search(rb"/MediaBox\s*\[\s*0 0 ([0-9.]+) ([0-9.]+)\s*\]", blob)
    return float(box.group(1)), float(box.group(2)), blob


def test_paper_figures_size_fonts_and_partial_data(tmp_path):
    paths = Paths(root=str(tmp_path))
    rows = [{"cell": c, "family": f, "model": m, "size": m.split("-")[-1], "compressed_weights": n, "spread_pct": y}
            for c, f, m, n, y in (("25%+4", "SmolLM2", "SmolLM2-135M", 106e6, 10.7), ("25%+4", "SmolLM2", "SmolLM2-360M", 315e6, 8.9),
                                  ("25%+4", "Qwen2.5", "Qwen2.5-0.5B", 358e6, 3.0), ("25%+4", "OLMo-2", "OLMo-2-0425-1B", 1074e6, 2.4),
                                  ("25%+4", "Qwen2.5", "Qwen2.5-1.5B", 1310e6, 2.3), ("25%+4", "SmolLM2", "SmolLM2-1.7B", 1611e6, 2.1),
                                  ("50%+4", "SmolLM2", "SmolLM2-135M", 106e6, 14.9))]   # no 50%+3 point: that panel says pending
    files, cap = paper_figures.fig_order_scale(rows, paths)
    w, h, blob = _pdf_info(files[0])
    assert files[0].endswith(os.path.join("paper", "generated", "fig_order_scale.pdf")) and os.path.exists(files[1])
    assert w == pytest.approx(12.2 / 2.54 * 72, abs=0.5) and h == pytest.approx(4.0 / 2.54 * 72, abs=0.5)
    assert b"/FontFile2" in blob and b"/Type3" not in blob                  # TrueType embedded, no Type 3 fonts
    assert "7 points" in cap
    q1 = [{"model": m, "seed": 0, "cell": c, "order": o, "ratio_mean_pct": -5.0 + i, "n_layers": 14}
          for m in ("SmolLM2-135M", "SmolLM2-360M") for i, c in enumerate(("25%+4", "50%+3")) for o in ("joint", "p_then_q", "q_then_p")]
    e2 = [{"model": "SmolLM2-135M", "cell": "25%+4", "n_layers": 7, "ratio_joint_pct": -4.0, "ratio_p_then_q_pct": -3.0,
           "ratio_q_then_p_pct": 2.0}]
    models = ["SmolLM2-135M", "SmolLM2-360M", "Qwen2.5-0.5B"]                 # Qwen2.5-0.5B: no data, a pending panel
    data = paper_figures.fig1_rows(q1, e2, models)
    assert len(data) == 3 * 6 * 3 and sum(r["e2_ratio_pct"] is not None for r in data) == 3
    files, cap = paper_figures.fig1_interaction(data, models, paths)
    w, h, blob = _pdf_info(files[0])
    assert w == pytest.approx(12.2 / 2.54 * 72, abs=0.5) and h == pytest.approx(0.30 * 12.2 / 2.54 * 72, abs=0.5)
    assert "bars (12)" in cap and "(E2, 3)" in cap
