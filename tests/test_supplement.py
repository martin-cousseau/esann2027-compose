"""The supplement is built from the analysis tables and agrees with the paper's macros."""
import csv
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_supplement_matches_macros(tmp_path):
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_supplement.py"), "--out", str(tmp_path), "--no-pdf"], check=True,
                   stdout=subprocess.DEVNULL)
    R = list(csv.DictReader(open(tmp_path / "complete_results.csv")))
    states = {(r["model"], r["state_id"]) for r in csv.DictReader(open(os.path.join(ROOT, "reports", "analysis", "e1_states.csv")))
              if float(r["seed"] or 0) == 0 and r["order"] in ("dense", "quant", "prune", "joint", "p_then_q", "q_then_p")}
    assert len(R) == len(states)
    mac = dict(re.findall(r"\\newcommand\{\\(res[A-Za-z]+)\}\{([^}]*)\}", open(os.path.join(ROOT, "paper", "generated", "macros.tex")).read()))
    num = lambda v: float(v.replace("\\ensuremath{", "").replace("}", ""))  # noqa: E731
    pick = lambda m, o, s, b: [r for r in R if r["model"] == m and r["order"] == o and float(r["sparsity_pct"]) == s and int(r["bits"]) == b][0]  # noqa: E731
    assert round(float(pick("SmolLM2-135M", "dense", 0.0, 16)["wikitext2_ppl"]), 2) == num(mac["resPplSmolSmallDense"])
    assert round(float(pick("SmolLM2-360M", "quant", 0.0, 3)["wikitext2_ppl"]), 2) == num(mac["resPplSmolMediumQuantBThree"])
    assert round(float(pick("SmolLM2-135M", "p_then_q", 25.0, 4)["I_Sr3"]), 3) == num(mac["resISrPaperSmolSmallSTwentyFiveBFourPQ"])
    T = list(csv.DictReader(open(tmp_path / "task_scores.csv")))
    t = [x for x in T if x["model"] == "SmolLM2-135M" and x["order"] == "p_then_q" and x["task"] == "piqa"][0]
    assert abs(float(t["retention"]) - float(t["score"]) / float(t["dense_score"])) < 1e-12


def test_layer_tables_match_macros(tmp_path):
    """The per-setting tables of the layer-level controls and of the weaker-first test reproduce the paper's macros."""
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "make_supplement.py"), "--out", str(tmp_path), "--no-pdf"], check=True,
                   stdout=subprocess.DEVNULL)
    sys.path.insert(0, ROOT)
    from scripts.analysis.latex import MODEL_TAG
    mac = dict(re.findall(r"\\newcommand\{\\(res[A-Za-z]+)\}\{([^}]*)\}", open(os.path.join(ROOT, "paper", "generated", "macros.tex")).read()))
    num = lambda v: float(v.replace("\\ensuremath{", "").replace("}", ""))  # noqa: E731
    S, B = {"25.0": "TwentyFive", "33.3": "ThirtyThree", "50.0": "Fifty"}, {"4": "Four", "3": "Three"}
    A = list(csv.DictReader(open(tmp_path / "layer_ablation.csv")))
    assert len(A) == int(mac["resETwoSplitCellsAllModels"])
    for r in A:   # every part of the split, per model and setting
        ct = MODEL_TAG[r["model"]] + "S" + S[r["sparsity_pct"]] + "B" + B[r["bits"]]
        for k in ("gap", "mask", "order", "grid"):
            assert abs(round(float(r[f"{k}_pct"]), 1) - num(mac[f"resETwoSplit{k.capitalize()}{ct}Pct"])) < 1e-9, (ct, k)
        assert abs(float(r["gap_pct"]) - float(r["mask_pct"]) - float(r["order_pct"]) - float(r["grid_pct"])) < 1e-9
    assert sum(float(r["I_E_joint_pct"]) < 0 for r in A) + sum(float(r["I_E_p_then_q_pct"]) < 0 for r in A) == int(mac["resETwoMaskFirstSubAllModels"])
    assert sum(float(r["gap_pct"]) > 0 for r in A) == int(mac["resETwoSplitGapPosAllModels"])
    N = list(csv.DictReader(open(tmp_path / "no_compensation.csv")))
    assert len(N) == int(mac["resENineCellsAllModels"])
    assert sum(float(r["I_E_mask_first_pct"]) < 0 for r in N) == int(mac["resENineSubPQStaleAllModels"])
    assert sum(float(r["I_E_q_then_p_pct"]) < 0 for r in N) == int(mac["resENineSubQPAllModels"])
    assert round(max(abs(float(r["cross_pct"])) for r in N), 2) == num(mac["resENineCrossMaxAbsAllModelsPct"])
    for r in N:   # absorption and the cross terms sum to the mask-first interaction
        assert abs(float(r["absorption_pct"]) + float(r["cross_pct"]) - float(r["I_E_mask_first_pct"])) < 1e-4   # float32 errors against the float64 terms
    W = list(csv.DictReader(open(tmp_path / "weaker_first.csv")))
    assert len(W) == int(mac["resQTwoCellsPplAllModels"])
    K = [r for r in W if r["rule_recommends"] == "q_then_p"]
    assert len(K) == int(mac["resESevenKimPredQPCellsPplAllModels"])
    assert sum(r["worst_ppl"] == "q_then_p" for r in K) == int(mac["resESevenKimPredQPWorstPplAllModels"])
    assert sum(r["best_ppl"] == "q_then_p" for r in K) == int(mac["resESevenKimPredQPBestPplAllModels"])
    assert sum(r["worst_ppl"] == "q_then_p" for r in W) == int(mac["resQTwoQPWorstPplAllModels"])
    assert sum(r["best_ppl"] == "q_then_p" for r in W) == int(mac["resQTwoQPBestPplAllModels"])
    assert min(float(r["sparsity_pct"]) for r in K) >= num(mac["resESevenKimPredQPMinSparsityPctAllModels"]) - 0.5
    Sd = list(csv.DictReader(open(tmp_path / "seeds.csv")))
    runs = {}
    for r in Sd:   # Q2's seed counts: runs with the three orders, Q->P worst / best
        if r["order"] in ("joint", "p_then_q", "q_then_p"):
            for k in (0, 1, 2):
                if r[f"ppl_seed{k}"]:
                    runs.setdefault((r["model"], r["sparsity_pct"], r["bits"], k), {})[r["order"]] = float(r[f"ppl_seed{k}"])
    full = {k: v for k, v in runs.items() if len(v) == 3}
    assert len(full) == int(mac["resSeedRunsAllBoth"])
    for bits, tag in (("4", "BFour"), ("3", "BThree")):
        sub = [v for k, v in full.items() if k[2] == bits]
        assert len(sub) == int(mac[f"resSeedRuns{'All'}{tag}"])
        assert sum(v["q_then_p"] > max(v["joint"], v["p_then_q"]) for v in sub) == int(mac[f"resSeedRunsQPWorstAll{tag}"])
    assert sum(v["q_then_p"] < min(v["joint"], v["p_then_q"]) for v in full.values()) == int(mac["resSeedRunsQPBestAllBoth"])
