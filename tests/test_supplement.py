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
