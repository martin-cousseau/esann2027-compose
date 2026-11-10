"""Tests of the v2 analysis (CPU, no network): python -m pytest -q tests/test_analysis.py

All data here are synthetic and live in pytest's temporary directories; nothing touches results/ or the
real output folders.
"""
import gzip
import hashlib
import json
import math
import os
import random
import re
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from scripts.analysis import stats as S  # noqa: E402
from scripts.analysis.common import P, Paths, Store, aligned_columns  # noqa: E402
from scripts.run_grid import state_id  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-135M"
NAMES = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj",
         "mlp.down_proj"]


# ----------------------------------------------------------------------------- fake result tree
def _items(rng, n, rate):
    rec = {"doc_id": list(range(n))}
    rec["acc"] = [float(rng.random() < rate) for _ in range(n)]
    rec["acc_norm"] = [float(rng.random() < rate) for _ in range(n)]
    return rec


def _diags(rng, base, blocks=2):
    out = []
    for layer in range(blocks):
        for name in NAMES:
            out.append({"layer": layer, "name": name, "shape": [64, 64], "mask_frac": 0.0, "zero_frac": 0.1,
                        "mask_kept": True, "max_levels": 8, "rel_err": base * (1 + rng.random())})
    return out


def make_fake_repo(root, n_items=60, seed=0):
    """A tiny results tree: SmolLM2-135M dense, 25% pruning, 4- and 3-bit, the three orders at 25%+4 and
    Q->P at 25%+3 (identical to 3-bit), items for every zero-shot task, a DET line, E2 lines, v1 files."""
    rng = random.Random(seed)
    v2 = os.path.join(root, "results", "v2")
    os.makedirs(os.path.join(v2, "E1"), exist_ok=True)
    os.makedirs(os.path.join(v2, "items"), exist_ok=True)
    states = [("dense", 0.0, 16, 15.0, 0.0, 0.62), ("prune", 0.25, 16, 16.5, 0.05, 0.60), ("quant", 0.0, 4, 21.0, 0.15, 0.57),
              ("quant", 0.0, 3, 58.0, 0.6, 0.45), ("joint", 0.25, 4, 23.0, 0.2, 0.55), ("p_then_q", 0.25, 4, 21.9, 0.18, 0.56),
              ("q_then_p", 0.25, 4, 24.0, 0.22, 0.54)]
    lines, q3 = [], None
    for order, s, b, ppl, kl, rate in states:
        diags = [] if order == "dense" else _diags(rng, {"prune": 0.01, "quant": 0.02}.get(order, 0.025))
        lines.append(_line(rng, v2, order, s, b, ppl, kl, rate, diags, n_items))
        if (order, b) == ("quant", 3):
            q3 = lines[-1]
    qp = json.loads(json.dumps(q3))  # Q->P at 25%+3: the 3-bit model itself
    sid = state_id(MODEL, {"order": "q_then_p", "sparsity": 0.25, "bits": 3, "seed": 0}, "float32")
    qp.update(state_id=sid, order="q_then_p", sparsity=0.25, items_file=f"results/v2/items/{sid}.json.gz")
    with gzip.open(os.path.join(v2, "items", f"{q3['state_id']}.json.gz"), "rb") as f:
        blob = f.read()
    _write_items(v2, sid, json.loads(blob))
    qp["items_sha256"] = _sha(os.path.join(v2, "items", f"{sid}.json.gz"))
    lines.append(qp)
    with open(os.path.join(v2, "E1", "SmolLM2-135M.jsonl"), "w") as f:
        for r in lines:
            f.write(json.dumps(r) + "\n")
        f.write('{"state_id": "cut line')  # a line cut by a kill must be ignored
    with open(os.path.join(v2, "E1", "failures__SmolLM2-135M.jsonl"), "w") as f:
        f.write(json.dumps({"state_id": state_id(MODEL, {"order": "joint", "sparsity": 0.5, "bits": 4, "seed": 0}, "float32"),
                            "error": "OutOfMemoryError: fake", "time": "2026-11-10T00:00:00+00:00"}) + "\n")
    joint = [r for r in lines if r["order"] == "joint"][0]
    os.makedirs(os.path.join(v2, "DET"), exist_ok=True)
    det = dict(joint, family="DET", tasks={}, items_file=None, items_sha256=None, c4=None)
    with open(os.path.join(v2, "DET", "SmolLM2-135M.jsonl"), "w") as f:
        f.write(json.dumps(det) + "\n")
    _fake_e2(rng, v2)
    _fake_v1(root, lines)
    return lines


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _write_items(v2, sid, items):
    with gzip.open(os.path.join(v2, "items", f"{sid}.json.gz"), "wt") as f:
        json.dump(items, f)


def _line(rng, v2, order, s, b, ppl, kl, rate, diags, n_items):
    sid = state_id(MODEL, {"order": order, "sparsity": s, "bits": b, "seed": 0}, "float32")
    items = {t: _items(rng, n_items, rate) for t in P.ZERO_SHOT_TASKS}
    _write_items(v2, sid, items)
    agg = {t: {"acc,none": sum(v["acc"]) / n_items, "acc_norm,none": sum(v["acc_norm"]) / n_items} for t, v in items.items()}
    return {"state_id": sid, "family": "E1", "model": MODEL, "order": order, "sparsity": s, "bits": b, "seed": 0,
            "dtype": "float32", "storage": P.storage(s, b, diags, 2.5 if b < 16 else None),
            "wikitext2": {"ppl": ppl, "nll_sum": 1000 * math.log(ppl), "kl": kl, "top1": 1 - kl / 2},
            "c4": {"ppl": 1.3 * ppl, "nll_sum": 1300 * math.log(ppl), "kl": 1.1 * kl, "top1": 1 - kl / 1.8},
            "tasks": agg, "main": {t: agg[t][f"{P.main_metric(t)},none"] for t in agg},
            "items_file": f"results/v2/items/{sid}.json.gz", "items_sha256": _sha(os.path.join(v2, "items", f"{sid}.json.gz")),
            "mean_rel_err": (sum(d["rel_err"] for d in diags) / len(diags)) if diags else 0.0, "diags": diags,
            "status": "ok", "calib_sha": "x", "wt_windows_sha": "y", "metadata": {"gpu": "fake"}}


def _fake_e2(rng, v2):
    os.makedirs(os.path.join(v2, "E2"), exist_ok=True)
    with open(os.path.join(v2, "E2", "SmolLM2-135M.jsonl"), "w") as f:
        for block in (0, 2):
            for name in NAMES:
                ep, eq = 0.001 * (1 + rng.random()), 0.002 * (1 + rng.random())
                cell = {"joint": 0.8 * (ep + eq), "p_then_q": 0.9 * (ep + eq), "q_then_p": 1.1 * (ep + eq),
                        "q_then_p_refit": 0.95 * (ep + eq), "q_then_p_pmask": 1.0 * (ep + eq),
                        "q_then_p_pmask_refit": 0.9 * (ep + eq), "p_then_q_stale": 1.05 * (ep + eq), "zeros_q_then_p": 0.3,
                        "mask_quality_q_then_p": 2 + rng.random(), "mask_jaccard_q_then_p_vs_P": 0.5 + 0.1 * rng.random()}
                f.write(json.dumps({"model": MODEL, "block": block, "name": name, "shape": [64, 64], "E_P": {"0.2500": ep},
                                    "E_Q": {"4": eq}, "zeros_Q": {"4": 0.2}, "cells": {"0.2500/4": cell},
                                    "h_corr_index": rng.random(), "h_offdiag_mean_abs_corr": rng.random(),
                                    "h_dead_columns": 0}) + "\n")


def _fake_v1(root, lines):
    res = os.path.join(root, "results")
    with open(os.path.join(res, "results_full.jsonl"), "w") as f:
        for r in lines:
            f.write(json.dumps({"model": MODEL, "order": r["order"], "sparsity": r["sparsity"], "bits": r["bits"], "seed": 0,
                                "ppl": r["wikitext2"]["ppl"] * 1.001, "diags": r["diags"], "main": {}}) + "\n")
    with open(os.path.join(res, "results_fix.jsonl"), "w") as f:
        f.write(json.dumps({"model": MODEL, "order": "joint", "sparsity": 0.25, "bits": 4, "seed": 0, "ppl": 23.01,
                            "diags": [], "main": {}}) + "\n")
    os.makedirs(os.path.join(root, "analysis"), exist_ok=True)
    with open(os.path.join(root, "analysis", "reproduce_paper.py"), "w") as f:
        f.write('raise SystemExit("never executed by the analysis")\nAFFECTED = [("SmolLM2-135M", "joint", 0.25, 4, 0)]\n')


# ----------------------------------------------------------------------------- bootstrap
def _two_states(n_per_task=400, k_flip=20, tasks=("t1", "t2", "t3"), seed=1):
    rng = np.random.default_rng(seed)
    cols = {}
    for t in tasks:
        a = (rng.random(n_per_task) < 0.6).astype(float)
        b = a.copy()
        ones = np.flatnonzero(a == 1)[:k_flip]
        b[ones] = 0.0  # B = A with exactly k_flip right answers turned wrong: known difference k_flip / n
        cols[t] = ([("A", "acc"), ("B", "acc")], np.vstack([a, b]))
    return cols


def test_bootstrap_known_difference_inside_interval():
    cols = _two_states()
    bs = S.PairedBootstrap(cols, B=2000, seed=7)
    known = 20 / 400
    for t in ("t1", "t2", "t3"):
        pt, lo, hi = bs.stat(lambda g: g(t, ("A", "acc")) - g(t, ("B", "acc")))
        assert pt == pytest.approx(known, abs=1e-12)
        assert lo <= known <= hi
        assert lo > 0  # paired: the 20 discordant items all favour A, so the interval excludes 0
    pt, lo, hi = bs.stat(lambda g: sum(g(t, ("A", "acc")) - g(t, ("B", "acc")) for t in ("t1", "t2", "t3")) / 3)
    assert lo <= known <= hi
    # same seed, same draws; resamples are per task, so another task's presence does not change them
    bs2 = S.PairedBootstrap({"t1": cols["t1"]}, B=2000, seed=7)
    f = lambda g: g("t1", ("A", "acc")) - g("t1", ("B", "acc"))  # noqa: E731
    assert bs.stat(f) == bs2.stat(f)
    with pytest.raises(S.MissingData):
        bs.stat(lambda g: g("t1", ("C", "acc")))


def test_bootstrap_unpaired_noise_is_wider_than_paired():
    cols = _two_states(k_flip=0)
    bs = S.PairedBootstrap(cols, B=2000, seed=3)
    pt, lo, hi = bs.stat(lambda g: g("t1", ("A", "acc")) - g("t1", ("B", "acc")))
    assert pt == lo == hi == 0.0  # identical vectors: every paired resample gives 0
    pt, lo, hi = bs.stat(lambda g: g("t1", ("A", "acc")))
    assert lo < pt < hi


# ----------------------------------------------------------------------------- McNemar and flips
def _binary(rng, tasks, n, p):
    return {t: {i: int(rng.random() < p) for i in range(n)} for t in tasks}


def test_mcnemar_agrees_with_extra_orders_and_scipy():
    from scipy.stats import binomtest
    rng = random.Random(5)
    a, b = _binary(rng, ["x", "y"], 300, 0.6), _binary(rng, ["x", "y"], 300, 0.5)
    mine = S.mcnemar(a, b, ["x", "y"])
    ref = P.extra().mcnemar(a, b)
    assert mine == ref
    for t in ("x", "y", "pooled"):
        x, y, p = mine[t]
        assert p == pytest.approx(binomtest(min(x, y), x + y, 0.5).pvalue, rel=1e-9)
    assert mine["pooled"][:2] == (mine["x"][0] + mine["y"][0], mine["x"][1] + mine["y"][1])
    assert S.mcnemar(a, b, ["x"])["pooled"] == ref["x"]


def test_flip_counts_split_and_agree_with_extra_orders():
    rng = random.Random(9)
    a, d = _binary(rng, ["x", "y"], 200, 0.55), _binary(rng, ["x", "y"], 200, 0.6)
    fc = S.flip_counts(a, d, ["x", "y"])
    ref = P.extra().flips(a, d)
    for t in ("x", "y"):
        cw = sum(d[t][k] == 1 and a[t][k] == 0 for k in a[t])
        wc = sum(d[t][k] == 0 and a[t][k] == 1 for k in a[t])
        assert (fc[t]["cw"], fc[t]["wc"]) == (cw, wc)
        assert fc[t]["flips"] == pytest.approx(ref[t])
    assert fc["pooled"]["flips"] == pytest.approx(ref["pooled"])


# ----------------------------------------------------------------------------- Sr and I_Sr arithmetic
def test_isr_arithmetic():
    n = 1000
    acc = {"dense": (0.600, 0.700, 0.400), "P": (0.550, 0.700, 0.380), "Q": (0.500, 0.650, 0.350),
           "PQ": (0.480, 0.640, 0.330)}
    tasks = tuple(P.PAPER_TASKS)
    cols = {}
    for i, t in enumerate(tasks):
        keys, rows = [], []
        for name, v in acc.items():
            k = int(round(v[i] * n))
            for m in ("acc", "acc_norm"):
                keys.append((name, m))
                rows.append([1.0] * k + [0.0] * (n - k))
        cols[t] = (keys, np.asarray(rows))
    bs = S.PairedBootstrap(cols, B=500, seed=1)
    spec = S.SR_SETS["sr3_paper"]
    sr = {k: sum(v) / sum(acc["dense"]) for k, v in acc.items()}
    pt, lo, hi = bs.stat(lambda g: S.isr(g, "PQ", "P", "Q", "dense", spec))
    assert pt == pytest.approx(sr["PQ"] - sr["P"] * sr["Q"], abs=1e-12)
    assert lo <= pt <= hi
    chance = {t: P.CHANCE[t] for t in tasks}
    cn = {k: sum(v[i] - chance[t] for i, t in enumerate(tasks)) / sum(acc["dense"][i] - chance[t] for i, t in enumerate(tasks))
          for k, v in acc.items()}
    spec_cn = dict(spec, chance=True)
    pt, _, _ = bs.stat(lambda g: S.isr(g, "PQ", "P", "Q", "dense", spec_cn, chance))
    assert pt == pytest.approx(cn["PQ"] - cn["P"] * cn["Q"], abs=1e-12)
    pt, _, _ = bs.stat(lambda g: S.sr(g, "dense", "dense", spec))
    assert pt == 1.0
    with pytest.raises(S.MissingData):
        bs.stat(lambda g: S.sr(g, "PQ", "dense", S.SR_SETS["sr9_cn"], {"arc_easy": 0.25}))


# ----------------------------------------------------------------------------- loaders and the full pipeline
def documented_names(md):
    """Macro names documented in MACROS.md: listed rows, plus matrix sections (column + model tag + row tag)."""
    names = set(re.findall(r"^\| `\\(res[A-Za-z]+)` \|", md, flags=re.M))
    for sec in md.split("\n## ")[1:]:
        m = re.search(r"Macro = `\\res` \+ column \+ `([A-Za-z]+)` \+ row tag", sec)
        if not m:
            continue
        rows = [ln for ln in sec.splitlines() if ln.startswith("|")]
        cols = [c.strip() for c in rows[0].strip("|").split("|")][3:]
        for ln in rows[2:]:
            cells = [c.strip() for c in ln.strip("|").split("|")]
            for c, v in zip(cols, cells[3:]):
                if v:
                    names.add("res" + c + m.group(1) + cells[0].strip("`"))
    return names



def test_store_and_alignment(tmp_path):
    make_fake_repo(str(tmp_path))
    store = Store(Paths(root=str(tmp_path)))
    assert len(store.states) == 8 and len(store.det) == 1 and len(store.failures) == 1
    d = store.dense("SmolLM2-135M")
    assert d is not None and store.prune_only("SmolLM2-135M", 0.25)["order"] == "prune"
    assert store.quant_only("SmolLM2-135M", 3)["bits"] == 3
    cols, notes = aligned_columns(store, store.model_states("SmolLM2-135M"), P.PAPER_TASKS)
    assert not notes and set(cols) == set(P.PAPER_TASKS)
    keys, X, ids = cols["arc_easy"]
    assert X.shape == (len(keys), 60) and ids == list(range(60))
    # a corrupted item file is detected by its sha256 and treated as missing
    sid = d["state_id"]
    with open(store.items_path(sid), "ab") as f:
        f.write(b"x")
    store2 = Store(Paths(root=str(tmp_path)))
    assert store2.items(sid) is None and "sha256" in store2.item_problems[sid]


def test_macros_and_tables_on_fake_results(tmp_path):
    root = str(tmp_path)
    lines = make_fake_repo(root)
    paths = Paths(root=root)
    os.makedirs(paths.reports, exist_ok=True)
    with open(os.path.join(paths.reports, "boolq_chance.json"), "w") as f:  # synthetic cache: exercises sr9_cn
        json.dump({"rate": 0.6, "n": 10, "counts": {"0": 4, "1": 6}, "dataset": "fake", "config": "boolq",
                   "split": "validation", "revision": "0" * 40}, f)
    from scripts.make_tables import run
    out = run(paths=paths, B=300, figures=True, fetch=False, log=lambda *a: None)
    with open(os.path.join(paths.generated, "macros.tex")) as f:
        tex = f.read()
    defs = re.findall(r"^\\newcommand\{\\([^}]*)\}\{(.*)\}$", tex, flags=re.M)
    names = [n for n, _ in defs]
    assert defs and len(names) == len(set(names))
    assert all(re.fullmatch(r"[A-Za-z]+", n) for n in names)
    vals = dict(defs)
    dense = [r for r in lines if r["order"] == "dense"][0]
    assert vals["resPplSmolSmallDense"] == f"{dense['wikitext2']['ppl']:.2f}"
    pq = [r for r in lines if r["order"] == "p_then_q"][0]
    assert vals["resKlSmolSmallSTwentyFiveBFourPQ"] == f"{pq['wikitext2']['kl']:.3f}"
    assert vals["resPplSmolMediumDense"] == "--"                     # planned, not in the fake results
    assert vals["resIdentityQPThreeSmolSmall"] == "yes"              # Q->P 25%+3 is the 3-bit line
    assert vals["resDetIdentical"] == "yes"
    assert vals["resQTwoCellsPpl"] == "1"
    isr = [x for x in out["e1"]["isr"] if x["set"] == "sr3_paper" and x["state_id"] == pq["state_id"]][0]
    assert vals["resISrPaperSmolSmallSTwentyFiveBFourPQ"] == "\\ensuremath{" + f"{isr['value']:+.3f}" + "}"
    assert vals["resSrCnSmolSmallSTwentyFiveBFourPQ"] != "--"
    with open(os.path.join(paths.generated, "MACROS.md")) as f:
        md = f.read()
    assert documented_names(md) == set(names)  # every macro is documented, nothing else is
    with open(os.path.join(paths.generated, "table_main_rows.tex")) as f:
        rows = f.read()
    assert "\\hline" not in rows and "SmolLM2-135M & dense & 16.00 &" in rows and "--" in rows
    with open(os.path.join(paths.generated, "fig_frontier_quant_SmolLM2-135M.dat")) as f:
        dat = [l for l in f.read().splitlines() if not l.startswith("#")]
    assert dat[0] == "bits bits_tcr bits_mask ppl kl" and len(dat) == 3
    with open(os.path.join(paths.reports, "missing.md")) as f:
        miss = f.read()
    assert "OutOfMemoryError: fake" in miss and "E1 SmolLM2-360M" in miss
    assert os.path.exists(os.path.join(paths.figures, "frontier_SmolLM2-135M.pdf"))
    assert os.path.exists(os.path.join(paths.reports, "summary.md"))
    e2 = [c for c in out["e2"]["cells"] if c["model"] == "SmolLM2-135M" and c.get("n_layers")][0]
    # E2 numbers agree with the review's own summarize() printout (one decimal)
    text = out["e2"]["review_text"]
    m = re.search(r"p_then_q\s+I_E/\(E_P\+E_Q\)\s+([+-]\d+\.\d)%", text)
    assert m and float(m.group(1)) == pytest.approx(round(e2["ratio_p_then_q_pct"], 1), abs=0.051)
    m = re.search(r"share of the penalty removed by q_then_p_refit\s+([+-]\d+\.\d)%", text)
    assert m and float(m.group(1)) == pytest.approx(round(e2["penalty_removed_q_then_p_refit_pct"], 1), abs=0.051)
