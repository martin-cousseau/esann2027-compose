"""E2: same-Hessian layer ablation (review_code/ablation/ablation_same_hessian.py, run by scripts/run_e2.py).

One line per (model, block, linear layer); every arm of a layer is compressed with the same dense Hessian.
The formulas are those of ablation_same_hessian.summarize (which only prints); its own printout is also
captured verbatim in reports/analysis/e2_review_summary.txt:
  interaction    I_E / (E_P + E_Q) per arm from sums over layers, and the share of sub-additive layers
  mechanism      share of the Q->P penalty (sum of E_QP - E_PQ) removed by each ablation arm
  stale grid     change of P->Q's summed error when the stale grid is imposed on it
  mask quality   pruning-only error with the Q->P mask relative to SparseGPT's mask (median), Jaccard (median)
Added here: the same by sublayer type, block 0 against deeper blocks, and the Spearman correlation between
each layer's interaction ratio and the off-diagonal correlation index of its Hessian (h_corr_index).
"""
import ast
import os
import re
import statistics as st
from fractions import Fraction

from .common import (COMBINED, LAYER_TYPES, MECH_ARMS, MQ, P, cell_label, finite, layer_type, short, v1_layer_layout,
                     write_csv, write_text)


def _cells_from_run_e2(root):
    path = os.path.join(root, "scripts", "run_e2.py")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        tree = ast.parse(f.read())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CELLS" for t in node.targets):
            text = ast.literal_eval(node.value)
            return [(float(Fraction(x.split(":")[0])), int(x.split(":")[1])) for x in text.split(",")]
    return None


def _block_plan(root):
    """The --blocks argument of the E2 chain (colab/chain_c.sh), e.g. 'stride2'; None if unknown."""
    path = os.path.join(root, "colab", "chain_c.sh")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        m = re.search(r"run_e2\.py[^\n]*--blocks\s+(\S+)", f.read())
    return m.group(1) if m else None


def _expected_blocks(plan, n_blocks):
    if plan is None or n_blocks is None:
        return None
    if plan == "all":
        return list(range(n_blocks))
    if plan.startswith("stride"):
        return list(range(0, n_blocks, int(plan[len("stride"):])))
    return [int(x) for x in plan.split(",")]


def load(paths):
    root = os.path.join(paths.results_v2, "E2")
    data, sources, dups = {}, {}, 0
    if not os.path.isdir(root):
        return data, sources, dups
    for f in sorted(os.listdir(root)):
        if not f.endswith(".jsonl") or f.startswith("failures__"):
            continue
        for r in P.read_jsonl(os.path.join(root, f)):
            if not isinstance(r, dict) or "cells" not in r:
                continue
            m = short(r["model"])
            seen = data.setdefault(m, {})
            k = (r["block"], r["name"])
            if k in seen:
                dups += 1
                continue
            seen[k] = r
            sources.setdefault(m, os.path.relpath(os.path.join(root, f), paths.root))
    return {m: [v[k] for k in sorted(v)] for m, v in data.items()}, sources, dups


def _ck(s, b):
    return f"{s:.4f}/{b}"


def _ratio(rows, ck, sk, bk, arm):
    sP = sum(r["E_P"][sk] for r in rows)
    sQ = sum(r["E_Q"][bk] for r in rows)
    sC = sum(r["cells"][ck][arm] for r in rows)
    return 100 * (sC - sP - sQ) / (sP + sQ) if rows and sP + sQ > 0 else None


def _share(rows, ck, sk, bk, arm):
    return sum(r["cells"][ck][arm] < r["E_P"][sk] + r["E_Q"][bk] for r in rows) / len(rows) if rows else None


def layer_ratio(r, ck, sk, bk, arm):
    ep, eq = r["E_P"][sk], r["E_Q"][bk]
    return 100 * (r["cells"][ck][arm] - ep - eq) / (ep + eq) if ep + eq > 0 else None


def cell_summary(model, rows, s, b, arms):
    ck, sk, bk = _ck(s, b), f"{s:.4f}", str(b)
    R = [r for r in rows if ck in r["cells"] and sk in r["E_P"] and bk in r["E_Q"]]
    out = {"model": model, "cell": cell_label(s, b), "sparsity": s, "bits": b, "n_layers": len(R),
           "blocks": " ".join(str(x) for x in sorted({r["block"] for r in R}))}
    if not R:
        return out, [], []
    arms = [a for a in arms if all(a in r["cells"][ck] for r in R)]
    out["E_P_sum"] = sum(r["E_P"][sk] for r in R)
    out["E_Q_sum"] = sum(r["E_Q"][bk] for r in R)
    for a in arms:
        out[f"ratio_{a}_pct"] = _ratio(R, ck, sk, bk, a)
        out[f"share_sub_{a}"] = _share(R, ck, sk, bk, a)
        out[f"ratio_block0_{a}_pct"] = _ratio([r for r in R if r["block"] == 0], ck, sk, bk, a)
        out[f"ratio_deeper_{a}_pct"] = _ratio([r for r in R if r["block"] > 0], ck, sk, bk, a)
    if "q_then_p" in arms and "p_then_q" in arms:
        pq = sum(r["cells"][ck]["p_then_q"] for r in R)
        pen = sum(r["cells"][ck]["q_then_p"] - r["cells"][ck]["p_then_q"] for r in R)
        out["qp_penalty_sum"], out["qp_penalty_pct_of_pq"] = pen, (100 * pen / pq if pq > 0 else None)
        for a in MECH_ARMS:
            if a in arms:
                closed = sum(r["cells"][ck]["q_then_p"] - r["cells"][ck][a] for r in R)
                out[f"penalty_removed_{a}_pct"] = 100 * closed / pen if pen > 0 else None
        if "p_then_q_stale" in arms and pq > 0:
            hurt = sum(r["cells"][ck]["p_then_q_stale"] - r["cells"][ck]["p_then_q"] for r in R)
            out["stale_grid_on_pq_pct"] = 100 * hurt / pq
    mq = [r["cells"][ck].get("mask_quality_q_then_p") for r in R]
    mq = [x for x in mq if finite(x)]
    jac = [r["cells"][ck].get("mask_jaccard_q_then_p_vs_P") for r in R]
    jac = [x for x in jac if finite(x)]
    out["mask_quality_median"] = st.median(mq) if mq else None
    out["mask_jaccard_median"] = st.median(jac) if jac else None
    zq = [r["zeros_Q"][bk] for r in R if bk in r.get("zeros_Q", {})]
    zqp = [r["cells"][ck].get("zeros_q_then_p") for r in R if finite(r["cells"][ck].get("zeros_q_then_p"))]
    out["zeros_quant_mean"] = st.mean(zq) if zq else None
    out["zeros_q_then_p_mean"] = st.mean(zqp) if zqp else None
    types = []
    for t in LAYER_TYPES:
        Rt = [r for r in R if layer_type(r["name"]) == t]
        for a in arms:
            types.append({"model": model, "cell": cell_label(s, b), "type": t, "arm": a, "n_layers": len(Rt),
                          "ratio_pct": _ratio(Rt, ck, sk, bk, a), "share_sub": _share(Rt, ck, sk, bk, a)})
    layers = []
    for r in R:
        x = {"model": model, "cell": cell_label(s, b), "block": r["block"], "name": r["name"], "type": layer_type(r["name"]),
             "E_P": r["E_P"][sk], "E_Q": r["E_Q"][bk], "h_corr_index": r.get("h_corr_index"),
             "h_offdiag_mean_abs_corr": r.get("h_offdiag_mean_abs_corr"), "h_dead_columns": r.get("h_dead_columns")}
        for a in arms:
            x[f"E_{a}"] = r["cells"][ck][a]
            x[f"ratio_{a}_pct"] = layer_ratio(r, ck, sk, bk, a)
        x["mask_quality_q_then_p"] = r["cells"][ck].get("mask_quality_q_then_p")
        x["mask_jaccard_q_then_p_vs_P"] = r["cells"][ck].get("mask_jaccard_q_then_p_vs_P")
        layers.append(x)
    return out, types, layers


def spearman(layers, arm):
    from scipy import stats as sps
    pts = [(x["h_corr_index"], x.get(f"ratio_{arm}_pct")) for x in layers]
    pts = [(a, b) for a, b in pts if finite(a) and finite(b)]
    if len(pts) < 3:
        return None, None, len(pts)
    res = sps.spearmanr([a for a, _ in pts], [b for _, b in pts])
    return float(res.statistic), float(res.pvalue), len(pts)


def review_printout(data, cells):
    """ablation_same_hessian.summarize's own text for every model (its log goes to a list)."""
    try:
        import ablation_same_hessian as ASH
    except Exception as e:  # noqa: BLE001
        return f"(review summarize() not available: {type(e).__name__}: {e})\n"
    lines = []
    for m, rows in data.items():
        try:
            ASH.summarize(rows, cells, COMBINED, log=lines.append)
        except Exception as e:  # noqa: BLE001  (e.g. a partial line set without every arm)
            lines.append(f"{m}: summarize() failed: {type(e).__name__}: {e}")
    return "\n".join(lines) + "\n"


def analyze(paths, missing):
    data, sources, dups = load(paths)
    cells = _cells_from_run_e2(paths.root)
    plan = _block_plan(paths.root)
    layout = v1_layer_layout(paths)
    res = {"cells": [], "types": [], "layers": [], "spearman": [], "coverage": [], "sources": sources,
           "duplicates": dups, "plan": plan, "cell_list": cells}
    models = [short(m) for m in MQ.CORE] + [m for m in data if m not in [short(x) for x in MQ.CORE]]
    if cells is None:  # scripts/run_e2.py not found: the cells present in the lines
        cells = sorted({(float(k.split("/")[0]), int(k.split("/")[1])) for rows in data.values() for r in rows
                        for k in r["cells"]}, key=lambda c: (-c[1], c[0])) or None
        res["cell_list"] = cells
    for m in models:
        rows = data.get(m, [])
        n_blocks, names = layout.get(m, (None, None))
        exp = _expected_blocks(plan, n_blocks)
        have = sorted({r["block"] for r in rows})
        cov = {"model": m, "n_lines": len(rows), "blocks_present": " ".join(map(str, have)), "source": sources.get(m)}
        if exp is not None and names:
            want = {(b, n) for b in exp for n in names}
            got = {(r["block"], r["name"]) for r in rows}
            cov.update(n_expected=len(want), n_missing=len(want - got),
                       blocks_missing=" ".join(map(str, sorted({b for b, _ in want - got}))))
            if want - got:
                missing.add("E2 same-Hessian ablation", m, f"{len(want - got)} of {len(want)} planned layer lines not available yet"
                            f" (blocks {cov['blocks_missing']})")
        elif not rows:
            missing.add("E2 same-Hessian ablation", m, "no lines yet")
        res["coverage"].append(cov)
        if not rows:
            continue
        for s, b in cells or []:
            summ, types, layers = cell_summary(m, rows, s, b, COMBINED)
            res["cells"].append(summ)
            res["types"] += types
            res["layers"] += layers
            for a in ("p_then_q", "joint", "q_then_p"):
                rho, p, n = spearman(layers, a)
                res["spearman"].append({"model": m, "cell": cell_label(s, b), "arm": a, "n_layers": n, "rho": rho, "p": p})
    res["review_text"] = review_printout(data, cells or [])
    return res


def write(res, paths):
    R = paths.reports
    write_csv(os.path.join(R, "e2_cells.csv"), res["cells"])
    write_csv(os.path.join(R, "e2_types.csv"), res["types"])
    write_csv(os.path.join(R, "e2_layers.csv"), res["layers"])
    write_csv(os.path.join(R, "e2_spearman.csv"), res["spearman"])
    write_csv(os.path.join(R, "e2_coverage.csv"), res["coverage"])
    write_text(os.path.join(R, "e2_review_summary.txt"),
               "# ablation_same_hessian.summarize() on results/v2/E2 (generated by scripts/make_tables.py)\n"
               + res["review_text"])
