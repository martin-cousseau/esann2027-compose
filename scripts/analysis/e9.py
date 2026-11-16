"""E9: the layer-level control without error compensation (scripts/run_e9.py), per (model, cell).

Sums over the layers of a model (as E2 does) of the uncompensated single-method errors E_P, E_Q and of the arms
p_then_q, q_then_p, p_then_q_stale; the interaction of each arm in % of E_P + E_Q; the three terms of the paper's
decomposition for p_then_q_stale (absorption, cross_PR, cross_RR) in % of E_P + E_Q and the largest relative residual
of the identity; the gap between the sequential orders and its split into a mask part (q_then_p - p_then_q_stale)
and a grid part (p_then_q_stale - p_then_q) (without compensation the update-order part is zero by construction);
the compensated single-method errors of the same layers; and the overlap of the pruning mask with the zeros of
quantization alone (weighted by the number of weights), without and with compensation.
Writes reports/analysis/e9_cells.csv; e2res["e9"] carries the result to the macros (latex.macros_e9).
"""
import json
import os

from .common import cell_label, short, write_csv

KEYS_NOCOMP = ("p_then_q", "q_then_p", "p_then_q_stale")


def load(paths):
    root = os.path.join(paths.results_v2, "E9")
    data = {}
    if not os.path.isdir(root):
        return data
    for fn in sorted(os.listdir(root)):
        if not fn.endswith(".jsonl"):
            continue
        rows, seen = [], set()
        for line in open(os.path.join(root, fn)):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            k = (r["block"], r["name"])
            if k not in seen:
                seen.add(k)
                rows.append(r)
        if rows:
            data[short(rows[0]["model"])] = rows
    return data


def _cells(rows):
    keys = sorted({k for r in rows for k in r["nocomp"]["cells"]}, key=lambda k: (-int(k.split("/")[1]), float(k.split("/")[0])))
    return [(float(k.split("/")[0]), int(k.split("/")[1]), k) for k in keys]


def cell_summary(model, rows, s, b, ck):
    sk, bk = f"{s:.4f}", str(b)
    R = [r for r in rows if ck in r["nocomp"]["cells"]]
    out = {"model": model, "cell": cell_label(s, b), "sparsity": s, "bits": b, "n_layers": len(R),
           "blocks": " ".join(str(x) for x in sorted({r["block"] for r in R}))}
    if not R:
        return out
    n = lambda r: r["shape"][0] * r["shape"][1]  # noqa: E731
    ep = sum(r["nocomp"]["E_P"][sk] for r in R)
    eq = sum(r["nocomp"]["E_Q"][bk] for r in R)
    add = ep + eq
    out.update(E_P_sum=ep, E_Q_sum=eq)
    E = {a: sum(r["nocomp"]["cells"][ck][a] for r in R) for a in KEYS_NOCOMP}
    for a in KEYS_NOCOMP:
        out[f"{a}_sum"] = E[a]
        out[f"ratio_{a}_pct"] = 100 * (E[a] - add) / add if add > 0 else None
        out[f"share_sub_{a}"] = sum((r["nocomp"]["cells"][ck][a] - r["nocomp"]["E_P"][sk] - r["nocomp"]["E_Q"][bk]) < 0 for r in R) / len(R)
    for t in ("absorption", "cross_PR", "cross_RR"):
        out[f"{t}_pct"] = 100 * sum(r["nocomp"]["cells"][ck][t] for r in R) / add if add > 0 else None
    out["cross_pct"] = out["cross_PR_pct"] + out["cross_RR_pct"] if add > 0 else None
    out["decomp_residual_rel_max"] = max(abs(r["nocomp"]["cells"][ck]["decomp_residual"]) /
                                         max(r["nocomp"]["E_P"][sk] + r["nocomp"]["E_Q"][bk], 1e-30) for r in R)
    pq = E["p_then_q"]
    out["gap_pct"] = 100 * (E["q_then_p"] - pq) / pq if pq > 0 else None
    out["mask_pct"] = 100 * (E["q_then_p"] - E["p_then_q_stale"]) / pq if pq > 0 else None
    out["grid_pct"] = 100 * (E["p_then_q_stale"] - pq) / pq if pq > 0 else None
    cp = sum(r["comp"]["E_P"][sk] for r in R)
    cq = sum(r["comp"]["E_Q"][bk] for r in R)
    out.update(comp_E_P_sum=cp, comp_E_Q_sum=cq, comp_over_nocomp_P=cp / ep if ep > 0 else None,
               comp_over_nocomp_Q=cq / eq if eq > 0 else None)
    for tag, get in (("nocomp", lambda r: r["nocomp"]["cells"][ck]), ("comp", lambda r: r["comp"]["cells"][ck])):
        w = sum(n(r) for r in R)
        for k in ("overlap_mask_zeros", "share_zeros_pruned", "zeros"):
            out[f"{tag}_{k}"] = sum(get(r)[k] * n(r) for r in R) / w
    return out


def analyze(paths):
    data = load(paths)
    cells = []
    for m, rows in data.items():
        for s, b, ck in _cells(rows):
            cells.append(cell_summary(m, rows, s, b, ck))
    return {"cells": cells, "models": sorted(data), "layers": {m: len(r) for m, r in data.items()}}


def write(res, paths):
    if res["cells"]:
        write_csv(os.path.join(paths.reports, "e9_cells.csv"), res["cells"])
