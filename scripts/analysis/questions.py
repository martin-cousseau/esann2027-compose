"""The paper's questions under the v2 metrics, plus the fp16-vs-fp32 comparison with the published runs.

Q1  layer level, from the per-layer diagnostics of the result lines: I_E / (E_P + E_Q) per cell and order,
    (a) as the ratio of mean errors over layers (the paper's Fig. 1, reproduce_paper.py), (b) as the share
    of sub-additive layers matched layer by layer (the review's analyze_results.py, section D), both also
    by sublayer type, and block 0 against blocks 1+ (block 0 is the only block calibrated on identical
    inputs in all runs).
Q2  per cell: the ranking of the three orders by WikiText-2 perplexity, C4 perplexity, KL, pooled flips
    and Sr; the cells where Q->P is worst; the worst/best perplexity ratio; agreement between metrics.
Q3  the 4-bit order spread (max - min) / min per model.
fp  for every published (model, order, s, b) at seed 0: v2 fp32 perplexity against the value the paper
    used (16-bit from results_full, or the 32-bit re-run from results_fix) and the relative difference.
"""
import os

from .common import COMBINED, LAYER_TYPES, ORDER_TXT, PAPER_ORDERS, cell_label, finite, layer_type, load_v1, write_csv

Q2_METRICS = (  # name, label, +1 = lower is better, -1 = higher is better; the first five define agreement
    ("wt_ppl", "WikiText-2 ppl", 1), ("c4_ppl", "C4 ppl", 1), ("wt_kl", "KL (WikiText-2)", 1),
    ("flips9", "flips (pooled, 9 tasks)", 1), ("sr3_v2", "Sr (3 tasks, v2 metric)", -1),
    ("c4_kl", "KL (C4)", 1), ("sr9_v2", "Sr-extended", -1), ("sr3_paper", "Sr (3 tasks, paper metric)", -1))
AGREEMENT_METRICS = ("wt_ppl", "c4_ppl", "wt_kl", "flips9", "sr3_v2")


# ----------------------------------------------------------------------------- Q1
def _lmap(r):
    return {(d["layer"], d["name"]): d["rel_err"] for d in (r.get("diags") or [])}


def _ratio(C, Pm, Qm, keys):
    sp, sq, sc = sum(Pm[k] for k in keys), sum(Qm[k] for k in keys), sum(C[k] for k in keys)
    return 100 * (sc - sp - sq) / (sp + sq) if keys and (sp + sq) > 0 else None


def q1(store, missing):
    rows = []
    for model in store.models():
        for seed in store.seeds(model):
            for s, b in store.cells(model, seed):
                pr, qr = store.prune_only(model, s, seed), store.quant_only(model, b, seed)
                for order in COMBINED:
                    if not store.is_planned(model, order, s, b, seed) and store.get(model, order, s, b, seed) is None:
                        continue
                    cr = store.get(model, order, s, b, seed)
                    what = f"{model} {cell_label(s, b)} {ORDER_TXT[order]} seed {seed}"
                    if cr is None or pr is None or qr is None:
                        lack = [n for n, x in (("combined", cr), ("prune-only", pr), ("quant-only", qr)) if x is None]
                        planned_p = store.is_planned(model, "prune", s, 16, seed)
                        reason = "missing: " + ", ".join(lack)
                        if pr is None and not planned_p:
                            reason += " (no prune-only state at this sparsity in the study plan)"
                        missing.add("Q1 layer interaction", what, reason)
                        continue
                    C, Pm, Qm = _lmap(cr), _lmap(pr), _lmap(qr)
                    keys = sorted(set(C) & set(Pm) & set(Qm))
                    if not keys:
                        missing.add("Q1 layer interaction", what, "no per-layer diagnostics")
                        continue
                    n = len(keys)
                    row = {"model": model, "seed": seed, "cell": cell_label(s, b), "sparsity": s, "bits": b,
                           "order": order, "n_layers": n,
                           "ratio_mean_pct": _ratio(C, Pm, Qm, keys),
                           "share_subadditive": sum(C[k] < Pm[k] + Qm[k] for k in keys) / n,
                           "n_subadditive": sum(C[k] < Pm[k] + Qm[k] for k in keys),
                           "ratio_block0_pct": _ratio(C, Pm, Qm, [k for k in keys if k[0] == 0]),
                           "ratio_blocks1plus_pct": _ratio(C, Pm, Qm, [k for k in keys if k[0] > 0]),
                           "ratio_from_mean_rel_err_pct": None}
                    ep, eq, ec = pr.get("mean_rel_err"), qr.get("mean_rel_err"), cr.get("mean_rel_err")
                    if all(finite(x) for x in (ep, eq, ec)) and ep + eq > 0:
                        row["ratio_from_mean_rel_err_pct"] = 100 * (ec - ep - eq) / (ep + eq)
                    for t in LAYER_TYPES:
                        kt = [k for k in keys if layer_type(k[1]) == t]
                        row[f"ratio_{t}_pct"] = _ratio(C, Pm, Qm, kt)
                        row[f"share_{t}"] = (sum(C[k] < Pm[k] + Qm[k] for k in kt) / len(kt)) if kt else None
                        row[f"n_{t}"] = len(kt)
                    rows.append(row)
    return rows


def q1_summary(rows, seed=0):
    """Counts as the paper states them (seed 0, the three paper orders), and pooled layer shares."""
    R = [r for r in rows if r["seed"] == seed and r["order"] in PAPER_ORDERS and r["ratio_mean_pct"] is not None]
    mf = [r for r in R if r["order"] in ("joint", "p_then_q")]
    out = {"mask_first_total": len(mf), "mask_first_subadditive": sum(r["ratio_mean_pct"] < 0 for r in mf)}
    for o in ("joint", "p_then_q", "q_then_p"):
        for b in (4, 3, None):
            rr = [r for r in R if r["order"] == o and (b is None or r["bits"] == b)]
            tag = f"{o}_{b}bit" if b else o
            v = [r["ratio_mean_pct"] for r in rr]
            out[f"{tag}_n"] = len(v)
            out[f"{tag}_min_pct"] = min(v) if v else None
            out[f"{tag}_max_pct"] = max(v) if v else None
            out[f"{tag}_n_subadditive"] = sum(x < 0 for x in v)
            out[f"{tag}_n_superadditive"] = sum(x > 0 for x in v)
            n_layers = sum(r["n_layers"] for r in rr)
            out[f"{tag}_layer_share_subadditive"] = (sum(r["n_subadditive"] for r in rr) / n_layers) if n_layers else None
            for t in LAYER_TYPES:
                nt = sum(r[f"n_{t}"] for r in rr)
                st = sum((r[f"share_{t}"] or 0) * r[f"n_{t}"] for r in rr)
                out[f"{tag}_layer_share_{t}"] = (st / nt) if nt else None
            b0 = [r for r in rr if r["ratio_block0_pct"] is not None and r["ratio_blocks1plus_pct"] is not None]
            out[f"{tag}_cells_block0_superadditive"] = sum(r["ratio_block0_pct"] > 0 for r in b0)
            out[f"{tag}_cells_block0_vs_rest_n"] = len(b0)
    return out


# ----------------------------------------------------------------------------- Q2 / Q3
def _metric(store, e1, sid, name):
    r = store.states.get(sid)
    if r is None:
        return None
    if name in ("wt_ppl", "wt_kl"):
        return (r.get("wikitext2") or {}).get(name[3:])
    if name in ("c4_ppl", "c4_kl"):
        return (r.get("c4") or {}).get(name[3:])
    b = e1["by_sid"].get(sid, {})
    if name == "flips9":
        return (b.get("flips", {}).get("pooled9") or {}).get("flips")
    if name.startswith("sr"):
        v = b.get("sr", {}).get(name)
        return v[0] if v else None
    return None


def _ranking(vals, sign):
    """'P->Q < joint < Q->P' from best to worst; '=' for exact ties."""
    order = sorted(vals, key=lambda o: (sign * vals[o], PAPER_ORDERS.index(o)))
    out = ORDER_TXT[order[0]]
    for prev, o in zip(order, order[1:]):
        out += (" = " if vals[o] == vals[prev] else " < ") + ORDER_TXT[o]
    return out, order


def q2(store, e1, missing, seed=0):
    rows = []
    for model in store.models():
        for s, b in store.cells(model, seed, PAPER_ORDERS):
            sids = {o: store.sid(model, o, s, b, seed) for o in PAPER_ORDERS}
            row = {"model": model, "seed": seed, "cell": cell_label(s, b), "sparsity": s, "bits": b}
            absent = [ORDER_TXT[o] for o, x in sids.items() if x is None]
            if absent:
                missing.add("Q2 order ranking", f"{model} {cell_label(s, b)} seed {seed}", "missing orders: " + ", ".join(absent))
            for name, _label, sign in Q2_METRICS:
                vals = {o: _metric(store, e1, x, name) for o, x in sids.items() if x}
                for o in PAPER_ORDERS:
                    row[f"{name}_{o}"] = vals.get(o)
                if len(vals) == 3 and all(finite(v) for v in vals.values()):
                    rk, order = _ranking(vals, sign)
                    row[f"{name}_ranking"] = rk
                    row[f"{name}_best"], row[f"{name}_worst"] = order[0], order[-1]
                    row[f"{name}_tie_at_worst"] = vals[order[-1]] == vals[order[-2]]
                    if name in ("wt_ppl", "c4_ppl"):
                        row[f"{name}_worst_best_ratio"] = max(vals.values()) / min(vals.values())
                        row[f"{name}_spread_pct"] = 100 * (max(vals.values()) - min(vals.values())) / min(vals.values())
                elif not absent:
                    missing.add("Q2 order ranking", f"{model} {cell_label(s, b)} {name}", "metric not available for every order")
            rows.append(row)
    return rows


def q2_summary(rows):
    out = {"cells": len(rows)}
    for name, _label, _sign in Q2_METRICS:
        R = [r for r in rows if r.get(f"{name}_ranking")]
        out[f"{name}_cells"] = len(R)
        out[f"{name}_qp_worst"] = sum(r[f"{name}_worst"] == "q_then_p" and not r[f"{name}_tie_at_worst"] for r in R)
        out[f"{name}_qp_worst_or_tied"] = sum(r[f"{name}_q_then_p"] == r[f"{name}_{r[f'{name}_worst']}"] for r in R)
        for o in PAPER_ORDERS:
            out[f"{name}_best_{o}"] = sum(r[f"{name}_best"] == o for r in R)
            for m in sorted({r["model"] for r in R}):
                out[f"{name}_best_{o}_{m}"] = sum(r[f"{name}_best"] == o for r in R if r["model"] == m)
        if name in ("wt_ppl", "c4_ppl"):
            rat = [(r[f"{name}_worst_best_ratio"], r["model"], r["cell"]) for r in R]
            out[f"{name}_max_worst_best_ratio"] = max(rat)[0] if rat else None
            out[f"{name}_max_worst_best_where"] = f"{max(rat)[1]} {max(rat)[2]}" if rat else None
    full = [r for r in rows if all(r.get(f"{m}_ranking") for m in AGREEMENT_METRICS)]
    out["agreement_cells"] = len(full)
    out["agreement_identical"] = sum(len({r[f"{m}_ranking"] for m in AGREEMENT_METRICS}) == 1 for r in full)
    for m in AGREEMENT_METRICS[1:]:
        R = [r for r in rows if r.get(f"{m}_ranking") and r.get("wt_ppl_ranking")]
        out[f"agree_with_wt_ppl_{m}_cells"] = len(R)
        out[f"agree_with_wt_ppl_{m}"] = sum(r[f"{m}_ranking"] == r["wt_ppl_ranking"] for r in R)
        out[f"same_best_as_wt_ppl_{m}"] = sum(r[f"{m}_best"] == r["wt_ppl_best"] for r in R)
    return out


def q3(rows):
    out = []
    for model in sorted({r["model"] for r in rows}, key=[r["model"] for r in rows].index):
        for b in (4, 3):
            for name in ("wt_ppl", "c4_ppl"):
                R = [r for r in rows if r["model"] == model and r["bits"] == b and r.get(f"{name}_spread_pct") is not None]
                expected = [r for r in rows if r["model"] == model and r["bits"] == b]
                v = [r[f"{name}_spread_pct"] for r in R]
                out.append({"model": model, "bits": b, "metric": name, "n_cells": len(v), "n_cells_planned": len(expected),
                            "spread_min_pct": min(v) if v else None, "spread_max_pct": max(v) if v else None,
                            "cells": " ".join(f"{r['cell']}:{r[f'{name}_spread_pct']:.6g}" for r in R)})
    return out


# ----------------------------------------------------------------------------- fp16 vs fp32
def fp_compare(store, paths, missing):
    v1 = load_v1(paths)
    if v1 is None:
        missing.add("fp16 vs fp32", "published runs", "results/results_full.jsonl or results_fix.jsonl not found")
        return [], None
    rows = []
    for k, r in sorted(v1["merged"].items()):
        m, o, s, b, sd = k
        if sd != 0:
            continue
        cur = store.get(m, o, s, b, 0)
        pub = r.get("ppl")
        fix = v1["fix"].get(k)
        row = {"model": m, "order": o, "sparsity": s, "bits": b, "config": cell_label(s, b) if o in COMBINED else
               {"dense": "dense", "prune": f"{round(s * 100)}% pruned", "quant": f"{b}-bit"}.get(o, o),
               "published_ppl": pub, "published_precision": r["precision"],
               "v2_ppl": (cur.get("wikitext2") or {}).get("ppl") if cur else None,
               "v2_state": cur["state_id"] if cur else None,
               "fix_fp32_ppl": fix.get("ppl") if fix else None}
        row["rel_delta"] = (row["v2_ppl"] / pub - 1) if (finite(row["v2_ppl"]) and finite(pub)) else None
        row["rel_delta_vs_fix"] = (row["v2_ppl"] / row["fix_fp32_ppl"] - 1) if (finite(row["v2_ppl"]) and finite(row["fix_fp32_ppl"])) else None
        if cur is None:
            missing.add("fp16 vs fp32", f"{m} {row['config']} {ORDER_TXT.get(o, o)}", "v2 state not available yet")
        rows.append(row)
    return rows, v1


def fp_summary(rows):
    R = [r for r in rows if r["rel_delta"] is not None]
    out = {"n_published": len(rows), "n_compared": len(R)}
    if R:
        w = max(R, key=lambda r: abs(r["rel_delta"]))
        out.update(max_abs_rel_delta=abs(w["rel_delta"]), max_where=f"{w['model']} {w['config']} {ORDER_TXT.get(w['order'], w['order'])}",
                   max_signed=w["rel_delta"])
        for prec in ("fp16", "fp32"):
            rr = [abs(r["rel_delta"]) for r in R if r["published_precision"] == prec]
            out[f"n_{prec}"] = len(rr)
            out[f"max_abs_rel_delta_{prec}"] = max(rr) if rr else None
        for m in sorted({r["model"] for r in R}):
            rr = [r["rel_delta"] for r in R if r["model"] == m]
            out[f"max_abs_rel_delta_{m}"] = max(abs(x) for x in rr)
            out[f"n_{m}"] = len(rr)
            out[f"median_rel_delta_{m}"] = sorted(rr)[len(rr) // 2] if len(rr) % 2 else \
                (sorted(rr)[len(rr) // 2 - 1] + sorted(rr)[len(rr) // 2]) / 2
    F = [r for r in rows if r["rel_delta_vs_fix"] is not None]
    out["n_vs_fix"] = len(F)
    out["max_abs_rel_delta_vs_fix"] = max(abs(r["rel_delta_vs_fix"]) for r in F) if F else None
    return out


def write(res, paths):
    R = paths.reports
    write_csv(os.path.join(R, "q1_layer_interaction.csv"), res["q1"])
    write_csv(os.path.join(R, "q2_order_ranking.csv"), res["q2"])
    write_csv(os.path.join(R, "q3_order_spread.csv"), res["q3"])
    write_csv(os.path.join(R, "fp16_vs_fp32.csv"), res["fp"])


def analyze(store, e1, paths, missing):
    res = {"q1": q1(store, missing)}
    res["q1_summary"] = q1_summary(res["q1"])
    res["q2"] = q2(store, e1, missing)
    res["q2_summary"] = q2_summary(res["q2"])
    res["q3"] = q3(res["q2"])
    res["fp"], v1 = fp_compare(store, paths, missing)
    res["fp_summary"] = fp_summary(res["fp"])
    res["v1_check"] = v1["check"] if v1 else None
    res["v1_files"] = v1["files"] if v1 else []
    return res
