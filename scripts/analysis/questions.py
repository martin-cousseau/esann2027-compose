"""The paper's questions under the v2 metrics, plus the fp16-vs-fp32 comparison with the published runs.

Q1  layer level, from the per-layer diagnostics of the result lines: I_E / (E_P + E_Q) per cell and order,
    (a) as the ratio of mean errors over layers (the paper's Fig. 1, reproduce_paper.py), (b) as the share
    of sub-additive layers matched layer by layer (the review's analyze_results.py, section D), both also
    by sublayer type, and block 0 against blocks 1+ (block 0 is the only block calibrated on identical
    inputs in all runs).
Q2  per cell: the ranking of the three orders by WikiText-2 perplexity, C4 perplexity, KL, pooled flips
    and Sr; the cells where Q->P is worst; the worst/best perplexity ratio; agreement between metrics.
Q3  the order spread (max - min) / min per model and bit-width, over every cell and over the grid cells
    (25/33/50%, E4's 60% left out).
E5  calibration seeds, 4- and 3-bit cells of every multi-seed model: best and worst order per seed, stability,
    Q->P worst, paired tests over seeds, mean and sd per order and cell, and the counts per scope and bit-width.
E6  family vs scale: per model and cell the spread, the Q->P penalty 100 (QP/best - 1) and the mask-first gap;
    per family and cell the least-squares slope against log10(compressed weights) (families never mixed); the
    families compared at matched scale (MATCHED).
fp  for every published (model, order, s, b) at seed 0: v2 fp32 perplexity against the value the paper
    used (16-bit from results_full, or the 32-bit re-run from results_fix) and the relative difference.
"""
import os

from .common import (ARMS, COMBINED, FAMILY_ORDER, GRID_SPARSITIES, LAYER_TYPES, MECH_ARMS, MQ, ORDER_TXT, PAPER_ORDERS,
                     cell_label, cell_sort_key, compressed_weights, family, finite, layer_type, load_v1, short, size_label, skey,
                     write_csv)

# Scope of the headline counts ("X of Y cells"): the paper's grid, i.e. the E1 cells of the core models
# (scripts/make_queues.py). Other cells (E4's 60%, E6's model) are in the per-cell CSV files.
PAPER_GRID = {(short(m), skey(u["sparsity"]), int(u["bits"])) for m, q in MQ.e1().items() for u in q
              if u["order"] in PAPER_ORDERS}


def in_grid(r):
    return (r["model"], skey(r["sparsity"]), int(r["bits"])) in PAPER_GRID

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
    R = [r for r in rows if r["seed"] == seed and r["order"] in PAPER_ORDERS and r["ratio_mean_pct"] is not None and in_grid(r)]
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


def q2_per_model(rows):
    """Per model (every model, seed 0), over its grid cells (25/33/50%, both bit-widths; the E1 cells for the core
    models): cells ranked by WikiText-2 ppl, how often each order is best, and Q->P strictly worst."""
    out = {}
    for r in rows:
        if r.get("seed", 0) != 0 or skey(r["sparsity"]) not in GRID_SPARSITIES:
            continue
        d = out.setdefault(r["model"], {"planned": 0, "cells": 0, "qp_worst": 0, **{f"best_{o}": 0 for o in PAPER_ORDERS}})
        d["planned"] += 1
        if r.get("wt_ppl_ranking"):
            d["cells"] += 1
            d[f"best_{r['wt_ppl_best']}"] += 1
            d["qp_worst"] += r["wt_ppl_worst"] == "q_then_p" and not r["wt_ppl_tie_at_worst"]
    return out


def q2_summary(rows):
    rows = [r for r in rows if in_grid(r)]
    out = {"cells": len(rows), "grid_cells": len(PAPER_GRID)}
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
    """Order spread range per model, bit-width and metric: over every cell (E1 and E4), and over the grid cells only
    (25/33/50%, E4's 60% left out); grid_complete = every planned grid cell has the three orders."""
    out = []
    for model in sorted({r["model"] for r in rows}, key=[r["model"] for r in rows].index):
        for b in (4, 3):
            for name in ("wt_ppl", "c4_ppl"):
                R = [r for r in rows if r["model"] == model and r["bits"] == b and r.get(f"{name}_spread_pct") is not None]
                expected = [r for r in rows if r["model"] == model and r["bits"] == b]
                v = [r[f"{name}_spread_pct"] for r in R]
                G = [r for r in R if skey(r["sparsity"]) in GRID_SPARSITIES]
                Ge = [r for r in expected if skey(r["sparsity"]) in GRID_SPARSITIES]
                g = [r[f"{name}_spread_pct"] for r in G]
                out.append({"model": model, "bits": b, "metric": name, "n_cells": len(v), "n_cells_planned": len(expected),
                            "spread_min_pct": min(v) if v else None, "spread_max_pct": max(v) if v else None,
                            "cells": " ".join(f"{r['cell']}:{r[f'{name}_spread_pct']:.6g}" for r in R),
                            "grid_n_cells": len(g), "grid_n_planned": len(Ge), "grid_complete": bool(Ge) and len(g) == len(Ge),
                            "grid_spread_min_pct": min(g) if g else None, "grid_spread_max_pct": max(g) if g else None})
    return out


def scale_cells(store, q2rows, seed=0):
    """Q3 / E6 per model and cell (seed 0): the order spread 100 (max - min) / min of WikiText-2 ppl over the three
    orders, the Q->P penalty 100 (ppl Q->P / best - 1) (0 when Q->P is best), the mask-first gap
    100 |joint - P->Q| / best, with the model's family and compressed weights (complete = all three orders)."""
    import math
    weights = {m: compressed_weights(store, m) for m in store.models()}
    rows = []
    for r in q2rows:
        if r.get("seed", 0) != seed:
            continue
        m, n = r["model"], weights.get(r["model"])
        vals = {o: r.get(f"wt_ppl_{o}") for o in PAPER_ORDERS}
        row = {"model": m, "family": family(m), "size": size_label(m), "compressed_weights": n,
               "log10_weights": math.log10(n) if finite(n) and n > 0 else None, "cell": r["cell"], "sparsity": r["sparsity"],
               "bits": r["bits"], "in_grid": skey(r["sparsity"]) in GRID_SPARSITIES, "complete": bool(r.get("wt_ppl_ranking"))}
        row.update({f"wt_ppl_{o}": vals[o] for o in PAPER_ORDERS})
        row.update(best=None, worst=None, spread_pct=None, qp_penalty_pct=None, gap_mask_first_pct=None, qp_worst=None,
                   c4_spread_pct=r.get("c4_ppl_spread_pct"))
        if row["complete"]:
            lo = min(vals.values())
            row.update(best=r["wt_ppl_best"], worst=r["wt_ppl_worst"], spread_pct=r["wt_ppl_spread_pct"],
                       qp_penalty_pct=100 * (vals["q_then_p"] / lo - 1),
                       gap_mask_first_pct=100 * abs(vals["joint"] - vals["p_then_q"]) / lo,
                       qp_worst=r["wt_ppl_worst"] == "q_then_p" and not r["wt_ppl_tie_at_worst"])
        rows.append(row)
    return rows


def _family_rank(fam):
    return FAMILY_ORDER.index(fam) if fam in FAMILY_ORDER else len(FAMILY_ORDER)


SCALE_KEYS = (("spread", "spread_pct"), ("gap_mask_first", "gap_mask_first_pct"), ("qp_penalty", "qp_penalty_pct"))


def scale_fit(cell_rows):
    """E6: per family and cell (never across families), the order spread, the mask-first gap and the Q->P penalty
    against log10(compressed weights): least-squares slope in percentage points per decade (numpy.polyfit, degree 1)
    over the models of the family that have the cell with all three orders, and whether the value falls with every
    larger model. Fewer than two models: values listed, no fit."""
    import math
    import numpy as np
    groups = {}
    for r in cell_rows:
        if r["complete"] and finite(r["compressed_weights"]):
            groups.setdefault((r["family"], r["sparsity"], r["bits"]), []).append(r)
    rows = []
    for (fam, s, b), pp in sorted(groups.items(), key=lambda kv: (_family_rank(kv[0][0]), cell_sort_key(kv[0][1:]))):
        pp = sorted(pp, key=lambda x: x["compressed_weights"])
        row = {"family": fam, "cell": cell_label(s, b), "sparsity": s, "bits": b, "models": " ".join(x["model"] for x in pp),
               "n_models": len(pp), "compressed_weights": " ".join(str(int(x["compressed_weights"])) for x in pp)}
        for key, col in SCALE_KEYS:
            row[col] = " ".join(f"{x[col]:.4g}" for x in pp)
        if len(pp) >= 2:
            lx = [math.log10(x["compressed_weights"]) for x in pp]
            for key, col in SCALE_KEYS:
                y = [x[col] for x in pp]
                row[f"slope_{key}_per_decade"] = float(np.polyfit(lx, y, 1)[0])
                row[f"{key}_shrinks_monotonically"] = all(b2 < a2 for a2, b2 in zip(y, y[1:]))
            row["range_ratio"] = pp[-1]["compressed_weights"] / pp[0]["compressed_weights"]
        rows.append(row)
    return rows


# Cross-family comparison at matched scale (compressed weights): tag, label, models (one per family where possible).
MATCHED = (("Small", "about 0.3B", ("SmolLM2-360M", "Qwen2.5-0.5B")),
           ("Billion", "about 1.1-1.6B", ("SmolLM2-1.7B", "Qwen2.5-1.5B", "OLMo-2-0425-1B")))


def matched_scale(cell_rows):
    """Per matched-scale group and cell: the spread and the Q->P penalty of each model of the group (different
    families at about the same number of compressed weights), their range and ratio, and the size ratio."""
    rows = []
    for tag, label, models in MATCHED:
        cells = sorted({(r["sparsity"], r["bits"]) for r in cell_rows if r["model"] in models}, key=cell_sort_key)
        for s, b in cells:
            R = [r for r in cell_rows if r["model"] in models and (r["sparsity"], r["bits"]) == (s, b)]
            ok = sorted([r for r in R if r["complete"] and finite(r["compressed_weights"])], key=lambda r: r["compressed_weights"])
            row = {"group": tag, "label": label, "cell": cell_label(s, b), "sparsity": s, "bits": b,
                   "models_planned": " ".join(r["model"] for r in R), "n_planned": len(R), "n_models": len(ok),
                   "models": " ".join(r["model"] for r in ok), "families": " ".join(r["family"] for r in ok),
                   "compressed_weights": " ".join(str(int(r["compressed_weights"])) for r in ok),
                   "spread_pct": " ".join(f"{r['model']}:{r['spread_pct']:.4g}" for r in ok),
                   "qp_penalty_pct": " ".join(f"{r['model']}:{r['qp_penalty_pct']:.4g}" for r in ok),
                   "complete": len(ok) == len(R)}
            if len(ok) >= 2:
                lo, hi = min(ok, key=lambda r: r["spread_pct"]), max(ok, key=lambda r: r["spread_pct"])
                plo, phi = min(ok, key=lambda r: r["qp_penalty_pct"]), max(ok, key=lambda r: r["qp_penalty_pct"])
                row.update(spread_min_pct=lo["spread_pct"], spread_max_pct=hi["spread_pct"],
                           spread_gap_pp=hi["spread_pct"] - lo["spread_pct"],
                           spread_ratio=hi["spread_pct"] / lo["spread_pct"] if lo["spread_pct"] > 0 else None,
                           largest_spread=hi["model"], smallest_spread=lo["model"],
                           qp_penalty_min_pct=plo["qp_penalty_pct"], qp_penalty_max_pct=phi["qp_penalty_pct"],
                           largest_qp_penalty=phi["model"], smallest_qp_penalty=plo["model"],
                           weights_ratio=ok[-1]["compressed_weights"] / ok[0]["compressed_weights"])
            rows.append(row)
    return rows


FIG_SCALE_CELLS = ("25%+4", "50%+4", "50%+3")


def fig_scale_rows(cell_rows):
    """The data of paper/generated/fig_order_scale.pdf (one row per plotted point)."""
    out = []
    for cell in FIG_SCALE_CELLS:
        for r in sorted([r for r in cell_rows if r["cell"] == cell and r["complete"] and finite(r["compressed_weights"])],
                        key=lambda r: (_family_rank(r["family"]), r["compressed_weights"])):
            out.append({k: r[k] for k in ("cell", "family", "model", "size", "compressed_weights", "spread_pct",
                                          "qp_penalty_pct", "best", "worst")})
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


# ----------------------------------------------------------------------------- E3: mechanism on full runs
def mechanism_e3(store, missing, seed=0):
    """Share of the Q->P penalty over P->Q (perplexity or KL of full sequential runs) removed by each arm,
    the model-level counterpart of E2's layer-error shares; and the stale grid imposed on P->Q."""
    rows = []
    for model in store.models():
        for s, b in store.cells(model, seed, ARMS):
            qp, pq = store.get(model, "q_then_p", s, b, seed), store.get(model, "p_then_q", s, b, seed)
            row = {"model": model, "cell": cell_label(s, b), "sparsity": s, "bits": b, "seed": seed}
            for arm in ARMS:
                if store.get(model, arm, s, b, seed) is None:
                    missing.add("E3 mechanism (full runs)", f"{model} {cell_label(s, b)} {ORDER_TXT[arm]}",
                                "not available yet" if store.is_planned(model, arm, s, b, seed) else "not planned")
            for metric, get in (("wt_ppl", lambda r: (r.get("wikitext2") or {}).get("ppl")),
                                ("c4_ppl", lambda r: (r.get("c4") or {}).get("ppl")),
                                ("wt_kl", lambda r: (r.get("wikitext2") or {}).get("kl"))):
                vqp, vpq = (get(qp) if qp else None), (get(pq) if pq else None)
                row[f"{metric}_q_then_p"], row[f"{metric}_p_then_q"] = vqp, vpq
                pen = (vqp - vpq) if finite(vqp) and finite(vpq) else None
                row[f"{metric}_penalty"] = pen
                for arm in MECH_ARMS:
                    r = store.get(model, arm, s, b, seed)
                    v = get(r) if r else None
                    row[f"{metric}_{arm}"] = v
                    row[f"{metric}_removed_{arm}_pct"] = 100 * (vqp - v) / pen if (pen and pen > 0 and finite(v)) else None
                r = store.get(model, "p_then_q_stale", s, b, seed)
                v = get(r) if r else None
                row[f"{metric}_p_then_q_stale"] = v
                row[f"{metric}_stale_on_pq_pct"] = 100 * (v / vpq - 1) if finite(v) and finite(vpq) and vpq else None
            rows.append(row)
    return rows


SEED_BITS = (4, 3)
SEED_PAIRS = (("joint", "p_then_q"), ("joint", "q_then_p"), ("p_then_q", "q_then_p"))


def seed_models(store):
    """Models planned with calibration seeds > 0 (E5), in the order of store.models()."""
    multi = {v["short"] for v in store.plan.values() if v["seed"] > 0}
    return [m for m in store.models() if m in multi]


def model_seeds(store, m):
    return sorted({0} | {v["seed"] for v in store.plan.values() if v["short"] == m})


def _wt_ppl(store, m, o, s, b, sd):
    return ((store.get(m, o, s, b, sd) or {}).get("wikitext2") or {}).get("ppl")


def _seed_cells(store, m, bits):
    """The (s, b) cells of a multi-seed model at these bit-widths (every seed-0 cell, E1 and E4)."""
    return [(s, b) for s, b in store.cells(m, 0, PAPER_ORDERS) if b in bits]


def _e5_cells(store, m):
    return {(round(v["sparsity"], 4), v["bits"]) for v in store.plan.values()
            if v["short"] == m and v["seed"] > 0 and v["order"] in PAPER_ORDERS}


def seed_stability(store, bits=SEED_BITS):
    """Cells of the models planned with several calibration seeds (E5), 4- and 3-bit: the best and the worst order
    by WikiText-2 ppl for every seed, whether the best is the same for all of them (macro R30) and whether Q->P is
    strictly the worst in every seed. complete = every seed has the three orders."""
    rows = []
    for m in seed_models(store):
        seeds, e5 = model_seeds(store, m), _e5_cells(store, m)
        for s, b in _seed_cells(store, m, bits):
            best, worst = {}, {}
            for sd in seeds:
                vals = {o: _wt_ppl(store, m, o, s, b, sd) for o in PAPER_ORDERS}
                if all(finite(v) for v in vals.values()):
                    best[sd] = min(PAPER_ORDERS, key=lambda o: (vals[o], PAPER_ORDERS.index(o)))
                    worst[sd] = "q_then_p" if vals["q_then_p"] > max(vals["joint"], vals["p_then_q"]) else \
                        max(PAPER_ORDERS, key=lambda o: (vals[o], -PAPER_ORDERS.index(o)))
            complete = len(best) == len(seeds)
            rows.append({"model": m, "bits": b, "cell": cell_label(s, b), "sparsity": s, "e5_cell": (s, b) in e5,
                         "seeds": " ".join(map(str, seeds)), "seeds_present": " ".join(map(str, sorted(best))),
                         "best_by_seed": " ".join(f"{sd}:{ORDER_TXT[o]}" for sd, o in sorted(best.items())),
                         "worst_by_seed": " ".join(f"{sd}:{ORDER_TXT[o]}" for sd, o in sorted(worst.items())),
                         "complete": complete, "stable": complete and len(set(best.values())) == 1,
                         "qp_worst_all_seeds": complete and all(o == "q_then_p" for o in worst.values())})
    return rows


def seed_pairs(store, bits=SEED_BITS):
    """E5 paired comparisons of the orders over calibration seeds (4- and 3-bit cells of the multi-seed models).

    Per (model, cell, pair of orders): the per-seed log ratio of WikiText-2 perplexity (> 0 = the second order
    is better), its mean and sample sd, a paired t-test over seeds (scipy ttest_rel on log ppl; with 3 seeds it
    has 2 degrees of freedom), and whether the sign is the same in every seed. Per (model, cell, seed): whether
    joint is the best of the three orders and whether it beats P->Q (the draft's "joint best in 11 of 12";
    among the mask-first orders, joint vs P->Q), the worst order and whether Q->P is strictly the worst;
    cell_complete = every seed of the cell has the three orders.
    """
    import math
    import statistics as st
    if isinstance(bits, int):
        bits = (bits,)
    rows, best = [], []
    for m in seed_models(store):
        seeds, e5 = model_seeds(store, m), _e5_cells(store, m)
        for s, b in _seed_cells(store, m, bits):
            ppl = {(o, sd): _wt_ppl(store, m, o, s, b, sd) for o in PAPER_ORDERS for sd in seeds}
            cell_best = []
            for sd in seeds:
                v = {o: ppl[(o, sd)] for o in PAPER_ORDERS}
                if all(finite(x) for x in v.values()):
                    bo = min(PAPER_ORDERS, key=lambda o: (v[o], PAPER_ORDERS.index(o)))
                    qw = v["q_then_p"] > max(v["joint"], v["p_then_q"])
                    wo = "q_then_p" if qw else max(PAPER_ORDERS, key=lambda o: (v[o], -PAPER_ORDERS.index(o)))
                    cell_best.append({"model": m, "bits": b, "cell": cell_label(s, b), "seed": sd, "best": bo, "worst": wo,
                                      "joint_best": bo == "joint", "joint_beats_pq": v["joint"] < v["p_then_q"],
                                      "qp_worst": qw, "e5_cell": (s, b) in e5})
            for x in cell_best:
                x["cell_complete"] = len(cell_best) == len(seeds)
            best += cell_best
            for a, c in SEED_PAIRS:
                d = [math.log(ppl[(c, sd)] / ppl[(a, sd)]) for sd in seeds if finite(ppl[(a, sd)]) and finite(ppl[(c, sd)])]
                row = {"model": m, "bits": b, "cell": cell_label(s, b), "order_a": a, "order_b": c, "n_seeds": len(d),
                       "complete": len(d) == len(seeds), "mean_log_ratio_b_over_a": st.mean(d) if d else None,
                       "sd_log_ratio": st.stdev(d) if len(d) > 1 else None,
                       "a_better_all_seeds": bool(d) and all(x > 0 for x in d), "b_better_all_seeds": bool(d) and all(x < 0 for x in d)}
                if len(d) >= 2:
                    from scipy import stats as sps
                    la = [math.log(ppl[(a, sd)]) for sd in seeds if finite(ppl[(a, sd)]) and finite(ppl[(c, sd)])]
                    lc = [math.log(ppl[(c, sd)]) for sd in seeds if finite(ppl[(a, sd)]) and finite(ppl[(c, sd)])]
                    t = sps.ttest_rel(lc, la)
                    row["t"], row["p_paired_t"] = float(t.statistic), float(t.pvalue)
                rows.append(row)
    return rows, best


def seed_ppl_orders(store, bits=SEED_BITS):
    """Mean and sample sd over calibration seeds of WikiText-2 ppl (also C4 ppl and KL), per model, bit-width, cell
    and order, for the E5 cells (and the quant-only state of each bit-width, cell = 'b-bit')."""
    import statistics as st
    rows = []
    for m in seed_models(store):
        seeds, e5 = model_seeds(store, m), _e5_cells(store, m)
        for b in bits:
            cfgs = [("quant", 0.0, b)] + [(o, s, bb) for s, bb in sorted(e5, key=cell_sort_key) if bb == b for o in PAPER_ORDERS]
            for o, s, bb in cfgs:
                if not any(store.is_planned(m, o, s, bb, sd) or store.get(m, o, s, bb, sd) for sd in seeds if sd > 0):
                    continue
                row = {"model": m, "bits": bb, "cell": f"{bb}-bit" if o == "quant" else cell_label(s, bb), "sparsity": s,
                       "order": o, "seeds_planned": " ".join(map(str, seeds))}
                lines = {sd: store.get(m, o, s, bb, sd) for sd in seeds}
                row["seeds_present"] = " ".join(str(sd) for sd, r in lines.items() if r is not None)
                for key, get in (("wt_ppl", lambda r: (r.get("wikitext2") or {}).get("ppl")),
                                 ("c4_ppl", lambda r: (r.get("c4") or {}).get("ppl")),
                                 ("wt_kl", lambda r: (r.get("wikitext2") or {}).get("kl"))):
                    v = {sd: get(r) for sd, r in lines.items() if r is not None and finite(get(r))}
                    row[f"{key}_n"] = len(v)
                    row[f"{key}_mean"] = st.mean(v.values()) if v else None
                    row[f"{key}_sd"] = st.stdev(v.values()) if len(v) > 1 else None
                    if key == "wt_ppl":
                        row["wt_ppl_sd_pct"] = 100 * row["wt_ppl_sd"] / row["wt_ppl_mean"] if finite(row["wt_ppl_sd"]) else None
                        for sd in seeds:
                            row[f"wt_ppl_seed{sd}"] = v.get(sd)
                row["complete"] = row["wt_ppl_n"] == len(seeds)
                rows.append(row)
    return rows


def seed_overview(store, stability, pairs, best, e1_seeds):
    """Counts for the E5 macros, per scope (every multi-seed model pooled = 'all', or one model) and bit-width
    (4, 3, or 'both'). Cells count only when every seed has the three orders (complete); (cell, seed) pairs only
    in complete cells; the seed sd only for configurations with three seeds."""
    out = []
    for m in [None] + seed_models(store):
        for b in (4, 3, None):
            def ok(r):
                return (m is None or r["model"] == m) and (b is None or int(r["bits"]) == b)
            S_ = [r for r in stability if ok(r) and r["complete"]]
            runs = [r for r in best if ok(r) and r["cell_complete"]]
            row = {"model": m or "all", "bits": b or "both", "cells": len(S_), "best_stable": sum(r["stable"] for r in S_),
                   "qp_worst_all_seeds": sum(r["qp_worst_all_seeds"] for r in S_), "runs": len(runs),
                   "runs_qp_worst": sum(r["qp_worst"] for r in runs), "runs_joint_best": sum(r["joint_best"] for r in runs),
                   "runs_joint_beats_pq": sum(r["joint_beats_pq"] for r in runs),
                   "models_complete": " ".join(x for x in seed_models(store) if any(r["model"] == x for r in S_))}
            for a, c in SEED_PAIRS:
                rr = [r for r in pairs if ok(r) and r["order_a"] == a and r["order_b"] == c and r["complete"]]
                row[f"pair_{a}_{c}_cells"] = len(rr)
                row[f"pair_{a}_{c}_a_all_seeds"] = sum(r["a_better_all_seeds"] for r in rr)
                row[f"pair_{a}_{c}_sig"] = sum(finite(r.get("p_paired_t")) and r["p_paired_t"] < 0.05 for r in rr)
            for name, sel in (("single", ("prune", "quant")), ("combined", COMBINED), ("any", ("prune", "quant") + COMBINED)):
                v = [100 * r["wt_ppl_sd"] / r["wt_ppl_mean"] for r in e1_seeds if r["order"] in sel and r["n"] >= 3
                     and (m is None or r["model"] == m) and (b is None or int(r["bits"]) == b)
                     and finite(r.get("wt_ppl_sd")) and finite(r.get("wt_ppl_mean"))]
                row[f"sd_max_{name}_pct"] = max(v) if v else None
            states = [r for r in store.states.values() if int(r.get("seed", 0)) > 0 and (m is None or short(r["model"]) == m)
                      and (b is None or int(r["bits"]) == b)]
            planned = [v for v in store.plan.values() if v["seed"] > 0 and (m is None or v["short"] == m)
                       and (b is None or v["bits"] == b)]
            row["states"], row["states_planned"] = len(states), len(planned)
            out.append(row)
    return out


def write(res, paths):
    R = paths.reports
    write_csv(os.path.join(R, "seed_order_stability.csv"), res["seed_stability"])
    write_csv(os.path.join(R, "seed_order_pairs.csv"), res["seed_pairs"])
    write_csv(os.path.join(R, "seed_best_order.csv"), res["seed_best"])
    write_csv(os.path.join(R, "seed_ppl_orders.csv"), res["seed_ppl"])
    write_csv(os.path.join(R, "seed_overview.csv"), res["seed_overview"])
    write_csv(os.path.join(R, "q1_layer_interaction.csv"), res["q1"])
    write_csv(os.path.join(R, "q2_order_ranking.csv"), res["q2"])
    write_csv(os.path.join(R, "q3_order_spread.csv"), res["q3"])
    write_csv(os.path.join(R, "e6_scale_spread.csv"), res["scale"])
    write_csv(os.path.join(R, "scale_family_cells.csv"), res["scale_cells"])
    write_csv(os.path.join(R, "scale_matched.csv"), res["scale_matched"])
    write_csv(os.path.join(R, "fig_order_scale.csv"), res["fig_scale"])
    write_csv(os.path.join(R, "fp16_vs_fp32.csv"), res["fp"])
    write_csv(os.path.join(R, "e3_mechanism.csv"), res["e3"])


def analyze(store, e1, paths, missing):
    res = {"q1": q1(store, missing)}
    res["q1_summary"] = q1_summary(res["q1"])
    res["q2"] = q2(store, e1, missing)
    res["q2_summary"] = q2_summary(res["q2"])
    res["q2_per_model"] = q2_per_model(res["q2"])
    res["q3"] = q3(res["q2"])
    res["scale_cells"] = scale_cells(store, res["q2"])
    res["scale"] = scale_fit(res["scale_cells"])
    res["scale_matched"] = matched_scale(res["scale_cells"])
    res["fig_scale"] = fig_scale_rows(res["scale_cells"])
    res["e3"] = mechanism_e3(store, missing)
    res["seed_stability"] = seed_stability(store)
    res["seed_pairs"], res["seed_best"] = seed_pairs(store)
    res["seed_ppl"] = seed_ppl_orders(store)
    res["seed_overview"] = seed_overview(store, res["seed_stability"], res["seed_pairs"], res["seed_best"], e1.get("seeds", []))
    res["fp"], v1 = fp_compare(store, paths, missing)
    res["fp_summary"] = fp_summary(res["fp"])
    res["v1_check"] = v1["check"] if v1 else None
    res["v1_files"] = v1["files"] if v1 else []
    return res
