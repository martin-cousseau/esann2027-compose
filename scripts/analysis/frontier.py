"""Storage frontier (E4): bits per weight under several accountings against WikiText-2 perplexity and KL.

Bits per weight come from each line's storage block (src_v2/pipeline.storage): tcr = b (1 - s), what the
paper's TCr counts; bitmap = tcr + 1 bit per weight for the mask when s > 0; entropy_actual = b * kept +
H2(mask fraction) per layer, size-weighted; symbol_entropy = plug-in entropy of the stored symbols per group
of 128 (credits rounding zeros too). The per-group scale/zero overhead (storage.group_overhead) is kept
separate. "Free zeros" = zero_frac_actual, the share of weights that are exactly zero after compression.

Also tests whether Q->P at 25% + 3-bit is the 3-bit model itself on every model (identical perplexity and
identical per-layer rel_err), as the paper observed in 16-bit.
"""
import os

from .common import COMBINED, PAPER_ORDERS, cell_label, finite, write_csv, write_json


def _point(store, sid, kind):
    r = store.states[sid]
    sto, wt, c4 = r.get("storage") or {}, r.get("wikitext2") or {}, r.get("c4") or {}
    return {"model": None, "state_id": sid, "kind": kind, "order": r["order"], "sparsity": r["sparsity"],
            "bits": r["bits"], "cell": cell_label(r["sparsity"], r["bits"]) if kind == "sparse+quant" else None,
            "bits_tcr": sto.get("tcr"), "bits_bitmap": sto.get("bitmap"), "bits_entropy_actual": sto.get("entropy_actual"),
            "bits_symbol_entropy": sto.get("symbol_entropy"), "group_overhead": sto.get("group_overhead"),
            "zero_frac_actual": sto.get("zero_frac_actual"), "mask_frac_actual": sto.get("mask_frac_actual"),
            "wt_ppl": wt.get("ppl"), "wt_kl": wt.get("kl"), "c4_ppl": c4.get("ppl"), "c4_kl": c4.get("kl"),
            "best_in_cell": None}


def analyze(store, missing, seed=0):
    points, best, identity = [], {}, []
    for model in store.models():
        d = store.dense(model)
        if d is not None:
            points.append(dict(_point(store, d["state_id"], "dense"), model=model))
        for b in store.quant_bits(model, seed):
            q = store.quant_only(model, b, seed)
            if q is None:
                missing.add("E4 frontier", f"{model} {b}-bit quant-only", "not available yet")
                continue
            points.append(dict(_point(store, q["state_id"], "quant"), model=model))
        for s, b in store.cells(model, seed):
            vals = {}
            for o in COMBINED:
                r = store.get(model, o, s, b, seed)
                if r is None:
                    continue
                p = dict(_point(store, r["state_id"], "sparse+quant"), model=model)
                points.append(p)
                if o in PAPER_ORDERS and finite(p["wt_ppl"]):
                    vals[o] = p
            if len(vals) == 3:
                o = min(PAPER_ORDERS, key=lambda x: (vals[x]["wt_ppl"], PAPER_ORDERS.index(x)))
                vals[o]["best_in_cell"] = True
                best[(model, s, b)] = vals[o]
            elif any(store.is_planned(model, o, s, b, seed) for o in PAPER_ORDERS):
                missing.add("E4 frontier", f"{model} {cell_label(s, b)} best order",
                            f"{len(vals)} of 3 paper orders available")
        identity.append(_identity(store, model, missing, seed))
    res = {"points": points, "best": best, "identity": identity}
    res["equal_storage"] = equal_storage(store, res, seed)
    return res


def _identity(store, model, missing, seed=0):
    qp, q3 = store.get(model, "q_then_p", 0.25, 3, seed), store.quant_only(model, 3, seed)
    row = {"model": model, "q_then_p_state": qp["state_id"] if qp else None, "quant3_state": q3["state_id"] if q3 else None}
    if qp is None or q3 is None:
        row["status"] = "pending"
        missing.add("Q->P 25%+3 = 3-bit identity", model, "Q->P 25%+3 or 3-bit state not available yet")
        return row
    a, b = qp.get("wikitext2") or {}, q3.get("wikitext2") or {}
    row["wt_ppl_qp"], row["wt_ppl_q3"] = a.get("ppl"), b.get("ppl")
    row["wt_ppl_equal"] = a.get("ppl") == b.get("ppl")
    row["wt_nll_sum_equal"] = a.get("nll_sum") == b.get("nll_sum")
    row["wt_ppl_abs_diff"] = abs(a["ppl"] - b["ppl"]) if finite(a.get("ppl")) and finite(b.get("ppl")) else None
    ca, cb = qp.get("c4") or {}, q3.get("c4") or {}
    row["c4_ppl_equal"] = (ca.get("ppl") == cb.get("ppl")) if (ca and cb) else None
    A = {(x["layer"], x["name"]): x["rel_err"] for x in qp.get("diags") or []}
    B = {(x["layer"], x["name"]): x["rel_err"] for x in q3.get("diags") or []}
    keys = sorted(set(A) & set(B))
    diffs = [abs(A[k] - B[k]) for k in keys if finite(A[k]) and finite(B[k])]
    row["n_layers"], row["n_layers_only_one"] = len(keys), len(set(A) ^ set(B))
    row["rel_err_max_abs_diff"] = max(diffs) if diffs else None
    row["rel_err_all_equal"] = bool(diffs) and all(x == 0 for x in diffs) and row["n_layers_only_one"] == 0
    za, zb = (qp.get("storage") or {}).get("zero_frac_actual"), (q3.get("storage") or {}).get("zero_frac_actual")
    row["zero_frac_qp"], row["zero_frac_q3"] = za, zb
    row["identical"] = bool(row["wt_ppl_equal"] and row["rel_err_all_equal"])
    row["status"] = "identical" if row["identical"] else "differs"
    return row


EQUAL_STORAGE = (("three", (0.5, 4), 3), ("four", (0.25, 4), 4))  # cell whose 1-bit-mask cost equals b-bit alone


def equal_storage(store, fr, seed=0):
    """Best order of 50%+4 against 3-bit alone, and of 25%+4 against 4-bit alone (macros R1-R5)."""
    rows = []
    for model in store.models():
        for tag, (s, b), qb in EQUAL_STORAGE:
            q, best = store.quant_only(model, qb, seed), fr["best"].get((model, s, b))
            qp = (q.get("wikitext2") or {}).get("ppl") if q else None
            bp = best["wt_ppl"] if best else None
            row = {"model": model, "comparison": tag, "cell": cell_label(s, b), "quant_bits": qb,
                   "best_order": best["order"] if best else None, "best_ppl": bp, "quant_ppl": qp,
                   "best_bits_mask": best["bits_bitmap"] if best else None,
                   "quant_bits_mask": (q.get("storage") or {}).get("bitmap") if q else None,
                   "available": finite(qp) and finite(bp)}
            row["gain_pct"] = 100 * (1 - bp / qp) if row["available"] else None
            row["gap_pct"] = 100 * (bp / qp - 1) if row["available"] else None
            rows.append(row)
    return rows


def write(res, paths):
    write_csv(os.path.join(paths.reports, "frontier_points.csv"), res["points"])
    write_csv(os.path.join(paths.reports, "frontier_equal_storage.csv"), res["equal_storage"])
    write_json(os.path.join(paths.reports, "identity_qp25_3bit.json"), res["identity"])
