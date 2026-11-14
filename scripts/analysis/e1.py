"""E1 statistics per model and state (and every later family that adds states: E3, E4, E5, E6).

For every state: WikiText-2 and C4 perplexity, KL to the dense model and top-1 agreement on both (from the
result line); accuracy per task from the per-item files with the v2 main metric and the paper's metric;
Sr (3 paper tasks, both metrics), Sr-extended (9 zero-shot tasks), chance-normalized Sr; flips against the
dense model per task and pooled, split into correct->wrong and wrong->correct; I_Sr for every combined state.

Intervals: paired bootstrap over items (stats.PairedBootstrap, 10,000 resamples by default, fixed seed,
items resampled within each task) for accuracy differences (state vs dense, order vs order in a cell), Sr,
Sr differences between orders and I_Sr. Exact McNemar tests between the orders of a cell, per task and
pooled, with review_code/ablation/extra_orders.mcnemar.
"""
import statistics as st

from . import stats as S
from .common import COMBINED, P, aligned_columns, cell_label, finite, short, skey, task_metric, write_csv

DIFF_SETS = (("mean9_v2", tuple(P.ZERO_SHOT_TASKS), "v2"), ("mean3_v2", tuple(P.PAPER_TASKS), "v2"),
             ("mean3_paper", tuple(P.PAPER_TASKS), "paper"))


def state_row(store, sid):
    r = store.states[sid]
    wt, c4, sto = r.get("wikitext2") or {}, r.get("c4") or {}, r.get("storage") or {}
    return {"model": short(r["model"]), "state_id": sid, "family": r.get("family"), "order": r["order"],
            "sparsity": r["sparsity"], "bits": r["bits"], "seed": r.get("seed", 0), "label": store.label(sid),
            "wt_ppl": wt.get("ppl"), "wt_nll_sum": wt.get("nll_sum"), "wt_kl": wt.get("kl"), "wt_top1": wt.get("top1"),
            "c4_ppl": c4.get("ppl"), "c4_kl": c4.get("kl"), "c4_top1": c4.get("top1"),
            "bits_tcr": sto.get("tcr"), "bits_bitmap": sto.get("bitmap"), "bits_entropy_actual": sto.get("entropy_actual"),
            "bits_symbol_entropy": sto.get("symbol_entropy"), "group_overhead": sto.get("group_overhead"),
            "mask_frac_actual": sto.get("mask_frac_actual"), "zero_frac_actual": sto.get("zero_frac_actual"),
            "mean_rel_err": r.get("mean_rel_err"), "n_layers": len(r.get("diags") or []),
            "has_items": store.items(sid) is not None, "source": store.source[sid]}


def _sort_key(store, sid):
    r = store.states[sid]
    rank = {"dense": 0, "quant": 1, "prune": 2}
    return (int(r.get("seed", 0)), rank.get(r["order"], 3), -int(r["bits"]), float(r["sparsity"]),
            COMBINED.index(r["order"]) if r["order"] in COMBINED else 99)


def _diffs(bs, a, b, base):
    """Paired accuracy differences a - b: per task (v2 metric) and the three task means of DIFF_SETS."""
    rows = []
    for t in P.ZERO_SHOT_TASKS:
        m = task_metric(t, "v2")
        try:
            pt, lo, hi = bs.stat(lambda g: g(t, (a, m)) - g(t, (b, m)))
        except S.MissingData:
            continue
        rows.append(dict(base, task=t, metric=m, diff=pt, lo=lo, hi=hi))
    for name, tasks, kind in DIFF_SETS:
        try:
            pt, lo, hi = bs.stat(lambda g: S.acc_mean(g, a, tasks, kind) - S.acc_mean(g, b, tasks, kind))
        except S.MissingData:
            continue
        rows.append(dict(base, task=name, metric=kind, diff=pt, lo=lo, hi=hi))
    return rows


def _missing_single(store, model, order, s, b, seed):
    if store.is_planned(model, order, s, b, seed):
        return "not available yet"
    return "not in the study plan (scripts/make_queues.py)"


def analyze(store, boolq_rate, missing, B=S.BOOT_B, seed=S.BOOT_SEED):
    chance = S.chance_levels(boolq_rate)
    out = {k: [] for k in ("states", "tasks", "sr", "flips", "isr", "acc_vs_dense", "acc_orders", "sr_orders",
                           "mcnemar")}
    out.update(checks={"item_vs_aggregate_max_abs": None, "item_vs_aggregate_n": 0, "doc_id_notes": [],
                       "flips_vs_extra_orders_max_abs": None},
               boot={"B": int(B), "seed": int(seed), "ci": list(S.CI)}, chance=chance, by_sid={})
    if boolq_rate is None:
        missing.add("E1 Sr", "chance-normalized Sr (sr9_cn)", "BoolQ majority-class rate unavailable")
    for model in store.models():
        sids = sorted(store.model_states(model), key=lambda x: _sort_key(store, x))
        for sid in sids:
            out["states"].append(state_row(store, sid))
            out["by_sid"][sid] = {"sr": {}, "isr": {}, "flips": {}}
        _model(store, model, sids, chance, missing, out, B, seed)
    out["seeds"] = seed_summary(store)
    return out


def _model(store, model, sids, chance, missing, out, B, seed):
    with_items = []
    for sid in sids:
        r = store.states[sid]
        if store.items(sid) is not None:
            with_items.append(sid)
        elif r.get("items_file"):
            missing.add("E1 tasks", sid, store.item_problems.get(sid, "items unavailable"))
        elif r.get("family") != "DET":
            missing.add("E1 tasks", sid, "no task evaluation in this state")
    if not with_items:
        return
    cols, notes = aligned_columns(store, with_items, P.ZERO_SHOT_TASKS)
    out["checks"]["doc_id_notes"] += notes
    bs = S.PairedBootstrap(cols, B=B, seed=seed)
    ck = out["checks"]
    for sid in with_items:
        items, agg = store.items(sid), store.states[sid].get("tasks") or {}
        for t in P.ZERO_SHOT_TASKS:
            if t not in items:
                missing.add("E1 tasks", f"{sid} {t}", "task absent from the items file")
                continue
            row = {"model": model, "state_id": sid, "task": t, "n_items": len(items[t]["doc_id"])}
            for m in ("acc", "acc_norm"):
                vals = items[t].get(m)
                row[m] = sum(vals) / len(vals) if vals else None
                a = (agg.get(t) or {}).get(f"{m},none")
                row[f"agg_{m}"] = a
                if row[m] is not None and finite(a):
                    d = abs(row[m] - a)
                    ck["item_vs_aggregate_max_abs"] = max(ck["item_vs_aggregate_max_abs"] or 0.0, d)
                    ck["item_vs_aggregate_n"] += 1
            row["v2_metric"] = task_metric(t, "v2")
            row["v2"] = row.get(row["v2_metric"])
            pm = task_metric(t, "paper")
            row["paper_metric"], row["paper"] = pm, (row.get(pm) if pm else None)
            out["tasks"].append(row)

    dense = store.dense(model)
    dsid = dense["state_id"] if dense else None
    for row in out["tasks"]:  # chance-normalized Sr needs dense headroom above chance on every task
        c = chance.get(row["task"])
        if dsid and row["state_id"] == dsid and c is not None and finite(row["v2"]) and row["v2"] <= c:
            ck.setdefault("dense_at_or_below_chance", []).append(
                {"model": model, "task": row["task"], "dense_v2": row["v2"], "chance": c})
    if dsid is None or dsid not in with_items:
        missing.add("E1 Sr/flips/I_Sr", f"{model}: every state", "dense state or its items not available yet")
        dsid = None
    if dsid:
        dbin = store.binary(dsid, "v2")
        for sid in with_items:
            for name, spec in S.SR_SETS.items():
                try:
                    pt, lo, hi = bs.stat(lambda g: S.sr(g, sid, dsid, spec, chance))
                except S.MissingData as e:
                    missing.add("E1 Sr", f"{sid} {name}", f"no data for {e.args[0]}")
                    continue
                out["sr"].append({"model": model, "state_id": sid, "set": name, "value": pt, "lo": lo, "hi": hi})
                out["by_sid"][sid]["sr"][name] = (pt, lo, hi)
            sbin = store.binary(sid, "v2")
            fc = S.flip_counts(sbin, dbin, P.ZERO_SHOT_TASKS)
            fc["pooled3"] = S.flip_counts(sbin, dbin, P.PAPER_TASKS)["pooled"]
            ref = S.flips_reference(sbin, dbin, P.ZERO_SHOT_TASKS)["pooled"]
            if finite(ref) and finite(fc["pooled"]["flips"]):
                d = abs(ref - fc["pooled"]["flips"])
                ck["flips_vs_extra_orders_max_abs"] = max(ck["flips_vs_extra_orders_max_abs"] or 0.0, d)
            for t, v in fc.items():
                out["flips"].append(dict({"model": model, "state_id": sid, "task": "pooled9" if t == "pooled" else t}, **v))
            out["by_sid"][sid]["flips"] = {"pooled9": fc["pooled"], "pooled3": fc["pooled3"]}
            if sid != dsid:
                out["acc_vs_dense"] += _diffs(bs, sid, dsid, {"model": model, "state_id": sid, "dense": dsid})

    # I_Sr for every combined state
    for sid in with_items:
        r = store.states[sid]
        if r["order"] not in COMBINED:
            continue
        s, b, sd = r["sparsity"], r["bits"], int(r.get("seed", 0))
        pr, qr = store.prune_only(model, s, sd), store.quant_only(model, b, sd)
        if pr is None or qr is None:
            what = []
            if pr is None:
                what.append(f"prune-only {round(s * 100)}% seed {sd}: " + _missing_single(store, model, "prune", s, 16, sd))
            if qr is None:
                what.append(f"quant-only {b}-bit seed {sd}: " + _missing_single(store, model, "quant", 0.0, b, sd))
            missing.add("E1 I_Sr", sid, "; ".join(what))
            continue
        if dsid is None or pr["state_id"] not in with_items or qr["state_id"] not in with_items:
            missing.add("E1 I_Sr", sid, "dense, prune-only or quant-only items not available")
            continue
        psid, qsid = pr["state_id"], qr["state_id"]
        for name, spec in S.SR_SETS.items():
            try:
                pt, lo, hi = bs.stat(lambda g: S.isr(g, sid, psid, qsid, dsid, spec, chance))
                parts = [bs.value(lambda g, x=x: S.sr(g, x, dsid, spec, chance)) for x in (sid, psid, qsid)]
            except S.MissingData as e:
                missing.add("E1 I_Sr", f"{sid} {name}", f"no data for {e.args[0]}")
                continue
            out["isr"].append({"model": model, "state_id": sid, "set": name, "value": pt, "lo": lo, "hi": hi,
                               "sr_pq": parts[0], "sr_p": parts[1], "sr_q": parts[2], "prune_state": psid,
                               "quant_state": qsid, "dense_state": dsid})
            out["by_sid"][sid]["isr"][name] = (pt, lo, hi)

    # orders within a cell: accuracy and Sr differences (bootstrap) and exact McNemar
    cells = {}
    for sid in with_items:
        r = store.states[sid]
        if r["order"] in COMBINED:
            cells.setdefault((skey(r["sparsity"]), int(r["bits"]), int(r.get("seed", 0))), {})[r["order"]] = sid
    for (s, b, sd), by_order in sorted(cells.items()):
        orders = [o for o in COMBINED if o in by_order]
        for i, oa in enumerate(orders):
            for ob in orders[i + 1:]:
                a, c = by_order[oa], by_order[ob]
                base = {"model": model, "cell": cell_label(s, b), "sparsity": s, "bits": b, "seed": sd,
                        "order_a": oa, "order_b": ob, "state_a": a, "state_b": c}
                out["acc_orders"] += _diffs(bs, a, c, base)
                if dsid:
                    for name, spec in S.SR_SETS.items():
                        try:
                            pt, lo, hi = bs.stat(lambda g: S.sr(g, a, dsid, spec, chance) - S.sr(g, c, dsid, spec, chance))
                        except S.MissingData:
                            continue
                        out["sr_orders"].append(dict(base, set=name, diff=pt, lo=lo, hi=hi))
                ia, ic = store.binary(a, "v2"), store.binary(c, "v2")
                m9 = S.mcnemar(ia, ic, P.ZERO_SHOT_TASKS)
                m3 = S.mcnemar(ia, ic, P.PAPER_TASKS)["pooled"]
                for t, (x, y, p) in list(m9.items()) + [("pooled3", m3)]:
                    out["mcnemar"].append(dict(base, task="pooled9" if t == "pooled" else t, a_only=x, b_only=y, p=p))


def seed_summary(store):
    """Mean and sample standard deviation over calibration seeds, for every configuration with 2+ seeds."""
    groups = {}
    for sid, r in store.states.items():
        k = (short(r["model"]), r["order"], skey(r["sparsity"]), int(r["bits"]))
        groups.setdefault(k, []).append(r)
    rows = []
    for (m, o, s, b), rs in sorted(groups.items()):
        if o == "dense" or len(rs) < 2:
            continue
        rs = sorted(rs, key=lambda r: int(r.get("seed", 0)))
        row = {"model": m, "order": o, "sparsity": s, "bits": b, "seeds": " ".join(str(int(r.get("seed", 0))) for r in rs),
               "n": len(rs)}
        for name, get in (("wt_ppl", lambda r: (r.get("wikitext2") or {}).get("ppl")),
                          ("c4_ppl", lambda r: (r.get("c4") or {}).get("ppl")),
                          ("wt_kl", lambda r: (r.get("wikitext2") or {}).get("kl"))):
            v = [get(r) for r in rs]
            v = [x for x in v if finite(x)]
            row[f"{name}_mean"] = st.mean(v) if v else None
            row[f"{name}_sd"] = st.stdev(v) if len(v) > 1 else None
            row[f"{name}_values"] = " ".join(repr(x) for x in v)
        rows.append(row)
    return rows


def write(out, paths):
    import os
    R = paths.reports
    states = []
    for row in out["states"]:
        x = dict(row)
        b = out["by_sid"].get(row["state_id"], {})
        for name in S.SR_SETS:
            v = b.get("sr", {}).get(name)
            x[name], x[f"{name}_lo"], x[f"{name}_hi"] = v if v else (None, None, None)
        f9 = b.get("flips", {}).get("pooled9") or {}
        x["flips9"], x["flips9_cw"], x["flips9_wc"], x["flips9_n"] = f9.get("flips"), f9.get("cw"), f9.get("wc"), f9.get("n")
        f3 = b.get("flips", {}).get("pooled3") or {}
        x["flips3"] = f3.get("flips")
        states.append(x)
    write_csv(os.path.join(R, "e1_states.csv"), states)
    for name in ("tasks", "sr", "flips", "isr", "acc_vs_dense", "acc_orders", "sr_orders", "mcnemar", "seeds"):
        write_csv(os.path.join(R, f"e1_{name}.csv"), out[name])
