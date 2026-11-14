"""BATCH check (audit finding 17, DECISIONS.md #38): do the lm-eval batch sizes change any result?

The four states whose benchmarks first ran with one lm-eval batch size (SmolLM2-135M dense, 4-bit and 50%+4 P->Q;
Qwen2.5-0.5B dense; scripts/make_queues.py batch()) were re-run with the batch groups used for every later state.
The re-runs are not study states (Store ignores the BATCH family); they are read here directly:
  results/v2/BATCH/<model>.jsonl                  one line per re-run state (same state_id as in E1)
  results/v2/items/<state_id>__BATCH.json.gz      its per-item results (the study file keeps its own name)
Per state, against the E1 line of the same state_id (as scripts/img_check.py compares lines):
  likelihood   WikiText-2 (and C4) NLL sum and perplexity identical? every layer's rel_err identical?
  items        per task and pooled over the 9 zero-shot tasks: items whose correctness differs between the two
               runs (v2 main metric, as the flips use; also whether acc or acc_norm differs)
Then the pooled 9-task flips of every E1/E4 state of the models with a BATCH dense line, recomputed against the
BATCH dense reference instead of the E1 one: change in percentage points, and whether the ranking of the three
orders by flips changes in any cell. Variant "both": the re-run items are also used for the re-run compressed
states (4-bit, 50%+4 P->Q). The dense state itself is left out of the flip changes (its own differences are the
item counts above).
"""
import os

from . import stats as S
from .common import (MQ, P, PAPER_ORDERS, binary_items, cell_label, finite, read_items_file, short, skey, state_id,
                     task_metric, write_csv, write_json)

FAMILY = "BATCH"
FLIP_FAMILIES = ("E1", "E4")


def planned():
    """[(model short, state_id, unit)] of the BATCH plan (scripts/make_queues.py, CHECKS)."""
    out = []
    for model, units in MQ.CHECKS[FAMILY]().items():
        for u in units:
            out.append((short(model), state_id(model, u, "float32"), u))
    return out


def load_lines(paths):
    """{state_id: (line, file)} of the usable BATCH lines (status ok; the first line of a state wins)."""
    root = os.path.join(paths.results_v2, FAMILY)
    out, dups = {}, 0
    if not os.path.isdir(root):
        return out, dups
    for f in sorted(os.listdir(root)):
        if not f.endswith(".jsonl") or "failures" in f:
            continue
        for r in P.read_jsonl(os.path.join(root, f)):
            if not isinstance(r, dict) or r.get("status") != "ok" or "state_id" not in r:
                continue
            if r["state_id"] in out:
                dups += 1
                continue
            out[r["state_id"]] = (r, paths.rel(os.path.join(root, f)))
    return out, dups


def _same(a, b, block, key):
    x, y = (a.get(block) or {}).get(key), (b.get(block) or {}).get(key)
    return (x == y) if (x is not None and y is not None) else None


def compare_items(new, old, tasks=None):
    """Per task: items common to both runs, items whose correctness differs (v2 main metric; acc or acc_norm)."""
    rows = []
    for t in tasks or P.ZERO_SHOT_TASKS:
        if t not in new or t not in old:
            continue
        pos = {d: i for i, d in enumerate(new[t]["doc_id"])}
        common = [(pos[d], j) for j, d in enumerate(old[t]["doc_id"]) if d in pos]
        m = task_metric(t, "v2")
        row = {"task": t, "metric": m, "n": len(common),
               "n_only_one": len(set(new[t]["doc_id"]) ^ set(old[t]["doc_id"]))}
        row["differ"] = sum(round(new[t][m][i]) != round(old[t][m][j]) for i, j in common)
        row["differ_any"] = sum(any(round(new[t][k][i]) != round(old[t][k][j]) for k in ("acc", "acc_norm")
                                    if k in new[t] and k in old[t]) for i, j in common)
        rows.append(row)
    return rows


def _flips(sbin, dbin):
    return S.flip_counts(sbin, dbin, P.ZERO_SHOT_TASKS)["pooled"]["flips"] if sbin and dbin else None


def analyze(store, paths, missing):
    lines, dups = load_lines(paths)
    states, tasks, batch_items = [], [], {}
    for model, sid, u in planned():
        label = f"{model} {store.label(sid)}"
        row = {"model": model, "state_id": sid, "label": store.label(sid), "order": u["order"], "sparsity": float(u["sparsity"]),
               "bits": int(u["bits"]), "available": False}
        new, src = lines.get(sid, (None, None))
        old = store.states.get(sid)
        if new is None:
            missing.add("BATCH check", label, "BATCH line not available yet")
            states.append(row)
            continue
        row.update(available=True, source=src, e1_source=store.source.get(sid),
                   batch_eval_batch=new.get("eval_batch"), e1_eval_batch=(old or {}).get("eval_batch"))
        if old is None:
            missing.add("BATCH check", label, "no E1 line with this state_id")
            states.append(row)
            continue
        row["wt_nll_identical"] = _same(new, old, "wikitext2", "nll_sum")
        row["wt_ppl_identical"] = _same(new, old, "wikitext2", "ppl")
        row["c4_nll_identical"] = _same(new, old, "c4", "nll_sum")
        a, b = (new.get("wikitext2") or {}).get("nll_sum"), (old.get("wikitext2") or {}).get("nll_sum")
        row["wt_nll_abs_diff"] = abs(a - b) if finite(a) and finite(b) else None
        la = {(d["layer"], d["name"]): d["rel_err"] for d in new.get("diags") or []}
        lb = {(d["layer"], d["name"]): d["rel_err"] for d in old.get("diags") or []}
        row["n_layers"] = len(lb)
        row["layers_identical"] = (set(la) == set(lb) and all(la[k] == lb[k] for k in la)) if (la or lb) else None
        items, problem = read_items_file(os.path.join(paths.root, new["items_file"]) if new.get("items_file") else None,
                                         new.get("items_sha256"))
        e1_items = store.items(sid)
        if items is None or e1_items is None:
            missing.add("BATCH check", f"{label} items", problem if items is None else
                        store.item_problems.get(sid, "E1 items unavailable"))
            states.append(row)
            continue
        batch_items[sid] = items
        cmp = compare_items(items, e1_items)
        for t in cmp:
            tasks.append(dict(t, model=model, state_id=sid, label=row["label"], share=t["differ"] / t["n"] if t["n"] else None))
        row.update(items_n=sum(t["n"] for t in cmp), items_differ=sum(t["differ"] for t in cmp),
                   items_differ_any=sum(t["differ_any"] for t in cmp), tasks=len(cmp),
                   items_only_one=sum(t["n_only_one"] for t in cmp))
        row["items_share"] = row["items_differ"] / row["items_n"] if row["items_n"] else None
        states.append(row)
    flips, cells = flips_vs_batch_dense(store, batch_items, missing)
    return {"states": states, "tasks": tasks, "flips": flips, "cells": cells, "duplicates": dups,
            "summary": summarize(states, tasks, flips, cells)}


def flips_vs_batch_dense(store, batch_items, missing):
    """Pooled 9-task flips of every E1/E4 state against the E1 dense reference and against the BATCH dense one."""
    rows, cells = [], []
    for model in sorted({m for m, _, _ in planned()}, key=store.models().index):
        dense = store.dense(model)
        dsid = dense["state_id"] if dense else None
        if dsid not in batch_items or store.items(dsid) is None:
            missing.add("BATCH check", f"{model} flips vs the BATCH dense reference", "BATCH or E1 dense items not available yet")
            continue
        d_e1, d_b = store.binary(dsid, "v2"), binary_items(batch_items[dsid], "v2")
        by_cell = {}
        for sid in sorted(store.model_states(model)):
            r = store.states[sid]
            if sid == dsid or r.get("family") not in FLIP_FAMILIES or store.items(sid) is None:
                continue
            sbin = store.binary(sid, "v2")
            sb2 = binary_items(batch_items[sid], "v2") if sid in batch_items else sbin
            f_e1, f_b, f_bb = _flips(sbin, d_e1), _flips(sbin, d_b), _flips(sb2, d_b)
            row = {"model": model, "state_id": sid, "label": store.label(sid), "family": r.get("family"), "order": r["order"],
                   "sparsity": r["sparsity"], "bits": r["bits"], "rerun_items_used": sid in batch_items,
                   "flips_e1_dense": f_e1, "flips_batch_dense": f_b, "flips_both": f_bb,
                   "change_pp": 100 * (f_b - f_e1) if finite(f_b) and finite(f_e1) else None,
                   "change_both_pp": 100 * (f_bb - f_e1) if finite(f_bb) and finite(f_e1) else None}
            rows.append(row)
            if r["order"] in PAPER_ORDERS and int(r.get("seed", 0)) == 0:
                by_cell.setdefault((skey(r["sparsity"]), int(r["bits"])), {})[r["order"]] = row
        from .questions import _ranking
        for (s, b), d in sorted(by_cell.items(), key=lambda kv: (-kv[0][1], kv[0][0])):
            if len(d) != 3:
                continue
            c = {"model": model, "cell": cell_label(s, b), "sparsity": s, "bits": b}
            for key, col in (("e1", "flips_e1_dense"), ("batch", "flips_batch_dense"), ("both", "flips_both")):
                c[f"ranking_{key}"] = _ranking({o: d[o][col] for o in PAPER_ORDERS}, 1)[0]
            c["changes"] = c["ranking_batch"] != c["ranking_e1"]
            c["changes_both"] = c["ranking_both"] != c["ranking_e1"]
            cells.append(c)
    return rows, cells


def summarize(states, tasks, flips, cells):
    av = [r for r in states if r.get("items_n")]
    out = {"planned": len(states), "available": sum(r["available"] for r in states),
           "compared_items": len(av), "nll_compared": sum(r.get("wt_nll_identical") is not None for r in states),
           "nll_identical": sum(bool(r.get("wt_nll_identical")) for r in states),
           "items_differ": sum(r["items_differ"] for r in av), "items_n": sum(r["items_n"] for r in av),
           "items_differ_any": sum(r["items_differ_any"] for r in av)}
    out["items_share"] = out["items_differ"] / out["items_n"] if out["items_n"] else None
    for t in P.ZERO_SHOT_TASKS:
        T = [x for x in tasks if x["task"] == t]
        out[f"task_{t}_differ"], out[f"task_{t}_n"] = sum(x["differ"] for x in T), sum(x["n"] for x in T)
    for key, col in (("", "change_pp"), ("_both", "change_both_pp")):
        F = [r for r in flips if finite(r[col])]
        w = max(F, key=lambda r: abs(r[col])) if F else None
        out[f"flips_max_change{key}_pp"] = abs(w[col]) if w else None
        out[f"flips_max_change{key}_signed_pp"] = w[col] if w else None
        out[f"flips_max_change{key}_where"] = (f"{w['model']} {w['label']}" if w[col] != 0 else "no state changes") if w else None
        out[f"ranking_changes{key}"] = sum(c["changes" + key] for c in cells)
    out["flips_states"] = len(flips)
    out["flips_models"] = " ".join(sorted({r["model"] for r in flips}))
    out["ranking_cells"] = len(cells)
    return out


def write(res, paths):
    R = paths.reports
    write_csv(os.path.join(R, "batch_states.csv"), res["states"])
    write_csv(os.path.join(R, "batch_items.csv"), res["tasks"])
    write_csv(os.path.join(R, "batch_flips.csv"), res["flips"])
    write_csv(os.path.join(R, "batch_flip_rankings.csv"), res["cells"])
    write_json(os.path.join(R, "batch_summary.json"), res["summary"])

