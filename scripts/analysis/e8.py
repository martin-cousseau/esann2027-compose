"""E8 (analysis part): does a smaller layer error predict a better model? (the draft's open question)

For every compressed state (seed 0) the result line holds mean_rel_err, the mean over all compressed layers of
the per-layer relative error (layer diagnostics, as in the paper); within one model every state compresses the
same layers, so its rank equals the rank of the summed layer error. Two tests per model:
  across states  Spearman rho between mean_rel_err and the KL to the dense model (WikiText-2, also C4) over
                 every compressed state of the model (prune only, quant only, combined orders, E3 arms);
  within cells   for each (s, b) cell with the three paper orders, does the order with the smallest layer
                 error also have the smallest KL? And are the two rankings identical?
The fourth order of E8 (joint with re-fit) needs GPU runs and was not run (budget, DECISIONS.md).
Also (layer error alone, every model, KL not needed): per bit-width, the cells where the joint sweep has the
strictly smallest mean layer error of the three orders (the draft: "the joint sweep always has the smallest layer
error on 4-bit cells"); e8_layer_error_by_order.csv.
"""
import os

from .common import COMBINED, MQ, PAPER_ORDERS, cell_label, finite, short, write_csv, write_json


def _states(store, model, seed=0):
    out = []
    for sid, r in store.states.items():
        if short(r["model"]) != model or int(r.get("seed", 0)) != seed or r["order"] == "dense":
            continue
        e, kl, c4 = r.get("mean_rel_err"), (r.get("wikitext2") or {}).get("kl"), (r.get("c4") or {}).get("kl")
        if finite(e) and finite(kl):
            out.append({"model": model, "state_id": sid, "order": r["order"], "sparsity": r["sparsity"], "bits": r["bits"],
                        "mean_rel_err": e, "wt_kl": kl, "c4_kl": c4 if finite(c4) else None,
                        "wt_ppl": (r.get("wikitext2") or {}).get("ppl")})
    return out


def _spearman(x, y):
    pts = [(a, b) for a, b in zip(x, y) if finite(a) and finite(b)]
    if len(pts) < 3:
        return None, None, len(pts)
    from scipy import stats as sps
    res = sps.spearmanr([a for a, _ in pts], [b for _, b in pts])
    return float(res.statistic), float(res.pvalue), len(pts)


def analyze(store, missing, seed=0):
    rows, models, cells, errors = [], [], [], []
    for model in store.models():
        st = _states(store, model, seed)
        if not st:
            missing.add("E8 layer error vs KL", model, "no compressed state with layer diagnostics yet")
            continue
        rows += st
        out = {"model": model, "n_states": len(st)}
        for key in ("wt_kl", "c4_kl"):
            rho, p, n = _spearman([r["mean_rel_err"] for r in st], [r[key] for r in st])
            out.update({f"rho_{key}": rho, f"p_{key}": p, f"n_{key}": n})
        comb = [r for r in st if r["order"] in COMBINED]
        rho, p, n = _spearman([r["mean_rel_err"] for r in comb], [r["wt_kl"] for r in comb])
        out.update({"rho_wt_kl_combined": rho, "p_wt_kl_combined": p, "n_combined": n})
        by = {}
        for r in st:
            if r["order"] in PAPER_ORDERS:
                by.setdefault((round(float(r["sparsity"]), 4), int(r["bits"])), {})[r["order"]] = r
        same_best = same_rank = n_cells = 0
        for (s, b), d in sorted(by.items()):
            if len(d) != 3:
                continue
            n_cells += 1
            re_rank = sorted(PAPER_ORDERS, key=lambda o: d[o]["mean_rel_err"])
            kl_rank = sorted(PAPER_ORDERS, key=lambda o: d[o]["wt_kl"])
            same_best += re_rank[0] == kl_rank[0]
            same_rank += re_rank == kl_rank
            cells.append({"model": model, "cell": cell_label(s, b), "sparsity": s, "bits": b,
                          "layer_error_ranking": " < ".join(re_rank), "kl_ranking": " < ".join(kl_rank),
                          "same_best": re_rank[0] == kl_rank[0], "same_ranking": re_rank == kl_rank,
                          "joint_smallest_error": d["joint"]["mean_rel_err"] < min(d["p_then_q"]["mean_rel_err"], d["q_then_p"]["mean_rel_err"])})
        out.update({"cells": n_cells, "cells_same_best": same_best, "cells_same_ranking": same_rank})
        models.append(out)
    for model in store.models():  # layer error alone (no KL needed): cells where joint has the smallest mean layer error
        for s, b in store.cells(model, seed, PAPER_ORDERS):
            d = {o: (store.get(model, o, s, b, seed) or {}).get("mean_rel_err") for o in PAPER_ORDERS}
            if all(finite(v) for v in d.values()):
                errors.append({"model": model, "cell": cell_label(s, b), "sparsity": s, "bits": b,
                               **{f"mean_rel_err_{o}": d[o] for o in PAPER_ORDERS},
                               "smallest": min(PAPER_ORDERS, key=lambda o: (d[o], PAPER_ORDERS.index(o))),
                               "joint_smallest": d["joint"] < min(d["p_then_q"], d["q_then_p"])})
    return {"states": rows, "models": models, "cells": cells, "errors": errors, "joint_smallest": joint_smallest(store, errors)}


def joint_smallest(store, errors):
    """Per model (and over all models, and over the core models), per bit-width: cells with the three orders and
    cells where the joint sweep has the strictly smallest mean layer error."""
    core = [short(m) for m in MQ.CORE]
    out = []
    for scope in ["all", "core"] + store.models():
        for b in (4, 3):
            R = [e for e in errors if e["bits"] == b and (scope == "all" or (scope == "core" and e["model"] in core) or e["model"] == scope)]
            out.append({"scope": scope, "bits": b, "cells": len(R), "joint_smallest": sum(e["joint_smallest"] for e in R),
                        "not_joint": " ".join(f"{e['model']} {e['cell']}" for e in R if not e["joint_smallest"])})
    return out


def write(res, paths):
    R = paths.reports
    write_csv(os.path.join(R, "e8_layer_error_vs_kl_states.csv"), res["states"])
    write_csv(os.path.join(R, "e8_layer_error_vs_kl_cells.csv"), res["cells"])
    write_csv(os.path.join(R, "e8_layer_error_by_order.csv"), res["errors"])
    write_json(os.path.join(R, "e8_layer_error_vs_kl.json"), {"models": res["models"], "joint_smallest": res["joint_smallest"]})
