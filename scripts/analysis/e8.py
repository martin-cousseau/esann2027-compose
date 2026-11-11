"""E8 (analysis part): does a smaller layer error predict a better model? (the draft's open question)

For every compressed state (seed 0) the result line holds mean_rel_err, the mean over all compressed layers of
the per-layer relative error (layer diagnostics, as in the paper); within one model every state compresses the
same layers, so its rank equals the rank of the summed layer error. Two tests per model:
  across states  Spearman rho between mean_rel_err and the KL to the dense model (WikiText-2, also C4) over
                 every compressed state of the model (prune only, quant only, combined orders, E3 arms);
  within cells   for each (s, b) cell with the three paper orders, does the order with the smallest layer
                 error also have the smallest KL? And are the two rankings identical?
The fourth order of E8 (joint with re-fit) needs GPU runs and was not run (budget, DECISIONS.md).
"""
import os

from .common import COMBINED, PAPER_ORDERS, cell_label, finite, short, write_csv, write_json


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
    rows, models, cells = [], [], []
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
            cells.append({"model": model, "cell": cell_label(s, b), "layer_error_ranking": " < ".join(re_rank),
                          "kl_ranking": " < ".join(kl_rank), "same_best": re_rank[0] == kl_rank[0], "same_ranking": re_rank == kl_rank})
        out.update({"cells": n_cells, "cells_same_best": same_best, "cells_same_ranking": same_rank})
        models.append(out)
    return {"states": rows, "models": models, "cells": cells}


def write(res, paths):
    R = paths.reports
    write_csv(os.path.join(R, "e8_layer_error_vs_kl_states.csv"), res["states"])
    write_csv(os.path.join(R, "e8_layer_error_vs_kl_cells.csv"), res["cells"])
    write_json(os.path.join(R, "e8_layer_error_vs_kl.json"), {"models": res["models"]})
