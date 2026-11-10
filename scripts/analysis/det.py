"""Determinism: the DET re-run (same state on a second L4) against the same state_id in the study families.

Exact equality is expected for wikitext2.nll_sum and for every per-layer rel_err; the maximum absolute
differences are reported (0.0 when identical), together with the other per-layer fields and the GPUs.
"""
import os

from .common import finite, write_json

LAYER_FIELDS = ("rel_err", "mask_frac", "zero_frac", "max_levels", "mask_kept")


def _gpu(r):
    md = r.get("metadata") or {}
    return md.get("nvidia_smi") or md.get("gpu")


def analyze(store, missing):
    out = []
    for sid, d in sorted(store.det.items()):
        row = {"state_id": sid, "det_source": store.det_source[sid], "det_gpu": _gpu(d),
               "det_ppl": (d.get("wikitext2") or {}).get("ppl"), "det_nll_sum": (d.get("wikitext2") or {}).get("nll_sum")}
        e = store.states.get(sid)
        if e is None:
            row["status"] = "pending: the same state is not in the study families yet"
            missing.add("DET determinism", sid, "counterpart state not available yet")
            out.append(row)
            continue
        ew, dw = e.get("wikitext2") or {}, d.get("wikitext2") or {}
        row.update(e1_source=store.source[sid], e1_gpu=_gpu(e), e1_ppl=ew.get("ppl"), e1_nll_sum=ew.get("nll_sum"),
                   same_calibration=e.get("calib_sha") == d.get("calib_sha"),
                   same_windows=e.get("wt_windows_sha") == d.get("wt_windows_sha"))
        if finite(ew.get("nll_sum")) and finite(dw.get("nll_sum")):
            row["nll_sum_abs_diff"] = abs(ew["nll_sum"] - dw["nll_sum"])
            row["nll_sum_equal"] = ew["nll_sum"] == dw["nll_sum"]
        if finite(ew.get("ppl")) and finite(dw.get("ppl")):
            row["ppl_abs_diff"] = abs(ew["ppl"] - dw["ppl"])
        A = {(x["layer"], x["name"]): x for x in e.get("diags") or []}
        B = {(x["layer"], x["name"]): x for x in d.get("diags") or []}
        keys = sorted(set(A) & set(B))
        row["n_layers_compared"], row["n_layers_only_one"] = len(keys), len(set(A) ^ set(B))
        for f in LAYER_FIELDS:
            diffs = []
            for k in keys:
                a, b = A[k].get(f), B[k].get(f)
                if isinstance(a, bool) or isinstance(b, bool):
                    diffs.append(0.0 if a == b else 1.0)
                elif finite(a) and finite(b):
                    diffs.append(abs(a - b))
            row[f"{f}_max_abs_diff"] = max(diffs) if diffs else None
            row[f"{f}_all_equal"] = all(x == 0 for x in diffs) if diffs else None
        row["status"] = "identical" if (row.get("nll_sum_equal") and row.get("rel_err_all_equal")
                                        and row["n_layers_only_one"] == 0) else "differs"
        out.append(row)
    if not store.det:
        missing.add("DET determinism", "results/v2/DET", "no DET lines")
    return out


def write(rows, paths):
    write_json(os.path.join(paths.reports, "det_determinism.json"), rows)
