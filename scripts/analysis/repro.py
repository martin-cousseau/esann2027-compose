"""Hardware/precision reproduction checks against the published v1 runs (macros R27-R28).

REPRO16: the same code in 16-bit on an L4 against the published 16-bit T4 runs (results/results_full.jsonl).
REPRO32: fp32 on an L4 against the paper's own fp32 re-runs on a T4 (results/results_fix.jsonl).
The planned units come from colab/chain_*.sh (family, model, queue, dtype) and colab/queues/*.json, so a unit
that has not run yet is reported as missing. REPRO lines are never merged into the study states (their
state_ids can coincide with study states).
"""
import glob
import json
import os
import re

from .common import P, finite, load_v1, short, skey, state_id, write_csv


def plan(root):
    out = {}
    for sh in sorted(glob.glob(os.path.join(root, "colab", "chain_*.sh"))):
        with open(sh) as f:
            for line in f:
                m = re.search(r"--family\s+(REPRO\w*)", line)
                if not m or line.lstrip().startswith("#"):
                    continue
                model = re.search(r"--model\s+(\S+)", line)
                queue = re.search(r"--queue\s+(\S+)", line)
                dtype = re.search(r"--dtype\s+(\S+)", line)
                qp = os.path.join(root, queue.group(1)) if queue else None
                if not (model and qp and os.path.exists(qp)):
                    continue
                with open(qp) as qf:
                    for u in json.load(qf):
                        sid = state_id(model.group(1), u, dtype.group(1) if dtype else "float32")
                        out[sid] = {"family": m.group(1), "model": short(model.group(1)), "order": u["order"],
                                    "sparsity": float(u["sparsity"]), "bits": int(u["bits"]), "seed": int(u.get("seed", 0))}
    return out


def analyze(paths, missing):
    v1 = load_v1(paths)
    planned = plan(paths.root)
    lines = {}
    for fam in sorted({v["family"] for v in planned.values()} | {"REPRO16", "REPRO32"}):
        for f in sorted(glob.glob(os.path.join(paths.results_v2, fam, "*.jsonl"))):
            if "failures__" in os.path.basename(f):
                continue
            for r in P.read_jsonl(f):
                if isinstance(r, dict) and r.get("status") == "ok" and r.get("state_id"):
                    lines.setdefault((fam, r["state_id"]), (r, paths.rel(f)))
    rows = []
    keys = sorted(set(lines) | {(v["family"], sid) for sid, v in planned.items()})
    for fam, sid in keys:
        r, src = lines.get((fam, sid), (None, None))
        info = planned.get(sid) or {"family": fam, "model": short(r["model"]), "order": r["order"], "sparsity": r["sparsity"],
                                    "bits": r["bits"], "seed": int(r.get("seed", 0))}
        k = (info["model"], info["order"], skey(info["sparsity"]), info["bits"], info["seed"])
        ref_name = "results_full (16-bit, T4)" if fam == "REPRO16" else "results_fix (fp32, T4)"
        ref = (v1["full"] if fam == "REPRO16" else v1["fix"]).get(k) if v1 else None
        ppl = (r.get("wikitext2") or {}).get("ppl") if r else None
        row = {"family": fam, "state_id": sid, "model": info["model"], "order": info["order"], "sparsity": info["sparsity"],
               "bits": info["bits"], "seed": info["seed"], "ppl": ppl, "source": src, "reference": ref_name,
               "ref_ppl": ref.get("ppl") if ref else None,
               "gpu": ((r or {}).get("metadata") or {}).get("gpu"), "dtype": (r or {}).get("dtype")}
        row["rel_delta_pct"] = 100 * (ppl / row["ref_ppl"] - 1) if finite(ppl) and finite(row["ref_ppl"]) else None
        if r is None:
            missing.add("REPRO (hardware/precision)", sid, "planned unit not available yet")
        elif ref is None:
            missing.add("REPRO (hardware/precision)", sid, f"no counterpart in {ref_name}")
        rows.append(row)
    return rows


def write(rows, paths):
    write_csv(os.path.join(paths.reports, "repro_hardware.csv"), rows)
