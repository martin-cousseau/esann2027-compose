#!/usr/bin/env python3
"""Phase-0 reproduction gate (runs on the runtime after the first E1 units of SmolLM2-135M).

PASS when the fp32 WikiText-2 perplexities of dense, 4-bit and 50%+4 P->Q match the closest published runs
(results/results_full.jsonl, 16-bit; the fp32 re-runs in results_fix.jsonl cover only SmolLM2-360M) within
0.5%. Writes logs/gate.json and exits 1 on failure so the chain stops before any further experiment.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.environ.get("ESANN_OUT", ROOT)


def rows(p, base=ROOT):
    p = os.path.join(base, p)
    return [json.loads(x) for x in open(p) if x.strip()] if os.path.exists(p) else []


ours = {(r["order"], round(r["sparsity"], 4), r["bits"]): r for r in rows("results/v2/E1/SmolLM2-135M.jsonl", OUT)
        if r.get("seed", 0) == 0}
pub = {(r["order"], round(r["sparsity"], 4), r["bits"]): r for r in rows("results/results_full.jsonl")
       if r["model"].endswith("SmolLM2-135M") and r.get("seed", 0) == 0}
out, ok = {"checks": []}, True
for key in [("dense", 0.0, 16), ("quant", 0.0, 4), ("p_then_q", 0.5, 4)]:
    if key not in ours:
        out["checks"].append({"cell": key, "status": "MISSING"})
        ok = False
        continue
    a, b = ours[key]["wikitext2"]["ppl"], pub[key]["ppl"]
    rel = abs(a - b) / b
    good = rel <= 0.005
    ok &= good
    out["checks"].append({"cell": key, "ours_fp32": a, "published_fp16": b, "rel_diff": rel,
                          "status": "PASS" if good else "FAIL"})
out["result"] = "PASS" if ok else "FAIL"
os.makedirs(os.path.join(OUT, "logs", "status"), exist_ok=True)
with open(os.path.join(OUT, "logs", "gate.json"), "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out))
sys.exit(0 if ok else 1)
