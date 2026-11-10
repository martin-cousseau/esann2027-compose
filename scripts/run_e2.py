#!/usr/bin/env python3
"""E2: same-Hessian layer ablation, a thin wrapper around the review's ablation_same_hessian.py.

The review script does the work (dense Hessians, every arm of arms.py on one H per layer, resume-safe
JSON lines). This wrapper only adds what the v2 protocol requires:
  * the fingerprint check against the locked runtime;
  * the full run_metadata (extra_orders.run_metadata + Colab fields) in every line instead of the
    script's shorter one;
  * per layer, the off-diagonal correlation index of H, mean |corr| exactly as in synthetic_layers.py
    (dead input columns, diag(H) = 0, are left out), and the mean |corr| over off-diagonal entries only;
  * a status file for the container-side watcher.

Usage (from /content/work):
  python scripts/run_e2.py --model HuggingFaceTB/SmolLM2-135M --blocks all \
      --locked colab/locked_fingerprint.json --runtime-name esann-e2-a --session e2-main
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src_v2 import pipeline as P  # noqa: E402  (also puts review_code/ablation on sys.path)
import ablation_same_hessian as ASH  # noqa: E402

CELLS = "1/4:4,1/3:4,1/2:4,1/4:3,1/3:3,1/2:3"


def corr_index(H):
    import torch
    d = torch.diag(H)
    live = d > 0
    Hl = H[live][:, live]
    D = torch.diag(Hl).sqrt()
    C = Hl / D[:, None] / D[None, :]
    off = C - torch.diag(torch.diag(C))
    n = C.shape[0]
    return {"h_corr_index": off.abs().mean().item(),                       # synthetic_layers.py's formula
            "h_offdiag_mean_abs_corr": (off.abs().sum() / max(n * n - n, 1)).item(),
            "h_dead_columns": int((~live).sum().item())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--blocks", default="all")
    ap.add_argument("--cells", default=CELLS)
    ap.add_argument("--locked", default=None)
    ap.add_argument("--runtime-name", default=None)
    ap.add_argument("--session", default=None)
    a = ap.parse_args()
    short = a.model.split("/")[-1]
    out = os.path.join(P.OUT, "results", "v2", "E2", f"{short}.jsonl")
    status = os.path.join(P.OUT, "logs", "status", f"E2__{short}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    os.makedirs(os.path.dirname(status), exist_ok=True)

    def write_status(stage):
        with open(status + ".tmp", "w") as f:
            json.dump({"stage": stage, "state_id": f"E2 {short} blocks={a.blocks}", "updated": P.utcnow(),
                       "done": None, "todo": None}, f)
        os.replace(status + ".tmp", status)

    P.set_determinism(0)
    fp = P.fingerprint()
    print("fingerprint", json.dumps(fp), flush=True)
    if a.locked:
        with open(a.locked) as f:
            P.check_fingerprint(fp, json.load(f))
        print("fingerprint matches the locked one", flush=True)
    write_status("running")

    # richer metadata in every line
    ASH.run_metadata = lambda repo, dtype, dev: P.run_metadata(
        str(dtype).replace("torch.", ""), a.runtime_name, a.session, {"family": "E2", "cells": a.cells})
    # correlation index of H for every layer
    orig = ASH.A.layer_ablation

    def layer_ablation_with_corr(core, W, H, cells, arms=ASH.A.ALL_ARMS, groupsize=128, mask_quality=True):
        res = orig(core, W, H, cells, arms, groupsize, mask_quality)
        res.update(corr_index(H))
        return res

    ASH.A.layer_ablation = layer_ablation_with_corr
    # the paper's core, with the strict overflow check (the review script loads core.py itself; make it strict)
    orig_load = ASH.load_core

    def load_core_strict(repo):
        core = orig_load(repo)
        P.extra().make_safeguard_strict(core.__dict__)
        return core

    ASH.load_core = load_core_strict
    sys.argv = ["ablation_same_hessian.py", "--repo", ROOT, "--model", a.model, "--cells", a.cells,
                "--blocks", a.blocks, "--dtype", "float32", "--out", out]
    ASH.main()
    write_status("finished")


if __name__ == "__main__":
    main()
