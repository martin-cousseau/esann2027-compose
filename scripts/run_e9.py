#!/usr/bin/env python3
"""E9: the layer-level control WITHOUT error compensation, i.e. the setting of the paper's decomposition (Section 3).

Per linear layer of the selected blocks, with the dense model's Hessian H (the same H as E2):
  * The sweeps of E2's arms run on diag(H). With a diagonal Hessian the OBS sweep propagates no error to other
    columns: pruning zeroes the weights with the smallest w^2 (H_jj + damping) and leaves the others unchanged,
    quantization rounds every weight to the nearest point of the same min-max grid. Errors are measured with the FULL
    H, as everywhere else. src/core.py is not modified: obs_sweep is simply called with torch.diag(torch.diag(H)).
  * Arms (uncompensated): p_then_q (grid fitted on the kept weights), q_then_p (mask chosen on rounded weights, grid of
    quantization alone) and p_then_q_stale (mask of pruning alone, grid of quantization alone). Without compensation
    the joint pass equals p_then_q and q_then_p_pmask equals p_then_q_stale (tests/test_e9.py checks both), so they
    are not run.
  * The exact terms of the decomposition for p_then_q_stale, whose error is Delta_PQ = P + R_K with P the pruned
    weights and R = R_M + R_K the rounding residual on pruned / kept entries:
        I_E = E_PQ - E_P - E_Q = (-||R_M||_H^2 + 2<P,R_K>_H - 2<R_M,R_K>_H) / tr(W H W^T)
    (absorption, cross_PR, cross_RR) and the residual of this identity (float64).
  * The overlap of the pruning mask M with the zeros Z that quantization alone creates, without and with
    compensation (the compensated single sweeps are the paper's): |M & Z| / |M| against |Z| / N.
One JSON line per layer in $ESANN_OUT/results/v2/E9/<model>.jsonl, resume-safe (layers already there are skipped).
Usage (from /content/work):
  python scripts/run_e9.py --model HuggingFaceTB/SmolLM2-135M --blocks stride2 --locked colab/locked_fingerprint.json
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import types  # noqa: E402

import torch  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from src_v2 import pipeline as P  # noqa: E402  (also puts review_code/ablation on sys.path)
import ablation_same_hessian as ASH  # noqa: E402
import arms as A  # noqa: E402

CELLS = "1/4:4,1/3:4,1/2:4,1/4:3,1/3:3,1/2:3"
NOCOMP_ARMS = ("p_then_q", "q_then_p", "p_then_q_stale")


def _tr(X, H, Y):
    """tr(X H Y^T) in float64."""
    return float(((X.double() @ H) * Y.double()).sum())


def layer_e9(core, W, H, cells, groupsize=128, arms=NOCOMP_ARMS):
    """Every E9 quantity for one linear layer (W: rows x cols, H: cols x cols, both on the same device)."""
    W = W.float()
    Hdiag = torch.diag(torch.diag(H))
    nocomp = types.SimpleNamespace(obs_sweep=lambda W_in, _H, **kw: core.obs_sweep(W_in, Hdiag, **kw),
                                   rel_recon_error=core.rel_recon_error)
    sparsities = sorted({s for s, _ in cells})
    bits_list = sorted({b for _, b in cells})
    E = lambda Wh: core.rel_recon_error(W, Wh, H)  # noqa: E731
    P0, Q0 = A.single_sweeps(nocomp, W, H, sparsities, bits_list, groupsize)
    Pc, Qc = A.single_sweeps(core, W, H, sparsities, bits_list, groupsize)
    H64 = H.double()
    norm = _tr(W, H64, W)
    out = {"nocomp": {"E_P": {A.key_s(s): E(P0[s][0]) for s in sparsities},
                      "E_Q": {str(b): E(Q0[b][0]) for b in bits_list},
                      "zeros_Q": {str(b): (Q0[b][0] == 0).float().mean().item() for b in bits_list}, "cells": {}},
           "comp": {"E_P": {A.key_s(s): E(Pc[s][0]) for s in sparsities},
                    "E_Q": {str(b): E(Qc[b][0]) for b in bits_list},
                    "zeros_Q": {str(b): (Qc[b][0] == 0).float().mean().item() for b in bits_list}, "cells": {}}}
    for s, b in cells:
        c = {}
        for arm in arms:
            Wh, m = A.run_arm(nocomp, arm, W, H, s, b, P0, Q0, groupsize)
            c[arm] = E(Wh)
            if arm in ("p_then_q", "q_then_p"):
                c[f"zeros_{arm}"] = (Wh == 0).float().mean().item()
            if arm == "q_then_p":
                mP = P0[s][1]
                c["mask_jaccard_q_then_p_vs_P"] = (m & mP).sum().item() / max((m | mP).sum().item(), 1)
            if arm == "p_then_q_stale":
                Wstale = Wh
        # decomposition of the uncompensated p_then_q_stale: Delta_P = P, Delta_Q = R, Delta_PQ = P + R_K
        M = P0[s][1]
        Dp, Dq, Dpq = W - P0[s][0], W - Q0[b][0], W - Wstale
        Rm, Rk = Dq * M, Dq * (~M)
        ep, eq, epq = _tr(Dp, H64, Dp) / norm, _tr(Dq, H64, Dq) / norm, _tr(Dpq, H64, Dpq) / norm
        c["absorption"] = -_tr(Rm, H64, Rm) / norm
        c["cross_PR"] = 2 * _tr(Dp, H64, Rk) / norm
        c["cross_RR"] = -2 * _tr(Rm, H64, Rk) / norm
        c["I_E_stale_f64"] = epq - ep - eq
        c["decomp_residual"] = (epq - ep - eq) - (c["absorption"] + c["cross_PR"] + c["cross_RR"])
        c["structure_residual"] = float(((Dpq - (Dp * M + Rk)).abs().max()))   # Delta_PQ = P + R_K entrywise
        for tag, Wq, Mm in (("nocomp", Q0[b][0], M), ("comp", Qc[b][0], Pc[s][1])):
            Z = Wq == 0
            both = (Mm & Z).sum().item()
            ov = {"overlap_mask_zeros": both / max(Mm.sum().item(), 1), "share_zeros_pruned": both / max(Z.sum().item(), 1),
                  "zeros": Z.float().mean().item()}
            (c.update(ov) if tag == "nocomp" else out["comp"]["cells"].setdefault(A.key_cell(s, b), {}).update(ov))
        out["nocomp"]["cells"][A.key_cell(s, b)] = c
    return out


def _done_layers(path):
    done = set()
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line)
                done.add((r["model"], r["block"], r["name"]))
            except (ValueError, KeyError):
                continue   # a line cut by a lost runtime
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--blocks", default="stride2")
    ap.add_argument("--cells", default=CELLS)
    ap.add_argument("--locked", default=None)
    ap.add_argument("--runtime-name", default=None)
    ap.add_argument("--session", default=None)
    a = ap.parse_args()
    import run_e2
    from torch import nn
    short = a.model.split("/")[-1]
    out = os.path.join(P.OUT, "results", "v2", "E9", f"{short}.jsonl")
    status = os.path.join(P.OUT, "logs", "status", f"E9__{short}.json")
    os.makedirs(os.path.dirname(status), exist_ok=True)

    def write_status(stage):
        with open(status + ".tmp", "w") as f:
            json.dump({"stage": stage, "state_id": f"E9 {short} blocks={a.blocks}", "updated": P.utcnow(),
                       "done": None, "todo": None}, f)
        os.replace(status + ".tmp", status)

    P.set_determinism(0)
    fp = P.fingerprint()
    if a.locked:
        with open(a.locked) as f:
            P.check_fingerprint(fp, json.load(f))
        print("fingerprint matches the locked one", flush=True)
    write_status("running")
    core = ASH.load_core(ROOT)
    P.extra().make_safeguard_strict(core.__dict__)
    meta = P.run_metadata("float32", a.runtime_name, a.session, {"family": "E9", "cells": a.cells})
    cells = ASH.parse_cells(a.cells)
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev).eval()
    model.config.use_cache = False
    c4 = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"}, split="validation")
    calib = ASH.get_calib(tok, c4, 128, 2048, 0)
    blocks = ASH.parse_blocks(a.blocks, len(core.get_layers(model)))
    done = _done_layers(out)
    print(f"{a.model}: blocks {blocks}, cells {cells}, {len(done)} layers already done", flush=True)
    t0 = time.time()
    Hs = ASH.dense_hessians(core, model, calib, dev, blocks)
    print(f"dense Hessians for {len(Hs)} layers in {time.time() - t0:.0f}s", flush=True)
    model.to("cpu")   # the weights are moved one layer at a time: frees the GPU for the float64 traces (Qwen2.5-3B OOM)
    if dev == "cuda":
        torch.cuda.empty_cache()
    layers = core.get_layers(model)
    for li in blocks:
        for name, m in layers[li].named_modules():
            if not isinstance(m, nn.Linear) or (a.model, li, name) in done:
                continue
            t1 = time.time()
            H = Hs[(li, name)]
            res = layer_e9(core, m.weight.data.float().to(H.device), H, cells)
            row = dict(model=a.model, block=li, name=name, shape=list(m.weight.shape), seconds=round(time.time() - t1, 1),
                       **res, **run_e2.corr_index(H), meta=meta)
            P.append_jsonl(out, row)
            c0 = res["nocomp"]["cells"][A.key_cell(*cells[0])]
            print(f"  block {li:2d} {name:22s} {row['seconds']:6.1f}s  absorption={c0['absorption']:.2e} "
                  f"cross={c0['cross_PR'] + c0['cross_RR']:.2e} resid={c0['decomp_residual']:.1e}", flush=True)
        for key in [k for k in Hs if k[0] == li]:
            del Hs[key]
        if dev == "cuda":
            torch.cuda.empty_cache()
    write_status("finished")


if __name__ == "__main__":
    main()
