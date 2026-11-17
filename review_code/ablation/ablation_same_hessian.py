"""Order and mechanism ablation on ONE shared Hessian per layer (review of the ESANN 2027 draft).

GPU run (a free Colab T4 is enough; see ablation_colab.ipynb):
    git clone https://github.com/antounyaacoub/esann2027-compose
    python ablation_same_hessian.py --repo esann2027-compose --model HuggingFaceTB/SmolLM2-135M
CPU smoke test (no download; a tiny random Llama model):
    python ablation_same_hessian.py --repo esann2027-compose --smoke-test

What it measures
  For every linear layer of the selected decoder blocks, H = 2XX^T is computed from the DENSE model's
  activations on the paper's calibration windows (same C4 shard, same sampler, same seed). That layer's
  dense weights are then compressed with every arm of arms.py using that one H, and the paper's
  normalized layer error E = tr(D H D^T) / tr(W H W^T) is recorded. Nothing is propagated between
  layers, so E_P, E_Q and every combination of a layer share the same H. In the paper's Fig. 1 they come
  from three separate sequential runs, whose deeper layers are calibrated on different inputs; only
  block 0 sees identical inputs in all runs (the smoke test checks that this script reproduces the
  paper's own block-0 errors).

Output
  One JSON line per (model, block, linear) in --out (resume-safe: finished layers are skipped), then a
  summary per model and cell:
    interaction    I_E / (E_P + E_Q) per arm, from sums over layers as in the paper's Fig. 1, and the
                   share of layers that are sub-additive;
    mechanism      share of the Q->P penalty (sum of E_QP - E_PQ) that each ablation arm removes;
    mask quality   pruning-only error with the mask Q->P selects, relative to SparseGPT's mask (median).

Cost on a T4 (estimated from the paper's logged sweep times, about 0.7 / 1.4 / 1.9 s per layer sweep
for the 135M / 360M / 0.5B models): the defaults (3 cells, every 4th block) need 29 sweeps per layer,
roughly 20 / 40 / 40 minutes per model. --blocks all multiplies this by about four.
"""
import argparse
import importlib.util
import json
import math
import os
import random
import statistics as st
import sys
import time
from collections import defaultdict
from fractions import Fraction

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import arms as A  # noqa: E402


# ----------------------------------------------------------------------------- setup helpers
def load_core(repo):
    """Import the paper's src/core.py from a clone of the repository."""
    path = os.path.join(repo, "src", "core.py")
    spec = importlib.util.spec_from_file_location("esann_core", path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    return core


def parse_cells(text):
    """'0.25:4,0.5:4,1/3:3' -> [(0.25, 4), (0.5, 4), (0.333..., 3)] (fractions kept exact, as in the paper)."""
    cells = []
    for item in text.split(","):
        s, b = item.split(":")
        cells.append((float(Fraction(s.strip())), int(b)))
    return cells


def parse_blocks(text, n_layers):
    if text == "all":
        return list(range(n_layers))
    if text.startswith("stride"):
        return list(range(0, n_layers, int(text[len("stride"):])))
    return [int(x) for x in text.split(",")]


def get_calib(tok, c4, n, seqlen, seed):
    """Identical to get_calib in the paper's notebook (cell 4): same windows for the same seed."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        doc = c4[rng.randint(0, len(c4) - 1)]["text"]
        ids = tok(doc, return_tensors="pt").input_ids
        if ids.shape[1] <= seqlen:
            continue
        s = rng.randint(0, ids.shape[1] - seqlen - 1)
        out.append(ids[:, s:s + seqlen])
    return out


@torch.no_grad()
def dense_hessians(core, model, calib, dev, blocks):
    """H = 2XX^T for every linear of the selected blocks, from the dense model's activations.

    Same running average as the paper's compress_model, but nothing is compressed, so every arm of a
    layer sees the same H. For block 0 this is exactly the paper's Hessian.
    """
    layers = core.get_layers(model)
    Hs, cnt, hooks = {}, {}, []

    def make_hook(key):
        def hook(mod, inp, out):
            x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
            t = x.shape[0]
            if key not in Hs:
                Hs[key] = torch.zeros(x.shape[1], x.shape[1], device=x.device)
                cnt[key] = 0
            Hs[key] *= cnt[key] / (cnt[key] + t)
            cnt[key] += t
            x = math.sqrt(2 / cnt[key]) * x
            Hs[key] += x.T @ x
        return hook

    for li in blocks:
        for name, m in layers[li].named_modules():
            if isinstance(m, nn.Linear):
                hooks.append(m.register_forward_hook(make_hook((li, name))))
    for ids in calib:
        model(ids.to(dev), use_cache=False)
    for h in hooks:
        h.remove()
    bad = {k: int((~torch.isfinite(v)).sum().item()) for k, v in Hs.items()}
    bad = {k: v for k, v in bad.items() if v}
    if bad:
        raise FloatingPointError(f"non-finite Hessian entries in {len(bad)} layers, e.g. {next(iter(bad))}: "
                                 "collect Hessians in float32 (--dtype float32)")
    return Hs


def run_metadata(repo, dtype, dev):
    import platform
    import subprocess
    md = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
          "dtype": str(dtype), "device": dev, "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG")}
    try:
        import transformers
        md["transformers"] = transformers.__version__
    except Exception:
        pass
    if dev == "cuda" and torch.cuda.is_available():
        md["gpu"] = torch.cuda.get_device_name(0)
        try:
            md["nvidia_smi"] = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,uuid",
                                               "--format=csv,noheader"], capture_output=True, text=True,
                                              timeout=20).stdout.strip()
        except Exception:
            pass
    try:
        md["repo_commit"] = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                                           text=True, timeout=20).stdout.strip()
    except Exception:
        pass
    return md


# ----------------------------------------------------------------------------- summary
def summarize(rows, cells, arms, log=print):
    by_model = defaultdict(list)
    for r in rows:
        by_model[r["model"]].append(r)
    for model, R in by_model.items():
        log(f"\n=== {model}: {len(R)} linear layers from blocks {sorted({r['block'] for r in R})}")
        for s, b in cells:
            ck, sk, bk = A.key_cell(s, b), A.key_s(s), str(b)
            R2 = [r for r in R if ck in r["cells"]]
            if not R2:
                continue
            sP = sum(r["E_P"][sk] for r in R2)
            sQ = sum(r["E_Q"][bk] for r in R2)
            log(f"  {round(s * 100):d}% + {b}-bit  (E_P sum {sP:.4g}, E_Q sum {sQ:.4g})")
            for arm in arms:
                sC = sum(r["cells"][ck][arm] for r in R2)
                sub = sum(r["cells"][ck][arm] < r["E_P"][sk] + r["E_Q"][bk] for r in R2)
                log(f"    {arm:22s} I_E/(E_P+E_Q) {100 * (sC - sP - sQ) / (sP + sQ):+7.1f}%   "
                    f"sub-additive layers {sub}/{len(R2)}")
            if "q_then_p" in arms and "p_then_q" in arms:
                pen = sum(r["cells"][ck]["q_then_p"] - r["cells"][ck]["p_then_q"] for r in R2)
                log(f"    Q->P penalty over P->Q (sum of layer errors): {pen:+.4g}"
                    f" ({100 * pen / sum(r['cells'][ck]['p_then_q'] for r in R2):+.1f}% of P->Q)")
                if pen > 0:
                    for arm in A.ABLATION_ARMS:
                        if arm in arms and arm != "p_then_q_stale":
                            closed = sum(r["cells"][ck]["q_then_p"] - r["cells"][ck][arm] for r in R2)
                            log(f"      share of the penalty removed by {arm:22s} {100 * closed / pen:+6.1f}%")
                if "p_then_q_stale" in arms:
                    hurt = sum(r["cells"][ck]["p_then_q_stale"] - r["cells"][ck]["p_then_q"] for r in R2)
                    log(f"      stale grid imposed on P->Q changes its error by "
                        f"{100 * hurt / sum(r['cells'][ck]['p_then_q'] for r in R2):+.1f}%")
            mq = [r["cells"][ck].get("mask_quality_q_then_p") for r in R2]
            mq = [x for x in mq if x is not None]
            if mq:
                jac = st.median(r["cells"][ck]["mask_jaccard_q_then_p_vs_P"] for r in R2)
                log(f"    Q->P mask used for pruning alone: median error ratio {st.median(mq):.2f}x SparseGPT's mask;"
                    f" median overlap (Jaccard) {jac:.2f}")


# ----------------------------------------------------------------------------- smoke test
def smoke_test(core, args):
    """Tiny random Llama model on CPU: checks the arms against the paper's compress_weight, checks that
    block-0 errors match the paper's own compress_model, and runs the summary end to end."""
    import copy
    from transformers import LlamaConfig, LlamaForCausalLM
    torch.manual_seed(0)
    cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=320, num_hidden_layers=3,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256)
    model = LlamaForCausalLM(cfg).float().eval()
    model.config.use_cache = False
    g = torch.Generator().manual_seed(1)
    calib = [torch.randint(0, cfg.vocab_size, (1, 128), generator=g) for _ in range(8)]
    cells = [(0.25, 4), (0.5, 4), (1 / 3, 3)]
    dev = "cpu"

    # 1. arms reproduce the paper's three orders exactly
    Hs = dense_hessians(core, model, calib, dev, blocks=[0, 1, 2])
    layer0 = core.get_layers(model)[0]
    worst = 0.0
    for name, m in layer0.named_modules():
        if isinstance(m, nn.Linear):
            for s, b in cells:
                d = A.check_against_core(core, m.weight.data.float(), Hs[(0, name)], s, b)
                worst = max(worst, max(d.values()))
    print(f"[1] arms vs paper's compress_weight (block 0, 3 cells, 3 orders): max |diff| = {worst:.3e}")
    assert worst == 0.0, "arms.py does not reproduce the paper's orders"

    # 2. block-0 errors equal the paper's own sequential pipeline (block 0 sees identical inputs)
    worst_rel = 0.0
    for order, s, b in [("prune", 0.5, 16), ("quant", 0.0, 4), ("p_then_q", 0.5, 4), ("q_then_p", 0.5, 4)]:
        mc = copy.deepcopy(model)
        diags = core.compress_model(mc, calib, s, b, order, dev, verbose=False)
        for d in diags:
            if d["layer"] != 0:
                continue
            W = layer0.get_submodule(d["name"]).weight.data.float()
            H = Hs[(0, d["name"])]
            if order in ("prune", "quant"):
                P, Q = A.single_sweeps(core, W, H, [s] if order == "prune" else [], [b] if order == "quant" else [])
                Wh = P[s][0] if order == "prune" else Q[b][0]
            else:
                P, Q = A.single_sweeps(core, W, H, [s], [b])
                Wh, _ = A.run_arm(core, order, W, H, s, b, P, Q)
            mine = core.rel_recon_error(W, Wh, H)
            worst_rel = max(worst_rel, abs(mine - d["rel_err"]) / max(d["rel_err"], 1e-30))
    print(f"[2] block-0 errors vs the paper's compress_model diags: max relative diff = {worst_rel:.2e}")
    assert worst_rel < 1e-4, "block-0 Hessians do not match the paper's pipeline"

    # 3. full ablation on all blocks + summary
    rows = []
    t0 = time.time()
    for li in range(cfg.num_hidden_layers):
        for name, m in core.get_layers(model)[li].named_modules():
            if isinstance(m, nn.Linear):
                res = A.layer_ablation(core, m.weight.data.float(), Hs[(li, name)], cells)
                rows.append(dict(model="tiny-random-llama", block=li, name=name, shape=list(m.weight.shape), **res))
    print(f"[3] ablation on {len(rows)} layers x {len(cells)} cells x {len(A.ALL_ARMS)} arms in {time.time() - t0:.1f}s")
    for r in rows:
        for c in r["cells"].values():
            for arm in A.ALL_ARMS:
                assert math.isfinite(c[arm]) and c[arm] >= 0
    summarize(rows, cells, A.ALL_ARMS)
    print("\nSMOKE TEST PASSED (random weights: the numbers above test the code path, not the paper's claims)")


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="path to a clone of esann2027-compose (for src/core.py)")
    ap.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M")
    ap.add_argument("--cells", default="1/4:4,1/2:4,1/3:3", help="sparsity:bits pairs")
    ap.add_argument("--blocks", default="stride4", help="'all', 'strideK' or a list such as 0,4,8")
    ap.add_argument("--arms", default=",".join(A.ALL_ARMS))
    ap.add_argument("--n-calib", type=int, default=128)
    ap.add_argument("--seqlen", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"],
                    help="model dtype for the calibration pass (float32 avoids the fp16 overflow)")
    ap.add_argument("--out", default="ablation_same_hessian.jsonl")
    ap.add_argument("--summary-only", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()

    core = load_core(args.repo)
    if args.smoke_test:
        smoke_test(core, args)
        return

    cells = parse_cells(args.cells)
    arms = tuple(a for a in args.arms.split(",") if a)
    rows = [json.loads(l) for l in open(args.out)] if os.path.exists(args.out) else []
    if args.summary_only:
        summarize([r for r in rows if r["model"] == args.model] or rows, cells, arms)
        return

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = getattr(torch, args.dtype)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(dev).eval()
    model.config.use_cache = False
    c4 = load_dataset("allenai/c4", data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                      split="validation")
    calib = get_calib(tok, c4, args.n_calib, args.seqlen, args.seed)
    blocks = parse_blocks(args.blocks, len(core.get_layers(model)))
    meta = run_metadata(args.repo, dtype, dev)
    meta.update(n_calib=args.n_calib, seqlen=args.seqlen, seed=args.seed, cells=args.cells, arms=list(arms))
    print(f"{args.model}: blocks {blocks}, cells {cells}, {len(arms)} arms | {meta.get('gpu', dev)}")

    done = {(r["model"], r["block"], r["name"]) for r in rows}
    t0 = time.time()
    Hs = dense_hessians(core, model, calib, dev, blocks)
    print(f"dense Hessians for {len(Hs)} layers in {time.time() - t0:.0f}s")
    layers = core.get_layers(model)
    with open(args.out, "a") as f:
        for li in blocks:
            for name, m in layers[li].named_modules():
                if not isinstance(m, nn.Linear) or (args.model, li, name) in done:
                    continue
                t1 = time.time()
                res = A.layer_ablation(core, m.weight.data.float(), Hs[(li, name)], cells, arms)
                row = dict(model=args.model, block=li, name=name, shape=list(m.weight.shape),
                           seconds=round(time.time() - t1, 1), **res, meta=meta)
                f.write(json.dumps(row) + "\n")
                f.flush()
                rows.append(row)
                c0 = res["cells"][A.key_cell(*cells[0])]
                print(f"  block {li:2d} {name:22s} {row['seconds']:6.1f}s  " + "  ".join(
                    f"{a}={c0[a]:.3e}" for a in ("p_then_q", "q_then_p", "q_then_p_refit") if a in c0))
            for key in [k for k in Hs if k[0] == li]:
                del Hs[key]  # this block is done: free its Hessians
            if dev == "cuda":
                torch.cuda.empty_cache()
    summarize([r for r in rows if r["model"] == args.model], cells, arms)


if __name__ == "__main__":
    main()
