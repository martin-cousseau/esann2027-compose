"""Add-ons for the paper's notebook (esann2027_compose.ipynb) or for src/core.py.

1. install_extra_orders: lets compress_model run the ablation arms of arms.py (q_then_p_refit,
   q_then_p_pmask, q_then_p_pmask_refit, p_then_q_stale) as full sequential runs, so the mechanism
   can be checked on perplexity and tasks, not only on layer error.
2. make_safeguard_strict: turns the silent non-finite-Hessian safeguard of obs_sweep into an error.
3. kl_to_dense: mean token-level KL(dense || compressed) and top-1 agreement on WikiText-2 windows.
4. lmeval_items, flips, mcnemar: per-item correctness from lm-eval (log_samples=True), the share of
   answers that change against the dense model (Dutta et al., 2024), and an exact paired test.
5. bits_per_weight: storage per weight with the sparsity mask counted (bitmap or entropy bound) and
   the per-group scale/zero overhead.
6. run_metadata: GPU, driver, GPU UUID, CUDA, package versions, Colab runtime tag, NVML
   virtualization mode (passthrough vs vGPU), compute dtype and repository commit.

Notebook use, after cell 3 (core) and cell 4 (data, load_model):
    %run -i extra_orders.py
    install_extra_orders(globals())          # compress_model now accepts the new orders
    make_safeguard_strict(globals())         # optional: fail loudly instead of degrading silently
    GRID = [("q_then_p_refit", 0.25, 3, 0), ("q_then_p_pmask", 0.25, 3, 0), ...]
Module use:
    import core; install_extra_orders(core.__dict__)
"""
import importlib.metadata
import math
import os
import subprocess
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import arms as A  # noqa: E402

EXTRA_ORDERS = A.ABLATION_ARMS


# ----------------------------------------------------------------------------- 1-2. orders and strict mode
def install_extra_orders(ns):
    """Re-bind compress_weight in a namespace (notebook globals or core.__dict__).

    compress_model looks compress_weight up in that namespace at call time, so after this call it
    accepts the ablation orders; the paper's own orders are passed through unchanged.
    """
    original = ns["compress_weight"]
    if getattr(original, "_extra_orders", False):
        return
    core_view = lambda: A.as_core(ns)  # looked up at every call, so a later make_safeguard_strict applies too

    def compress_weight(W, H, sparsity, bits, order, groupsize=128):
        if order not in EXTRA_ORDERS:
            return original(W, H, sparsity, bits, order, groupsize)
        P, Q = A.single_sweeps(core_view(), W, H, [sparsity], [bits], groupsize)
        return A.run_arm(core_view(), order, W, H, sparsity, bits, P, Q, groupsize)

    compress_weight._extra_orders = True
    ns["compress_weight"] = compress_weight


def make_safeguard_strict(ns, counter=None):
    """Raise instead of zeroing non-finite Hessian entries (the paper's safeguard only prints a warning,
    and the notebook outputs are not kept, so a degraded run looks like a normal one)."""
    original = ns["obs_sweep"]
    if getattr(original, "_strict", False):
        return

    def obs_sweep(W, H, *args, **kwargs):
        bad = int((~torch.isfinite(H)).sum().item())
        if bad:
            if counter is not None:
                counter.append(bad)
            raise FloatingPointError(f"{bad} non-finite Hessian entries: re-run this configuration in float32")
        return original(W, H, *args, **kwargs)

    obs_sweep._strict = True
    ns["obs_sweep"] = obs_sweep


# ----------------------------------------------------------------------------- 3. KL to the dense model
@torch.no_grad()
def kl_to_dense(model, dense_model, enc_ids, seqlen, dev, max_windows=None, chunk=512):
    """Mean KL(p_dense || p_compressed) per token (nats) and top-1 agreement, on the same windows as the
    paper's perplexity. Keep a dense copy for dense_model (load_model(name)[0] in the notebook)."""
    n = enc_ids.numel() // seqlen
    if max_windows:
        n = min(n, max_windows)
    kl_sum, agree, tokens = 0.0, 0, 0
    for i in range(n):
        ids = enc_ids[:, i * seqlen:(i + 1) * seqlen].to(dev)
        lc = model(ids, use_cache=False).logits[0, :-1]
        ld = dense_model(ids, use_cache=False).logits[0, :-1]
        for a in range(0, lc.shape[0], chunk):
            pc = torch.log_softmax(lc[a:a + chunk].float(), -1)
            pd = torch.log_softmax(ld[a:a + chunk].float(), -1)
            kl_sum += (pd.exp() * (pd - pc)).sum().item()
            agree += (pd.argmax(-1) == pc.argmax(-1)).sum().item()
            tokens += pc.shape[0]
    return {"kl_nats_per_token": kl_sum / tokens, "top1_agreement": agree / tokens, "kl_tokens": tokens}


# ----------------------------------------------------------------------------- 4. per-item results, flips, McNemar
def lmeval_items(lm, tasks, limit, main_metric, num_fewshot=0):
    """Run lm-eval with per-item logging. Returns (aggregate results, {task: {doc_id: 0/1}}).

    main_metric is the notebook's MAIN_METRIC, e.g. {"hellaswag": "acc_norm,none", ...}.
    Save the per-item dict with each result line so flips and paired tests can be computed later.
    """
    from lm_eval import simple_evaluate
    out = simple_evaluate(model=lm, tasks=tasks, num_fewshot=num_fewshot, limit=limit, bootstrap_iters=0,
                          log_samples=True, verbosity="ERROR")
    items = {}
    for t in tasks:
        key = main_metric[t].split(",")[0]
        items[t] = {int(s["doc_id"]): int(round(float(s[key]))) for s in out["samples"][t]}
    return out, items


def flips(items_a, items_b):
    """Share of items whose correctness differs between two runs, per task and pooled."""
    res, n_all, d_all = {}, 0, 0
    for t in items_a:
        common = sorted(set(items_a[t]) & set(items_b.get(t, {})))
        d = sum(items_a[t][k] != items_b[t][k] for k in common)
        res[t] = d / len(common) if common else float("nan")
        n_all += len(common)
        d_all += d
    res["pooled"] = d_all / n_all if n_all else float("nan")
    return res


def mcnemar(items_a, items_b):
    """Exact two-sided McNemar test (binomial on discordant items), per task and pooled.

    Returns {task: (right in A only, right in B only, p-value)}.
    """
    def p_exact(b, c):
        n = b + c
        if n == 0:
            return 1.0
        tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2 ** n
        return min(1.0, 2 * tail)

    res, B, C = {}, 0, 0
    for t in items_a:
        common = set(items_a[t]) & set(items_b.get(t, {}))
        b = sum(items_a[t][k] == 1 and items_b[t][k] == 0 for k in common)
        c = sum(items_a[t][k] == 0 and items_b[t][k] == 1 for k in common)
        res[t] = (b, c, p_exact(b, c))
        B, C = B + b, C + c
    res["pooled"] = (B, C, p_exact(B, C))
    return res


# ----------------------------------------------------------------------------- 5. storage
def bits_per_weight(sparsity, bits, mask="bitmap", groupsize=128, scale_bits=16, zero_bits=None):
    """Storage per original weight, in bits.

    mask='none' reproduces the paper's TCr accounting (16 * (1 - TCr)); 'bitmap' adds 1 bit per weight
    (SparseGPT's own accounting); 'entropy' adds H(s), the lower bound for an i.i.d. mask. Quantized
    formats also pay one scale (scale_bits) and one zero point (zero_bits, default = bits) per group;
    that overhead is the same for every configuration at a given b, so it does not change rankings.
    """
    s = float(sparsity)
    if bits >= 16:
        value, overhead = 16 * (1 - s), 0.0
    else:
        value = bits * (1 - s)
        overhead = (scale_bits + (bits if zero_bits is None else zero_bits)) / groupsize
    if s <= 0 or mask == "none":
        m = 0.0
    elif mask == "bitmap":
        m = 1.0
    elif mask == "entropy":
        m = 0.0 if s >= 1 else -(s * math.log2(s) + (1 - s) * math.log2(1 - s))
    else:
        raise ValueError(mask)
    return value + m + overhead


# ----------------------------------------------------------------------------- 6. metadata
def run_metadata(repo=None, dtype=None):
    """Everything needed to tell runs apart later; store it in every result line."""
    import platform
    md = {"python": platform.python_version(), "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
          "colab_release_tag": os.environ.get("COLAB_RELEASE_TAG"), "compute_dtype": str(dtype)}
    for dist in ("transformers", "lm_eval", "datasets", "accelerate"):
        try:
            md[dist] = importlib.metadata.version(dist)
        except Exception:
            md[dist] = None
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        md.update(gpu=p.name, capability=list(torch.cuda.get_device_capability(0)),
                  gpu_mem_gb=round(p.total_memory / 1e9, 2))
        try:
            q = "name,driver_version,uuid,pci.bus_id,memory.total,clocks.max.sm"
            md["nvidia_smi"] = subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"],
                                              capture_output=True, text=True, timeout=20).stdout.strip()
        except Exception:
            pass
        try:
            import pynvml  # pip install nvidia-ml-py
            pynvml.nvmlInit()
            mode = pynvml.nvmlDeviceGetVirtualizationMode(pynvml.nvmlDeviceGetHandleByIndex(0))
            md["virtualization_mode"] = {0: "none (bare metal)", 1: "passthrough", 2: "vgpu",
                                         3: "host_vgpu", 4: "host_vsga"}.get(mode, str(mode))
        except Exception as e:
            md["virtualization_mode"] = f"unavailable ({type(e).__name__})"
    if repo:
        try:
            md["repo_commit"] = subprocess.run(["git", "-C", repo, "rev-parse", "HEAD"], capture_output=True,
                                               text=True, timeout=20).stdout.strip()
        except Exception:
            pass
    return md
