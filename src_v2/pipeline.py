"""v2 pipeline around the paper's compression code.

Nothing in the compression algorithm is re-implemented here. The OBS sweep, the three orders, the
sequential compression, the layer diagnostics and the perplexity come from the paper's src/core.py
(imported, not copied). The ablation arms, the strict overflow check, KL to the dense model, flips,
McNemar, bits per weight and run_metadata come from the review's review_code/ablation/extra_orders.py.

This module adds what the v2 protocol needs around them:
  * determinism flags and seeds;
  * the paper's calibration sampler (same windows for the same seed), the C4 evaluation windows
    (disjoint from every calibration seed we use), WikiText-2 windows;
  * one pass over evaluation windows that gives perplexity (same formula as core.perplexity), the
    token-level KL(dense || compressed) and top-1 agreement;
  * lm-eval with per-item logging (both acc and acc_norm kept for every item);
  * storage accounting (TCr, bitmap, mask entropy, symbol entropy) and run metadata with Colab fields;
  * a fingerprint check that refuses to run on a different GPU type or package set.
"""
import contextlib
import datetime as dt
import gzip
import hashlib
import importlib.util
import json
import math
import os
import random
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Where runners write results, logs, caches and models. On Colab: /content/out, outside the git checkout,
# so updating the code never touches live result files. In the container it defaults to the repository.
OUT = os.environ.get("ESANN_OUT", REPO)
REVIEW_ABLATION = os.path.join(REPO, "review_code", "ablation")
if REVIEW_ABLATION not in sys.path:
    sys.path.insert(0, REVIEW_ABLATION)

SEQLEN = 2048
GROUPSIZE = 128
N_CALIB = 128
C4_SHARD = "en/c4-validation.00000-of-00008.json.gz"
C4_EVAL_WINDOWS = 256
C4_EVAL_SEED = 20261109          # fixed, unrelated to the calibration seeds
CALIB_SEEDS_EXCLUDED = (0, 1, 2)  # C4 evaluation windows avoid every document these seeds calibrate on

PAPER_TASKS = ["arc_easy", "piqa", "hellaswag"]
EXTENDED_TASKS = ["arc_challenge", "winogrande", "openbookqa", "boolq", "sciq", "lambada_openai"]
ZERO_SHOT_TASKS = PAPER_TASKS + EXTENDED_TASKS
# lm-eval batch groups: short-input tasks run at 3x the batch of the long-passage tasks (BoolQ passages,
# SciQ supports, LAMBADA paragraphs), whose longest requests set the memory peak. Fixed sizes keep the
# evaluation deterministic (no OOM probing).
LONG_INPUT_TASKS = {"boolq", "sciq", "lambada_openai"}
FEWSHOT_TASKS = {"mmlu": 5}   # only for models >= 1.5B, as a sanity check
# v2 main metric per task (the protocol: acc_norm for these five, acc otherwise); both are always stored.
ACC_NORM_TASKS = {"arc_easy", "arc_challenge", "hellaswag", "openbookqa", "piqa"}
# the paper's own choice, kept for continuity (Sr as printed in the draft)
PAPER_MAIN_METRIC = {"arc_easy": "acc", "piqa": "acc", "hellaswag": "acc_norm"}
CHANCE = {"arc_easy": 0.25, "arc_challenge": 0.25, "hellaswag": 0.25, "openbookqa": 0.25, "piqa": 0.5,
          "winogrande": 0.5, "boolq": None, "sciq": 0.25, "lambada_openai": 0.0}
# boolq: chance is the majority-class rate of the validation split, computed from the data at analysis time


def main_metric(task):
    return "acc_norm" if task in ACC_NORM_TASKS else "acc"


# ----------------------------------------------------------------------------- determinism
def set_determinism(seed=0):
    """Rule 7 of the protocol. CUBLAS_WORKSPACE_CONFIG must be set before CUDA is initialised, so the
    runner also sets it at process start; setting it here again is harmless."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("highest")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def determinism_state():
    import torch
    return {"allow_tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "allow_tf32_cudnn": torch.backends.cudnn.allow_tf32,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


# ----------------------------------------------------------------------------- the paper's code
def load_core(strict=True, extra_orders=True, repo=REPO):
    """A fresh instance of the paper's src/core.py.

    extra_orders: compress_model also accepts the review's ablation arms (q_then_p_refit, ...).
    strict: a non-finite Hessian raises FloatingPointError instead of being zeroed with a warning.
    The order matters (install first, then strict): the installed compress_weight looks obs_sweep up
    at call time, so the strict check applies to every order.
    """
    import extra_orders as X
    path = os.path.join(repo, "src", "core.py")
    spec = importlib.util.spec_from_file_location(f"esann_core_{random.getrandbits(32):08x}", path)
    core = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(core)
    if extra_orders:
        X.install_extra_orders(core.__dict__)
    if strict:
        X.make_safeguard_strict(core.__dict__)
    return core


def extra():
    import extra_orders as X
    return X


def code_hash(repo=REPO):
    """Hash of the compression code (paper's core.py and the review's arms/extra_orders). Every unit of the
    study must share it; evaluation-code changes are tracked by the commit recorded in run_metadata."""
    files = ["src/core.py", "review_code/ablation/arms.py", "review_code/ablation/extra_orders.py"]
    h = hashlib.sha256()
    for f in files:
        with open(os.path.join(repo, f), "rb") as fh:
            h.update(f.encode() + b"\0" + fh.read() + b"\0")
    return h.hexdigest()[:16]


# ----------------------------------------------------------------------------- data
def load_c4_validation():
    from datasets import load_dataset
    return load_dataset("allenai/c4", data_files={"validation": C4_SHARD}, split="validation")


def load_wikitext2_test():
    from datasets import load_dataset
    return load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")


def get_calib(tok, c4, n=N_CALIB, seqlen=SEQLEN, seed=0):
    """The paper's sampler (notebook cell 4), imported from the review's verbatim copy."""
    from ablation_same_hessian import get_calib as _get_calib
    return _get_calib(tok, c4, n, seqlen, seed)


def calib_documents(tok, c4, n=N_CALIB, seqlen=SEQLEN, seed=0):
    """Replays the sampler's random stream and returns [(doc_index, start), ...] of the n windows.

    Must stay in lockstep with get_calib: same rng calls in the same order (tested)."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        i = rng.randint(0, len(c4) - 1)
        ids = tok(c4[i]["text"], return_tensors="pt").input_ids
        if ids.shape[1] <= seqlen:
            continue
        s = rng.randint(0, ids.shape[1] - seqlen - 1)
        out.append((i, s))
    return out


def c4_eval_windows(tok, c4, n=C4_EVAL_WINDOWS, seqlen=SEQLEN, seed=C4_EVAL_SEED, exclude=()):
    """n windows of seqlen tokens from distinct C4-validation documents that no calibration seed uses.

    Returns (LongTensor [n, seqlen], [(doc_index, start), ...])."""
    import torch
    exclude = set(exclude)
    rng = random.Random(seed)
    rows, where, used, tries = [], [], set(), 0
    while len(rows) < n:
        tries += 1
        if tries > 200 * n + 100000:
            raise RuntimeError(f"could only find {len(rows)} eligible C4 documents")
        i = rng.randint(0, len(c4) - 1)
        if i in exclude or i in used:
            continue
        ids = tok(c4[i]["text"], return_tensors="pt").input_ids
        if ids.shape[1] <= seqlen:
            continue
        s = rng.randint(0, ids.shape[1] - seqlen - 1)
        rows.append(ids[0, s:s + seqlen])
        where.append((i, s))
        used.add(i)
    return torch.stack(rows, 0), where


def wikitext2_windows(tok, wt, seqlen=SEQLEN):
    """The paper's WikiText-2 windows: the whole test split joined by blank lines, cut into seqlen chunks."""
    ids = tok("\n\n".join(wt["text"]), return_tensors="pt").input_ids
    n = ids.numel() // seqlen
    return ids[0, :n * seqlen].reshape(n, seqlen)


# ----------------------------------------------------------------------------- models
def load_model(name, dtype="float32", dev="cuda"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, dtype=getattr(torch, dtype)).to(dev).eval()
    model.config.use_cache = False
    return model, tok


def compressed_weight_count(core, model):
    """Number of weights in the linear layers of the decoder blocks (what the paper compresses)."""
    import torch.nn as nn
    return sum(m.weight.numel() for layer in core.get_layers(model)
               for m in layer.modules() if isinstance(m, nn.Linear))


# ----------------------------------------------------------------------------- perplexity, KL, top-1
def lm_metrics(model, windows, dev, dense=None, chunk=512):
    """Perplexity over the windows, plus KL(dense || model) per token and top-1 agreement if dense is given.

    The perplexity uses exactly core.perplexity's formula (sum of token NLLs in float, window by window,
    divided by n * (seqlen - 1)), so on the WikiText-2 windows it equals core.perplexity.
    KL and agreement follow extra_orders.kl_to_dense, computed in the same pass.
    """
    import torch
    import torch.nn.functional as F
    n, seqlen = windows.shape
    nll, kl_sum, agree, tokens = 0.0, 0.0, 0, 0
    with torch.no_grad():
        for i in range(n):
            ids = windows[i:i + 1].to(dev)
            logits = model(ids, use_cache=False).logits.float()
            nll += F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum").item()
            if dense is not None:
                lc = logits[0, :-1]
                ld = dense(ids, use_cache=False).logits[0, :-1]
                for a in range(0, lc.shape[0], chunk):
                    pc = torch.log_softmax(lc[a:a + chunk].float(), -1)
                    pd = torch.log_softmax(ld[a:a + chunk].float(), -1)
                    kl_sum += (pd.exp() * (pd - pc)).sum().item()
                    agree += (pd.argmax(-1) == pc.argmax(-1)).sum().item()
                    tokens += pc.shape[0]
                del ld
            del logits
    out = {"ppl": math.exp(nll / (n * (seqlen - 1))), "nll_sum": nll, "windows": n, "seqlen": seqlen}
    if dense is not None:
        out.update(kl=kl_sum / tokens, top1=agree / tokens, kl_tokens=tokens)
    return out


# ----------------------------------------------------------------------------- lm-eval with per-item logs
PER_ITEM_KEYS = ("acc", "acc_norm", "perplexity", "exact_match", "f1")


def run_lmeval(model, tok, tasks, batch_size, num_fewshot=0, limit=None, max_length=None):
    """lm-eval on an in-memory model. Returns (aggregates, items).

    aggregates: {task: {metric: value}} with every numeric metric lm-eval reports (acc, acc_norm, stderr...)
    items:      {task: {"doc_id": [...], "acc": [...], "acc_norm": [...], ...}} in doc_id order
    """
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM
    kw = {"pretrained": model, "tokenizer": tok, "batch_size": batch_size}
    if max_length:
        kw["max_length"] = max_length
    lm = HFLM(**kw)
    out = simple_evaluate(model=lm, tasks=list(tasks), num_fewshot=num_fewshot, limit=limit, bootstrap_iters=0,
                          log_samples=True, verbosity="ERROR", random_seed=0, numpy_random_seed=1234,
                          torch_random_seed=1234, fewshot_random_seed=1234)
    agg = {}
    for t, r in out["results"].items():
        agg[t] = {k: v for k, v in r.items() if isinstance(v, (int, float)) and not isinstance(v, bool)}
    items = {}
    for t, samples in out.get("samples", {}).items():
        samples = sorted(samples, key=lambda s: int(s["doc_id"]))
        rec = {"doc_id": [int(s["doc_id"]) for s in samples]}
        for k in PER_ITEM_KEYS:
            if samples and k in samples[0]:
                vals = []
                for s in samples:
                    v = s[k]
                    if isinstance(v, (list, tuple)):  # some lm-eval versions keep (value, ...) tuples
                        v = v[0]
                    vals.append(float(v))
                rec[k] = vals
        items[t] = rec
    n_samples = {t: len(v["doc_id"]) for t, v in items.items()}
    for t in agg:
        if t in n_samples:
            agg[t]["n_items"] = n_samples[t]
    versions = {t: v for t, v in (out.get("versions") or {}).items()}
    nshot = {t: v for t, v in (out.get("n-shot") or {}).items()}
    return agg, items, {"versions": versions, "n_shot": nshot}


def write_items(path, items):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with gzip.open(tmp, "wt") as f:
        json.dump(items, f, separators=(",", ":"))
    os.replace(tmp, path)
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def read_items(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)


def binary_items(items, metric="main"):
    """{task: {doc_id: 0/1}} as extra_orders.flips / mcnemar expect. metric='main' uses the v2 main metric."""
    out = {}
    for t, rec in items.items():
        m = main_metric(t) if metric == "main" else metric
        if m not in rec:
            continue
        out[t] = {d: int(round(v)) for d, v in zip(rec["doc_id"], rec[m])}
    return out


# ----------------------------------------------------------------------------- storage accounting
def h2(p):
    if p <= 0 or p >= 1:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def storage(sparsity, bits, diags=None, symbol_entropy=None, groupsize=GROUPSIZE):
    """Bits per weight under several accountings (group overhead reported separately).

    tcr:      16 (1 - TCr) = b (1 - s), what the paper counts;
    bitmap:   b (1 - s) + 1 bit per weight when s > 0 (SparseGPT's accounting);
    entropy:  b (1 - s) + H2(s_layer) per weight, with the actual mask fraction of every layer
              (weighted by layer size), the i.i.d. lower bound for the mask;
    symbol_entropy: plug-in entropy of the stored symbols per group of 128 (zeros included), which also
              credits rounding zeros of plain quantization; None when not measured.
    group_overhead: (16 + b) / groupsize bits per weight for a 16-bit scale and a b-bit zero point.
    """
    s, b = float(sparsity), int(bits)
    value = (16 if b >= 16 else b) * (1 - s)   # = 16 (1 - TCr); identical to extra_orders.bits_per_weight's value term
    out = {"tcr": value,
           "bitmap": value + (1.0 if s > 0 else 0.0),
           "entropy_nominal": value + h2(s),
           "group_overhead": (16 + b) / groupsize if b < 16 else 0.0}
    if diags:
        tot = sum(d["shape"][0] * d["shape"][1] for d in diags)
        ent = sum(d["shape"][0] * d["shape"][1] * h2(d["mask_frac"]) for d in diags) / tot
        kept = sum(d["shape"][0] * d["shape"][1] * (1 - d["mask_frac"]) for d in diags) / tot
        out["entropy_actual"] = (16 if b >= 16 else b) * kept + ent
        out["mask_frac_actual"] = 1 - kept
        out["zero_frac_actual"] = sum(d["shape"][0] * d["shape"][1] * d["zero_frac"] for d in diags) / tot
    out["symbol_entropy"] = symbol_entropy
    return out


def group_symbol_entropy(W, groupsize=GROUPSIZE):
    """Mean plug-in entropy (bits per weight) of the values inside each (row, group of `groupsize` columns).

    For a quantized layer every group holds at most 2^b distinct values, so this is the empirical entropy
    of its symbols, zeros included (it rewards free rounding zeros as well as mask zeros). Plug-in
    estimates on 128 samples are biased low by at most about (levels - 1) / (2 * 128 * ln 2) bits.
    """
    import torch
    rows, cols = W.shape
    g = cols // groupsize
    if g == 0:
        return None
    x = W[:, :g * groupsize].reshape(rows * g, groupsize)
    xs, _ = torch.sort(x, dim=1)
    new = torch.ones_like(xs, dtype=torch.bool)
    new[:, 1:] = xs[:, 1:] != xs[:, :-1]
    run_id = torch.cumsum(new.to(torch.int64), dim=1) - 1
    counts = torch.zeros(x.shape[0], groupsize, device=W.device, dtype=torch.float64)
    counts.scatter_add_(1, run_id, torch.ones_like(run_id, dtype=torch.float64))
    p = counts / groupsize
    ent = -(torch.where(p > 0, p * torch.log2(p), torch.zeros_like(p))).sum(1)
    return float(ent.mean().item())


def model_symbol_entropy(core, model, groupsize=GROUPSIZE):
    import torch.nn as nn
    num, den = 0.0, 0
    for layer in core.get_layers(model):
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                e = group_symbol_entropy(m.weight.data.float(), groupsize)
                if e is None:
                    continue
                num += e * m.weight.numel()
                den += m.weight.numel()
    return num / den if den else None


# ----------------------------------------------------------------------------- metadata and fingerprint
def utcnow():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def run_metadata(dtype, runtime_name=None, session_name=None, extra_fields=None):
    """extra_orders.run_metadata (GPU name, UUID, driver, CUDA, packages, virtualization mode, commit)
    plus the Colab fields of the protocol."""
    X = extra()
    md = X.run_metadata(repo=REPO, dtype=dtype)
    md.update(colab_runtime_name=runtime_name, colab_session=session_name,
              colab_image=os.environ.get("COLAB_RELEASE_TAG"), code_hash=code_hash(),
              determinism=determinism_state(), python_executable=sys.executable)
    if extra_fields:
        md.update(extra_fields)
    return md


def fingerprint():
    """The fields that must be identical on every runtime of the study."""
    import importlib.metadata as im
    import subprocess
    import torch
    fp = {"gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
          "colab_image": os.environ.get("COLAB_RELEASE_TAG"),
          "torch": torch.__version__}
    for d in ("transformers", "lm_eval", "datasets", "accelerate", "huggingface_hub", "numpy"):
        try:
            fp[d] = im.version(d)
        except Exception:  # noqa: BLE001
            fp[d] = None
    try:
        fp["gpu_uuid"] = subprocess.run(["nvidia-smi", "--query-gpu=uuid", "--format=csv,noheader"],
                                        capture_output=True, text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        fp["gpu_uuid"] = None
    try:
        fp["commit"] = subprocess.run(["git", "-C", REPO, "rev-parse", "HEAD"], capture_output=True,
                                      text=True, timeout=20).stdout.strip()
    except Exception:  # noqa: BLE001
        fp["commit"] = None
    return fp


LOCKED_KEYS = ("gpu_name", "colab_image", "torch", "transformers", "lm_eval", "datasets", "accelerate")


def check_fingerprint(fp, locked):
    """Raise if any locked field differs (GPU type, image, package versions). UUID and commit may differ."""
    bad = {k: (fp.get(k), locked.get(k)) for k in LOCKED_KEYS if k in locked and fp.get(k) != locked.get(k)}
    if bad:
        raise RuntimeError("fingerprint mismatch (found, locked): " + json.dumps(bad))
    return True


# ----------------------------------------------------------------------------- misc
@contextlib.contextmanager
def timer(store, key):
    import time
    t0 = time.time()
    yield
    store[key] = round(time.time() - t0, 2)


def append_jsonl(path, obj):
    """Append one line atomically enough for a single writer: write, flush, fsync."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    line = json.dumps(obj, separators=(",", ":")) + "\n"
    with open(path, "a") as f:
        f.write(line)
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path):
    if not os.path.exists(path):
        return []
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut by a kill: ignored, the unit is re-run
    return out


# ----------------------------------------------------------------------------- exact compact model export
class GridRecorder:
    """Records, for every compress_weight call, the grid returned by its LAST obs_sweep call and the mask.

    For every order (paper orders and review arms) the last sweep inside compress_weight produces the final
    weights, so its grid is the one the weights lie on. Install after load_core (it wraps the strict
    obs_sweep, so the overflow check still runs); records come in compress_model's layer order.
    """

    def __init__(self, core):
        self.core, self.records, self._last = core, [], None
        self._sweep, self._cw = core.obs_sweep, core.compress_weight
        rec = self

        def obs_sweep(W, H, *a, **kw):
            out = rec._sweep(W, H, *a, **kw)
            rec._last = out[2]
            return out

        def compress_weight(W, H, sparsity, bits, order, groupsize=128):
            rec._last = None
            Wh, m = rec._cw(W, H, sparsity, bits, order, groupsize)
            rec.records.append({"grid": rec._last, "mask": m.detach().clone()})
            return Wh, m

        core.__dict__["obs_sweep"] = obs_sweep
        core.__dict__["compress_weight"] = compress_weight

    def reset(self):
        self.records = []

    def uninstall(self):
        self.core.__dict__["obs_sweep"] = self._sweep
        self.core.__dict__["compress_weight"] = self._cw


def _linears_in_order(core, model):
    import torch.nn as nn
    for li, layer in enumerate(core.get_layers(model)):
        lins = {n: m for n, m in layer.named_modules() if isinstance(m, nn.Linear)}  # compress_model's order
        for n, m in lins.items():
            yield li, n, m


def _pack_nibbles(q):
    import torch
    q = q.to(torch.uint8).reshape(-1)
    if q.numel() % 2:
        q = torch.cat([q, q.new_zeros(1)])
    return (q[0::2] | (q[1::2] << 4)).contiguous()


def _unpack_nibbles(p, n):
    import torch
    out = torch.empty(p.numel() * 2, dtype=torch.uint8, device=p.device)
    out[0::2] = p & 0x0F
    out[1::2] = p >> 4
    return out[:n]


def _packbits(mask):
    import numpy as np
    import torch
    return torch.from_numpy(np.packbits(mask.reshape(-1).cpu().numpy().astype(np.uint8)))


def _unpackbits(p, n, dev):
    import numpy as np
    import torch
    return torch.from_numpy(np.unpackbits(p.cpu().numpy())[:n].astype(bool)).to(dev)


def export_compressed(core, model, records, bits, path, meta, groupsize=GROUPSIZE):
    """Write the compressed linear weights of the decoder blocks in an exact compact form.

    Quantized layers: integer codes (4-bit packed two per byte when bits <= 4, else uint8), fp32 scale and
    uint8 zero point per (row, group), and the sparsity mask as a bitmap (needed only to tell mask zeros from
    rounding zeros: decoding gives exact zeros either way). Pruned-only layers: fp32 kept values + bitmap.
    Every layer is decoded and compared bit for bit before writing; a layer that does not round-trip is
    stored as plain fp32 and flagged. Returns a summary dict (bytes, exact layers, fallbacks).
    """
    import torch
    from safetensors.torch import save_file
    tensors, layers_meta, fallbacks = {}, [], []
    lin = list(_linears_in_order(core, model))
    assert len(lin) == len(records), (len(lin), len(records))
    for (li, n, m), rec in zip(lin, records):
        W = m.weight.data.float()
        rows, cols = W.shape
        key = f"{li}.{n}"
        mask = rec["mask"].to(W.device)
        entry = {"key": key, "shape": [rows, cols]}
        if bits < 16 and rec["grid"]:
            G = len(rec["grid"])
            scale = torch.stack([rec["grid"][g][0].float() for g in range(G)], 1)
            zero = torch.stack([rec["grid"][g][1].float() for g in range(G)], 1)
            sc = scale.repeat_interleave(groupsize, 1)[:, :cols]
            ze = zero.repeat_interleave(groupsize, 1)[:, :cols]
            q = torch.clamp(torch.round(W / sc + ze), 0, 2 ** bits - 1)
            Wr = sc * (q - ze)
            if mask.any():
                Wr[mask] = 0
            if torch.equal(Wr, W) and bool((zero >= 0).all()) and bool((zero <= 255).all()):
                codes = _pack_nibbles(q) if bits <= 4 else q.to(torch.uint8).reshape(-1)
                tensors[key + ".codes"] = codes.cpu()
                tensors[key + ".scale"] = scale.cpu().contiguous()
                tensors[key + ".zero"] = zero.to(torch.uint8).cpu().contiguous()
                if mask.any():
                    tensors[key + ".mask"] = _packbits(mask)
                entry.update(kind="quant", bits=bits, groups=G, packed=bits <= 4)
            else:
                tensors[key + ".weight"] = W.cpu().contiguous()
                entry.update(kind="fp32")
                fallbacks.append(key)
        elif mask.any():
            tensors[key + ".values"] = W[~mask].cpu().contiguous()
            tensors[key + ".mask"] = _packbits(mask)
            entry.update(kind="sparse_fp32")
        else:
            tensors[key + ".weight"] = W.cpu().contiguous()
            entry.update(kind="fp32")
        layers_meta.append(entry)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    meta = dict(meta, format="esann2027-v2-compact-1", groupsize=groupsize, layers=layers_meta,
                fp32_fallback_layers=fallbacks)
    save_file(tensors, path, metadata={"esann2027": json.dumps(meta)})
    with open(path, "rb") as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    return {"path": path, "bytes": os.path.getsize(path), "sha256": sha, "layers": len(layers_meta),
            "fp32_fallback_layers": fallbacks}


def load_compressed_into(core, model, path, groupsize=None):
    """Decode a file written by export_compressed into an already loaded dense model (same architecture).
    The result equals the evaluated compressed model bit for bit."""
    import torch
    from safetensors import safe_open
    with safe_open(path, framework="pt") as f:
        meta = json.loads(f.metadata()["esann2027"])
        gs = groupsize or meta["groupsize"]
        by_key = {e["key"]: e for e in meta["layers"]}
        for li, n, m in _linears_in_order(core, model):
            e = by_key[f"{li}.{n}"]
            dev, dt_ = m.weight.device, m.weight.dtype
            rows, cols = e["shape"]
            if e["kind"] == "quant":
                codes = f.get_tensor(e["key"] + ".codes").to(dev)
                q = (_unpack_nibbles(codes, rows * cols) if e["packed"] else codes).reshape(rows, cols).float()
                scale = f.get_tensor(e["key"] + ".scale").to(dev)
                zero = f.get_tensor(e["key"] + ".zero").to(dev).float()
                sc = scale.repeat_interleave(gs, 1)[:, :cols]
                ze = zero.repeat_interleave(gs, 1)[:, :cols]
                W = sc * (q - ze)
                if e["key"] + ".mask" in f.keys():
                    W[_unpackbits(f.get_tensor(e["key"] + ".mask"), rows * cols, dev).reshape(rows, cols)] = 0
            elif e["kind"] == "sparse_fp32":
                mask = _unpackbits(f.get_tensor(e["key"] + ".mask"), rows * cols, dev).reshape(rows, cols)
                W = torch.zeros(rows, cols, device=dev)
                W[~mask] = f.get_tensor(e["key"] + ".values").to(dev)
            else:
                W = f.get_tensor(e["key"] + ".weight").to(dev)
            m.weight.data = W.to(dt_)
    return meta
