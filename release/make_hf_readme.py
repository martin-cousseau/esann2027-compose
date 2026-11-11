#!/usr/bin/env python3
"""Write the model card (README.md) of the Hugging Face model repository from the result files.

Every number in the card (counts, bits per weight, perplexities, KL) is read from results/v2/*/*.jsonl, the
lines written on the GPU runtimes; nothing is typed by hand. With --check-hub, every state's file is also
looked up in the Hub repository and its size and sha256 (from the result line) are compared.

    python release/make_hf_readme.py                      # writes release/README_hf_models.md
    python release/make_hf_readme.py --check-hub          # also verifies the files on the Hub ($HF_TOKEN)
"""
import argparse
import collections

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from scripts.analysis.common import ORDER_TXT, Store, cell_label, finite, short  # noqa: E402

HF_REPO = "martincousseau/esann2027-compose-v2"
GIT = "https://github.com/martin-cousseau/esann2027-compose"
BRANCH = "claude/busy-hopper-ghtynk"
WANDB = "https://wandb.ai/martin-css/esann2027-compose-v2"
BASE = {"SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M", "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M",
        "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B", "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B"}
FAMILY_TXT = {"E1": "the paper's grid in fp32", "E3": "Q->P mechanism arms", "E4": "extra frontier points",
              "E5": "calibration seeds 1 and 2", "E6": "scale within one family"}


def fmt(v, nd):
    return f"{v:.{nd}f}" if finite(v) else "--"


def config(r):
    o = r["order"]
    if o == "dense":
        return "dense"
    if o == "quant":
        return f"{int(r['bits'])}-bit"
    if o == "prune":
        return f"{round(100 * float(r['sparsity']))}% pruned"
    return f"{cell_label(r['sparsity'], r['bits'])} {ORDER_TXT.get(o, o)}"


def rows_by_model(store):
    out = collections.OrderedDict()
    for m in store.models():
        rr = [r for r in store.states.values() if short(r["model"]) == m and (r.get("model_export") or {}).get("repo_path")]
        rr.sort(key=lambda r: (int(r.get("seed", 0)), r["order"] != "dense", r["order"] not in ("quant", "prune"),
                               -int(r["bits"]), float(r["sparsity"]), r["order"]))
        if rr:
            out[m] = rr
    return out


def check_hub(by_model):
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    info = {}
    for f in api.list_repo_tree(HF_REPO, repo_type="model", recursive=True):
        if getattr(f, "lfs", None) is not None:
            info[f.path] = (f.size, f.lfs.sha256)
        elif hasattr(f, "size"):
            info[f.path] = (f.size, None)
    bad = []
    for rr in by_model.values():
        for r in rr:
            e = r["model_export"]
            got = info.get(e["repo_path"])
            if got is None:
                bad.append((e["repo_path"], "not on the Hub"))
            elif got[0] != e["bytes"] or (got[1] and got[1] != e["sha256"]):
                bad.append((e["repo_path"], f"size/sha differ: hub {got}, line {(e['bytes'], e['sha256'])}"))
    return bad, len(info)


def card(store, by_model):
    n = sum(len(v) for v in by_model.values())
    fam = collections.Counter(r.get("family") for rr in by_model.values() for r in rr)
    L = ["---", "license: apache-2.0", "base_model:"] + [f"- {BASE[m]}" for m in by_model if m in BASE] + [
        "tags:", "- pruning", "- quantization", "- model-compression", "- sparsegpt", "- gptq", "- obs", "---", "",
        "# Do pruning and quantization compose? Compressed model states (ESANN 2027 study, v2)", "",
        f"This repository holds **{n} compressed model states** of {len(by_model)} base models, every state evaluated in the"
        " v2 study of the order of unstructured pruning and weight quantization under Hessian-based (OBS) compensation:"
        " pruning alone, quantization alone, and the two combined in three orders (prune then quantize, quantize then"
        " prune, and both in one joint sweep), plus the ablation arms that test why quantizing first hurts.", "",
        "- **Code, result lines and per-item benchmark logs:** " + f"[{GIT}]({GIT}/tree/{BRANCH}) (branch `{BRANCH}`)",
        f"- **One Weights & Biases run per state (metrics, layer diagnostics, links back here):** [{WANDB}]({WANDB})",
        "- **Paper:** in preparation for ESANN 2027; citation below once available.", "",
        "Each file stores the compressed linear layers of the decoder blocks in an exact compact form (integer codes,"
        " per-group scale and zero point, sparsity bitmap). Decoding gives the evaluated fp32 weights bit for bit;"
        " embeddings, norms and the LM head are the unchanged base model.", "",
        "## Load a state", "", "```bash", "pip install torch transformers safetensors huggingface_hub numpy",
        "python load_esann2027.py models/SmolLM2-135M/SmolLM2-135M__p_then_q__s0.5000__b4__seed0__float32.safetensors", "```", "",
        "```python", "from huggingface_hub import hf_hub_download", "import load_esann2027 as L  # this repository's loader", "",
        f'path = hf_hub_download("{HF_REPO}", "models/SmolLM2-135M/SmolLM2-135M__p_then_q__s0.5000__b4__seed0__float32.safetensors")',
        "model, meta = L.load(path)   # transformers model in fp32, compressed weights in place", "```", "",
        "## File names", "",
        "`models/<model>/<model>__<order>__s<sparsity>__b<bits>__seed<seed>__float32.safetensors`, with orders `dense`,"
        " `prune` (pruning only, `b16`), `quant` (quantization only, `s0.0000`), `joint`, `p_then_q`, `q_then_p` and the"
        " ablation arms `q_then_p_refit` (grid re-fitted after pruning), `q_then_p_pmask` (the mask pruning alone would"
        " choose), `q_then_p_pmask_refit` (both) and `p_then_q_stale` (P->Q with the grid of the unpruned weights)."
        " `seed` is the calibration-sample seed.", "",
        "## Format `esann2027-v2-compact-1`", "",
        "JSON metadata under the safetensors metadata key `esann2027` (state, commit, code hash, group size, and one entry"
        " per layer: key `<block>.<module>`, shape, kind). Per kind:", "",
        "| kind | tensors | decoding |", "|---|---|---|",
        "| `quant` | `<key>.codes` uint8 (two 4-bit codes per byte, low nibble first, when bits <= 4), `<key>.scale` fp32 and"
        " `<key>.zero` uint8 per (row, group of 128 columns), optional `<key>.mask` bitmap | `W = scale * (code - zero)`, then"
        " `W[mask] = 0` |",
        "| `sparse_fp32` | `<key>.mask` bitmap, `<key>.values` fp32 kept weights (row-major) | zeros, kept values in place |",
        "| `fp32` | `<key>.weight` | as stored |", "",
        "Bitmaps use `numpy.packbits` order. The bitmap is stored to tell pruned weights from weights that round to"
        " zero; the bits-per-weight columns below count it as 1 bit per weight whenever the state is pruned.", "",
        "## How the states were made", "",
        "All states: fp32 model and Hessians on one GPU type (NVIDIA L4), TF32 off, deterministic kernels, the OBS"
        " sweep of the paper's code (`src/core.py`), 128 calibration windows of 2048 tokens from C4, asymmetric min-max"
        " quantization grid per group of 128 weights. Evaluation: WikiText-2 test perplexity, C4 validation perplexity"
        " (256 windows), KL divergence and top-1 agreement with the dense model, and (E1, E4) nine zero-shot tasks with"
        " lm-evaluation-harness 0.4.13 on full splits with per-item logs. Every result line records the GPU, driver,"
        " Colab image, library versions, commit and code hash; see the GitHub repository.", "",
        "States by experiment family: " + ", ".join(f"{k} ({FAMILY_TXT.get(k, k)}): {v}" for k, v in sorted(fam.items())) + ".", ""]
    L += ["## States", "",
          "bits/w = bits per weight of the compressed linear layers with a 1-bit mask counted for pruned states (the"
          " per-group scale and zero point add (16 + b)/128 bits for quantized states and are not included); PPL ="
          " WikiText-2 test perplexity; KL = mean KL divergence to the dense model on WikiText-2. Pruning-only states are"
          " counted at 16 bits per kept weight (the base models' precision; they are stored here in fp32). The dense"
          " states are the base models themselves and have no file here.", ""]
    for m, rr in by_model.items():
        L += [f"<details><summary><b>{m}</b> ({len(rr)} states; base model <code>{BASE.get(m, m)}</code>)</summary>", "",
              "| state | family | seed | bits/w | PPL | C4 PPL | KL | file |", "|---|---|---|---|---|---|---|---|"]
        for r in rr:
            sto, wt, c4 = r.get("storage") or {}, r.get("wikitext2") or {}, r.get("c4") or {}
            e = r["model_export"]
            L.append(f"| {config(r)} | {r.get('family')} | {r.get('seed', 0)} | {fmt(sto.get('bitmap'), 2)} | {fmt(wt.get('ppl'), 2)}"
                     f" | {fmt(c4.get('ppl'), 2)} | {fmt(wt.get('kl'), 3)} | [{os.path.basename(e['repo_path'])}]({e['repo_path']}) |")
        L += ["", "</details>", ""]
    L += ["## License", "",
          "Derived from the base models, all released under Apache-2.0; the compressed states keep that license.", "",
          "## Citation", "", "A citation will be added when the ESANN 2027 paper is available.", ""]
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "release", "README_hf_models.md"))
    ap.add_argument("--check-hub", action="store_true")
    a = ap.parse_args()
    store = Store()
    by_model = rows_by_model(store)
    if a.check_hub:
        bad, n_hub = check_hub(by_model)
        print(f"hub files: {n_hub}; states checked: {sum(len(v) for v in by_model.values())}; problems: {len(bad)}")
        for p, why in bad[:20]:
            print("  ", p, why)
    with open(a.out, "w") as f:
        f.write(card(store, by_model))
    print(f"wrote {a.out}: {sum(len(v) for v in by_model.values())} states, {len(by_model)} models")
    return 0


if __name__ == "__main__":
    sys.exit(main())
