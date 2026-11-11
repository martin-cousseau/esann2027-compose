#!/usr/bin/env python3
"""Load one compressed model state of the ESANN 2027 v2 study into a Hugging Face transformers model.

Standalone (torch, numpy, safetensors, transformers, huggingface_hub). Each file holds the compressed linear
layers of the decoder blocks in an exact compact form; everything else (embeddings, norms, lm_head) is the
unchanged base model. Decoding reproduces the evaluated weights bit for bit (fp32).

    python load_esann2027.py models/SmolLM2-135M/SmolLM2-135M__p_then_q__s0.5000__b4__seed0__float32.safetensors

Format "esann2027-v2-compact-1" (JSON metadata under the safetensors metadata key "esann2027"):
  layers[i] = {"key": "<block>.<module>", "shape": [rows, cols], "kind": ...}, <module> as in named_modules() of
  the decoder layer (e.g. "self_attn.q_proj", "mlp.down_proj"), and per kind the tensors
    quant        <key>.codes  uint8 integer codes, row-major; two 4-bit codes per byte (low nibble first)
                              when "packed" is true (bits <= 4), one per byte otherwise
                 <key>.scale  fp32 [rows, groups], <key>.zero uint8 [rows, groups], groups of "groupsize"
                              columns: W = scale * (code - zero)
                 <key>.mask   optional bitmap of pruned weights (numpy.packbits order), set to exactly 0
    sparse_fp32  <key>.mask   bitmap of pruned weights, <key>.values fp32 kept weights in row-major order
    fp32         <key>.weight plain fp32 weight
"""
import json
import sys

import numpy as np
import torch


def _bits(packed, n):
    return torch.from_numpy(np.unpackbits(packed.numpy())[:n].astype(bool))


def decode(path):
    """{key: fp32 weight tensor} and the metadata of one file."""
    from safetensors import safe_open
    out = {}
    with safe_open(path, framework="pt") as f:
        meta = json.loads(f.metadata()["esann2027"])
        assert meta.get("format") == "esann2027-v2-compact-1", meta.get("format")
        gs = meta["groupsize"]
        for e in meta["layers"]:
            k, (rows, cols) = e["key"], e["shape"]
            if e["kind"] == "quant":
                c = f.get_tensor(k + ".codes")
                if e["packed"]:
                    q = torch.empty(c.numel() * 2, dtype=torch.uint8)
                    q[0::2], q[1::2] = c & 0x0F, c >> 4
                    c = q[: rows * cols]
                q = c.reshape(rows, cols).float()
                sc = f.get_tensor(k + ".scale").repeat_interleave(gs, 1)[:, :cols]
                ze = f.get_tensor(k + ".zero").float().repeat_interleave(gs, 1)[:, :cols]
                W = sc * (q - ze)
                if k + ".mask" in f.keys():
                    W[_bits(f.get_tensor(k + ".mask"), rows * cols).reshape(rows, cols)] = 0
            elif e["kind"] == "sparse_fp32":
                m = _bits(f.get_tensor(k + ".mask"), rows * cols).reshape(rows, cols)
                W = torch.zeros(rows, cols)
                W[~m] = f.get_tensor(k + ".values")
            else:
                W = f.get_tensor(k + ".weight")
            out[k] = W
    return out, meta


BASE = {"SmolLM2-135M": "HuggingFaceTB/SmolLM2-135M", "SmolLM2-360M": "HuggingFaceTB/SmolLM2-360M",
        "SmolLM2-1.7B": "HuggingFaceTB/SmolLM2-1.7B", "Qwen2.5-0.5B": "Qwen/Qwen2.5-0.5B"}


def load(path, base=None, device="cpu"):
    """The base model in fp32 with the decoded compressed weights put in place."""
    from transformers import AutoModelForCausalLM
    weights, meta = decode(path)
    base = base or BASE[path.replace("\\", "/").split("/")[-1].split("__")[0]]
    model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.float32).to(device).eval()
    layers = model.model.layers
    for k, W in weights.items():
        block, name = k.split(".", 1)
        mod = layers[int(block)].get_submodule(name)
        assert tuple(mod.weight.shape) == tuple(W.shape), (k, mod.weight.shape, W.shape)
        mod.weight.data = W.to(device=device, dtype=torch.float32)
    return model, meta


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    p = sys.argv[1]
    if not p.endswith(".safetensors"):
        sys.exit("give the path of one .safetensors file")
    import os
    if not os.path.exists(p):  # a path inside the Hugging Face repo
        from huggingface_hub import hf_hub_download
        p = hf_hub_download("martincousseau/esann2027-compose-v2", p)
    model, meta = load(p)
    w = [m.weight for m in model.model.layers.modules() if isinstance(m, torch.nn.Linear)]
    zeros = sum(int((x == 0).sum()) for x in w) / sum(x.numel() for x in w)
    print(f"loaded {meta.get('state_id', p)}: {len(meta['layers'])} compressed layers, {100 * zeros:.1f}% exact zeros")
