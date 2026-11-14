---
license: apache-2.0
base_model:
- HuggingFaceTB/SmolLM2-135M
- HuggingFaceTB/SmolLM2-360M
- Qwen/Qwen2.5-0.5B
- HuggingFaceTB/SmolLM2-1.7B
tags:
- pruning
- quantization
- model-compression
- sparsegpt
- gptq
- obs
---

# Do pruning and quantization compose? Compressed model states (ESANN 2027 study, v2)

This repository holds **157 compressed model states** of 4 base models, every state evaluated in the v2 study of the order of unstructured pruning and weight quantization under Hessian-based (OBS) compensation: pruning alone, quantization alone, and the two combined in three orders (prune then quantize, quantize then prune, and both in one joint sweep), plus the ablation arms that test why quantizing first hurts.

- **Code, result lines and per-item benchmark logs:** [https://github.com/martin-cousseau/esann2027-compose](https://github.com/martin-cousseau/esann2027-compose/tree/claude/busy-hopper-ghtynk) (branch `claude/busy-hopper-ghtynk`)
- **One Weights & Biases run per state (metrics, layer diagnostics, links back here):** [https://wandb.ai/martin-css/esann2027-compose-v2](https://wandb.ai/martin-css/esann2027-compose-v2)
- **Paper:** in preparation for ESANN 2027; citation below once available.

Each file stores the compressed linear layers of the decoder blocks in an exact compact form (integer codes, per-group scale and zero point, sparsity bitmap). Decoding gives the evaluated fp32 weights bit for bit; embeddings, norms and the LM head are the unchanged base model.

## Load a state

```bash
pip install torch transformers safetensors huggingface_hub numpy
python load_esann2027.py models/SmolLM2-135M/SmolLM2-135M__p_then_q__s0.5000__b4__seed0__float32.safetensors
```

```python
from huggingface_hub import hf_hub_download
import load_esann2027 as L  # this repository's loader

path = hf_hub_download("martincousseau/esann2027-compose-v2", "models/SmolLM2-135M/SmolLM2-135M__p_then_q__s0.5000__b4__seed0__float32.safetensors")
model, meta = L.load(path)   # transformers model in fp32, compressed weights in place
```

## File names

`models/<model>/<model>__<order>__s<sparsity>__b<bits>__seed<seed>__float32.safetensors`, with orders `dense`, `prune` (pruning only, `b16`), `quant` (quantization only, `s0.0000`), `joint`, `p_then_q`, `q_then_p` and the ablation arms `q_then_p_refit` (grid re-fitted after pruning), `q_then_p_pmask` (the mask pruning alone would choose), `q_then_p_pmask_refit` (both) and `p_then_q_stale` (P->Q with the grid of the unpruned weights). `seed` is the calibration-sample seed.

## Format `esann2027-v2-compact-1`

JSON metadata under the safetensors metadata key `esann2027` (state, commit, code hash, group size, and one entry per layer: key `<block>.<module>`, shape, kind). Per kind:

| kind | tensors | decoding |
|---|---|---|
| `quant` | `<key>.codes` uint8 (two 4-bit codes per byte, low nibble first, when bits <= 4), `<key>.scale` fp32 and `<key>.zero` uint8 per (row, group of 128 columns), optional `<key>.mask` bitmap | `W = scale * (code - zero)`, then `W[mask] = 0` |
| `sparse_fp32` | `<key>.mask` bitmap, `<key>.values` fp32 kept weights (row-major) | zeros, kept values in place |
| `fp32` | `<key>.weight` | as stored |

Bitmaps use `numpy.packbits` order. The bitmap is stored to tell pruned weights from weights that round to zero; the bits-per-weight columns below count it as 1 bit per weight whenever the state is pruned.

## How the states were made

All states: fp32 model and Hessians on one GPU type (NVIDIA L4), TF32 off, deterministic kernels, the OBS sweep of the paper's code (`src/core.py`), 128 calibration windows of 2048 tokens from C4, asymmetric min-max quantization grid per group of 128 weights. Evaluation: WikiText-2 test perplexity, C4 validation perplexity (256 windows), KL divergence and top-1 agreement with the dense model, and (E1, E4) nine zero-shot tasks with lm-evaluation-harness 0.4.13 on full splits with per-item logs. Every result line records the GPU, driver, Colab image, library versions, commit and code hash; see the GitHub repository.

States by experiment family: E1 (the paper's grid in fp32): 72, E3 (Q->P mechanism arms): 40, E4 (extra frontier points): 15, E5 (calibration seeds 1 and 2): 20, E6 (scale within one family): 10.

## States

bits/w = bits per weight of the compressed linear layers with a 1-bit mask counted for pruned states (the per-group scale and zero point add (16 + b)/128 bits for quantized states and are not included); PPL = WikiText-2 test perplexity; KL = mean KL divergence to the dense model on WikiText-2. Pruning-only states are counted at 16 bits per kept weight (the base models' precision; they are stored here in fp32). The dense states are the base models themselves and have no file here.

<details><summary><b>SmolLM2-135M</b> (45 states; base model <code>HuggingFaceTB/SmolLM2-135M</code>)</summary>

| state | family | seed | bits/w | PPL | C4 PPL | KL | file |
|---|---|---|---|---|---|---|---|
| 25% pruned | E1 | 0 | 13.00 | 16.48 | 21.63 | 0.067 | [SmolLM2-135M__prune__s0.2500__b16__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__prune__s0.2500__b16__seed0__float32.safetensors) |
| 33% pruned | E1 | 0 | 11.67 | 18.37 | 23.58 | 0.173 | [SmolLM2-135M__prune__s0.3333__b16__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__prune__s0.3333__b16__seed0__float32.safetensors) |
| 50% pruned | E1 | 0 | 9.00 | 32.82 | 39.17 | 0.740 | [SmolLM2-135M__prune__s0.5000__b16__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__prune__s0.5000__b16__seed0__float32.safetensors) |
| 60% pruned | E4 | 0 | 7.40 | 71.43 | 80.24 | 1.503 | [SmolLM2-135M__prune__s0.6000__b16__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__prune__s0.6000__b16__seed0__float32.safetensors) |
| 8-bit | E1 | 0 | 8.00 | 15.44 | 20.59 | 0.002 | [SmolLM2-135M__quant__s0.0000__b8__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__quant__s0.0000__b8__seed0__float32.safetensors) |
| 4-bit | E1 | 0 | 4.00 | 21.94 | 27.83 | 0.346 | [SmolLM2-135M__quant__s0.0000__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__quant__s0.0000__b4__seed0__float32.safetensors) |
| 3-bit | E1 | 0 | 3.00 | 57.55 | 62.31 | 1.273 | [SmolLM2-135M__quant__s0.0000__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__quant__s0.0000__b3__seed0__float32.safetensors) |
| 2-bit | E4 | 0 | 2.00 | 94279.66 | 25405.10 | 8.620 | [SmolLM2-135M__quant__s0.0000__b2__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__quant__s0.0000__b2__seed0__float32.safetensors) |
| 25%+4 joint | E1 | 0 | 4.00 | 23.50 | 29.73 | 0.414 | [SmolLM2-135M__joint__s0.2500__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__joint__s0.2500__b4__seed0__float32.safetensors) |
| 25%+4 P->Q | E1 | 0 | 4.00 | 21.99 | 27.59 | 0.348 | [SmolLM2-135M__p_then_q__s0.2500__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__p_then_q__s0.2500__b4__seed0__float32.safetensors) |
| 25%+4 Q->P | E1 | 0 | 4.00 | 24.35 | 30.23 | 0.449 | [SmolLM2-135M__q_then_p__s0.2500__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p__s0.2500__b4__seed0__float32.safetensors) |
| 33%+4 joint | E1 | 0 | 3.67 | 26.23 | 32.23 | 0.514 | [SmolLM2-135M__joint__s0.3333__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__joint__s0.3333__b4__seed0__float32.safetensors) |
| 33%+4 P->Q | E1 | 0 | 3.67 | 24.44 | 30.24 | 0.458 | [SmolLM2-135M__p_then_q__s0.3333__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__p_then_q__s0.3333__b4__seed0__float32.safetensors) |
| 33%+4 Q->P | E1 | 0 | 3.67 | 27.07 | 33.30 | 0.553 | [SmolLM2-135M__q_then_p__s0.3333__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p__s0.3333__b4__seed0__float32.safetensors) |
| 50%+4 joint | E1 | 0 | 3.00 | 44.47 | 54.34 | 1.043 | [SmolLM2-135M__joint__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__joint__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 P->Q | E1 | 0 | 3.00 | 41.20 | 48.37 | 0.966 | [SmolLM2-135M__p_then_q__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__p_then_q__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 P->Q stale grid | E3 | 0 | 3.00 | 41.22 | 49.57 | 0.976 | [SmolLM2-135M__p_then_q_stale__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__p_then_q_stale__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 Q->P | E1 | 0 | 3.00 | 47.33 | 56.63 | 1.112 | [SmolLM2-135M__q_then_p__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 Q->P P-mask | E3 | 0 | 3.00 | 46.78 | 56.38 | 1.086 | [SmolLM2-135M__q_then_p_pmask__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p_pmask__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 Q->P P-mask refit | E3 | 0 | 3.00 | 44.96 | 52.35 | 1.046 | [SmolLM2-135M__q_then_p_pmask_refit__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p_pmask_refit__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 Q->P refit | E3 | 0 | 3.00 | 49.49 | 57.35 | 1.149 | [SmolLM2-135M__q_then_p_refit__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p_refit__s0.5000__b4__seed0__float32.safetensors) |
| 60%+4 joint | E4 | 0 | 2.60 | 97.56 | 105.03 | 1.799 | [SmolLM2-135M__joint__s0.6000__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__joint__s0.6000__b4__seed0__float32.safetensors) |
| 60%+4 P->Q | E4 | 0 | 2.60 | 104.20 | 117.87 | 1.871 | [SmolLM2-135M__p_then_q__s0.6000__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__p_then_q__s0.6000__b4__seed0__float32.safetensors) |
| 60%+4 Q->P | E4 | 0 | 2.60 | 137.79 | 135.98 | 2.151 | [SmolLM2-135M__q_then_p__s0.6000__b4__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p__s0.6000__b4__seed0__float32.safetensors) |
| 25%+3 joint | E1 | 0 | 3.25 | 68.77 | 72.30 | 1.447 | [SmolLM2-135M__joint__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__joint__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 P->Q | E1 | 0 | 3.25 | 53.14 | 55.20 | 1.192 | [SmolLM2-135M__p_then_q__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__p_then_q__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 P->Q stale grid | E3 | 0 | 3.25 | 58.96 | 59.91 | 1.297 | [SmolLM2-135M__p_then_q_stale__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__p_then_q_stale__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 Q->P | E1 | 0 | 3.25 | 57.55 | 62.31 | 1.273 | [SmolLM2-135M__q_then_p__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 Q->P P-mask | E3 | 0 | 3.25 | 52.17 | 54.68 | 1.178 | [SmolLM2-135M__q_then_p_pmask__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p_pmask__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 Q->P P-mask refit | E3 | 0 | 3.25 | 56.56 | 57.58 | 1.259 | [SmolLM2-135M__q_then_p_pmask_refit__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p_pmask_refit__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 Q->P refit | E3 | 0 | 3.25 | 65.62 | 67.73 | 1.402 | [SmolLM2-135M__q_then_p_refit__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p_refit__s0.2500__b3__seed0__float32.safetensors) |
| 33%+3 joint | E1 | 0 | 3.00 | 62.21 | 66.75 | 1.346 | [SmolLM2-135M__joint__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__joint__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 P->Q | E1 | 0 | 3.00 | 53.24 | 54.86 | 1.205 | [SmolLM2-135M__p_then_q__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__p_then_q__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 P->Q stale grid | E3 | 0 | 3.00 | 60.74 | 61.51 | 1.339 | [SmolLM2-135M__p_then_q_stale__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__p_then_q_stale__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 Q->P | E1 | 0 | 3.00 | 68.59 | 73.98 | 1.445 | [SmolLM2-135M__q_then_p__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 Q->P P-mask | E3 | 0 | 3.00 | 60.44 | 62.72 | 1.323 | [SmolLM2-135M__q_then_p_pmask__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p_pmask__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 Q->P P-mask refit | E3 | 0 | 3.00 | 64.18 | 66.63 | 1.397 | [SmolLM2-135M__q_then_p_pmask_refit__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p_pmask_refit__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 Q->P refit | E3 | 0 | 3.00 | 74.11 | 75.94 | 1.527 | [SmolLM2-135M__q_then_p_refit__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p_refit__s0.3333__b3__seed0__float32.safetensors) |
| 50%+3 joint | E1 | 0 | 2.50 | 99.24 | 114.20 | 1.833 | [SmolLM2-135M__joint__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__joint__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 P->Q | E1 | 0 | 2.50 | 86.03 | 92.39 | 1.671 | [SmolLM2-135M__p_then_q__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__p_then_q__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 P->Q stale grid | E3 | 0 | 2.50 | 84.87 | 82.77 | 1.652 | [SmolLM2-135M__p_then_q_stale__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__p_then_q_stale__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 Q->P | E1 | 0 | 2.50 | 152.51 | 147.58 | 2.249 | [SmolLM2-135M__q_then_p__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 Q->P P-mask | E3 | 0 | 2.50 | 115.80 | 118.32 | 1.975 | [SmolLM2-135M__q_then_p_pmask__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p_pmask__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 Q->P P-mask refit | E3 | 0 | 2.50 | 98.22 | 96.95 | 1.812 | [SmolLM2-135M__q_then_p_pmask_refit__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p_pmask_refit__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 Q->P refit | E3 | 0 | 2.50 | 146.92 | 135.85 | 2.193 | [SmolLM2-135M__q_then_p_refit__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-135M/SmolLM2-135M__q_then_p_refit__s0.5000__b3__seed0__float32.safetensors) |

</details>

<details><summary><b>SmolLM2-360M</b> (65 states; base model <code>HuggingFaceTB/SmolLM2-360M</code>)</summary>

| state | family | seed | bits/w | PPL | C4 PPL | KL | file |
|---|---|---|---|---|---|---|---|
| 25% pruned | E1 | 0 | 13.00 | 12.08 | 16.74 | 0.051 | [SmolLM2-360M__prune__s0.2500__b16__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__prune__s0.2500__b16__seed0__float32.safetensors) |
| 33% pruned | E1 | 0 | 11.67 | 12.86 | 17.58 | 0.118 | [SmolLM2-360M__prune__s0.3333__b16__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__prune__s0.3333__b16__seed0__float32.safetensors) |
| 50% pruned | E1 | 0 | 9.00 | 18.63 | 23.99 | 0.497 | [SmolLM2-360M__prune__s0.5000__b16__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__prune__s0.5000__b16__seed0__float32.safetensors) |
| 60% pruned | E4 | 0 | 7.40 | 34.64 | 44.30 | 1.106 | [SmolLM2-360M__prune__s0.6000__b16__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__prune__s0.6000__b16__seed0__float32.safetensors) |
| 8-bit | E1 | 0 | 8.00 | 11.47 | 16.15 | 0.001 | [SmolLM2-360M__quant__s0.0000__b8__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__quant__s0.0000__b8__seed0__float32.safetensors) |
| 4-bit | E1 | 0 | 4.00 | 13.27 | 18.06 | 0.143 | [SmolLM2-360M__quant__s0.0000__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__quant__s0.0000__b4__seed0__float32.safetensors) |
| 3-bit | E1 | 0 | 3.00 | 21.97 | 27.28 | 0.650 | [SmolLM2-360M__quant__s0.0000__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__quant__s0.0000__b3__seed0__float32.safetensors) |
| 2-bit | E4 | 0 | 2.00 | 32421.62 | 11417.49 | 7.903 | [SmolLM2-360M__quant__s0.0000__b2__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__quant__s0.0000__b2__seed0__float32.safetensors) |
| 25%+4 joint | E1 | 0 | 4.00 | 13.20 | 17.88 | 0.139 | [SmolLM2-360M__joint__s0.2500__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__joint__s0.2500__b4__seed0__float32.safetensors) |
| 25%+4 P->Q | E1 | 0 | 4.00 | 13.34 | 18.02 | 0.146 | [SmolLM2-360M__p_then_q__s0.2500__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q__s0.2500__b4__seed0__float32.safetensors) |
| 25%+4 Q->P | E1 | 0 | 4.00 | 14.38 | 19.17 | 0.222 | [SmolLM2-360M__q_then_p__s0.2500__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p__s0.2500__b4__seed0__float32.safetensors) |
| 33%+4 joint | E1 | 0 | 3.67 | 14.01 | 18.83 | 0.205 | [SmolLM2-360M__joint__s0.3333__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__joint__s0.3333__b4__seed0__float32.safetensors) |
| 33%+4 P->Q | E1 | 0 | 3.67 | 14.19 | 19.02 | 0.215 | [SmolLM2-360M__p_then_q__s0.3333__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q__s0.3333__b4__seed0__float32.safetensors) |
| 33%+4 Q->P | E1 | 0 | 3.67 | 15.58 | 20.45 | 0.302 | [SmolLM2-360M__q_then_p__s0.3333__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p__s0.3333__b4__seed0__float32.safetensors) |
| 50%+4 joint | E1 | 0 | 3.00 | 20.48 | 26.32 | 0.589 | [SmolLM2-360M__joint__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__joint__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 P->Q | E1 | 0 | 3.00 | 21.29 | 26.84 | 0.631 | [SmolLM2-360M__p_then_q__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 P->Q stale grid | E3 | 0 | 3.00 | 21.01 | 27.03 | 0.617 | [SmolLM2-360M__p_then_q_stale__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q_stale__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 Q->P | E1 | 0 | 3.00 | 23.63 | 29.48 | 0.727 | [SmolLM2-360M__q_then_p__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 Q->P P-mask | E3 | 0 | 3.00 | 22.17 | 28.12 | 0.670 | [SmolLM2-360M__q_then_p_pmask__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p_pmask__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 Q->P P-mask refit | E3 | 0 | 3.00 | 21.76 | 27.89 | 0.654 | [SmolLM2-360M__q_then_p_pmask_refit__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p_pmask_refit__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 Q->P refit | E3 | 0 | 3.00 | 23.04 | 28.65 | 0.702 | [SmolLM2-360M__q_then_p_refit__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p_refit__s0.5000__b4__seed0__float32.safetensors) |
| 60%+4 joint | E4 | 0 | 2.60 | 38.47 | 49.06 | 1.217 | [SmolLM2-360M__joint__s0.6000__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__joint__s0.6000__b4__seed0__float32.safetensors) |
| 60%+4 P->Q | E4 | 0 | 2.60 | 39.01 | 48.31 | 1.233 | [SmolLM2-360M__p_then_q__s0.6000__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q__s0.6000__b4__seed0__float32.safetensors) |
| 60%+4 Q->P | E4 | 0 | 2.60 | 43.96 | 58.27 | 1.343 | [SmolLM2-360M__q_then_p__s0.6000__b4__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p__s0.6000__b4__seed0__float32.safetensors) |
| 25%+3 joint | E1 | 0 | 3.25 | 20.00 | 25.01 | 0.564 | [SmolLM2-360M__joint__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__joint__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 P->Q | E1 | 0 | 3.25 | 20.38 | 25.35 | 0.575 | [SmolLM2-360M__p_then_q__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 P->Q stale grid | E3 | 0 | 3.25 | 20.31 | 25.31 | 0.578 | [SmolLM2-360M__p_then_q_stale__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q_stale__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 Q->P | E1 | 0 | 3.25 | 21.97 | 27.28 | 0.650 | [SmolLM2-360M__q_then_p__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 Q->P P-mask | E3 | 0 | 3.25 | 22.72 | 26.52 | 0.686 | [SmolLM2-360M__q_then_p_pmask__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p_pmask__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 Q->P P-mask refit | E3 | 0 | 3.25 | 21.52 | 26.49 | 0.632 | [SmolLM2-360M__q_then_p_pmask_refit__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p_pmask_refit__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 Q->P refit | E3 | 0 | 3.25 | 25.85 | 26.79 | 0.805 | [SmolLM2-360M__q_then_p_refit__s0.2500__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p_refit__s0.2500__b3__seed0__float32.safetensors) |
| 33%+3 joint | E1 | 0 | 3.00 | 20.03 | 25.10 | 0.567 | [SmolLM2-360M__joint__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__joint__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 P->Q | E1 | 0 | 3.00 | 20.68 | 26.07 | 0.596 | [SmolLM2-360M__p_then_q__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 P->Q stale grid | E3 | 0 | 3.00 | 20.65 | 26.06 | 0.595 | [SmolLM2-360M__p_then_q_stale__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q_stale__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 Q->P | E1 | 0 | 3.00 | 28.88 | 29.77 | 0.923 | [SmolLM2-360M__q_then_p__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 Q->P P-mask | E3 | 0 | 3.00 | 22.52 | 27.82 | 0.674 | [SmolLM2-360M__q_then_p_pmask__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p_pmask__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 Q->P P-mask refit | E3 | 0 | 3.00 | 23.04 | 28.52 | 0.702 | [SmolLM2-360M__q_then_p_pmask_refit__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p_pmask_refit__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 Q->P refit | E3 | 0 | 3.00 | 29.71 | 29.95 | 0.946 | [SmolLM2-360M__q_then_p_refit__s0.3333__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p_refit__s0.3333__b3__seed0__float32.safetensors) |
| 50%+3 joint | E1 | 0 | 2.50 | 29.14 | 34.72 | 0.938 | [SmolLM2-360M__joint__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__joint__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 P->Q | E1 | 0 | 2.50 | 29.62 | 37.18 | 0.954 | [SmolLM2-360M__p_then_q__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 P->Q stale grid | E3 | 0 | 2.50 | 30.66 | 37.69 | 0.987 | [SmolLM2-360M__p_then_q_stale__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q_stale__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 Q->P | E1 | 0 | 2.50 | 49.29 | 52.46 | 1.462 | [SmolLM2-360M__q_then_p__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 Q->P P-mask | E3 | 0 | 2.50 | 37.59 | 47.29 | 1.200 | [SmolLM2-360M__q_then_p_pmask__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p_pmask__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 Q->P P-mask refit | E3 | 0 | 2.50 | 39.20 | 46.99 | 1.235 | [SmolLM2-360M__q_then_p_pmask_refit__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p_pmask_refit__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 Q->P refit | E3 | 0 | 2.50 | 62.75 | 51.20 | 1.693 | [SmolLM2-360M__q_then_p_refit__s0.5000__b3__seed0__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p_refit__s0.5000__b3__seed0__float32.safetensors) |
| 4-bit | E5 | 1 | 4.00 | 13.23 | 18.06 | 0.140 | [SmolLM2-360M__quant__s0.0000__b4__seed1__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__quant__s0.0000__b4__seed1__float32.safetensors) |
| 25%+4 joint | E5 | 1 | 4.00 | 13.15 | 17.84 | 0.135 | [SmolLM2-360M__joint__s0.2500__b4__seed1__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__joint__s0.2500__b4__seed1__float32.safetensors) |
| 25%+4 P->Q | E5 | 1 | 4.00 | 13.25 | 18.04 | 0.146 | [SmolLM2-360M__p_then_q__s0.2500__b4__seed1__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q__s0.2500__b4__seed1__float32.safetensors) |
| 25%+4 Q->P | E5 | 1 | 4.00 | 14.30 | 19.13 | 0.218 | [SmolLM2-360M__q_then_p__s0.2500__b4__seed1__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p__s0.2500__b4__seed1__float32.safetensors) |
| 33%+4 joint | E5 | 1 | 3.67 | 14.09 | 18.84 | 0.209 | [SmolLM2-360M__joint__s0.3333__b4__seed1__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__joint__s0.3333__b4__seed1__float32.safetensors) |
| 33%+4 P->Q | E5 | 1 | 3.67 | 14.24 | 19.03 | 0.219 | [SmolLM2-360M__p_then_q__s0.3333__b4__seed1__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q__s0.3333__b4__seed1__float32.safetensors) |
| 33%+4 Q->P | E5 | 1 | 3.67 | 15.49 | 20.46 | 0.299 | [SmolLM2-360M__q_then_p__s0.3333__b4__seed1__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p__s0.3333__b4__seed1__float32.safetensors) |
| 50%+4 joint | E5 | 1 | 3.00 | 20.38 | 25.94 | 0.585 | [SmolLM2-360M__joint__s0.5000__b4__seed1__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__joint__s0.5000__b4__seed1__float32.safetensors) |
| 50%+4 P->Q | E5 | 1 | 3.00 | 21.27 | 26.92 | 0.626 | [SmolLM2-360M__p_then_q__s0.5000__b4__seed1__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q__s0.5000__b4__seed1__float32.safetensors) |
| 50%+4 Q->P | E5 | 1 | 3.00 | 23.71 | 29.75 | 0.732 | [SmolLM2-360M__q_then_p__s0.5000__b4__seed1__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p__s0.5000__b4__seed1__float32.safetensors) |
| 4-bit | E5 | 2 | 4.00 | 13.22 | 18.02 | 0.145 | [SmolLM2-360M__quant__s0.0000__b4__seed2__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__quant__s0.0000__b4__seed2__float32.safetensors) |
| 25%+4 joint | E5 | 2 | 4.00 | 13.20 | 17.87 | 0.142 | [SmolLM2-360M__joint__s0.2500__b4__seed2__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__joint__s0.2500__b4__seed2__float32.safetensors) |
| 25%+4 P->Q | E5 | 2 | 4.00 | 13.27 | 17.96 | 0.146 | [SmolLM2-360M__p_then_q__s0.2500__b4__seed2__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q__s0.2500__b4__seed2__float32.safetensors) |
| 25%+4 Q->P | E5 | 2 | 4.00 | 14.29 | 19.07 | 0.219 | [SmolLM2-360M__q_then_p__s0.2500__b4__seed2__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p__s0.2500__b4__seed2__float32.safetensors) |
| 33%+4 joint | E5 | 2 | 3.67 | 14.16 | 18.82 | 0.213 | [SmolLM2-360M__joint__s0.3333__b4__seed2__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__joint__s0.3333__b4__seed2__float32.safetensors) |
| 33%+4 P->Q | E5 | 2 | 3.67 | 14.46 | 19.08 | 0.235 | [SmolLM2-360M__p_then_q__s0.3333__b4__seed2__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q__s0.3333__b4__seed2__float32.safetensors) |
| 33%+4 Q->P | E5 | 2 | 3.67 | 15.67 | 20.50 | 0.311 | [SmolLM2-360M__q_then_p__s0.3333__b4__seed2__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p__s0.3333__b4__seed2__float32.safetensors) |
| 50%+4 joint | E5 | 2 | 3.00 | 20.51 | 25.85 | 0.587 | [SmolLM2-360M__joint__s0.5000__b4__seed2__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__joint__s0.5000__b4__seed2__float32.safetensors) |
| 50%+4 P->Q | E5 | 2 | 3.00 | 21.16 | 26.93 | 0.625 | [SmolLM2-360M__p_then_q__s0.5000__b4__seed2__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__p_then_q__s0.5000__b4__seed2__float32.safetensors) |
| 50%+4 Q->P | E5 | 2 | 3.00 | 23.44 | 29.12 | 0.722 | [SmolLM2-360M__q_then_p__s0.5000__b4__seed2__float32.safetensors](models/SmolLM2-360M/SmolLM2-360M__q_then_p__s0.5000__b4__seed2__float32.safetensors) |

</details>

<details><summary><b>Qwen2.5-0.5B</b> (37 states; base model <code>Qwen/Qwen2.5-0.5B</code>)</summary>

| state | family | seed | bits/w | PPL | C4 PPL | KL | file |
|---|---|---|---|---|---|---|---|
| 25% pruned | E1 | 0 | 13.00 | 13.63 | 20.21 | 0.045 | [Qwen2.5-0.5B__prune__s0.2500__b16__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__prune__s0.2500__b16__seed0__float32.safetensors) |
| 33% pruned | E1 | 0 | 11.67 | 14.53 | 21.12 | 0.111 | [Qwen2.5-0.5B__prune__s0.3333__b16__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__prune__s0.3333__b16__seed0__float32.safetensors) |
| 50% pruned | E1 | 0 | 9.00 | 20.15 | 26.86 | 0.440 | [Qwen2.5-0.5B__prune__s0.5000__b16__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__prune__s0.5000__b16__seed0__float32.safetensors) |
| 60% pruned | E4 | 0 | 7.40 | 39.13 | 44.88 | 1.102 | [Qwen2.5-0.5B__prune__s0.6000__b16__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__prune__s0.6000__b16__seed0__float32.safetensors) |
| 8-bit | E1 | 0 | 8.00 | 13.07 | 19.62 | 0.000 | [Qwen2.5-0.5B__quant__s0.0000__b8__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__quant__s0.0000__b8__seed0__float32.safetensors) |
| 4-bit | E1 | 0 | 4.00 | 14.31 | 21.07 | 0.098 | [Qwen2.5-0.5B__quant__s0.0000__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__quant__s0.0000__b4__seed0__float32.safetensors) |
| 3-bit | E1 | 0 | 3.00 | 21.84 | 28.33 | 0.514 | [Qwen2.5-0.5B__quant__s0.0000__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__quant__s0.0000__b3__seed0__float32.safetensors) |
| 2-bit | E4 | 0 | 2.00 | 7398.83 | 3043.48 | 6.320 | [Qwen2.5-0.5B__quant__s0.0000__b2__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__quant__s0.0000__b2__seed0__float32.safetensors) |
| 25%+4 joint | E1 | 0 | 4.00 | 14.63 | 21.43 | 0.122 | [Qwen2.5-0.5B__joint__s0.2500__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__joint__s0.2500__b4__seed0__float32.safetensors) |
| 25%+4 P->Q | E1 | 0 | 4.00 | 14.77 | 21.50 | 0.129 | [Qwen2.5-0.5B__p_then_q__s0.2500__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__p_then_q__s0.2500__b4__seed0__float32.safetensors) |
| 25%+4 Q->P | E1 | 0 | 4.00 | 15.07 | 21.78 | 0.148 | [Qwen2.5-0.5B__q_then_p__s0.2500__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__q_then_p__s0.2500__b4__seed0__float32.safetensors) |
| 33%+4 joint | E1 | 0 | 3.67 | 15.64 | 22.33 | 0.187 | [Qwen2.5-0.5B__joint__s0.3333__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__joint__s0.3333__b4__seed0__float32.safetensors) |
| 33%+4 P->Q | E1 | 0 | 3.67 | 15.63 | 22.43 | 0.190 | [Qwen2.5-0.5B__p_then_q__s0.3333__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__p_then_q__s0.3333__b4__seed0__float32.safetensors) |
| 33%+4 Q->P | E1 | 0 | 3.67 | 16.07 | 22.86 | 0.213 | [Qwen2.5-0.5B__q_then_p__s0.3333__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__q_then_p__s0.3333__b4__seed0__float32.safetensors) |
| 50%+4 joint | E1 | 0 | 3.00 | 21.91 | 28.74 | 0.523 | [Qwen2.5-0.5B__joint__s0.5000__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__joint__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 P->Q | E1 | 0 | 3.00 | 22.18 | 28.95 | 0.541 | [Qwen2.5-0.5B__p_then_q__s0.5000__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__p_then_q__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 P->Q stale grid | E3 | 0 | 3.00 | 23.05 | 29.93 | 0.577 | [Qwen2.5-0.5B__p_then_q_stale__s0.5000__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__p_then_q_stale__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 Q->P | E1 | 0 | 3.00 | 22.39 | 30.24 | 0.550 | [Qwen2.5-0.5B__q_then_p__s0.5000__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__q_then_p__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 Q->P P-mask | E3 | 0 | 3.00 | 22.79 | 29.93 | 0.567 | [Qwen2.5-0.5B__q_then_p_pmask__s0.5000__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__q_then_p_pmask__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 Q->P P-mask refit | E3 | 0 | 3.00 | 22.83 | 30.00 | 0.573 | [Qwen2.5-0.5B__q_then_p_pmask_refit__s0.5000__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__q_then_p_pmask_refit__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 Q->P refit | E3 | 0 | 3.00 | 22.60 | 30.01 | 0.563 | [Qwen2.5-0.5B__q_then_p_refit__s0.5000__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__q_then_p_refit__s0.5000__b4__seed0__float32.safetensors) |
| 60%+4 joint | E4 | 0 | 2.60 | 42.75 | 49.44 | 1.190 | [Qwen2.5-0.5B__joint__s0.6000__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__joint__s0.6000__b4__seed0__float32.safetensors) |
| 60%+4 P->Q | E4 | 0 | 2.60 | 47.08 | 51.02 | 1.286 | [Qwen2.5-0.5B__p_then_q__s0.6000__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__p_then_q__s0.6000__b4__seed0__float32.safetensors) |
| 60%+4 Q->P | E4 | 0 | 2.60 | 48.26 | 55.50 | 1.316 | [Qwen2.5-0.5B__q_then_p__s0.6000__b4__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__q_then_p__s0.6000__b4__seed0__float32.safetensors) |
| 25%+3 joint | E1 | 0 | 3.25 | 21.53 | 28.29 | 0.505 | [Qwen2.5-0.5B__joint__s0.2500__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__joint__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 P->Q | E1 | 0 | 3.25 | 21.92 | 28.69 | 0.521 | [Qwen2.5-0.5B__p_then_q__s0.2500__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__p_then_q__s0.2500__b3__seed0__float32.safetensors) |
| 25%+3 Q->P | E1 | 0 | 3.25 | 21.84 | 28.33 | 0.514 | [Qwen2.5-0.5B__q_then_p__s0.2500__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__q_then_p__s0.2500__b3__seed0__float32.safetensors) |
| 33%+3 joint | E1 | 0 | 3.00 | 21.95 | 28.70 | 0.525 | [Qwen2.5-0.5B__joint__s0.3333__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__joint__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 P->Q | E1 | 0 | 3.00 | 22.96 | 29.50 | 0.566 | [Qwen2.5-0.5B__p_then_q__s0.3333__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__p_then_q__s0.3333__b3__seed0__float32.safetensors) |
| 33%+3 Q->P | E1 | 0 | 3.00 | 23.15 | 29.40 | 0.580 | [Qwen2.5-0.5B__q_then_p__s0.3333__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__q_then_p__s0.3333__b3__seed0__float32.safetensors) |
| 50%+3 joint | E1 | 0 | 2.50 | 28.62 | 35.45 | 0.792 | [Qwen2.5-0.5B__joint__s0.5000__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__joint__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 P->Q | E1 | 0 | 2.50 | 29.96 | 36.54 | 0.841 | [Qwen2.5-0.5B__p_then_q__s0.5000__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__p_then_q__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 P->Q stale grid | E3 | 0 | 2.50 | 32.08 | 38.49 | 0.909 | [Qwen2.5-0.5B__p_then_q_stale__s0.5000__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__p_then_q_stale__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 Q->P | E1 | 0 | 2.50 | 34.63 | 42.36 | 0.976 | [Qwen2.5-0.5B__q_then_p__s0.5000__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__q_then_p__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 Q->P P-mask | E3 | 0 | 2.50 | 36.61 | 43.92 | 1.040 | [Qwen2.5-0.5B__q_then_p_pmask__s0.5000__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__q_then_p_pmask__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 Q->P P-mask refit | E3 | 0 | 2.50 | 34.02 | 41.63 | 0.963 | [Qwen2.5-0.5B__q_then_p_pmask_refit__s0.5000__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__q_then_p_pmask_refit__s0.5000__b3__seed0__float32.safetensors) |
| 50%+3 Q->P refit | E3 | 0 | 2.50 | 35.01 | 42.90 | 0.989 | [Qwen2.5-0.5B__q_then_p_refit__s0.5000__b3__seed0__float32.safetensors](models/Qwen2.5-0.5B/Qwen2.5-0.5B__q_then_p_refit__s0.5000__b3__seed0__float32.safetensors) |

</details>

<details><summary><b>SmolLM2-1.7B</b> (10 states; base model <code>HuggingFaceTB/SmolLM2-1.7B</code>)</summary>

| state | family | seed | bits/w | PPL | C4 PPL | KL | file |
|---|---|---|---|---|---|---|---|
| 25% pruned | E6 | 0 | 13.00 | 8.51 | 12.37 | 0.046 | [SmolLM2-1.7B__prune__s0.2500__b16__seed0__float32.safetensors](models/SmolLM2-1.7B/SmolLM2-1.7B__prune__s0.2500__b16__seed0__float32.safetensors) |
| 50% pruned | E6 | 0 | 9.00 | 13.51 | 17.29 | 0.506 | [SmolLM2-1.7B__prune__s0.5000__b16__seed0__float32.safetensors](models/SmolLM2-1.7B/SmolLM2-1.7B__prune__s0.5000__b16__seed0__float32.safetensors) |
| 4-bit | E6 | 0 | 4.00 | 8.71 | 12.58 | 0.070 | [SmolLM2-1.7B__quant__s0.0000__b4__seed0__float32.safetensors](models/SmolLM2-1.7B/SmolLM2-1.7B__quant__s0.0000__b4__seed0__float32.safetensors) |
| 3-bit | E6 | 0 | 3.00 | 13.01 | 15.58 | 0.468 | [SmolLM2-1.7B__quant__s0.0000__b3__seed0__float32.safetensors](models/SmolLM2-1.7B/SmolLM2-1.7B__quant__s0.0000__b3__seed0__float32.safetensors) |
| 25%+4 joint | E6 | 0 | 4.00 | 9.00 | 12.87 | 0.101 | [SmolLM2-1.7B__joint__s0.2500__b4__seed0__float32.safetensors](models/SmolLM2-1.7B/SmolLM2-1.7B__joint__s0.2500__b4__seed0__float32.safetensors) |
| 25%+4 P->Q | E6 | 0 | 4.00 | 9.06 | 12.91 | 0.106 | [SmolLM2-1.7B__p_then_q__s0.2500__b4__seed0__float32.safetensors](models/SmolLM2-1.7B/SmolLM2-1.7B__p_then_q__s0.2500__b4__seed0__float32.safetensors) |
| 25%+4 Q->P | E6 | 0 | 4.00 | 9.19 | 13.07 | 0.122 | [SmolLM2-1.7B__q_then_p__s0.2500__b4__seed0__float32.safetensors](models/SmolLM2-1.7B/SmolLM2-1.7B__q_then_p__s0.2500__b4__seed0__float32.safetensors) |
| 50%+4 joint | E6 | 0 | 3.00 | 14.13 | 18.08 | 0.549 | [SmolLM2-1.7B__joint__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-1.7B/SmolLM2-1.7B__joint__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 P->Q | E6 | 0 | 3.00 | 14.11 | 18.23 | 0.548 | [SmolLM2-1.7B__p_then_q__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-1.7B/SmolLM2-1.7B__p_then_q__s0.5000__b4__seed0__float32.safetensors) |
| 50%+4 Q->P | E6 | 0 | 3.00 | 15.02 | 19.62 | 0.608 | [SmolLM2-1.7B__q_then_p__s0.5000__b4__seed0__float32.safetensors](models/SmolLM2-1.7B/SmolLM2-1.7B__q_then_p__s0.5000__b4__seed0__float32.safetensors) |

</details>

## License

Derived from the base models, all released under Apache-2.0; the compressed states keep that license.

## Citation

A citation will be added when the ESANN 2027 paper is available.
