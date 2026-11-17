---
license: apache-2.0
tags:
- pruning
- quantization
- model-compression
pretty_name: ESANN 2027 v2 study, result lines and per-item logs
---

# Do pruning and quantization compose? Results, supplementary tables and per-item logs (ESANN 2027 study, v2)

The results of the v2 study: one JSON line per compressed model state (or per layer for the two layer-level controls),
uploaded by the GPU runners as each run finished, one gzip file of per-item benchmark correctness per state with
benchmarks, and the paper's supplementary tables. The same files are in the GitHub repository, which also holds the
code that made them and the code that analyses them.

- **Supplementary tables of the paper:** [`supplement/`](./supplement) (`supplement.pdf`, CSV files, `README.md` with
  every column), generated from the analysis tables; nothing in them is typed by hand.
- **Code, analysis:** https://github.com/martin-cousseau/esann2027-compose/tree/claude/busy-hopper-ghtynk
- **Compressed model states (exact codes, scales, masks) and a loader:** https://huggingface.co/martincousseau/esann2027-compose-v2
- **One Weights & Biases run per state:** https://wandb.ai/martin-css/esann2027-compose-v2

## Layout

- `supplement/`: `supplement.pdf` (every state of every model at calibration seed 0 with all its measurements, the
  per-task scores, the layer-level controls and the weaker-first test per model and setting) and its CSV files:
  `complete_results.csv`, `task_scores.csv`, `seeds.csv`, `layer_ablation.csv`, `no_compensation.csv`,
  `weaker_first.csv`; `README.md` defines every column.
- `results/v2/<family>/<model>*.jsonl`: one line per model state. Families: `E1` (the grid on SmolLM2-135M,
  SmolLM2-360M and Qwen2.5-0.5B), `E6` (the same grid on OLMo-2-1B, Qwen2.5-1.5B, Qwen2.5-3B and SmolLM2-1.7B),
  `E6B` (states of the larger models re-run with benchmarks), `E5` (calibration seeds 1 and 2 for the three small
  models and Qwen2.5-1.5B), `E3` (quantize-then-prune mechanism arms), `E4` (extra frontier points: 2-bit, 60%
  pruning); per layer: `E2` (one dense Hessian shared by all orders) and `E9` (the same layers without error
  compensation, with the terms of the paper's decomposition); checks: `DET` (determinism on a second GPU), `REPRO16`
  / `REPRO32` (precision and hardware), `BATCH` (evaluation batch size), `IMG` (Colab image check).
- `results/v2/items/<state_id>.json.gz`: per task, the item ids and per-item `acc` / `acc_norm` from
  lm-evaluation-harness 0.4.13 on full splits.

Each state line records the state (model, order, sparsity, bits, calibration seed), WikiText-2 and C4 perplexity, KL
divergence and top-1 agreement with the dense model, benchmark aggregates, bits per weight under several storage
accountings, per-layer diagnostics, timings, and metadata (GPU, driver, Colab image, library versions, commit, code
hash, determinism flags). Field definitions: `src_v2/pipeline.py`, `scripts/run_grid.py`, `scripts/run_e2.py` and
`scripts/run_e9.py` in the GitHub repository.

## License

Apache-2.0. The results are measured on SmolLM2, Qwen2.5 and OLMo 2 base models; the Qwen2.5-3B results are
measurements only (its weights, under the Qwen research license, are not redistributed here or in the model
repository).
