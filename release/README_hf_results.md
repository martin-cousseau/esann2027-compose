---
license: apache-2.0
tags:
- pruning
- quantization
- model-compression
pretty_name: ESANN 2027 v2 study, result lines and per-item logs
---

# Do pruning and quantization compose? Result lines and per-item logs (ESANN 2027 study, v2)

The raw results of the v2 study, uploaded by the GPU runners as each model state finished: one JSON line per
compressed model state and one gzip file of per-item benchmark correctness per state with benchmarks. The same
files are in the GitHub repository, which also holds the code that made them and the code that analyses them.

- **Code, analysis, paper:** https://github.com/martin-cousseau/esann2027-compose/tree/claude/busy-hopper-ghtynk
- **Compressed model states (exact codes, scales, masks) and a loader:** https://huggingface.co/martincousseau/esann2027-compose-v2
- **One Weights & Biases run per state:** https://wandb.ai/martin-css/esann2027-compose-v2

## Layout

- `results/v2/<family>/<model>.jsonl`: one line per model state. Families: `E1` (the paper's grid in fp32), `E3`
  (quantize-then-prune mechanism arms), `E4` (extra frontier points: 2-bit, 60% pruning), `E5` (calibration seeds 1
  and 2), `E6` (SmolLM2-1.7B), `E2` (same-Hessian layer ablation, one line per layer), and checks: `DET`
  (determinism on a second machine), `REPRO16` / `REPRO32` (precision and hardware), `IMG` (Colab image check).
- `results/v2/items/<state_id>.json.gz`: per task, the item ids and per-item `acc` / `acc_norm` from
  lm-evaluation-harness 0.4.13 on full splits.

Each line records the state (model, order, sparsity, bits, calibration seed), WikiText-2 and C4 perplexity, KL
divergence and top-1 agreement with the dense model, benchmark aggregates, bits per weight under several storage
accountings, per-layer diagnostics, timings, and metadata (GPU, driver, Colab image, library versions, commit, code
hash, determinism flags). Field definitions: `src_v2/pipeline.py` and `scripts/run_grid.py` in the GitHub repository.

## License

Apache-2.0, like the base models (SmolLM2, Qwen2.5) the results are measured on.
