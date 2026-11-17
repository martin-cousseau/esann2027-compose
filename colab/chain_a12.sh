#!/bin/bash
# Runtime A (2026-11-13 19:10 UTC, DECISIONS.md #45): replaces the rest of chain_a11 so that Qwen2.5-3B runs on this
# runtime (a third runtime is refused). Waits for chain_a11's running step (E6B SmolLM2-1.7B), then the E6B re-runs of
# Qwen2.5-1.5B, Qwen2.5-1.5B to the full grid, and Qwen2.5-3B (perplexity only; no second fp32 copy for KL, no export).
# The SmolLM2 seed queues move to runtime B (chain_b10); this runtime no longer writes E5 files.
cd /content/work
export ESANN_OUT=/content/out
while pgrep -f "run_grid.py --family E6B --model HuggingFaceTB/SmolLM2-1.7B" >/dev/null; do sleep 60; done
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-a7"
X="--export-models --hf-models-repo martincousseau/esann2027-compose-v2"
python scripts/run_grid.py --family E6B --model HuggingFaceTB/SmolLM2-1.7B --queue colab/queues/E6B__SmolLM2-1.7B.json --eval-batch 8 $H --session e6b-smol
python scripts/run_grid.py --family E6B --model Qwen/Qwen2.5-1.5B --queue colab/queues/E6B__Qwen2.5-1.5B.json --eval-batch 8 $H --session e6b-qwen
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E6__Qwen2.5-1.5B.json --eval-batch 8 $H $X --session e6-qwen
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-3B --queue colab/queues/E6__Qwen2.5-3B.json --eval-batch 4 --no-dense-copy $H --session e6-qwen3b
echo "CHAIN A12 DONE"
