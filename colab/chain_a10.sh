#!/bin/bash
# Runtime A, after chain_a9 (DECISIONS.md #40): E6 Qwen2.5-1.5B to the full grid, E6 Qwen2.5-3B (perplexity only, no
# KL copy, no model export: its licence is not Apache-2.0), then E5 seeds of SmolLM2-135M (4- and 3-bit cells).
cd /content/work
export ESANN_OUT=/content/out
until grep -q "CHAIN A9 DONE" /content/out/logs/runner/chain_a9.log 2>/dev/null; do sleep 60; done
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-a6"
X="--export-models --hf-models-repo martincousseau/esann2027-compose-v2"
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E6__Qwen2.5-1.5B.json --eval-batch 8 $H $X --session e6-qwen
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-3B --queue colab/queues/E6__Qwen2.5-3B.json --eval-batch 4 --no-dense-copy $H --session e6-qwen3b
python scripts/run_grid.py --family E5 --model HuggingFaceTB/SmolLM2-135M --queue colab/queues/E5__SmolLM2-135M.json --eval-batch 16 $H $X --session e5-135m
echo "CHAIN A10 DONE"
