#!/bin/bash
# Runtime B (2026-11-12, DECISIONS.md #38): the BATCH check (four early states re-evaluated with the batch groups
# of every later state; no model export), then E5 seeds 1 and 2 of Qwen2.5-0.5B (perplexity, C4, KL).
cd /content/work
export ESANN_OUT=/content/out
COMMON="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-b5"
python scripts/run_grid.py --family BATCH --model HuggingFaceTB/SmolLM2-135M --queue colab/queues/BATCH__SmolLM2-135M.json --eval-batch 16 $COMMON --session batch
python scripts/run_grid.py --family BATCH --model Qwen/Qwen2.5-0.5B --queue colab/queues/BATCH__Qwen2.5-0.5B.json --eval-batch 8 $COMMON --session batch
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E5__Qwen2.5-0.5B.json --eval-batch 8 --export-models --hf-models-repo martincousseau/esann2027-compose-v2 $COMMON --session e5-qwen
echo "CHAIN B7 DONE"
