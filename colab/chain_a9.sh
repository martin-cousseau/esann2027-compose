#!/bin/bash
# Runtime A (2026-11-12, DECISIONS.md #38): E6 on Qwen2.5-1.5B (11 states; perplexity, C4, KL), scale within Qwen.
cd /content/work
export ESANN_OUT=/content/out
COMMON="--locked colab/locked_fingerprint.json --export-models --hf-repo martincousseau/esann2027-v2-results --hf-models-repo martincousseau/esann2027-compose-v2 --runtime-name esann-a6"
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E6__Qwen2.5-1.5B.json --eval-batch 8 $COMMON --session e6-qwen
echo "CHAIN A9 DONE"
