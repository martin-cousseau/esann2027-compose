#!/bin/bash
# Runtime B (2026-11-15, DECISIONS.md #50): part c of the Qwen2.5-3B grid completion (Q->P at 25%+3, the 33%+3 and
# 50%+3 cells), into its own result file results/v2/E6/Qwen2.5-3B__c.jsonl (one writer per file). Perplexity only.
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-b8 || exit 1
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-b8"
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-3B --queue colab/queues/E6__Qwen2.5-3B__c.json --out /content/out/results/v2/E6/Qwen2.5-3B__c.jsonl --eval-batch 4 --no-dense-copy $H --session e6-qwen3b-c
echo "CHAIN B13 DONE"
