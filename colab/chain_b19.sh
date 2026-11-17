#!/bin/bash
# Runtime B (esann-b13), 2026-11-16 17:05 UTC, DECISIONS.md #64: the other 6 Qwen2.5-1.5B seed-2 states missing at 4 bits
# (Qwen2.5-1.5B__d.jsonl). Each finished state is uploaded to the Hub by run_grid.
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-b13 || exit 1
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-b13"
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E5__Qwen2.5-1.5B__d.json --out /content/out/results/v2/E5/Qwen2.5-1.5B__d.jsonl --eval-batch 8 $H --session e5-qwen15-d
echo "CHAIN B19 DONE"
