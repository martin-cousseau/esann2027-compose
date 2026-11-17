#!/bin/bash
# Runtime B (2026-11-15, DECISIONS.md #56): the 3-bit calibration seeds 1 and 2 of Qwen2.5-1.5B (20 states), into its
# own result file results/v2/E5/Qwen2.5-1.5B__b.jsonl (one writer per file); runtime A runs the 4-bit seeds after the
# last Qwen2.5-3B states. Perplexity, C4 and KL, no benchmarks, no export.
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-b9 || exit 1
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-b9"
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E5__Qwen2.5-1.5B__b.json --out /content/out/results/v2/E5/Qwen2.5-1.5B__b.jsonl --eval-batch 8 $H --session e5-qwen15-b
echo "CHAIN B14 DONE"
