#!/bin/bash
# Runtime A (2026-11-15, DECISIONS.md #55, #56): the last 5 Qwen2.5-3B states after the compute units ran out (#54):
# part b (25%+3 joint and P->Q) and part c (50%+3 in the three orders). The existing lines of Qwen2.5-3B__b and __c are
# put here first, so this runtime is the single writer of both files and the 8 finished states are skipped.
# Perplexity only, no export (#40).
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-a10 || exit 1
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-a10"
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-3B --queue colab/queues/E6__Qwen2.5-3B__b.json --out /content/out/results/v2/E6/Qwen2.5-3B__b.jsonl --eval-batch 4 --no-dense-copy $H --session e6-qwen3b-b
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-3B --queue colab/queues/E6__Qwen2.5-3B__c.json --out /content/out/results/v2/E6/Qwen2.5-3B__c.jsonl --eval-batch 4 --no-dense-copy $H --session e6-qwen3b-c
# then the 4-bit calibration seeds 1 and 2 of Qwen2.5-1.5B (DECISIONS.md #56; this runtime writes E5/Qwen2.5-1.5B.jsonl)
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E5__Qwen2.5-1.5B.json --eval-batch 8 $H --session e5-qwen15
echo "CHAIN A16 DONE"
