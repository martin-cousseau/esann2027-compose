#!/bin/bash
# Runtime B (esann-b11), 2026-11-16, DECISIONS.md #60: the rest of the same-Hessian ablation (E2) on OLMo-2-1B (every
# second block, 17 of 56 layers done) and Qwen2.5-3B (every fourth, 9 of 63 done), resumed from the Hub copies; then 5
# of the 10 Qwen2.5-1.5B seed-2 states still missing at 4 bits, into their own result file (Qwen2.5-1.5B__d.jsonl).
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-b11 || exit 1
( while true; do python scripts/hub_sync.py put results/v2/E2 >/dev/null 2>&1; sleep 600; done ) &
SYNC=$!
python scripts/hub_sync.py get results/v2/E2/OLMo-2-0425-1B.jsonl results/v2/E2/Qwen2.5-3B.jsonl
E="--locked colab/locked_fingerprint.json --runtime-name esann-b11 --session e2-large"
python scripts/run_e2.py --model allenai/OLMo-2-0425-1B --blocks stride2 $E
python scripts/run_e2.py --model Qwen/Qwen2.5-3B --blocks stride4 $E
python scripts/hub_sync.py put results/v2/E2
kill $SYNC 2>/dev/null
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-b11"
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E5__Qwen2.5-1.5B__d.json --out /content/out/results/v2/E5/Qwen2.5-1.5B__d.jsonl --eval-batch 8 $H --session e5-qwen15-d
echo "CHAIN B16 DONE"
bash colab/next_loop.sh
