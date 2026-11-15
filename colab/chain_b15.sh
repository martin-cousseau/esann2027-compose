#!/bin/bash
# Runtime B (esann-b10), 2026-11-15 22:40 UTC, DECISIONS.md #58: the same-Hessian ablation (E2) on Qwen2.5-1.5B and
# Qwen2.5-3B, every fourth block (7 of 28 and 9 of 36). Follow-up jobs come through colab/next_loop.sh.
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-b10 || exit 1
( while true; do python scripts/hub_sync.py put results/v2/E2 results/v2/E9 >/dev/null 2>&1; sleep 600; done ) &
SYNC=$!
E="--locked colab/locked_fingerprint.json --runtime-name esann-b10 --session e2-large"
python scripts/run_e2.py --model Qwen/Qwen2.5-1.5B --blocks stride4 $E
python scripts/run_e2.py --model Qwen/Qwen2.5-3B --blocks stride4 $E
python scripts/hub_sync.py put results/v2/E2
kill $SYNC 2>/dev/null
echo "CHAIN B15 DONE"
bash colab/next_loop.sh
