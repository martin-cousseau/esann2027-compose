#!/bin/bash
# Runtime B (esann-b12), 2026-11-16 15:45 UTC, DECISIONS.md #61: E9 on Qwen2.5-3B again (every fourth block), resumed from
# the 6 layers on the Hub; chain_b17's run stopped on a GPU out-of-memory error at block 0's down_proj (the model stayed
# on the GPU next to the 63 Hessians and the float64 copies; run_e9.py now moves the model to the CPU first).
cd /content/work
export ESANN_OUT=/content/out
( while true; do python scripts/hub_sync.py put results/v2/E9 >/dev/null 2>&1; sleep 300; done ) &
SYNC=$!
python scripts/hub_sync.py get results/v2/E9/Qwen2.5-3B.jsonl
python scripts/run_e9.py --model Qwen/Qwen2.5-3B --blocks stride4 --locked colab/locked_fingerprint.json --runtime-name esann-b12 --session e9-qwen3b
python scripts/hub_sync.py put results/v2/E9
kill $SYNC 2>/dev/null
echo "CHAIN B18 DONE"
