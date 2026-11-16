#!/bin/bash
# Runtime B (esann-b12), 2026-11-16 14:20 UTC, DECISIONS.md #61: the rest of E2 on Qwen2.5-3B (9 of 63 layers done,
# resumed from the Hub), then E9 on Qwen2.5-3B (every fourth block). About 75 minutes.
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-b12 || exit 1
( while true; do python scripts/hub_sync.py put results/v2/E2 results/v2/E9 >/dev/null 2>&1; sleep 300; done ) &
SYNC=$!
python scripts/hub_sync.py get results/v2/E2/Qwen2.5-3B.jsonl results/v2/E9/Qwen2.5-3B.jsonl
python scripts/run_e2.py --model Qwen/Qwen2.5-3B --blocks stride4 --locked colab/locked_fingerprint.json --runtime-name esann-b12 --session e2-large
python scripts/hub_sync.py put results/v2/E2
python scripts/run_e9.py --model Qwen/Qwen2.5-3B --blocks stride4 --locked colab/locked_fingerprint.json --runtime-name esann-b12 --session e9-large
python scripts/hub_sync.py put results/v2/E9
kill $SYNC 2>/dev/null
echo "CHAIN B17 DONE"
