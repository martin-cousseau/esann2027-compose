#!/bin/bash
# Runtime A (esann-a13), 2026-11-16 14:20 UTC, DECISIONS.md #61: E9 (no compensation) on SmolLM2-1.7B, Qwen2.5-1.5B (every
# fourth block) and OLMo-2-1B (every second), the blocks of E2. Short runs only (about 75 minutes): the session that
# renews the runtime tokens has been paused by its usage limit, and the runtimes were reclaimed during each pause.
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-a13 || exit 1
( while true; do python scripts/hub_sync.py put results/v2/E9 >/dev/null 2>&1; sleep 300; done ) &
SYNC=$!
python scripts/hub_sync.py get results/v2/E9/SmolLM2-1.7B.jsonl results/v2/E9/Qwen2.5-1.5B.jsonl results/v2/E9/OLMo-2-0425-1B.jsonl
E="--locked colab/locked_fingerprint.json --runtime-name esann-a13 --session e9-large"
python scripts/run_e9.py --model HuggingFaceTB/SmolLM2-1.7B --blocks stride4 $E
python scripts/hub_sync.py put results/v2/E9
python scripts/run_e9.py --model Qwen/Qwen2.5-1.5B --blocks stride4 $E
python scripts/hub_sync.py put results/v2/E9
python scripts/run_e9.py --model allenai/OLMo-2-0425-1B --blocks stride2 $E
python scripts/hub_sync.py put results/v2/E9
kill $SYNC 2>/dev/null
echo "CHAIN A19 DONE"
