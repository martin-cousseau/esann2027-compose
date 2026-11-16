#!/bin/bash
# Runtime A (esann-a12), 2026-11-16, DECISIONS.md #60: E9, the layer-level control without error compensation
# (scripts/run_e9.py), on the seven models: the three small ones every second block (as E2), SmolLM2-1.7B,
# Qwen2.5-1.5B and Qwen2.5-3B every fourth, OLMo-2-1B every second (as E2). E9 lines go to the Hub every 10 minutes.
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-a12 || exit 1
( while true; do python scripts/hub_sync.py put results/v2/E9 >/dev/null 2>&1; sleep 600; done ) &
SYNC=$!
E="--locked colab/locked_fingerprint.json --runtime-name esann-a12 --session e9"
for m in HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-360M Qwen/Qwen2.5-0.5B; do
  python scripts/run_e9.py --model $m --blocks stride2 $E
done
python scripts/hub_sync.py put results/v2/E9
python scripts/run_e9.py --model HuggingFaceTB/SmolLM2-1.7B --blocks stride4 $E
python scripts/run_e9.py --model Qwen/Qwen2.5-1.5B --blocks stride4 $E
python scripts/hub_sync.py put results/v2/E9
python scripts/run_e9.py --model allenai/OLMo-2-0425-1B --blocks stride2 $E
python scripts/run_e9.py --model Qwen/Qwen2.5-3B --blocks stride4 $E
python scripts/hub_sync.py put results/v2/E9
kill $SYNC 2>/dev/null
echo "CHAIN A18 DONE"
bash colab/next_loop.sh
