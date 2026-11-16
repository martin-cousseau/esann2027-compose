#!/bin/bash
# Runtime A (esann-a12), 2026-11-16, DECISIONS.md #60: E9, the layer-level control without error compensation
# (scripts/run_e9.py), on the seven models: the three small ones every second block (as E2), SmolLM2-1.7B,
# Qwen2.5-1.5B and Qwen2.5-3B every fourth, OLMo-2-1B every second (as E2). E9 lines go to the Hub every 10 minutes.
# Then 5 of the 10 Qwen2.5-1.5B seed-2 states missing at 4 bits (Qwen2.5-1.5B__c.jsonl). Colab allows two runtimes:
# chain_c2.sh is not used, its work is split between this chain and chain_b16.sh.
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
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-a12"
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E5__Qwen2.5-1.5B__c.json --out /content/out/results/v2/E5/Qwen2.5-1.5B__c.jsonl --eval-batch 8 $H --session e5-qwen15-c
echo "CHAIN A18 DONE"
bash colab/next_loop.sh
