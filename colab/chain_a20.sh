#!/bin/bash
# Runtime A (esann-a14), 2026-11-16 17:05 UTC, DECISIONS.md #64 (you asked to launch the missing seed states): the 3
# Qwen2.5-1.5B seed-2 states still missing at 3 bits (resumed from the Hub copy of Qwen2.5-1.5B__b.jsonl), then 4 of the
# 10 missing at 4 bits (Qwen2.5-1.5B__c.jsonl). Each finished state is uploaded to the Hub by run_grid.
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-a14 || exit 1
python scripts/hub_sync.py get results/v2/E5/Qwen2.5-1.5B__b.jsonl
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-a14"
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E5__Qwen2.5-1.5B__b.json --out /content/out/results/v2/E5/Qwen2.5-1.5B__b.jsonl --eval-batch 8 $H --session e5-qwen15-b
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E5__Qwen2.5-1.5B__c.json --out /content/out/results/v2/E5/Qwen2.5-1.5B__c.jsonl --eval-batch 8 $H --session e5-qwen15-c
echo "CHAIN A20 DONE"
