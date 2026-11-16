#!/bin/bash
# Runtime C (esann-c2), 2026-11-16, DECISIONS.md #60: the 3 Qwen2.5-1.5B seed-2 states still missing at 3 bits (resumed
# from the Hub copy of Qwen2.5-1.5B__b.jsonl), then 5 of the 10 missing at 4 bits into Qwen2.5-1.5B__c.jsonl.
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-c2 || exit 1
python scripts/hub_sync.py get results/v2/E5/Qwen2.5-1.5B__b.jsonl
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-c2"
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E5__Qwen2.5-1.5B__b.json --out /content/out/results/v2/E5/Qwen2.5-1.5B__b.jsonl --eval-batch 8 $H --session e5-qwen15-b
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E5__Qwen2.5-1.5B__c.json --out /content/out/results/v2/E5/Qwen2.5-1.5B__c.jsonl --eval-batch 8 $H --session e5-qwen15-c
echo "CHAIN C2 DONE"
bash colab/next_loop.sh
