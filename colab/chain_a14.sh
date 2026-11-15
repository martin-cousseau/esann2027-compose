#!/bin/bash
# Runtime A (2026-11-15, DECISIONS.md #49): once chain_a13 has ended (its last step is the Qwen2.5-3B base design),
# the last five states of the Qwen2.5-3B grid (P->Q and Q->P at 33%+3, the three orders at 50%+3), taken from runtime
# B's queue so that both runtimes end at about the same time. A writes its own result file,
# results/v2/E6/Qwen2.5-3B__c.jsonl (one writer per result file, #44). Perplexity only, no export (#40).
cd /content/work
export ESANN_OUT=/content/out
while pgrep -f "colab/chain_a13[.]sh" >/dev/null; do sleep 60; done
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-a8"
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-3B --queue colab/queues/E6__Qwen2.5-3B__c.json --out /content/out/results/v2/E6/Qwen2.5-3B__c.jsonl --eval-batch 4 --no-dense-copy $H --session e6-qwen3b-c
echo "CHAIN A14 DONE"
