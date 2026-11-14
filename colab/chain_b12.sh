#!/bin/bash
# Runtime B (2026-11-14, DECISIONS.md #47): once chain_b11 has ended (OLMo-2-1B, then the SmolLM2 seeds), the
# completion of the Qwen2.5-3B grid: pruning alone at 33%, the 33%+4 cell and the three 3-bit cells in the three
# orders (13 states), while runtime A runs the 3B base design (chain_a13.sh). B writes its own result file,
# results/v2/E6/Qwen2.5-3B__b.jsonl, so that every result file keeps one writer (#44). Perplexity only and no model
# export, as on A (#40).
cd /content/work
export ESANN_OUT=/content/out
while pgrep -f "colab/chain_b11[.]sh" >/dev/null; do sleep 60; done
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-b7"
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-3B --queue colab/queues/E6__Qwen2.5-3B__b.json --out /content/out/results/v2/E6/Qwen2.5-3B__b.jsonl --eval-batch 4 --no-dense-copy $H --session e6-qwen3b-b
echo "CHAIN B12 DONE"
