#!/bin/bash
# Optional tail (DECISIONS.md #44): E6 Qwen2.5-3B (perplexity only; a second fp32 copy for KL does not fit on the L4),
# started on whichever runtime finishes first, if the compute budget allows. Usage: bash colab/chain_tail_qwen3b.sh <runtime-name>
cd /content/work
export ESANN_OUT=/content/out
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name $1"
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-3B --queue colab/queues/E6__Qwen2.5-3B.json --eval-batch 4 --no-dense-copy $H --session e6-qwen3b
echo "CHAIN QWEN3B DONE"
