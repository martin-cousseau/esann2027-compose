#!/bin/bash
# Runtime A, after chain_a7: E6 minimal (SmolLM2-1.7B: dense, 4-bit, 3-bit, 25%/50% + 4-bit in P->Q and joint;
# perplexity, C4, KL; no benchmarks for budget).
cd /content/work
export ESANN_OUT=/content/out
WAIT_PID=$(python -c "import json; print(json.load(open('/content/out/logs/chain_pids.json')).get('chain_a7', 0))")
while [ "$WAIT_PID" != "0" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
COMMON="--locked colab/locked_fingerprint.json --export-models --hf-repo martincousseau/esann2027-v2-results --hf-models-repo martincousseau/esann2027-compose-v2 --runtime-name esann-a5 --session e6-main"
python scripts/run_grid.py --family E6 --model HuggingFaceTB/SmolLM2-1.7B --queue colab/queues/E6__SmolLM2-1.7B.json --eval-batch 8 $COMMON
echo "CHAIN A8 DONE"
