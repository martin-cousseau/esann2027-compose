#!/bin/bash
# Runtime A, after chain_a3 (reproduction investigation): rest of E1 SmolLM2-135M, then E1 SmolLM2-360M.
cd /content/work
export ESANN_OUT=/content/out
WAIT_PID=$(python -c "import json; print(json.load(open('/content/out/logs/chain_pids.json')).get('chain_a3', 0))")
while [ "$WAIT_PID" != "0" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
COMMON="--locked colab/locked_fingerprint.json --export-models --hf-repo martincousseau/esann2027-v2-results --hf-models-repo martincousseau/esann2027-compose-v2 --runtime-name esann-e1-a3 --session e1-main"
python scripts/run_grid.py --family E1 --model HuggingFaceTB/SmolLM2-135M --queue colab/queues/E1__SmolLM2-135M.json --eval-batch 16 $COMMON
python scripts/run_grid.py --family E1 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/E1__SmolLM2-360M.json --eval-batch 16 $COMMON
echo "CHAIN A4 DONE"
