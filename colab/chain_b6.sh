#!/bin/bash
# Runtime B, replaces the rest of chain_b4 and all of chain_b5 (DECISIONS.md #32): after the running E4
# SmolLM2-360M runner exits, run E5 (SmolLM2-360M calibration seeds) before E3 (SmolLM2-360M, then Qwen2.5-0.5B),
# so that a budget cut, if one is needed, falls on E3 Qwen2.5-0.5B.
cd /content/work
export ESANN_OUT=/content/out
WAIT_PID=$(pgrep -f "run_grid.py --family E4 --model HuggingFaceTB/SmolLM2-360M" | head -1)
while [ -n "$WAIT_PID" ] && kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
COMMON="--locked colab/locked_fingerprint.json --export-models --hf-repo martincousseau/esann2027-v2-results --hf-models-repo martincousseau/esann2027-compose-v2 --runtime-name esann-b4"
python scripts/run_grid.py --family E5 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/E5__SmolLM2-360M.json --eval-batch 16 $COMMON --session e5-main
python scripts/run_grid.py --family E3 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/E3__SmolLM2-360M.json --eval-batch 16 $COMMON --session e3-main
python scripts/run_grid.py --family E3 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E3__Qwen2.5-0.5B.json --eval-batch 8 $COMMON --session e3-main
echo "CHAIN B6 DONE"
