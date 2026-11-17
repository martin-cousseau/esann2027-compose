#!/bin/bash
# Runtime B, after chain_b7 (DECISIONS.md #40): E6 SmolLM2-1.7B to the full grid, E6 OLMo-2-1B (a third family:
# base design, then the full grid), then the 3-bit E5 seeds of SmolLM2-360M and Qwen2.5-0.5B.
cd /content/work
export ESANN_OUT=/content/out
until grep -q "CHAIN B7 DONE" /content/out/logs/runner/chain_b7.log 2>/dev/null; do sleep 60; done
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-b5"
X="--export-models --hf-models-repo martincousseau/esann2027-compose-v2"
python scripts/run_grid.py --family E6 --model HuggingFaceTB/SmolLM2-1.7B --queue colab/queues/E6__SmolLM2-1.7B.json --eval-batch 8 $H $X --session e6-smol
python scripts/run_grid.py --family E6 --model allenai/OLMo-2-0425-1B --queue colab/queues/E6__OLMo-2-0425-1B.json --eval-batch 8 $H $X --session e6-olmo
python scripts/run_grid.py --family E5 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/E5__SmolLM2-360M.json --eval-batch 16 $H $X --session e5-360m
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E5__Qwen2.5-0.5B__b3.json --eval-batch 8 $H $X --session e5-qwen
echo "CHAIN B8 DONE"
