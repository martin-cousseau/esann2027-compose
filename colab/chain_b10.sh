#!/bin/bash
# Runtime B (2026-11-13 19:10 UTC, DECISIONS.md #45): replaces the rest of chain_b9. Waits for chain_b9's running step
# (E5 Qwen2.5-0.5B, 4-bit seeds; re-running the same queue afterwards only skips its finished states), then the 3-bit
# Qwen2.5-0.5B seeds, SmolLM2-1.7B to the full grid, OLMo-2-1B, and the SmolLM2-360M (3-bit) and SmolLM2-135M seeds
# (moved here from runtime A, whose copy of E5/SmolLM2-360M was removed; its existing lines were pushed here).
cd /content/work
export ESANN_OUT=/content/out
while pgrep -f "run_grid.py --family E5 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E5__Qwen2.5-0.5B.json" >/dev/null; do sleep 60; done
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-b6"
X="--export-models --hf-models-repo martincousseau/esann2027-compose-v2"
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E5__Qwen2.5-0.5B.json --eval-batch 8 $H $X --session e5-qwen
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-0.5B --queue colab/queues/E5__Qwen2.5-0.5B__b3.json --eval-batch 8 $H $X --session e5-qwen3
python scripts/run_grid.py --family E6 --model HuggingFaceTB/SmolLM2-1.7B --queue colab/queues/E6__SmolLM2-1.7B.json --eval-batch 8 $H $X --session e6-smol
python scripts/run_grid.py --family E6 --model allenai/OLMo-2-0425-1B --queue colab/queues/E6__OLMo-2-0425-1B.json --eval-batch 8 $H $X --session e6-olmo
python scripts/run_grid.py --family E5 --model HuggingFaceTB/SmolLM2-360M --queue colab/queues/E5__SmolLM2-360M.json --eval-batch 16 $H $X --session e5-360m
python scripts/run_grid.py --family E5 --model HuggingFaceTB/SmolLM2-135M --queue colab/queues/E5__SmolLM2-135M.json --eval-batch 16 $H $X --session e5-135m
echo "CHAIN B10 DONE"
