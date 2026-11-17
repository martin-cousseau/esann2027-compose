#!/bin/bash
# Runtime A (2026-11-15, DECISIONS.md #50): resumes after the third stop (both runtimes reclaimed while this session's
# worker restarted). The existing lines of E6/Qwen2.5-3B and E6/Qwen2.5-3B__b are put here first (one writer per
# result file), so finished states are skipped: the last two base-design states of Qwen2.5-3B, then part b of its grid
# completion. Perplexity only, no model export (#40).
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-a9 || exit 1
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-a9"
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-3B --queue colab/queues/E6__Qwen2.5-3B.json --eval-batch 4 --no-dense-copy $H --session e6-qwen3b
python scripts/run_grid.py --family E6 --model Qwen/Qwen2.5-3B --queue colab/queues/E6__Qwen2.5-3B__b.json --out /content/out/results/v2/E6/Qwen2.5-3B__b.jsonl --eval-batch 4 --no-dense-copy $H --session e6-qwen3b-b
echo "CHAIN A15 DONE"
