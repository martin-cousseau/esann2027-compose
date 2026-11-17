#!/bin/bash
# Runtime A (esann-a11), 2026-11-15 22:40 UTC, DECISIONS.md #58. After the loss of esann-a10/b9 (session paused by its
# usage limit, tokens not renewed): the same-Hessian ablation (E2) on two larger models, so that the paper's
# shared-Hessian statements and the split of the quantize-first penalty (mask / grid / update order) are not
# limited to the three small models; then the 3 Qwen2.5-1.5B seed states still missing at 3 bits. Every fourth block
# on SmolLM2-1.7B (6 of 24), every second on OLMo-2-1B (8 of 16). Follow-up jobs come through colab/next_loop.sh.
cd /content/work
export ESANN_OUT=/content/out
bash colab/image_gate.sh esann-a11 || exit 1
( while true; do python scripts/hub_sync.py put results/v2/E2 results/v2/E9 >/dev/null 2>&1; sleep 600; done ) &
SYNC=$!
E="--locked colab/locked_fingerprint.json --runtime-name esann-a11 --session e2-large"
python scripts/run_e2.py --model HuggingFaceTB/SmolLM2-1.7B --blocks stride4 $E
python scripts/run_e2.py --model allenai/OLMo-2-0425-1B --blocks stride2 $E
python scripts/hub_sync.py put results/v2/E2
H="--locked colab/locked_fingerprint.json --hf-repo martincousseau/esann2027-v2-results --runtime-name esann-a11"
python scripts/hub_sync.py get results/v2/E5/Qwen2.5-1.5B__b.jsonl
python scripts/run_grid.py --family E5 --model Qwen/Qwen2.5-1.5B --queue colab/queues/E5__Qwen2.5-1.5B__b.json --out /content/out/results/v2/E5/Qwen2.5-1.5B__b.jsonl --eval-batch 8 $H --session e5-qwen15-b
kill $SYNC 2>/dev/null
echo "CHAIN A17 DONE"
bash colab/next_loop.sh
