#!/bin/bash
# Image gate (DECISIONS.md #26, #39, #44): if this runtime's image is not in colab/locked_fingerprint.json, recompute
# one known state (IMG) with an image-free copy of the lock, then wait until the orchestrator has compared it bit for
# bit (scripts/img_check.py) and admitted the image in the lock (git pull every 2 minutes).
# Usage: bash colab/image_gate.sh <runtime-name>
cd /content/work
export ESANN_OUT=/content/out
locked_ok() { python - <<'PY'
import json, sys
sys.path.insert(0, ".")
from src_v2 import pipeline as P
try:
    P.check_fingerprint(P.fingerprint(), json.load(open("colab/locked_fingerprint.json")))
except RuntimeError as e:
    print(e); sys.exit(1)
PY
}
if locked_ok; then echo "IMAGE ADMITTED already"; exit 0; fi
python -c "import json; d=json.load(open('colab/locked_fingerprint.json')); d.pop('colab_image'); json.dump(d, open('/content/lock_noimage.json', 'w'))"
python scripts/run_grid.py --family IMG --model HuggingFaceTB/SmolLM2-135M --queue colab/queues/IMG__SmolLM2-135M.json \
  --eval-batch 16 --locked /content/lock_noimage.json --hf-repo martincousseau/esann2027-v2-results --runtime-name "$1" --session img
echo "IMG DONE, waiting for admission"
until locked_ok >/dev/null 2>&1; do sleep 120; git pull -q; done
echo "IMAGE ADMITTED"
