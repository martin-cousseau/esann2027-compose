#!/usr/bin/env python3
"""Idempotent runtime bootstrap (run from /content/work after the repository is cloned at a commit).

1. installs the exact package versions of requirements-v2.txt (torch stays at the image's version,
   enforced with a constraint so no dependency can move it);
2. checks the repository is at the requested commit and that review_code/ is present;
3. reports whether a Hugging Face token file exists (never its content);
4. prints a one-line fingerprint: GPU, UUID, image, torch, transformers, lm_eval, datasets, commit.

The determinism flags are set by every runner at process start (src_v2/pipeline.set_determinism and
CUBLAS_WORKSPACE_CONFIG before CUDA initialises), not here: environment changes made by this script
would not reach later processes.

Usage: python colab/bootstrap.py --commit <sha> [--requirements requirements-v2.txt]
"""
import argparse
import os
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--commit", required=True)
    ap.add_argument("--requirements", default=os.path.join(ROOT, "requirements-v2.txt"))
    ap.add_argument("--skip-install", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    head = sh(["git", "-C", ROOT, "rev-parse", "HEAD"]).stdout.strip()
    if not head.startswith(args.commit) and not args.commit.startswith(head):
        print(f"FAIL repository at {head}, expected {args.commit}")
        return 1
    print(f"PASS repository at {head}")
    ok = all(os.path.exists(os.path.join(ROOT, "review_code", p)) for p in
             ("ablation/arms.py", "ablation/extra_orders.py", "ablation/ablation_same_hessian.py"))
    print(("PASS" if ok else "FAIL") + " review_code present")
    if not ok:
        return 1
    import importlib.metadata as im
    torch_v = im.version("torch")
    if not args.skip_install and os.path.exists(args.requirements):
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as c:
            c.write(f"torch=={torch_v}\n")
            cons = c.name
        r = sh([sys.executable, "-m", "pip", "install", "-q", "--progress-bar", "off", "-r", args.requirements,
                "-c", cons])
        if r.returncode != 0:
            print("FAIL pip install:", r.stderr.strip()[-1500:])
            return 1
        print(f"PASS pinned packages installed ({time.time() - t0:.0f}s)")
    tok = os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))
    print(f"INFO hugging face token file present: {'yes' if tok else 'no'}")
    fp = sh([sys.executable, "-c", "import sys, json; sys.path.insert(0, %r); from src_v2 import pipeline as P; "
                                   "print(json.dumps(P.fingerprint()))" % ROOT])
    if fp.returncode != 0:
        print("FAIL fingerprint:", fp.stderr.strip()[-1500:])
        return 1
    print("FINGERPRINT " + fp.stdout.strip())
    print(f"bootstrap done in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
