#!/usr/bin/env python3
"""Compact status of a background runner on this runtime (meant for short execute_code polls).

Usage: python colab/poll.py --family E1 --model SmolLM2-135M [--pid 1234] [--wait 240]
--wait N: block up to N seconds (< 500) until the result file grows or the runner exits, then report.
Prints at most ~12 lines: alive?, units done, status stage, last log lines, GPU memory.
"""
import argparse
import json
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def count_lines(path):
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def alive(pid):
    if not pid:
        return None
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    try:  # a zombie counts as dead
        with open(f"/proc/{pid}/status") as f:
            return "zombie" not in f.read().lower()
    except OSError:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--model", required=True, help="short name, e.g. SmolLM2-135M")
    ap.add_argument("--pid", type=int, default=None)
    ap.add_argument("--wait", type=int, default=0)
    ap.add_argument("--log-lines", type=int, default=4)
    args = ap.parse_args()
    res = os.path.join(ROOT, "results", "v2", args.family, f"{args.model}.jsonl")
    log = os.path.join(ROOT, "logs", "runner", f"{args.family}__{args.model}.log")
    st = os.path.join(ROOT, "logs", "status", f"{args.family}__{args.model}.json")
    n0 = count_lines(res)
    t0 = time.time()
    while args.wait and time.time() - t0 < min(args.wait, 500):
        if count_lines(res) != n0 or alive(args.pid) is False:
            break
        time.sleep(10)
    n = count_lines(res)
    status = json.load(open(st)) if os.path.exists(st) else {}
    tail = []
    if os.path.exists(log):
        with open(log, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 4000))
            tail = f.read().decode("utf8", "replace").splitlines()[-args.log_lines:]
    mem = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                          "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    fails = count_lines(os.path.join(ROOT, "results", "v2", args.family, "failures.jsonl"))
    print(json.dumps({"alive": alive(args.pid), "done_lines": n, "new_lines": n - n0, "failures": fails,
                      "stage": status.get("stage"), "unit": status.get("state_id"), "updated": status.get("updated"),
                      "gpu": mem, "waited_s": round(time.time() - t0)}))
    for line in tail:
        print("  " + line[:220])


if __name__ == "__main__":
    main()
