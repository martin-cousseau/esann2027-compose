#!/usr/bin/env python3
"""Detached keepalive (DECISIONS.md #46): executes a trivial cell on every runtime of the private config every few
minutes, re-reading the config at each cycle (so the tokens the orchestrator refreshes are picked up) and never writing
it. It runs outside the agent's background tasks, so it survives a restart of the agent's worker, which ended the
watcher, and with it the keepalive, twice (DECISIONS.md #42, #46). It cannot refresh the hourly proxy token itself
(only the Colab connector can), so it bridges an interruption of up to about an hour. Prints no secret.

    setsid nohup python scripts/keepalive_loop.py --minutes 4 --hours 40 >> /root/.config/esann/keepalive.log 2>&1 &

Stops when --stop-file exists or after --hours.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import colab_keepalive as KA  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/.config/esann/runtimes.json")
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--hours", type=float, default=40.0)
    ap.add_argument("--stop-file", default="/root/.config/esann/keepalive.stop")
    a = ap.parse_args()
    kernels = {}
    t_end = time.time() + 3600 * a.hours
    while time.time() < t_end and not os.path.exists(a.stop_file):
        try:
            with open(a.config) as f:
                rts = json.load(f)["runtimes"]
        except Exception as ex:  # noqa: BLE001  (the watcher may be rewriting the file)
            rts = []
            print(f"{dt.datetime.now(dt.timezone.utc):%m-%d %H:%M:%S} config unreadable: {type(ex).__name__}", flush=True)
        for rt in rts:
            rt = dict(rt)
            if not rt.get("kernel_id") and kernels.get(rt["name"]):
                rt["kernel_id"] = kernels[rt["name"]]
            try:
                s = KA.ping(rt)
                kernels[rt["name"]] = rt.get("kernel_id")
                status = f"ok {s:.1f}s"
            except Exception as ex:  # noqa: BLE001
                status = f"fail {type(ex).__name__}"
            print(f"{dt.datetime.now(dt.timezone.utc):%m-%d %H:%M:%S} {rt['name']} {status}", flush=True)
        time.sleep(60 * a.minutes)
    print("keepalive loop ends", flush=True)


if __name__ == "__main__":
    sys.exit(main())
