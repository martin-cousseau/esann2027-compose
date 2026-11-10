#!/usr/bin/env python3
"""Container-side watcher for one Colab runtime: mirrors results and exits on the first event.

Run it in the background (Bash run_in_background); its exit is the orchestrator's wake-up signal.
Every --interval seconds it mirrors /content/work/results/v2 into results/v2 and the runner status
files into logs/status/<runtime>/, through the runtime's Jupyter server (scripts/colab_jupyter.py).
It exits with one summary line when:
  QUEUE_DONE     a runner status says "finished"
  FAILURE        a failures.jsonl grew
  STALL          a runner log has not changed for --stall-min minutes while its queue is unfinished
  TOKEN_REFRESH  the proxy token expires within 6 minutes (refresh it with get_runtime, restart the watcher)
  HEARTBEAT      --heartbeat-min minutes passed (for NEWS.md updates)
  UNREACHABLE    5 consecutive HTTP failures (runtime reclaimed or deleted?)

Runtimes come from a JSON file outside the repository (default /root/.config/esann/runtimes.json, mode 600):
  {"runtimes": [{"name": "esann-e1-a", "url": "...", "token": "...", "expires": "RFC 3339"}, ...]}
Tokens are never printed.
"""
import argparse
import datetime as dt
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import colab_jupyter as CJ  # noqa: E402
import colab_keepalive as KA  # noqa: E402

REMOTE = "/content/out"  # ESANN_OUT on the runtimes (results, logs, caches, models)


def now():
    return dt.datetime.now(dt.timezone.utc)


def parse_time(s):
    s = s.replace("Z", "+00:00")
    if "." in s:
        head, tail = s.split(".", 1)
        frac = "".join(ch for ch in tail if ch.isdigit())[:6]
        tz = tail[len("".join(ch for ch in tail if ch.isdigit())):]
        s = f"{head}.{frac}{tz}"
    t = dt.datetime.fromisoformat(s)
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)  # Jupyter's last_modified may carry no zone


def snapshot(runtime):
    """Remote status files and runner-log modification times."""
    status, logs = {}, {}
    try:
        for e in CJ.listing(f"{REMOTE}/logs/status"):
            if e["name"].endswith(".json"):
                raw = CJ._req("GET", "/files/" + e["path"].lstrip("/"), raw=True)
                status[e["name"][:-5]] = json.loads(raw)
    except Exception as ex:  # noqa: BLE001
        if "404" not in str(ex):
            raise
    try:
        for e in CJ.listing(f"{REMOTE}/logs/runner"):
            logs[e["name"][:-4]] = parse_time(e["last_modified"])
    except Exception as ex:  # noqa: BLE001
        if "404" not in str(ex):
            raise
    return status, logs


def count_failures(local_results):
    n = 0
    for dirpath, _, files in os.walk(local_results):
        for f in files:
            if f.startswith("failures"):
                n += sum(1 for line in open(os.path.join(dirpath, f)) if line.strip())
    return n


def count_lines(local_results):
    out = {}
    for dirpath, _, files in os.walk(local_results):
        for f in files:
            if f.endswith(".jsonl") and not f.startswith("failures"):
                key = os.path.relpath(os.path.join(dirpath, f), local_results)
                out[key] = sum(1 for line in open(os.path.join(dirpath, f)) if line.strip())
    return out


def use(rt):
    os.environ["COLAB_URL"], os.environ["COLAB_PROXY_TOKEN"] = rt["url"], rt["token"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/.config/esann/runtimes.json")
    ap.add_argument("--interval", type=int, default=90)
    ap.add_argument("--stall-min", type=int, default=25)
    ap.add_argument("--heartbeat-min", type=int, default=60)
    ap.add_argument("--exit-on-unit", action="store_true", help="also exit when any new result line appears")
    ap.add_argument("--keepalive-min", type=float, default=4.0,
                    help="execute a trivial cell on every runtime this often (Colab reclaims client-idle runtimes)")
    a = ap.parse_args()
    with open(a.config) as f:
        rts = json.load(f)["runtimes"]
    local_results = os.path.join(ROOT, "results", "v2")
    t_start, errors = now(), {r["name"]: 0 for r in rts}
    fail0 = count_failures(local_results)
    lines0 = count_lines(local_results)
    seen_finished = None
    last_ping = {r["name"]: 0.0 for r in rts}
    ka_fail = {r["name"]: 0 for r in rts}
    while True:
        for rt in rts:
            if time.time() - last_ping[rt["name"]] >= 60 * a.keepalive_min:
                try:
                    KA.ping(rt)
                    last_ping[rt["name"]], ka_fail[rt["name"]] = time.time(), 0
                except Exception as ex:  # noqa: BLE001
                    ka_fail[rt["name"]] += 1
                    if ka_fail[rt["name"]] >= 3:
                        print(f"UNREACHABLE {rt['name']}: keepalive failed 3 times: {type(ex).__name__}: {str(ex)[:120]}")
                        return 0
        try:  # remember the keepalive kernel ids
            with open(a.config, "w") as f:
                json.dump({"runtimes": rts}, f)
        except Exception:  # noqa: BLE001
            pass
        status, logs = {}, {}
        for rt in rts:
            use(rt)
            try:
                CJ.pull(f"{REMOTE}/results/v2", local_results)
                st, lg = snapshot(rt["name"])
                os.makedirs(os.path.join(ROOT, "logs", "status", rt["name"]), exist_ok=True)
                for k, v in st.items():
                    with open(os.path.join(ROOT, "logs", "status", rt["name"], f"{k}.json"), "w") as f:
                        json.dump(v, f)
                status.update({f"{rt['name']}:{k}": v for k, v in st.items()})
                logs.update({f"{rt['name']}:{k}": v for k, v in lg.items()})
                errors[rt["name"]] = 0
            except Exception as ex:  # noqa: BLE001
                errors[rt["name"]] += 1
                if errors[rt["name"]] >= 5:
                    print(f"UNREACHABLE {rt['name']}: {type(ex).__name__}: {str(ex)[:160]}")
                    return 0
        a.runtime = ",".join(r["name"] for r in rts)
        lines = count_lines(local_results)
        summary = "; ".join(f"{k}:{v.get('stage')}({v.get('done')}/{(v.get('done') or 0) + (v.get('todo') or 0)})"
                            for k, v in sorted(status.items()))
        finished = sorted(k for k, v in status.items() if v.get("stage") == "finished")
        if seen_finished is None:
            seen_finished = set(finished)  # queues already finished when the watcher started do not count
        new_finished = [k for k in finished if k not in seen_finished]
        # a queue that already finished on an earlier runtime (re-run after a restore) is not news
        known_path = "/root/.config/esann/finished_queues.json"
        known = set(json.load(open(known_path))) if os.path.exists(known_path) else set()
        fresh = [k for k in new_finished if k.split(":", 1)[1] not in known]
        seen_finished.update(new_finished)
        known.update(k.split(":", 1)[1] for k in finished)
        with open(known_path, "w") as f:
            json.dump(sorted(known), f)
        if fresh:
            print(f"QUEUE_DONE {a.runtime} {fresh} | {summary}")
            return 0
        if count_failures(local_results) > fail0:
            print(f"FAILURE {a.runtime} failures={count_failures(local_results)} | {summary}")
            return 0
        latest = {}  # newest log time per runtime (chains log to one file, runners to another)
        for k, v in logs.items():
            rn = k.split(":")[0]
            latest[rn] = max(latest.get(rn, v), v)
        for k, v in status.items():
            rn = k.split(":")[0]
            if v.get("stage") != "finished" and rn in latest and (now() - latest[rn]).total_seconds() > 60 * a.stall_min:
                print(f"STALL {rn} {k}: no log activity for {(now() - latest[rn]).total_seconds() / 60:.0f} min | {summary}")
                return 0
        if a.exit_on_unit and lines != lines0:
            print(f"UNIT_DONE {a.runtime} | {summary}")
            return 0
        if min((parse_time(r["expires"]) - now()).total_seconds() for r in rts) < 360:
            print(f"TOKEN_REFRESH {a.runtime} | {summary}")
            return 0
        if (now() - t_start).total_seconds() > 60 * a.heartbeat_min:
            print(f"HEARTBEAT {a.runtime} | {summary}")
            return 0
        time.sleep(a.interval)


if __name__ == "__main__":
    sys.exit(main())
