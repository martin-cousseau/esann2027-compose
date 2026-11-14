#!/usr/bin/env python3
"""Runtime-side keepalive: executes a trivial cell in a kernel of THIS runtime's own Jupyter server every few
minutes, so the runtime stays assigned while the orchestrator is away (the external keepalive needs a proxy
token that expires after an hour). Runs as a background process on the runtime; needs websocket-client.
Usage: nohup python colab/local_keepalive.py --minutes 4 &
"""
import argparse
import json
import subprocess
import sys
import time
import urllib.request
import uuid


def servers():
    out = subprocess.run([sys.executable, "-m", "jupyter", "server", "list", "--json"], capture_output=True, text=True).stdout
    res = []
    for line in out.splitlines():
        try:
            res.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not res:  # older Colab images run the classic notebook server
        out = subprocess.run([sys.executable, "-m", "jupyter", "notebook", "list", "--json"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            try:
                res.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return res


def api(base, token, method, path, body=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(base.rstrip("/") + path, data=data, method=method)
    if token:
        req.add_header("Authorization", f"token {token}")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=30) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def ping(base, token, kid):
    import websocket
    ws_url = base.replace("http://", "ws://").replace("https://", "wss://").rstrip("/") + f"/api/kernels/{kid}/channels?session_id={uuid.uuid4().hex}"
    hdr = [f"Authorization: token {token}"] if token else []
    ws = websocket.create_connection(ws_url, header=hdr, timeout=30)
    try:
        mid = uuid.uuid4().hex
        ws.send(json.dumps({"header": {"msg_id": mid, "username": "local-keepalive", "session": uuid.uuid4().hex,
                                       "msg_type": "execute_request", "version": "5.3", "date": ""},
                            "parent_header": {}, "metadata": {}, "channel": "shell", "buffers": [],
                            "content": {"code": "0", "silent": False, "store_history": False, "user_expressions": {},
                                        "allow_stdin": False, "stop_on_error": True}}))
        t0 = time.time()
        while time.time() - t0 < 30:
            m = json.loads(ws.recv())
            if m.get("parent_header", {}).get("msg_id") == mid and m.get("msg_type") == "execute_reply":
                return True
        return False
    finally:
        ws.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=4.0)
    ap.add_argument("--idle-exit-min", type=float, default=15.0,
                    help="exit when no chain/runner process has existed for this long (lets Colab free the runtime)")
    a = ap.parse_args()
    kid, idle_since = None, None
    while True:
        busy = subprocess.run(["pgrep", "-f", "chain_|run_grid.py|run_e2.py"], capture_output=True, text=True).stdout.strip()
        if busy:
            idle_since = None
        else:
            idle_since = idle_since or time.time()
            if time.time() - idle_since >= 60 * a.idle_exit_min:
                print(time.strftime("%H:%M:%S"), "no work for", a.idle_exit_min, "min: keepalive exits", flush=True)
                return 0
        try:
            srv = servers()
            s = srv[0]
            base, token = s.get("url") or f"http://127.0.0.1:{s['port']}/", s.get("token", "")
            if kid is None or kid not in {k["id"] for k in api(base, token, "GET", "/api/kernels")}:
                kid = api(base, token, "POST", "/api/kernels", {"name": "python3"})["id"]
            ok = ping(base, token, kid)
            print(time.strftime("%H:%M:%S"), "local keepalive", "ok" if ok else "no reply", base, flush=True)
        except Exception as e:  # noqa: BLE001
            print(time.strftime("%H:%M:%S"), "local keepalive failed:", type(e).__name__, str(e)[:200], flush=True)
        time.sleep(60 * a.minutes)


if __name__ == "__main__":
    main()
