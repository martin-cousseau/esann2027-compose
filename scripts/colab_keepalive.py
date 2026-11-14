#!/usr/bin/env python3
"""Keep a Colab runtime assigned by executing a trivial cell in a kernel over the Jupyter websocket.

Colab reclaimed our runtimes 20-35 min after the last code execution coming from a client, even while a
kernel cell and GPU jobs were running and while the file API was polled every 90 s. Executing code through
the kernel channel is what the connector's execute_code does, so the watcher calls ping() every few minutes.

The runtime URL and proxy token come from the caller (never printed). A dedicated kernel named in
rt["kernel_id"] is created on first use (POST /api/kernels) and reused.
Standalone test:  python scripts/colab_keepalive.py --config /root/.config/esann/runtimes.json
"""
import argparse
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
import uuid


def _api(rt, method, path, body=None, timeout=60):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(rt["url"].rstrip("/") + path, data=data, method=method)
    req.add_header("X-Colab-Runtime-Proxy-Token", rt["token"])
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    return json.loads(raw) if raw else None


def ensure_kernel(rt):
    kernels = _api(rt, "GET", "/api/kernels") or []
    ids = {k["id"] for k in kernels}
    if rt.get("kernel_id") in ids:
        return rt["kernel_id"]
    k = _api(rt, "POST", "/api/kernels", {"name": "python3"})
    rt["kernel_id"] = k["id"]
    return k["id"]


def ping(rt, timeout=45):
    """Execute `0` in the keepalive kernel; returns the round-trip seconds. Raises on failure."""
    import websocket  # pip install websocket-client
    kid = ensure_kernel(rt)
    session = uuid.uuid4().hex
    url = rt["url"].replace("https://", "wss://", 1).rstrip("/") + f"/api/kernels/{kid}/channels?session_id={session}"
    proxy = urllib.parse.urlparse(os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy") or "")
    kw = {"header": [f"X-Colab-Runtime-Proxy-Token: {rt['token']}"], "timeout": timeout,
          "sslopt": {"cert_reqs": ssl.CERT_REQUIRED, "ca_certs": os.environ.get("SSL_CERT_FILE") or None}}
    if proxy.hostname:
        kw.update(http_proxy_host=proxy.hostname, http_proxy_port=proxy.port, proxy_type="http")
    t0 = time.time()
    ws = websocket.create_connection(url, **kw)
    try:
        msg_id = uuid.uuid4().hex
        ws.send(json.dumps({
            "header": {"msg_id": msg_id, "username": "keepalive", "session": session, "msg_type": "execute_request",
                       "version": "5.3", "date": ""},
            "parent_header": {}, "metadata": {}, "channel": "shell", "buffers": [],
            "content": {"code": "0", "silent": False, "store_history": False, "user_expressions": {},
                        "allow_stdin": False, "stop_on_error": True}}))
        while time.time() - t0 < timeout:
            m = json.loads(ws.recv())
            if m.get("parent_header", {}).get("msg_id") == msg_id and m.get("msg_type") == "execute_reply":
                if m["content"].get("status") != "ok":
                    raise RuntimeError(f"keepalive execute_reply status {m['content'].get('status')}")
                return time.time() - t0
        raise TimeoutError("no execute_reply")
    finally:
        ws.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="/root/.config/esann/runtimes.json")
    a = ap.parse_args()
    cfg = json.load(open(a.config))
    for rt in cfg["runtimes"]:
        try:
            print(f"{rt['name']}: keepalive ok in {ping(rt):.1f}s (kernel {rt['kernel_id'][:8]})")
        except Exception as e:  # noqa: BLE001
            print(f"{rt['name']}: keepalive FAILED: {type(e).__name__}: {str(e)[:200]}")
    with open(a.config, "w") as f:
        json.dump(cfg, f)
    return 0


if __name__ == "__main__":
    sys.exit(main())
