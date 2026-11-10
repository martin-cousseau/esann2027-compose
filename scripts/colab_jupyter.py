#!/usr/bin/env python3
"""Container-side access to a Colab runtime's Jupyter server (contents API).

Used for the two transfers that must not pass through an agent transcript:
  put-token  writes this container's $HF_TOKEN to /root/.cache/huggingface/token on the runtime. The value
             is read from the environment here and never printed or logged. Set mode 600 afterwards with a
             one-line execute_code call (os.chmod), which contains no secret.
  pull       mirrors new or changed files from a remote directory into a local directory (result lines,
             per-item files, logs), so bulk results never go through a tool output.
  ls         lists a remote directory.

The runtime URL and its short-lived proxy token (from the Colab connector's create_runtime / get_runtime)
are read from the environment variables COLAB_URL and COLAB_PROXY_TOKEN; they are never printed.

Examples:
  COLAB_URL=... COLAB_PROXY_TOKEN=... python scripts/colab_jupyter.py put-token
  COLAB_URL=... COLAB_PROXY_TOKEN=... python scripts/colab_jupyter.py pull /content/work/results/v2 results/v2
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

TOKEN_PATH = "root/.cache/huggingface/token"


def _req(method, path, body=None, raw=False, timeout=120):
    url = os.environ["COLAB_URL"].rstrip("/") + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("X-Colab-Runtime-Proxy-Token", os.environ["COLAB_PROXY_TOKEN"])
    if data is not None:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        payload = r.read()
    return payload if raw else (json.loads(payload) if payload else None)


def api_path(remote):
    return "/api/contents/" + urllib.parse.quote(remote.lstrip("/"))


def exists(remote):
    try:
        _req("GET", api_path(remote) + "?content=0")
        return True
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return False
        raise


def mkdirs(remote):
    parts = remote.strip("/").split("/")
    for i in range(1, len(parts) + 1):
        p = "/".join(parts[:i])
        if not exists(p):
            _req("PUT", api_path(p), {"type": "directory"})


def put_token():
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        print("FAIL HF_TOKEN is not set in this container")
        return 1
    mkdirs(os.path.dirname(TOKEN_PATH))
    _req("PUT", api_path(TOKEN_PATH), {"type": "file", "format": "text", "content": tok})
    info = _req("GET", api_path(TOKEN_PATH) + "?content=0")
    ok = info and info.get("size") == len(tok)
    print(("PASS" if ok else "FAIL") + f" token file written to /{TOKEN_PATH} (size check {'ok' if ok else 'mismatch'}; "
          "value not shown). Next: chmod 600 with execute_code.")
    return 0 if ok else 1


def push(local_root, remote_root, include_suffix=(".jsonl", ".json.gz")):
    """Upload local result files to the runtime (restores a resume point on a fresh runtime)."""
    import base64
    n = 0
    for dirpath, _, files in os.walk(local_root):
        for f in sorted(files):
            if not f.endswith(include_suffix) or f.startswith(".") or f.startswith("archived_"):
                continue
            lp = os.path.join(dirpath, f)
            rel = os.path.relpath(lp, local_root)
            rp = remote_root.strip("/") + "/" + rel
            mkdirs(os.path.dirname(rp))
            with open(lp, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            _req("PUT", api_path(rp), {"type": "file", "format": "base64", "content": b64}, timeout=300)
            n += 1
    print(f"pushed {n} files from {local_root} to /{remote_root.strip('/')}")
    return 0


def listing(remote):
    d = _req("GET", api_path(remote) + "?content=1")
    return d.get("content") or []


def walk(remote):
    for e in listing(remote):
        if e["type"] == "directory":
            yield from walk(e["path"])
        else:
            yield e


def pull(remote, local, include=None):
    os.makedirs(local, exist_ok=True)
    man_path = os.path.join(local, ".pull_manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    root = remote.strip("/")
    n_new, n_bytes = 0, 0
    for e in walk(remote):
        rel = e["path"][len(root):].lstrip("/")
        if include and not any(s in rel for s in include):
            continue
        key = f"{e.get('last_modified')}|{e.get('size')}"
        dst = os.path.join(local, rel)
        if man.get(rel) == key and os.path.exists(dst):
            continue
        data = _req("GET", "/files/" + urllib.parse.quote(e["path"]), raw=True, timeout=600)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        tmp = dst + ".part"
        with open(tmp, "wb") as f:
            f.write(data)
        os.replace(tmp, dst)
        man[rel] = key
        n_new += 1
        n_bytes += len(data)
    with open(man_path, "w") as f:
        json.dump(man, f, indent=0, sort_keys=True)
    print(f"pulled {n_new} new or changed files ({n_bytes / 1024:.0f} KiB) from {remote} into {local}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("put-token")
    p = sub.add_parser("pull")
    p.add_argument("remote")
    p.add_argument("local")
    p.add_argument("--include", nargs="*", default=None, help="only paths containing one of these substrings")
    pu = sub.add_parser("push")
    pu.add_argument("local")
    pu.add_argument("remote")
    ls = sub.add_parser("ls")
    ls.add_argument("remote")
    a = ap.parse_args()
    for v in ("COLAB_URL", "COLAB_PROXY_TOKEN"):
        if not os.environ.get(v):
            print(f"FAIL {v} is not set")
            return 1
    if a.cmd == "put-token":
        return put_token()
    if a.cmd == "pull":
        return pull(a.remote, a.local, a.include)
    if a.cmd == "push":
        return push(a.local, a.remote)
    for e in listing(a.remote):
        print(f"{e['type'][:4]} {e.get('size') or 0:>10} {e.get('last_modified', '')[:19]} {e['name']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
