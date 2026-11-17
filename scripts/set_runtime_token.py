#!/usr/bin/env python3
"""Update one runtime's proxy token in the private watcher config (keeps kernel_id). Prints no secret.
Usage: python scripts/set_runtime_token.py NAME URL TOKEN EXPIRES [--config PATH]"""
import json
import os
import sys

cfg_path = "/root/.config/esann/runtimes.json"
args = [a for a in sys.argv[1:] if not a.startswith("--config")]
name, url, token, expires = args[:4]
cfg = json.load(open(cfg_path)) if os.path.exists(cfg_path) else {"runtimes": []}
for rt in cfg["runtimes"]:
    if rt["name"] == name:
        rt.update(url=url, token=token, expires=expires)
        break
else:
    cfg["runtimes"].append({"name": name, "url": url, "token": token, "expires": expires})
old = os.umask(0o077)
with open(cfg_path, "w") as f:
    json.dump(cfg, f)
os.umask(old)
print(f"{name}: token updated, expires {expires}; runtimes in config: {[r['name'] for r in cfg['runtimes']]}")
