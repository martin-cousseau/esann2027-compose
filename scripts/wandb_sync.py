#!/usr/bin/env python3
"""Sync finished results to a public Weights & Biases project: one W&B run per model state (idempotent).

The result files (results/v2/*/*.jsonl, written on the GPU runtimes) stay the single source of truth; this
script mirrors them to W&B with links to the git commit and to the compressed model on Hugging Face, so git,
HF and W&B point at each other. Re-running it updates runs whose result line changed and skips the rest.

The API key is read from $WANDB_API_KEY or /root/.config/esann/wandb_api_key and is never printed.
  python scripts/wandb_sync.py --setup     # create the project (public) with a description and links
  python scripts/wandb_sync.py             # sync every new or changed result line
  python scripts/wandb_sync.py --e2        # also sync the E2 same-Hessian ablation (one run per model)
"""
import argparse
import glob
import hashlib
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENTITY, PROJECT = "martin-css", "esann2027-compose-v2"
GIT = "https://github.com/martin-cousseau/esann2027-compose"
HF_MODELS = "https://huggingface.co/martincousseau/esann2027-compose-v2"
STATE = os.path.join(ROOT, "results", "v2", ".wandb_synced.json")
DESCRIPTION = ("ESANN 2027 v2 study 'Do Pruning and Quantization Compose?': one run per compressed model state "
               "(fp32, NVIDIA L4, Colab image locked). Code and result lines: " + GIT +
               " ; compressed models (exact codes + scales + masks): " + HF_MODELS)


def api_key():
    k = os.environ.get("WANDB_API_KEY")
    if not k and os.path.exists("/root/.config/esann/wandb_api_key"):
        k = open("/root/.config/esann/wandb_api_key").read().strip()
    if not k:
        sys.exit("no W&B API key")
    os.environ["WANDB_API_KEY"] = k
    os.environ.setdefault("WANDB_SILENT", "true")
    os.environ.setdefault("WANDB_DIR", "/root/.cache/esann-wandb")  # keep W&B's local run files out of the repo
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
    return k


def graphql(query, variables):
    """Plain HTTP call to W&B's GraphQL endpoint (this wandb version ships no gql client)."""
    import requests
    r = requests.post("https://api.wandb.ai/graphql", json={"query": query, "variables": variables},
                      auth=("api", api_key()), timeout=60)
    r.raise_for_status()
    out = r.json()
    if out.get("errors"):
        raise RuntimeError(json.dumps(out["errors"])[:500])
    return out["data"]


def setup():
    """Create or update the project as public (access USER_READ = anyone can view)."""
    q = """mutation upsertModel($name: String!, $entityName: String!, $access: String, $description: String) {
      upsertModel(input: {name: $name, entityName: $entityName, access: $access, description: $description}) {
        model { id name entityName access } } }"""
    m = graphql(q, {"name": PROJECT, "entityName": ENTITY, "access": "USER_READ",
                    "description": DESCRIPTION})["upsertModel"]["model"]
    print(f"project {m['entityName']}/{m['name']} access={m['access']} -> https://wandb.ai/{ENTITY}/{PROJECT}")


def flat(prefix, d, out):
    for k, v in (d or {}).items():
        key = f"{prefix}/{k}" if prefix else k
        if isinstance(v, dict):
            flat(key, v, out)
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out[key] = v
    return out


def run_id(family, state_id):
    """One W&B run per (family, state): E1 and its re-checks (DET, IMG, REPRO) stay separate runs."""
    return "r" + hashlib.sha1(f"{family}|{state_id}".encode()).hexdigest()[:24]


def sync_row(row):
    import wandb
    md = row.get("metadata") or {}
    commit = md.get("repo_commit")
    exp = row.get("model_export") or {}
    cfg = {k: row.get(k) for k in ("state_id", "family", "model", "order", "sparsity", "bits", "seed", "dtype", "tcr",
                                   "n_calib", "seqlen", "groupsize", "compressed_weights", "eval_batch", "status")}
    cfg.update({"storage": row.get("storage"), "git_repo": GIT, "git_commit": commit,
                "git_commit_url": f"{GIT}/commit/{commit}" if commit else None,
                "hf_model_url": f"{HF_MODELS}/blob/main/{exp['repo_path']}" if exp.get("repo_path") else None,
                "hf_model_sha256": exp.get("sha256"), "code_hash": md.get("code_hash"), "gpu": md.get("gpu"),
                "nvidia_smi": md.get("nvidia_smi"), "colab_image": md.get("colab_image"),
                "colab_runtime": md.get("colab_runtime_name"), "torch": md.get("torch"),
                "transformers": md.get("transformers"), "lm_eval": md.get("lm_eval"),
                "virtualization_mode": md.get("virtualization_mode"), "started": row.get("started"),
                "finished": row.get("finished"), "items_file": row.get("items_file"),
                "items_url": f"{GIT}/blob/claude/busy-hopper-ghtynk/{row['items_file']}" if row.get("items_file") else None})
    summ = {}
    for split in ("wikitext2", "c4"):
        for k in ("ppl", "kl", "top1"):
            v = (row.get(split) or {}).get(k)
            if v is not None:
                summ[f"{split}/{k}"] = v
    for t, r in (row.get("tasks") or {}).items():
        for k in ("acc,none", "acc_norm,none", "perplexity,none"):
            if k in r:
                summ[f"tasks/{t}/{k.split(',')[0]}"] = r[k]
    flat("main", row.get("main"), summ)
    flat("bits", row.get("storage"), summ)
    flat("time", row.get("timings"), summ)
    summ["mean_rel_err"] = row.get("mean_rel_err")
    short = row["model"].split("/")[-1]
    tags = [row["order"], f"{row['bits']}bit", f"s{float(row['sparsity']):.2f}", row.get("family") or "", short]
    run = wandb.init(entity=ENTITY, project=PROJECT, id=run_id(row.get("family"), row["state_id"]),
                     name=f"{row.get('family')}:{row['state_id']}",
                     group=short, job_type=row.get("family"), tags=[t for t in tags if t], config=cfg,
                     resume="allow", reinit="finish_previous", settings=wandb.Settings(silent=True))
    run.summary.update({k: v for k, v in summ.items() if v is not None})
    if row.get("diags"):
        cols = ["layer", "name", "rows", "cols", "mask_frac", "zero_frac", "mask_kept", "max_levels", "rel_err", "sec"]
        tab = wandb.Table(columns=cols)
        for d in row["diags"]:
            tab.add_data(d["layer"], d["name"], d["shape"][0], d["shape"][1], d["mask_frac"], d["zero_frac"],
                         d["mask_kept"], d["max_levels"], d["rel_err"], d.get("sec"))
        run.log({"layers": tab})
    run.finish()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setup", action="store_true")
    ap.add_argument("--e2", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--prune-old", action="store_true",
                    help="delete runs of the project whose id is not a current (family, state) id")
    a = ap.parse_args()
    api_key()
    if a.setup:
        setup()
        return
    state = json.load(open(STATE)) if os.path.exists(STATE) else {}
    n = 0
    for f in sorted(glob.glob(os.path.join(ROOT, "results", "v2", "*", "*.jsonl"))):
        if os.path.basename(f).startswith("failures") or os.sep + "E2" + os.sep in f:
            continue
        for line in open(f):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("status") not in ("ok", "failed_check") or not row.get("state_id"):
                continue
            h = hashlib.sha1(line.strip().encode()).hexdigest()
            key = f"{row.get('family')}|{row['state_id']}"
            if state.get(key) == h:
                continue
            try:
                sync_row(row)
            except Exception as e:  # noqa: BLE001  (one W&B failure must not stop the others)
                print(f"W&B sync failed for {key}: {type(e).__name__}: {str(e)[:160]}")
                continue
            state[key] = h
            n += 1
            with open(STATE, "w") as fh:
                json.dump(state, fh, indent=0, sort_keys=True)
            if a.limit and n >= a.limit:
                break
    print(f"synced {n} runs to https://wandb.ai/{ENTITY}/{PROJECT}")
    if a.prune_old:
        import wandb
        keep = {run_id(k.split("|", 1)[0], k.split("|", 1)[1]) for k in state}
        api = wandb.Api(api_key=os.environ["WANDB_API_KEY"], timeout=60)
        gone = 0
        for r in api.runs(f"{ENTITY}/{PROJECT}"):
            if r.id not in keep:
                r.delete()
                gone += 1
        print(f"deleted {gone} runs with old ids")


if __name__ == "__main__":
    main()
