"""BoolQ chance level = majority-class rate of the validation labels, with provenance.

The dataset is the one lm-eval 0.4.13's boolq task reads (dataset_path / dataset_name / validation_split of
lm_eval/tasks/super_glue/boolq/default.yaml, read from the installed package when available). The labels
are downloaded once from the Hugging Face Hub (parquet file of that split, pinned to the dataset commit),
counted, and the result is cached in reports/analysis/boolq_chance.json with the URL, commit, file sha256 and
counts. The downloaded file goes to a fresh temporary directory, is read with pyarrow only, and is deleted.

    python -m scripts.analysis.boolq_chance          # (re)fetch and print
"""
import datetime as dt
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.request

from .common import Paths, write_json

DEFAULT_TASK = {"dataset_path": "aps/super_glue", "dataset_name": "boolq", "validation_split": "validation"}
UA = {"User-Agent": "python-urllib (analysis script)"}


def lmeval_task_config():
    try:
        import importlib.util
        spec = importlib.util.find_spec("lm_eval")
        if spec is None or not spec.submodule_search_locations:
            return None
        path = os.path.join(list(spec.submodule_search_locations)[0], "tasks", "super_glue", "boolq", "default.yaml")
        if not os.path.exists(path):
            return None
        cfg = {}
        with open(path) as f:
            for line in f:
                if ":" in line and not line.startswith(" "):
                    k, v = line.split(":", 1)
                    cfg[k.strip()] = v.strip()
        import importlib.metadata as im
        return {"file": path, "lm_eval_version": im.version("lm_eval"),
                **{k: cfg.get(k) for k in ("task", "dataset_path", "dataset_name", "validation_split")}}
    except Exception:  # noqa: BLE001
        return None


def _get(url, timeout):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def fetch(timeout=60):
    cfg = lmeval_task_config() or {}
    repo = cfg.get("dataset_path") or DEFAULT_TASK["dataset_path"]
    name = cfg.get("dataset_name") or DEFAULT_TASK["dataset_name"]
    split = cfg.get("validation_split") or DEFAULT_TASK["validation_split"]
    info = json.loads(_get(f"https://huggingface.co/api/datasets/{repo}/revision/main", timeout))
    sha = info["sha"]
    files = [s["rfilename"] for s in info.get("siblings", []) if s["rfilename"].startswith(f"{name}/{split}-")
             and s["rfilename"].endswith(".parquet")]
    if not files:
        raise RuntimeError(f"no parquet file for {repo}/{name}/{split} at {sha}")
    import pyarrow.parquet as pq
    tmp = tempfile.mkdtemp(prefix="boolq_labels_")
    counts, n, shas, urls = {}, 0, {}, []
    try:
        for fn in sorted(files):
            url = f"https://huggingface.co/datasets/{repo}/resolve/{sha}/{fn}"
            blob = _get(url, timeout)
            path = os.path.join(tmp, os.path.basename(fn))
            with open(path, "wb") as f:
                f.write(blob)
            shas[fn] = hashlib.sha256(blob).hexdigest()
            urls.append(url)
            labels = pq.read_table(path, columns=["label"]).column("label").to_pylist()
            for x in labels:
                counts[str(x)] = counts.get(str(x), 0) + 1
            n += len(labels)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    major = max(counts, key=counts.get)
    return {"rate": counts[major] / n, "majority_label": major, "counts": counts, "n": n,
            "label_names": "0 = no / False, 1 = yes / True (lm-eval doc_to_choice ['no', 'yes'], target = label)",
            "dataset": repo, "config": name, "split": split, "revision": sha, "files": urls, "sha256": shas,
            "lm_eval_task": cfg or None, "retrieved_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}


def load_or_fetch(paths, allow_fetch=True, refresh=False):
    """(rate or None, record or None, note). The cached record is used unless refresh=True."""
    cache = os.path.join(paths.reports, "boolq_chance.json")
    if os.path.exists(cache) and not refresh:
        with open(cache) as f:
            rec = json.load(f)
        return rec.get("rate"), rec, f"cached in {paths.rel(cache)}"
    if not allow_fetch:
        return None, None, "not cached and fetching disabled"
    try:
        rec = fetch()
    except Exception as e:  # noqa: BLE001
        return None, None, f"fetch failed: {type(e).__name__}: {str(e)[:160]}"
    write_json(cache, rec)
    return rec["rate"], rec, f"fetched from the Hugging Face Hub, cached in {paths.rel(cache)}"


if __name__ == "__main__":
    rate, rec, note = load_or_fetch(Paths(), refresh="--refresh" in sys.argv)
    print(note)
    print(json.dumps(rec, indent=1) if rec else "no record")
