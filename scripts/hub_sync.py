#!/usr/bin/env python3
"""Copy result files between a runtime's output folder and the Hub results dataset.

  python scripts/hub_sync.py put results/v2/E2 results/v2/E9      upload every *.jsonl under these folders of $ESANN_OUT
  python scripts/hub_sync.py get results/v2/E5/Qwen2.5-1.5B.jsonl  fetch a result file that is not there yet (to resume)

Layer-level families (E2, E9) write JSON lines that only the container-side watcher mirrored; `put`, called every few
minutes by the chain scripts, keeps a copy on the Hub so that a lost runtime loses at most the layer in progress.
`get` never overwrites a file that exists. A line still being written is uploaded as it is: readers skip a last line
that does not parse. The token is read from the Hugging Face cache or $HF_TOKEN by huggingface_hub; it is never printed.
"""
import glob
import json
import os
import sys

REPO = os.environ.get("ESANN_HF_RESULTS", "martincousseau/esann2027-v2-results")
OUT = os.environ.get("ESANN_OUT", "/content/out")


def main():
    from huggingface_hub import HfApi, hf_hub_download
    cmd, args = sys.argv[1], sys.argv[2:]
    api = HfApi()
    if cmd == "put":
        n = 0
        for d in args:
            for p in sorted(glob.glob(os.path.join(OUT, d, "*.jsonl"))):
                rel = os.path.relpath(p, OUT)
                api.upload_file(path_or_fileobj=p, path_in_repo=rel, repo_id=REPO, repo_type="dataset",
                                commit_message=f"{rel}: {sum(1 for _ in open(p))} lines")
                n += 1
        print(f"hub_sync put: {n} files")
    elif cmd == "get":
        for rel in args:
            dst = os.path.join(OUT, rel)
            if os.path.exists(dst):
                print(f"hub_sync get: {rel} already there ({sum(1 for _ in open(dst))} lines), kept")
                continue
            try:
                src = hf_hub_download(REPO, rel, repo_type="dataset", force_download=True)
            except Exception as e:  # not on the Hub yet: the runner starts from an empty file
                print(f"hub_sync get: {rel} not on the Hub ({type(e).__name__}), starting empty")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            kept = []
            for line in open(src):
                try:
                    json.loads(line)
                    kept.append(line if line.endswith("\n") else line + "\n")
                except ValueError:
                    pass  # a line cut while it was uploaded: that state or layer simply runs again
            with open(dst, "w") as f:
                f.writelines(kept)
            print(f"hub_sync get: {rel} fetched ({len(kept)} complete lines)")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
