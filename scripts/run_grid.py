#!/usr/bin/env python3
"""Resume-safe queue runner: one JSON line per finished unit (model state), safe to kill at any time.

A unit is one model state: (model, order, sparsity, bits, calibration seed, dtype). For each unit the
runner restores the dense weights, compresses with the paper's compress_model (src/core.py, imported),
then measures WikiText-2 and C4 perplexity, KL to the dense model and top-1 agreement, and the lm-eval
tasks with per-item logs. The line is appended only when everything for the unit is done, so a kill
loses at most the unit in progress; finished units are skipped on restart.

Usage (on the GPU runtime, from the repository root):
    python scripts/run_grid.py --family E1 --model HuggingFaceTB/SmolLM2-135M \
        --queue colab/queues/E1__SmolLM2-135M.json --locked colab/locked_fingerprint.json \
        --runtime-name esann-e1-a --session e1-main
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # before torch initialises CUDA

import argparse  # noqa: E402
import copy  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402

import traceback  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src_v2 import pipeline as P  # noqa: E402


def short(model):
    return model.split("/")[-1]


def state_id(model, u, dtype):
    return f"{short(model)}__{u['order']}__s{float(u['sparsity']):.4f}__b{int(u['bits'])}__seed{int(u.get('seed', 0))}__{dtype}"


class Log:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "a", buffering=1)

    def __call__(self, *a):
        msg = " ".join(str(x) for x in a)
        line = f"[{P.utcnow()}] {msg}"
        print(line, flush=True)
        self.f.write(line + "\n")


def write_status(path, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    kw["updated"] = P.utcnow()
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(kw, f)
    os.replace(tmp, path)


def tensor_sha(t):
    return hashlib.sha256(t.cpu().numpy().tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------- uploads (optional)
class Uploader:
    """Uploads files to a Hugging Face repo (one commit per batch) if a token file exists on this machine.

    Without a token the files stay on the runtime and are pulled by the orchestrator through the Jupyter
    API. Failed batches stay queued and are retried at the next flush. background=True runs each flush
    in a thread (used for model files, so the GPU keeps evaluating while they upload)."""

    def __init__(self, repo_id, log, repo_type="dataset", background=False):
        self.repo_id, self.log, self.repo_type, self.background = repo_id, log, repo_type, background
        self.pending, self.threads = [], []
        tokfile = os.path.expanduser("~/.cache/huggingface/token")
        self.enabled = bool(repo_id) and os.path.exists(tokfile)

    def push(self, files, message):
        if not self.enabled:
            return
        self.pending.append((list(files), message))
        if self.background:
            import threading
            t = threading.Thread(target=self.flush, daemon=True)
            t.start()
            self.threads.append(t)
        else:
            self.flush()

    def flush(self):
        if not self.enabled or not self.pending:
            return
        from huggingface_hub import CommitOperationAdd, HfApi
        api = HfApi()
        batch, self.pending = self.pending, []
        for files, message in batch:
            try:
                ops = [CommitOperationAdd(path_in_repo=r, path_or_fileobj=l) for l, r in files]
                api.create_commit(repo_id=self.repo_id, repo_type=self.repo_type, operations=ops,
                                  commit_message=message)
            except Exception as e:  # noqa: BLE001
                self.log(f"upload to {self.repo_id} failed ({message}): {type(e).__name__}: {str(e)[:160]}; queued")
                self.pending.append((files, message))

    def join(self, timeout=None):
        for t in self.threads:
            t.join(timeout)


# ----------------------------------------------------------------------------- one unit
def unit_checks(u, diags, wt):
    s, b = float(u["sparsity"]), int(u["bits"])
    c = {"ppl_finite": math.isfinite(wt["ppl"])}
    if diags:
        c["rel_err_finite"] = all(math.isfinite(d["rel_err"]) for d in diags)
        if s > 0:
            c["mask_frac_ok"] = all(abs(d["mask_frac"] - s) < 2e-3 for d in diags)
            c["mask_kept"] = all(d["mask_kept"] for d in diags)
        if b < 16:
            c["levels_ok"] = all(d["max_levels"] <= 2 ** b for d in diags)
    c["all_ok"] = all(c.values())
    return c


def eval_tasks(model, tok, tasks, bs, log, fewshot=0, limit=None):
    import torch
    try:
        agg, items, meta = P.run_lmeval(model, tok, tasks, bs, num_fewshot=fewshot, limit=limit)
        return agg, items, meta, bs
    except torch.cuda.OutOfMemoryError:
        log(f"OOM in lm-eval at batch {bs}; retrying once at batch {max(1, bs // 4)}")
        gc.collect()
        torch.cuda.empty_cache()
        bs2 = max(1, bs // 4)
        agg, items, meta = P.run_lmeval(model, tok, tasks, bs2, num_fewshot=fewshot, limit=limit)
        return agg, items, meta, bs2


def run_unit(ctx, u, log):
    import torch
    args, model, tok, dense, dense_sd, core = ctx["args"], ctx["model"], ctx["tok"], ctx["dense"], ctx["dense_sd"], ctx["core"]
    sid = state_id(args.model, u, args.dtype)
    t, started = {}, P.utcnow()
    P.set_determinism(0)
    ctx["recorder"].reset()
    model.load_state_dict(dense_sd)
    seed = int(u.get("seed", 0))
    if seed not in ctx["calibs"]:
        ctx["calibs"][seed] = P.get_calib(tok, ctx["c4"], P.N_CALIB, P.SEQLEN, seed)
        ctx["calib_sha"][seed] = tensor_sha(torch.cat(ctx["calibs"][seed], 0))
    s, b, order = float(u["sparsity"]), int(u["bits"]), u["order"]
    write_status(ctx["status"], state_id=sid, stage="compress", done=ctx["n_done"], todo=ctx["n_todo"])
    diags = []
    with P.timer(t, "compress"):
        if order != "dense":
            diags = core.compress_model(model, ctx["calibs"][seed], s, b, order, args.device, groupsize=P.GROUPSIZE,
                                        verbose=True, log=log)
    if args.device == "cuda":
        torch.cuda.synchronize()
    export = None
    if order != "dense" and args.export_models:
        with P.timer(t, "export"):
            mpath = os.path.join(ctx["models_root"], f"{sid}.safetensors")
            export = P.export_compressed(core, model, ctx["recorder"].records, b, mpath,
                                         {"state_id": sid, "model": args.model, "order": order, "sparsity": s,
                                          "bits": b, "seed": seed, "dtype": args.dtype,
                                          "commit": ctx["fp"].get("commit"), "code_hash": P.code_hash()})
        export["repo_path"] = f"models/{short(args.model)}/{sid}.safetensors"
        ctx["model_uploader"].push([(mpath, export["repo_path"])], f"model {sid}")
        log(f"  exported {export['bytes'] / 2**20:.1f} MiB, fp32 fallbacks {len(export['fp32_fallback_layers'])}")
    ctx["recorder"].reset()
    sym = P.model_symbol_entropy(core, model) if (b < 16 and order != "dense") else None
    kl_on = u.get("kl", True)
    write_status(ctx["status"], state_id=sid, stage="wikitext2", done=ctx["n_done"], todo=ctx["n_todo"])
    with P.timer(t, "wikitext2"):
        wt = P.lm_metrics(model, ctx["wt_win"], args.device, dense=dense if kl_on else None)
    log(f"  {sid}: wikitext2 ppl {wt['ppl']:.4f}" + (f" KL {wt['kl']:.5f} top1 {wt['top1']:.4f}" if kl_on else ""))
    c4m = None
    if u.get("c4", True):
        write_status(ctx["status"], state_id=sid, stage="c4", done=ctx["n_done"], todo=ctx["n_todo"])
        with P.timer(t, "c4"):
            c4m = P.lm_metrics(model, ctx["c4_win"], args.device, dense=dense if kl_on else None)
        log(f"  {sid}: c4 ppl {c4m['ppl']:.4f}" + (f" KL {c4m['kl']:.5f} top1 {c4m['top1']:.4f}" if kl_on else ""))
    tasks = u.get("tasks", P.ZERO_SHOT_TASKS)
    agg, items, tmeta, bs_used = {}, {}, {}, None
    if tasks:
        write_status(ctx["status"], state_id=sid, stage="tasks", done=ctx["n_done"], todo=ctx["n_todo"])
        with P.timer(t, "tasks"):
            agg, items, tmeta, bs_used = eval_tasks(model, tok, tasks, args.eval_batch, log,
                                                    limit=2 if args.smoke else u.get("limit"))
    fs = {}
    for task, k in (u.get("fewshot") or {}).items():
        write_status(ctx["status"], state_id=sid, stage=f"fewshot:{task}", done=ctx["n_done"], todo=ctx["n_todo"])
        with P.timer(t, f"fewshot_{task}"):
            a2, i2, m2, _ = eval_tasks(model, tok, [task], max(1, args.eval_batch // 4), log, fewshot=int(k),
                                       limit=u.get("limit"))
        fs[task] = {"num_fewshot": int(k), "results": a2, "meta": m2}
        items.update({f"{kk}@{k}shot": vv for kk, vv in i2.items()})
    items_abs = os.path.join(ctx["items_root"], f"{sid}.json.gz")
    items_rel = os.path.relpath(items_abs, ROOT)
    sha = P.write_items(items_abs, items) if items else None
    checks = unit_checks(u, diags, wt)
    main = {task: agg[task].get(f"{P.main_metric(task)},none") for task in agg if task in P.ZERO_SHOT_TASKS}
    row = {
        "state_id": sid, "family": args.family, "model": args.model, "order": order, "sparsity": s, "bits": b,
        "seed": seed, "dtype": args.dtype, "tcr": ctx["core"].tcr(s, b) if b <= 16 else None,
        "storage": P.storage(s, b, diags, sym), "wikitext2": wt, "c4": c4m,
        "tasks": agg, "main": main, "fewshot": fs, "task_meta": tmeta, "eval_batch": bs_used,
        "items_file": items_rel if items else None, "items_sha256": sha,
        "mean_rel_err": (sum(d["rel_err"] for d in diags) / len(diags)) if diags else 0.0,
        "diags": diags, "checks": checks, "timings": t, "started": started, "finished": P.utcnow(),
        "n_calib": P.N_CALIB, "seqlen": P.SEQLEN, "groupsize": P.GROUPSIZE,
        "calib_sha": ctx["calib_sha"].get(seed), "wt_windows_sha": ctx["wt_sha"], "c4_windows_sha": ctx["c4_sha"],
        "compressed_weights": ctx["n_comp"], "model_export": export,
        "status": "ok" if checks["all_ok"] else "failed_check",
        "metadata": P.run_metadata(args.dtype, args.runtime_name, args.session,
                                   {"unit": u, "fp16_flag": args.dtype != "float32"}),
    }
    P.append_jsonl(ctx["out"], row)
    files = [(ctx["out"], os.path.relpath(ctx["out"], ROOT))] + ([(items_abs, items_rel)] if items else [])
    ctx["uploader"].push(files, f"result {sid}")
    log(f"DONE {sid} status={row['status']} ppl={wt['ppl']:.4f} times={t}")
    return row


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--queue", required=True, help="JSON list of units")
    ap.add_argument("--out", default=None, help="default results/v2/<family>/<model>.jsonl")
    ap.add_argument("--locked", default=None, help="locked fingerprint JSON; refuse to run if it differs")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    ap.add_argument("--eval-batch", type=int, default=32)
    ap.add_argument("--runtime-name", default=None)
    ap.add_argument("--session", default=None)
    ap.add_argument("--hf-repo", default=None, help="private HF dataset for result uploads (needs a token file)")
    ap.add_argument("--hf-models-repo", default=None, help="HF model repo for the compressed models")
    ap.add_argument("--export-models", action="store_true", help="write every compressed state in exact compact form")
    ap.add_argument("--max-units", type=int, default=None)
    ap.add_argument("--no-dense-copy", action="store_true", help="skip KL (no dense copy on the GPU)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--smoke", action="store_true",
                    help="code-path test only: 4 calibration windows of 256 tokens, 2 eval windows, 2 items per task")
    args = ap.parse_args()
    if args.smoke:
        P.N_CALIB, P.SEQLEN, P.C4_EVAL_WINDOWS = 4, 256, 2

    out = args.out or os.path.join(ROOT, "results", "v2", args.family, f"{short(args.model)}.jsonl")
    base = os.path.dirname(os.path.abspath(out)) if args.smoke else ROOT  # smoke runs never touch results/ or logs/
    log = Log(os.path.join(base, "logs", "runner", f"{args.family}__{short(args.model)}.log"))
    status = os.path.join(base, "logs", "status", f"{args.family}__{short(args.model)}.json")
    items_root = os.path.join(base, "items") if args.smoke else os.path.join(ROOT, "results", "v2", "items")
    import torch
    P.set_determinism(0)
    fp = P.fingerprint()
    log(f"start family={args.family} model={args.model} fingerprint={json.dumps(fp)}")
    if args.locked:
        with open(args.locked) as f:
            P.check_fingerprint(fp, json.load(f))
        log("fingerprint matches the locked one")
    with open(args.queue) as f:
        units = json.load(f)
    done = {r["state_id"] for r in P.read_jsonl(out) if r.get("status") in ("ok", "failed_check")}
    fails = {}
    for r in P.read_jsonl(os.path.join(os.path.dirname(out), f"failures__{short(args.model)}.jsonl")):
        fails[r["state_id"]] = fails.get(r["state_id"], 0) + 1
    gave_up = {k for k, v in fails.items() if v >= 2}
    if gave_up:
        log(f"skipping {len(gave_up)} units that failed twice: {sorted(gave_up)}")
    todo = [u for u in units if state_id(args.model, u, args.dtype) not in done | gave_up]
    if args.max_units:
        todo = todo[:args.max_units]
    log(f"{len(units)} units in queue, {len(units) - len(todo)} already done, {len(todo)} to run")
    if not todo:
        write_status(status, stage="finished", done=len(units), todo=0)
        return 0

    core = P.load_core(strict=True)
    recorder = P.GridRecorder(core)
    model, tok = P.load_model(args.model, args.dtype, args.device)
    dense = None if args.no_dense_copy else copy.deepcopy(model)
    dense_sd = {k: v.detach().clone().cpu() for k, v in model.state_dict().items()}
    c4, wt = P.load_c4_validation(), P.load_wikitext2_test()
    cache = os.path.join(base, "cache", short(args.model) + ("_smoke" if args.smoke else ""))
    os.makedirs(cache, exist_ok=True)
    wt_win = P.wikitext2_windows(tok, wt, P.SEQLEN)
    if args.smoke:
        wt_win = wt_win[:2]
    cpath = os.path.join(cache, f"c4_eval_{P.C4_EVAL_WINDOWS}x{P.SEQLEN}_seed{P.C4_EVAL_SEED}.pt")
    if os.path.exists(cpath):
        blob = torch.load(cpath)
        c4_win, c4_where = blob["windows"], blob["where"]
    else:
        excl = set()
        for sd in P.CALIB_SEEDS_EXCLUDED:
            excl |= {i for i, _ in P.calib_documents(tok, c4, P.N_CALIB, P.SEQLEN, sd)}
        c4_win, c4_where = P.c4_eval_windows(tok, c4, n=P.C4_EVAL_WINDOWS, seqlen=P.SEQLEN, exclude=excl)
        torch.save({"windows": c4_win, "where": c4_where, "excluded_docs": sorted(excl)}, cpath)
    ctx = dict(items_root=items_root, recorder=recorder, fp=fp,
               models_root=os.path.join(base if args.smoke else ROOT, "models", short(args.model)),
               model_uploader=Uploader(args.hf_models_repo, log, repo_type="model", background=True),
               args=args, model=model, tok=tok, dense=dense, dense_sd=dense_sd, core=core, c4=c4, wt_win=wt_win,
               c4_win=c4_win, calibs={}, calib_sha={}, wt_sha=tensor_sha(wt_win), c4_sha=tensor_sha(c4_win),
               out=out, status=status, n_done=len(units) - len(todo), n_todo=len(todo),
               n_comp=P.compressed_weight_count(core, model), uploader=Uploader(args.hf_repo, log))
    log(f"model loaded ({args.dtype}); compressed weights {ctx['n_comp']}; wikitext windows {tuple(wt_win.shape)}; "
        f"c4 windows {tuple(c4_win.shape)}; dense copy {'yes' if dense is not None else 'no'}")
    failures = 0
    for u in todo:
        sid = state_id(args.model, u, args.dtype)
        log(f"BEGIN {sid}")
        try:
            run_unit(ctx, u, log)
        except Exception as e:  # noqa: BLE001
            failures += 1
            err = f"{type(e).__name__}: {str(e)[:500]}"
            log(f"FAILED {sid}: {err}\n{traceback.format_exc()[-2000:]}")
            P.append_jsonl(os.path.join(os.path.dirname(out), f"failures__{short(args.model)}.jsonl"),
                           {"state_id": sid, "family": args.family, "model": args.model, "unit": u, "error": err,
                            "time": P.utcnow()})
            gc.collect()
            if args.device == "cuda":
                torch.cuda.empty_cache()
        ctx["n_done"] += 1
        ctx["uploader"].flush()
    ctx["model_uploader"].join(timeout=1800)
    ctx["model_uploader"].flush()
    ctx["uploader"].flush()
    write_status(status, stage="finished", done=ctx["n_done"], todo=0, failures=failures,
                 upload_backlog=len(ctx["uploader"].pending) + len(ctx["model_uploader"].pending))
    log(f"queue finished: {failures} failures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
