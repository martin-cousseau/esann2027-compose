#!/usr/bin/env python3
"""Phase-0 GPU-side checks. One PASS/FAIL/INFO line per check; never prints a token.

Usage (from /content/work after bootstrap): python colab/check_runtime.py [--skip-lmeval] [--skip-tests]
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import argparse  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RESULTS = {"ok": True}

MODELS = ["HuggingFaceTB/SmolLM2-135M", "HuggingFaceTB/SmolLM2-360M", "Qwen/Qwen2.5-0.5B",
          "HuggingFaceTB/SmolLM2-1.7B", "Qwen/Qwen2.5-1.5B", "Qwen/Qwen2.5-3B"]
EXPECTED_SPLITS = {"arc_easy": 2376, "piqa": 1838, "hellaswag": 10042, "arc_challenge": 1172, "winogrande": 1267,
                   "openbookqa": 500, "boolq": 3270, "sciq": 1000, "lambada_openai": 5153}


def line(status, name, detail=""):
    if status == "FAIL":
        RESULTS["ok"] = False
    print(f"{status:4s} | {name} | {detail}", flush=True)


def check(name, fn):
    try:
        detail = fn()
        line("PASS", name, detail if detail is not None else "")
    except Exception as e:  # noqa: BLE001
        line("FAIL", name, f"{type(e).__name__}: {str(e)[:300]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-lmeval", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    args = ap.parse_args()
    import torch
    from src_v2 import pipeline as P
    P.set_determinism(0)

    def nvsmi():
        q = "name,driver_version,uuid,memory.total,clocks.max.sm,pci.bus_id"
        return subprocess.run(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"], capture_output=True,
                              text=True, timeout=30, check=True).stdout.strip()
    check("nvidia-smi", nvsmi)

    def cuda():
        assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
        return f"torch {torch.__version__} cuda {torch.version.cuda} cudnn {torch.backends.cudnn.version()} " \
               f"device {torch.cuda.get_device_name(0)} capability {torch.cuda.get_device_capability(0)}"
    check("torch.cuda", cuda)

    def virt():
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(0)
        mode = pynvml.nvmlDeviceGetVirtualizationMode(h)
        names = {0: "none (bare metal)", 1: "passthrough", 2: "vgpu", 3: "host_vgpu", 4: "host_vsga"}
        return f"{names.get(mode, mode)}; driver {pynvml.nvmlSystemGetDriverVersion()}"
    check("NVML virtualization mode", virt)

    def resources():
        import psutil
        p = torch.cuda.get_device_properties(0)
        du = shutil.disk_usage("/content" if os.path.exists("/content") else "/")
        return (f"GPU mem {p.total_memory / 2**30:.1f} GiB | RAM {psutil.virtual_memory().total / 2**30:.1f} GiB | "
                f"disk free {du.free / 2**30:.0f} GiB | CPUs {os.cpu_count()}")
    check("GPU memory, RAM, disk", resources)

    def matmul():
        P.set_determinism(0)
        a = torch.randn(4096, 4096, device="cuda")
        b = torch.randn(4096, 4096, device="cuda")
        for _ in range(3):
            a @ b
        torch.cuda.synchronize()
        n, t0 = 20, time.time()
        for _ in range(n):
            a @ b
        torch.cuda.synchronize()
        dt = (time.time() - t0) / n
        tf = 2 * 4096 ** 3 / dt / 1e12
        return f"4096^2 fp32 matmul {dt * 1e3:.2f} ms = {tf:.1f} TFLOPS (TF32 off: {not torch.backends.cuda.matmul.allow_tf32})"
    check("fp32 matmul timing", matmul)

    def configs():
        from huggingface_hub import hf_hub_download
        out = []
        for m in MODELS:
            p = hf_hub_download(m, "config.json")
            c = json.load(open(p))
            out.append(f"{m.split('/')[-1]}:L{c.get('num_hidden_layers')}/h{c.get('hidden_size')}/"
                       f"{c.get('torch_dtype') or c.get('dtype')}")
        tok = os.path.exists(os.path.expanduser("~/.cache/huggingface/token"))
        return f"{'with' if tok else 'without'} token: " + ", ".join(out)
    check("config.json for 6 models", configs)

    def data():
        c4 = P.load_c4_validation()
        wt = P.load_wikitext2_test()
        ex = c4[0]["text"][:60].replace("\n", " ")
        return f"C4 validation shard 0: {len(c4)} docs (first: {ex!r}...); WikiText-2 test rows: {len(wt)}"
    check("C4 and WikiText-2", data)

    def splits():
        from lm_eval.tasks import TaskManager, get_task_dict
        tm = TaskManager()
        td = get_task_dict(list(EXPECTED_SPLITS), tm)
        got, bad = {}, {}
        for name, task in td.items():
            n = len(list(task.eval_docs))
            got[name] = n
            if EXPECTED_SPLITS.get(name) not in (None, n):
                bad[name] = (n, EXPECTED_SPLITS[name])
        assert not bad, f"split sizes differ (found, expected): {bad}"
        return json.dumps(got)
    check("task split sizes", splits)

    if not args.skip_lmeval:
        def lmeval():
            model, tok = P.load_model("HuggingFaceTB/SmolLM2-135M", "float32", "cuda")
            agg, items, meta = P.run_lmeval(model, tok, P.ZERO_SHOT_TASKS, batch_size=8, limit=2)
            missing = [t for t in P.ZERO_SHOT_TASKS if t not in items]
            assert not missing, f"no per-item records for {missing}"
            agg5, items5, _ = P.run_lmeval(model, tok, ["mmlu"], batch_size=4, num_fewshot=5, limit=1)
            n5 = sum(len(v["doc_id"]) for v in items5.values())
            tiny = []
            try:
                aggt, itemst, _ = P.run_lmeval(model, tok, ["tinyArc", "tinyHellaswag", "tinyMMLU", "tinyWinogrande"],
                                               batch_size=8, limit=2)
                tiny = sorted(itemst)
            except Exception as e:  # noqa: BLE001
                tiny = [f"tinyBenchmarks unavailable: {type(e).__name__}: {str(e)[:120]}"]
            counts = ", ".join("%s:%d" % (t, len(items[t]["doc_id"])) for t in P.ZERO_SHOT_TASKS)
            return f"zero-shot tasks OK ({counts}); mmlu 5-shot items {n5}; tiny: {tiny}"
        check("lm-eval limit=2 on SmolLM2-135M", lmeval)

    if not args.skip_tests:
        def review_tests():
            r1 = subprocess.run([sys.executable, "review_code/ablation/ablation_same_hessian.py", "--repo", ".",
                                 "--smoke-test"], capture_output=True, text=True, cwd=ROOT, timeout=900)
            r2 = subprocess.run([sys.executable, "review_code/ablation/test_extra_orders.py", "--repo", "."],
                                capture_output=True, text=True, cwd=ROOT, timeout=900)
            assert "SMOKE TEST PASSED" in r1.stdout, r1.stdout[-500:] + r1.stderr[-500:]
            assert "ALL CHECKS PASSED" in r2.stdout, r2.stdout[-500:] + r2.stderr[-500:]
            return "ablation_same_hessian --smoke-test PASSED; test_extra_orders ALL CHECKS PASSED"
        check("review test scripts", review_tests)

        def unit_tests():
            r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-m", "network or not network",
                                "tests/test_pipeline.py"], capture_output=True, text=True, cwd=ROOT, timeout=1800)
            tail = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr[-300:]
            assert r.returncode == 0, r.stdout[-1500:]
            return tail
        check("pytest (CPU + CUDA + network)", unit_tests)

    print(f"RESULT runtime checks: {'PASS' if RESULTS['ok'] else 'FAIL'}")
    return 0 if RESULTS["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
