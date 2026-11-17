#!/usr/bin/env python3
"""Phase-0 container-side checks. Prints PASS/FAIL per line; never prints the token.

Usage: python scripts/check_env.py [--repo-id NAME] [--log logs/env_check.txt]
"""
import argparse
import datetime as dt
import io
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINES = []


def log(status, name, detail=""):
    line = f"{status:4s} | {name} | {detail}".rstrip(" |")
    print(line, flush=True)
    LINES.append(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-name", default="esann2027-v2-results")
    ap.add_argument("--log", default=os.path.join(ROOT, "logs", "env_check.txt"))
    args = ap.parse_args()
    ok = True
    LINES.append(f"# container checks, {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')}")

    # 1. token
    tok = os.environ.get("HF_TOKEN", "")
    if tok:
        log("PASS", "1 HF_TOKEN set", "non-empty (value never printed)")
    else:
        log("FAIL", "1 HF_TOKEN set", "missing or empty")
        return finish(args, False)

    # 2. identity
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        who = api.whoami(token=tok)
        user = who.get("name")
        at = (who.get("auth") or {}).get("accessToken") or {}
        role = at.get("role")
        write = role == "write"
        if role == "fineGrained":
            fg = at.get("fineGrained") or {}
            perms = set()
            for sc in fg.get("scoped", []) or []:
                perms.update(sc.get("permissions", []) or [])
            perms.update(fg.get("global", []) or [])
            write = any(p.startswith("repo.write") or p == "repo.content.write" for p in perms) or "repo.write" in perms
        log("PASS" if write else "FAIL", "2 HF identity", f"user={user} role={role} write_scope={write}")
        ok &= bool(write)
    except Exception as e:  # noqa: BLE001
        log("FAIL", "2 HF identity", f"{type(e).__name__}: {str(e)[:200]}")
        return finish(args, False)

    # 3. private results repo, upload + delete
    repo_id = f"{user}/{args.repo_name}"
    try:
        api.create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
        info = api.repo_info(repo_id, repo_type="dataset")
        if not info.private:
            log("FAIL", "3 results repo private", f"{repo_id} exists but is PUBLIC")
            return finish(args, False)
        name = f"_phase0/test_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%S')}.txt"
        api.upload_file(path_or_fileobj=io.BytesIO(b"phase0 write test\n"), path_in_repo=name,
                        repo_id=repo_id, repo_type="dataset", commit_message="phase0 write test")
        files = api.list_repo_files(repo_id, repo_type="dataset")
        assert name in files, "uploaded file not listed"
        api.delete_file(name, repo_id=repo_id, repo_type="dataset", commit_message="phase0 cleanup")
        files = api.list_repo_files(repo_id, repo_type="dataset")
        assert name not in files, "test file still present after delete"
        log("PASS", "3 results repo", f"{repo_id} private=True upload+delete OK")
    except Exception as e:  # noqa: BLE001
        log("FAIL", "3 results repo", f"{type(e).__name__}: {str(e)[:200]}")
        ok = False

    # 4. review documents present in context/
    need = {"context/review.md": "Major issue 1", "context/explainer.md": None,
            "context/review_code.zip": None, "context/esann2027_paper_draft.pdf": None}
    for rel, sig in need.items():
        p = os.path.join(ROOT, rel)
        if not os.path.exists(p) or os.path.getsize(p) == 0:
            log("FAIL", f"4 document {rel}", "missing")
            ok = False
            continue
        if sig and sig not in open(p, encoding="utf8", errors="replace").read():
            log("FAIL", f"4 document {rel}", f"signature {sig!r} not found")
            ok = False
            continue
        log("PASS", f"4 document {rel}", f"{os.path.getsize(p)} bytes")

    # 5. repository: commit, review code unpacked, push permission (dry run only, nothing is pushed)
    commit = subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    branch = subprocess.run(["git", "-C", ROOT, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True).stdout.strip()
    log("PASS", "5a repository", f"branch={branch} HEAD={commit}")
    rc_ok = all(os.path.exists(os.path.join(ROOT, "review_code", f)) for f in
                ("ablation/arms.py", "ablation/extra_orders.py", "ablation/ablation_same_hessian.py",
                 "reanalysis/analyze_results.py"))
    log("PASS" if rc_ok else "FAIL", "5b review_code unpacked", "")
    ok &= rc_ok
    dry = subprocess.run(["git", "-C", ROOT, "push", "--dry-run", "origin", f"HEAD:refs/heads/{branch}"],
                         capture_output=True, text=True, timeout=120)
    push_ok = dry.returncode == 0
    log("PASS" if push_ok else "FAIL", "5c push permission (dry run; nothing pushed)",
        (dry.stderr or dry.stdout).strip().splitlines()[-1][:160] if (dry.stderr or dry.stdout).strip() else "")
    ok &= push_ok

    # 6. baseline re-analysis
    out = subprocess.run([sys.executable, "-I", os.path.join(ROOT, "review_code/reanalysis/analyze_results.py"), ROOT,
                          "--csv", os.path.join(ROOT, "results/v2/storage_points_v1.csv")],
                         capture_output=True, text=True, timeout=600)
    with open(os.path.join(ROOT, "logs", "reanalysis_v1_output.txt"), "w") as f:
        f.write(out.stdout + ("\n[stderr]\n" + out.stderr if out.stderr else ""))
    good = "max |recomputed - printed| = 0.000" in out.stdout
    log("PASS" if good else "FAIL", "6 baseline re-analysis", "max |recomputed - printed| = 0.000" if good else
        f"rc={out.returncode} stderr={out.stderr.strip()[-200:]}")
    ok &= good
    return finish(args, ok)


def finish(args, ok):
    LINES.append(f"RESULT container checks: {'PASS' if ok else 'FAIL'}")
    print(LINES[-1])
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    with open(args.log, "a") as f:
        f.write("\n".join(LINES) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
