# Decisions

Every judgment call, with the reason and the date (UTC). Newest last.

1. **2026-11-09, Explainer document missing.** The Drive folder holds only "Review" and "Code". Phase 0 rule
   says this fails the check; the author chose in session to proceed and to fold the Explainer in if it is
   added later. Every review claim is re-verified independently anyway.
2. **2026-11-09, where commits go.** The fork `martin-cousseau/esann2027-compose` is public. The author chose
   in session to push to it, with a draft PR only. The private inputs (review text, draft PDF, Drive exports)
   stay out of git (`context/` is in `.gitignore`) and are backed up to the private HF dataset
   `martincousseau/esann2027-v2-results` instead.
3. **2026-11-09, branch name.** The session is bound to branch `claude/busy-hopper-ghtynk`; it plays the role
   of `exp/review-2026-11` in the brief. Nothing is pushed to `main` or to the author's repository.
4. **2026-11-09, GPU type.** Rates found (logs/budget.md): L4 about 1.7-1.94 CU/h, A100 5.4-9, G4 8.7-10+.
   G4 is above "at most about twice L4", so only L4 is piloted. T4 cannot hold SmolLM2-1.7B in fp32 with a
   dense copy for KL (needs about 18-20 GB), so it fails the choice rule. Lock decided after the pilot.
5. **2026-11-09, units are model states.** A result line is identified by
   `model__order__s{sparsity}__b{bits}__seed{seed}__{dtype}`; families reuse states (dense, single
   methods) instead of re-evaluating them.
6. **2026-11-09, main metric per task.** The v2 protocol's main metric is acc_norm for ARC-E, ARC-C,
   HellaSwag, OpenBookQA and PIQA, acc otherwise. The paper used acc for ARC-E and PIQA. Both are stored per
   item; the headline Sr is reported with the v2 metric and the paper's metric side by side.
7. **2026-11-09, C4 evaluation windows.** 256 windows of 2048 tokens from distinct C4-validation (shard 0)
   documents, sampled with seed 20261109, excluding every document used by calibration seeds 0, 1 and 2
   (the paper calibrates on the same shard).
8. **2026-11-09, token transfer to Colab.** The container can reach the runtime's Jupyter server directly
   (the connector's connection URL and proxy token). The HF token is therefore written to
   `/root/.cache/huggingface/token` by `scripts/colab_jupyter.py put-token`, which reads `$HF_TOKEN` from the
   container environment, so the value never appears in any tool call, output, log or commit; mode 600 is
   set with a one-line `execute_code` (no secret in it). This replaces the execute_code transfer of section
   3.4 with a channel that exposes less. Bulk results come back the same way (`colab_jupyter.py pull`).
9. **2026-11-09, image lock.** Colab image `release-colab-external-images_20261106-060046_RC00` (the latest
   when the pilot was created). Later runtimes are created with this version string; if Colab refuses it,
   the run stops and the mismatch is logged.
10. **2026-11-09, budget freedom (author, in session).** The 100 CU are one pool, to be allocated freely
    and spent on speed (parallel runtimes) as needed. The allocation table in logs/budget.md is now
    indicative only; the total stays capped at 100 CU (stop at about 5 CU estimated left).
11. **2026-11-09, models published at the end (author, in session).** Every compressed state is exported
    in an exact compact form (integer codes, per-group fp32 scale and uint8 zero point, mask bitmap; pruned-only
    layers as fp32 kept values + bitmap), verified bit for bit before writing, and uploaded from the runtime
    to `martincousseau/esann2027-compose-v2` (private until the end of the session, then public with a README,
    linked from the fork). Round trip tested for every order and arm (tests/test_pipeline.py).
12. **2026-11-09, tinyBenchmarks dropped.** lm-eval 0.4.13 needs a git install of the tinyBenchmarks package
    for its metrics; it is first in the shrink order, so it is not run.
13. **2026-11-09, GPU subagents do not poll.** Colab tool calls time out after 60 s on the client side, and a
    subagent cannot sleep between polls. Waiting is done by a container-side watcher
    (`scripts/watch_runtime.py`, run in the background by the orchestrator), which mirrors results through
    the Jupyter API and exits on events. The orchestrator makes the Colab calls; subagents write family code
    and analyse pulled results (the brief's fallback mode).
14. **2026-11-10, W&B (author, in session).** Every finished result line is mirrored to the public W&B
    project https://wandb.ai/martin-css/esann2027-compose-v2 (one run per model state, idempotent,
    `scripts/wandb_sync.py`), with links to the git commit, the per-item file and the HF model file. Sync runs
    from the container after results are pulled, so the GPU runner has no W&B dependency. The API key lives
    outside the repository (mode 600) and is never printed or committed.
15. **2026-11-10, two runtimes at once.** Colab refused a third concurrent L4 ("Precondition check failed").
    E2 therefore shares runtime A with E1 (E2's sweeps are CPU/launch bound, E1's evaluation GPU bound);
    results do not depend on co-location, only the logged timings do.
16. **2026-11-10, runtime outputs outside the checkout.** Runners write results, logs, caches and models to
    `ESANN_OUT=/content/out` on the runtimes, not into the git checkout: committing the pulled results made
    `git checkout` of new code collide with the live result files. The container keeps the same paths
    (`results/v2/...`); the watcher mirrors `/content/out/results/v2` into them.
17. **2026-11-10, lm-eval batch sizes.** Batch 64 ran out of memory on long BoolQ/SciQ requests (fp32 logits
    and log-softmax over a 49k vocabulary). Fixed sizes: 16 for SmolLM2 models, 8 for Qwen2.5 (152k vocabulary);
    the OOM retry now runs outside the except block, and the dense copy waits on the CPU during lm-eval.
18. **2026-11-10, keep the kernel busy.** Colab reclaimed both runtimes about 20 minutes after the last code
    execution, although runner processes were using the GPU (and the pilot earlier the same way): the idle
    check looks at the kernel, not at background processes. Chains are now started from a kernel cell that
    blocks until they finish, so the runtime stays alive while it works and frees itself after its queue is
    done. Finished units were all safe (resume-safe runners, results pulled every 90 s); the two units in
    progress were lost and are re-run. Finished result files are pushed back to fresh runtimes so the runners
    skip them.
19. **2026-11-10, lm-eval batch groups.** The benchmarks took about 75% of each unit at batch 16. Short-input
    tasks (ARC-E/C, HellaSwag, OBQA, PIQA, WinoGrande) now run at 3x the batch of the long-passage tasks
    (BoolQ, SciQ, LAMBADA), which set the memory peak: SmolLM2 48/16, Qwen 24/8. Fixed sizes, so the
    evaluation stays deterministic. The first units (both dense models, and the 3 gate units of SmolLM2-135M)
    ran at a single batch size; batch size changes only padding, so per-item results can differ at most on
    near-tied items (checked at the end by re-evaluating the dense models).
20. **2026-11-10, E3 and E4 scope.** Measured cost per E1 unit (L4): about 10 min (SmolLM2-135M), about 21 min
    (360M), about 22 min (Qwen2.5-0.5B), three quarters of it lm-eval. With E1 at about 20 L4-hours, E3 runs
    with WikiText-2 and C4 perplexity, KL and top-1 only (no benchmarks; the mechanism question is about
    the perplexity/KL gap), and E4 adds 5 frontier states per model (2-bit, 60% alone, 60%+4-bit in the three
    orders) with the full E1 metrics. If CU remain at the end, E3's 50%+4 and 25%+3 cells get the paper's
    three benchmarks.
21. **2026-11-10, runtime names in metadata.** Units run on the replacement runtimes `esann-e1-a2`/`-b2` carry
    `colab_runtime_name` = `esann-e1-a`/`-b` (the chain scripts' names); the GPU UUID in `nvidia_smi` and
    `logs/runtimes.jsonl` identify the machine. chain_b2 records `esann-e1-b2` correctly.
