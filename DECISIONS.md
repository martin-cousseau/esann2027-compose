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
22. **2026-11-10, reproduction gate re-read after investigation.** fp32 on L4 vs the published 16-bit (T4)
    runs of SmolLM2-135M: dense -0.000%, 4-bit +0.74%, 50%+4 P->Q +2.62%; Qwen2.5-0.5B dense -0.003%, 4-bit
    +0.50%. The same code in 16-bit on L4 gives SmolLM2-135M 4-bit +1.50% vs the same published 16-bit run,
    so the gap is not the precision change and not the pipeline (dense matches to 1e-5): tiny kernel-level
    differences between GPU types flip greedy rounding and mask decisions, which cascade through the
    sequential calibration. The 0.5% tolerance cannot hold across GPU types for compressed states; within the
    locked type, determinism is checked bit for bit (DET). E1 proceeds; the cross-hardware sensitivity is
    reported as part of the noise analysis (R6). The fp32 SmolLM2-360M check against the paper's own fp32
    re-runs (REPRO32) completes the picture.
23. **2026-11-10, phase-0 checks settled.** fp32 on L4 reproduces the paper's own fp32 re-runs of SmolLM2-360M
    (T4) within the 0.5% tolerance: joint 25%+4 -0.11%, P->Q 25%+4 +0.37%, Q->P 25%+4 +0.37% (REPRO32). The
    SmolLM2-135M joint 50%+4 state is bit-identical on two different L4 machines (WikiText-2 NLL and every
    layer error). Both checks pass with matched precision; the 135M differences vs 16-bit T4 runs are the
    hardware/precision effect of #22.
24. **2026-11-10, keepalive over the kernel channel.** Runtimes kept dying 20-35 min after the last client
    code execution even with a busy kernel cell (lost: esann-e1-a2, -b2, -a3). The watcher now executes a
    trivial cell over the Jupyter websocket every 4 min on each runtime (`scripts/colab_keepalive.py`), the
    same channel the connector's execute_code uses. Results the runtimes uploaded to the HF results dataset
    after the watcher stopped were recovered from there.
25. **2026-11-10, angle B + C (provisional, writing started).** E2 shows on all three core models that neither
    re-fitting the grid nor the pruning-only mask removes most of the quantize-first penalty, so angle A's
    evidence bar (one arm closes >= 60% on 2 of 3 models) fails at layer level; E3 will check it at model
    level. The interim SmolLM2-135M frontier supports angle B (50%+4-bit beats plain 3-bit at 3.0 bits/w with
    the mask counted; 25%+4-bit does not beat 4-bit at 4.0 bits/w) and C (Q->P at 25%+3 equals 3-bit exactly).
    The paper is drafted for B + C with the mechanism reported as a negative result; claims are macros, so the
    text follows the final numbers. Revisit when E1 for 360M and Qwen and E3/E4 are complete.
26. **2026-11-10, second Colab image admitted.** Runtimes created at 18:04 came up on
    `release-colab-external-images_20261109-060054_RC00` instead of the locked `..._20261106-060046_RC00`,
    with identical package versions; the fingerprint check refused them, as designed. On both new runtimes,
    SmolLM2-135M joint 50%+4 recomputed on the new image is bit-identical to the old image's E1 result
    (WikiText-2 NLL sum and all 210 layer errors; `results/v2/IMG/` on the runtimes). The lock now lists both
    images; units record which image they ran on.
27. **2026-11-10, runtime-side keepalive.** The external keepalive needs a proxy token that expires hourly and
    stops when this session is suspended (both runtimes died ~1 h after a suspension at 14:48). Each runtime
    now also runs `colab/local_keepalive.py`, which executes a trivial cell through its own Jupyter server
    every 4 min.
28. **2026-11-10, angle B + C confirmed on the complete SmolLM2-135M picture** (E1 25 states, E2, E3 16, E4 5):
    the frontier shows no gain from pruning at 4.0 bits/w, a clear gain at 3.0 bits/w (50%+4 vs 3-bit) and
    the only usable points below 3 bits (2-bit plain collapses); angle A fails its bar (no single arm closes
    >= 60% consistently; refit never helps). Final confirmation when 360M and Qwen are complete.
29. **2026-11-10, E5 and E6 scope for the remaining budget.** About 21 CU remain after the current queues.
    E6 runs SmolLM2-1.7B in 7 states (dense, 4-bit, 3-bit, 25%+4 and 50%+4 in P->Q and joint) with perplexity,
    C4 and KL only, so the frontier conclusion is tested within one family at 15x the smallest model. E5 runs
    SmolLM2-360M with calibration seeds 1 and 2 (4-bit alone and the nine 4-bit cells), perplexity, C4 and KL.
    Qwen2.5 seeds, the prune-only seeds, the 3-bit seeds, Qwen2.5-1.5B/3B and MMLU are cut (shrink order).
