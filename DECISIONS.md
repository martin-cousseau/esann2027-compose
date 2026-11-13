# Decisions

Every judgment call, with the reason and the date (UTC). Newest last.

1. **2026-11-09, Explainer document missing.** The Drive folder holds only "Review" and "Code". Phase 0 rule
   says this fails the check; the author chose in session to proceed and to fold the Explainer in if it is
   added later. Every review claim is re-verified independently anyway.
2. **2026-11-09, where commits go.** The fork `martin-cousseau/esann2027-compose` is public. The author chose
   in session to push to it, with a draft PR only. The private inputs (review text, draft PDF, Drive exports)
   stay out of git (`context/` is in `.gitignore`) and are backed up to a private HF dataset instead
   (done 2026-11-11 in a separate private dataset, `martincousseau/esann2027-v2-private-context`, so the
   results dataset can be made public without carrying them in its history).
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
30. **2026-11-10, keepalive stops when the work does.** The runtime-side keepalive exits after 15 min without
    any chain or runner process, so a finished runtime is freed by Colab within about half an hour even when
    this session is away; follow-on chains (E6 on A, E5 on B) are armed to start as soon as the current chain
    ends, so no runtime idles between queues.
31. **2026-11-11, E6 gets Q->P and pruning alone (4 more states).** The 7-state E6 of #29 cannot answer E6's
    own question (does the 4-bit order spread shrink within a family?), because Q->P, the order that makes the
    spread, was cut, and without pruning alone neither the interaction nor the order rules of E7 can be
    computed at this scale. Added after the 7 states, in this order: Q->P 50%+4, Q->P 25%+4, prune 50%,
    prune 25% (perplexity, C4, KL). Cost about 2 h of L4 (about 4.5 CU at the planning rate), which brings the
    plan to about 95 CU at the planning rate (about 83 at the reported 1.94 CU/h). The two prune-only states
    run last and are dropped if a budget check before them says otherwise. The E5/E6 queues are now written by
    `scripts/make_queues.py` (e5, e6 = as run; e5_full, e6_full = the mission's design), so the analysis plan
    lists only states that are meant to exist.
32. **2026-11-11, runtime B order: seeds before E3.** Measured unit times (SmolLM2-360M about 8 min without
    benchmarks, Qwen2.5-0.5B about 9 min, 20 min with them) put the armed queues at about 97 CU at the planning
    rate (2.23 CU/h; about 84 at the reported 1.94 CU/h), above the stop line of about 95. E5 (calibration seeds,
    the review's R6) now runs before E3 SmolLM2-360M and E3 Qwen2.5-0.5B (`colab/chain_b6.sh`, which replaced
    the rest of chain_b4 and all of chain_b5 at 01:05 UTC without touching the running E4 runner), so a cut,
    if needed, falls on E3 Qwen2.5-0.5B, whose layer-level counterpart (E2) is complete. On runtime A the two
    E6 prune-only states run last for the same reason. The decision to cut is taken from the CU estimate at
    about 07:00 UTC.
33. **2026-11-11, angle B + C confirmed on the three core models (E1 75/75, E2, E4 15/15).** The frontier with the
    mask counted gives the same picture on every model: no gain from pruning at 4.0 bits/w (25%+4 within -0.5% to
    +2.2% of 4-bit), a gain at 3.0 bits/w on both SmolLM2 models (50%+4 and 33%+3 beat 3-bit) but a tie on
    Qwen2.5-0.5B, and the only usable points below 3 bits (plain 2-bit collapses on all three). Q->P at 25%+3 is the
    3-bit model on all three. The paper states the Qwen tie and the 4-bit ties plainly; the abstract's "does not beat
    it" becomes "ties it". Angle A stays rejected (no arm closes 60% of the Q->P gap on 2 of 3 models in E2). The
    order paragraph changes: P->Q is no longer the usual winner (7 of 18 cells); a mask-first order is best in all 18
    and Q->P is strictly worst in 16 of 18.
34. **2026-11-11, budget at 05:10 UTC: keep E6 whole, order E3 Qwen2.5 by information.** About 80 CU spent at the
    planning rate; the armed queues need about 7.4 runtime-hours more (A: 6 E6 states, about 25 min each; B: E3
    SmolLM2-360M then E3 Qwen2.5-0.5B, 32 states of 8-9 min), about 97 CU in total, 2 above the stop line. E6 keeps
    its prune-only states (E7 asks for the order rules on E6). E3 Qwen2.5-0.5B now runs its cells in the order 50%+3,
    50%+4, 33%+3, 25%+3 (`scripts/make_queues.py`; same 16 states): 25%+3 is the cell where Q->P beats P->Q in E1,
    so it has no penalty to explain and is the one to drop (4 states, about 1.3 CU) if the 09:00 check says so.
35. **2026-11-11 06:05 UTC, E3 Qwen2.5-0.5B reduced to its two 50% cells.** Measured times (1.7B states about
    28 min, SmolLM2-360M E3 states about 9 min) raised the projection to about 98.5 CU at the planning rate (about 86
    at the reported 1.94 CU/h). E3 Qwen2.5-0.5B now runs 50%+3 and 50%+4 only (8 states): in E1 the Q->P penalty on
    Qwen2.5-0.5B is +15.6% at 50%+3, +0.9% at 50%+4, +0.9% at 33%+3 and negative at 25%+3, so the two cut cells have
    almost no penalty to explain. Saves about 3 CU; projection about 95.5 at the planning rate. E6 stays whole.
36. **2026-11-11, public release on Hugging Face (as the author asked on 2026-11-10).** The model repository
    `martincousseau/esann2027-compose-v2` (every exported state, verified by size and sha256 against its result
    line; generated card; standalone loader) and the results dataset `martincousseau/esann2027-v2-results` (the
    complete `results/v2` tree of this branch and a card) are public; both link to this branch and to the public W&B
    project. Before the switch, both were scanned for tokens and keys (none). The private inputs stay in the separate
    private dataset `martincousseau/esann2027-v2-private-context` (anonymous access refused, checked).
37. **2026-11-11, response to the independent audit (`reports/audit.md`: 2 blockers, 5 major, 14 minor, 7 nits;
    every number re-derived, 0 mismatches).** Changed in the paper: the "pays only when it replaces a bit" rule is
    gone (33%+3 at 3.0 bits also beats 3-bit on both small SmolLM2 models; new macros), the order claim is "by
    perplexity and KL" with the accuracy outcome of Kim et al.'s test reported (15 of 18), the mechanism claim is
    per layer, with the E3 stale-grid control (-1.3% to +14.1%) given as the reason full-run shares are weak
    evidence; Harma et al.'s App. N (sparsity first also wins with SparseGPT + 4-bit GPTQ, verified from the primary
    source) is cited and OBR is no longer cited for a weight-only pipeline; the code link points to the v2 branch;
    cell sets, E3/E6 scope, the C4 disjointness, the theory wording (1/2 factor, supports, sufficient conditions,
    Cholesky factor), the Bouriel scope (20%), the joint-mode attribution, Fig. 1 (ticks, caption, header comment)
    and the captions (best order chosen on the test set, gaps below the seed spread) are fixed. Outside the paper:
    FINDINGS R3/R4 and items 2 and 5 corrected, a caveats section (lm-eval batch for four states, Cholesky fallback,
    test-set order choice), the Drive ids removed from the current files, and QUESTIONS.md lists what only the
    author can decide (authors, code link after a merge, review code licence, Drive sharing, an open 1-GPU-hour check).
    Not changed: notation collisions (nit 25) and the harness author form (nit 26), for space and because either
    form is acceptable.
38. **2026-11-12, more runs with the larger budget (the author's account has 120 CU, not 100).** The first author's
    new draft makes the order effect the main contribution and attributes its decrease to the model family rather
    than to scale; E6 on SmolLM2-1.7B already shows the 4-bit order spread falling within one family. Three runs
    test what that draft now relies on: (1) Qwen2.5-1.5B in the E6 design (11 states), for scale within Qwen; (2)
    calibration seeds 1 and 2 for the 4-bit cells of Qwen2.5-0.5B (20 states), where the gaps between orders are
    0.1-1%; (3) the BATCH check of audit finding 17 (the four states whose benchmarks ran with one batch size,
    among them both dense references), compared item by item with E1. About 20 CU at the planning rate. Check runs
    now write their per-item files under a family-suffixed name, so they never overwrite a study state's file
    (`scripts/run_grid.py`).
39. **2026-11-12, third Colab image admitted.** The runtimes created at 19:47 came up on
    `release-colab-external-images_20261111-060450_RC00` with identical package versions (torch 2.14.1+cu130,
    transformers 5.22.1, lm_eval 0.4.13). SmolLM2-135M joint 50%+4 recomputed on it is bit-identical to its E1
    result (WikiText-2 NLL sum and all 210 layer errors; `scripts/img_check.py`,
    `results/v2/IMG/SmolLM2-135M__esann-a6.jsonl`), as for #26. The lock lists the three images.
40. **2026-11-12, the 120-CU plan (the author confirms 120 CU available).** To strengthen the first author's three
    questions, the GPU work now adds, in priority order: (1) the full E1 grid (without 8-bit) at SmolLM2-1.7B and
    Qwen2.5-1.5B, so "Q->P is the worst order" is tested at 3 bits beyond 0.5B; (2) Qwen2.5-3B (base design,
    perplexity only: a second fp32 copy for KL does not fit on the L4; no model export, its licence is not
    Apache-2.0), the third size within Qwen; (3) OLMo-2-0425-1B (Apache-2.0, ungated, Llama-like; a smoke run passed),
    a third family at about 1.1B compressed weights, so family and scale can be separated (two families at about
    0.3B, three at about 1.3B, three sizes within SmolLM2 and within Qwen); (4) calibration seeds 1 and 2 for the
    3-bit cells of SmolLM2-360M and Qwen2.5-0.5B and for every cell of SmolLM2-135M. About 85 CU at the planning
    rate on top of #38; benchmarks at scale for the model-level interaction come last, if the balance allows.
41. **2026-11-12, the author package: his paper, revised, not ours.** The first author's 12 Nov draft (order effect
    as the main contribution) is revised in place rather than replaced by "Counting the Mask": his framing is
    supported by the fp32 data (Q->P worst in 16 of 18 cells again, the same four weaker-first cells), and replacing
    a first author's framing six days before the deadline is a collaboration risk the data do not require. The
    package (private, `context/author_package/`, gitignored) holds: `revised/main.tex` (his file; every number a
    macro from `paper/generated/macros.tex`), `diff/main_diff.pdf` (latexdiff against his draft, with the numbers
    inlined by `scripts/flatten_macros.py` so the diff shows values, not macro names, and with both reference
    lists inlined so bibliography fixes show), and `NOTE.md`, rendered from a template by
    `scripts/render_template.py` (every number from the macros, tables from `scripts/note_tables.py`). Three of
    his statements change because the data contradict them: "family more than scale" (E6), the stale-grid
    explanation of the Q->P penalty (E2/E3), and the 16-bit values (25.92, "twice"). One disagreement is reported
    rather than hidden: by accuracy over nine tasks, Q->P beats P->Q in 3 of the 4 weaker-first cells (on his three
    tasks, in fewer; E7 with his three-task metric gives the count). The build is `context/author_package/build.sh`; it fails on undefined macros, on a flattened
    text that differs from the macro version, and reports the page count.
42. **2026-11-13, after the stop.** The v3 runtimes stopped on 12 Nov at about 21:35 UTC because the container
    watcher (the client keepalive) ended with the session's pause; about 4 L4-hours were used. Finished states are on
    the Hub and the runner resumes from them, so nothing is re-run. Two scope choices for the restart: (1) no
    benchmarks for the larger models for now: E6 measures perplexity and KL, so Table 2 keeps its rows (three small
    models with fp32 intervals, two 7-8B rows); adding the three paper tasks at 25%+4 for SmolLM2-1.7B, Qwen2.5-1.5B
    and OLMo-2-1B would cost about 12 states (roughly 25 CU at the planning rate) and is offered to the author
    (note, decision 7); (2) the paper names only models whose runs are complete (today SmolLM2-1.7B, two 4-bit cells),
    so the abstract, setup and Q3 grow with the data instead of announcing it. The watcher must run with the longest
    background timeout and a heartbeat below it, so that a timeout cannot silently end the keepalive again.
43. **2026-11-13, the package audit.** An independent audit of package v1 (`context/author_package/note_src/audit.md`,
    private) found every macro value correct and three blockers in the wording: the nine-task accuracy result in the
    four weaker-first cells was only implicit (now stated next to the claim; the abstract and conclusion say "by
    perplexity" / "for perplexity and KL"); "its penalty grows with compression" was false for Qwen2.5-0.5B at 4 bits in
    fp32 (now "at 3 bits ... on all three models"); "what decides is whether the grid is fitted after the mask is
    known" was contradicted by the re-fit test (now "the order, mask before grid; why is open", at the layer level). The
    majors are fixed as well: E2 is every second block and in aggregate; the 50%+3 Q->P case turns super-additive with
    one Hessian; the family effect fades at 60%+4 and the SmolLM2 scale trend is not monotone; the equal-storage 3-bit
    win holds on the small SmolLM2 models only; Table 2 names its tasks; the IJCNN models re-quantized the pruned
    weights; the run count, which the paper does not break down, is dropped; the note no longer claims printed page
    numbers or runs that have not happened. To stay at 6 pages, the unit-check sentence and the layer-error sentence of
    Q2 are cut (offered back in the note). The Fig. 1 legend drops the internal label "(E2)" (`paper_figures.py`).
44. **2026-11-13 18:40 UTC, benchmarks at 1-2B and the restart (you asked for both; 114.97 CU available).** Two new
    L4 runtimes (esann-a7, esann-b6) run, in priority order: (1) the nine zero-shot tasks on the I_Sr states (dense, P
    alone, Q alone, P->Q at 25%+4 and 33%+3) of SmolLM2-1.7B, Qwen2.5-1.5B and OLMo-2-1B, so Table 2 gets rows at
    1-2B; states that had run without benchmarks are re-run as family E6B, and the analysis replaces an E6 line by its
    E6B twin only after checking that the WikiText-2 NLL sum is identical; SmolLM2-1.7B also gets its weaker-first cell
    (50%+4, pruning alone and the three orders) with benchmarks, to test the small models' accuracy reversal at 1.7B;
    (2) the rest of the E6 grids (Qwen2.5-1.5B, SmolLM2-1.7B, OLMo-2-1B); (3) the remaining E5 seeds (Qwen2.5-0.5B at
    4 and 3 bits, SmolLM2-360M at 3 bits, SmolLM2-135M). Qwen2.5-3B is an optional tail, started on the first free
    runtime if the budget allows. Estimate: about 53 L4-hours without the tail (about 90 CU at 1.71 CU/h, 118 at the
    planning rate of 2.23), 61 with it. Each result file has exactly one writer runtime, and its existing lines are
    pushed to that runtime first, because the watcher mirrors remote files over local ones. The runtimes came up on a
    new image (20261112), so each first recomputes a known state (`colab/image_gate.sh`) and waits for admission. The
    watcher runs with the two-hour background limit and a heartbeat below it, so a background timeout cannot silently
    end the keepalive again (#42).
