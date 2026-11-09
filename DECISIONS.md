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
