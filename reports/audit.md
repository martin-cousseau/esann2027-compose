# Independent audit: "Counting the Mask" (ESANN 2027 draft) and its experimental record

Audited state: commit `d90af48` (2026-11-11 08:54 UTC, clean tree). The repository changed while the audit ran
(E3 Qwen2.5-0.5B landed at 08:46, the paper was revised in `b6d2628`), so every check below was re-run on `d90af48`.
Method: own Python code reading `results/v2/**.jsonl`, `results/v2/items/*.json.gz` and `results/*.jsonl` directly
(nothing imported from `scripts/analysis`); primary sources fetched from arXiv, PMLR, ACL Anthology, IJCAI, iclr.cc,
Zenodo and the SparseGPT GitHub repository (OpenReview refused automated access on the audit day).

## Summary

**Verdict: the numbers are sound, but the text claims more than the data show. Do not submit as is.** All 95 text
macros shown in the PDF and all 68 Table 1 macros match my recomputation from the raw files, with 0 mismatches. A
rebuild from `d90af48` gives the same PDF text. No measured number is typed by hand, and no secret is tracked. The
problems are in the claims:

- The paper's central rule, "pruning pays only when it replaces a bit", is contradicted by its own 33%+3 states.
- The order claim holds for perplexity only.
- The mechanism claim holds on SmolLM2 only.
- Two cited works are misrepresented (Harma et al., Guo et al.).
- Authorship is unconfirmed.

Everything can be fixed in the text; no new GPU runs are needed.

| Severity | Count |
|---|---|
| blocker | 2 |
| major | 5 |
| minor | 14 |
| nit | 7 |

## Findings

**1. [blocker] Section 5 heading (`main.tex:349`) and Conclusion (`main.tex:475-476`): "pruning pays only when it replaces a bit (of precision)" is contradicted by the paper's own data.**
- The comparison: 33%+3 costs B = 3·(2/3) + 1 = 3.0 bits per weight, the same as plain 3-bit. `MACROS.md` lists BitsMask 3.00 for both. Unlike 50%+4, it keeps 3-bit precision, so it replaces no bit.
- Best 33%+3 order against plain 3-bit (WikiText-2 ppl / KL / flips):
  - SmolLM2-135M, P→Q: 53.24 vs 57.55 (−7.5%), KL 1.205 vs 1.273, flips 27.0 vs 27.4%.
  - SmolLM2-360M, joint: 20.03 vs 21.97 (−8.8%), KL 0.567 vs 0.650, flips 19.9 vs 21.5%.
  - Qwen2.5-0.5B, joint: 21.95 vs 21.84 (+0.5%).
- These points are drawn in Fig. 1: on both SmolLM2 panels a blue dot sits at x = 3.0 below the black 3-bit square. FINDINGS R1 records them; the paper omits them. Sec. 3 (`main.tex:261-262`) lists only 50%+4 at 3.0 bits.
- Fix: report 33%+3. Replace the rule with what the data show: at 3.0 bits, pruning pays on the two small SmolLM2 models whether or not it buys a bit; at 4.0 bits it never pays; at 1.7B it loses. Reword the heading and the conclusion. Drop the loaded "Counted honestly".

**2. [blocker] Author block (`main.tex:157-162`): the author list and affiliation are unconfirmed placeholders.**
- The paper shows "A. Yaacoub" and "[Affiliation to be confirmed]".
- NEWS.md ("Decisions waiting on you") says the author was taken from README.md. The code, models and runs live under other accounts (martin-cousseau, martincousseau, martin-css). Ref. [4] has four authors.
- ESANN reviewing is single-blind and the deadline is 18.11.2026.
- Fix: the authors confirm the author list and the affiliations before submission.

**3. [major] Abstract (`main.tex:181-182`), Conclusion (`:472-473`), Results (`:396-409`): "Quantizing first is the worst order in most settings, also where a weaker-first rule predicts it to be best" holds for perplexity and KL only, and the paper does not say so.**
- By WikiText-2 ppl the claim holds: Q→P is strictly worst in 21 of 23 cells (16 of 18 E1 cells).
- By 9-task accuracy (Sr-extended, v2 metric, my computation over 18 E1 cells):
  - Q→P is the best order in 5 cells and the worst in only 10.
  - Kim et al.'s rule is right in 15 of 18 cells, and in 3 of the 4 cells where it predicts Q→P.
  - `reports/analysis/e7_summary.json`, `sr9_v2|all`, agrees: correct 15, predicted_qp 4, predicted_qp_correct 3, and 10 correct among the 11 cells where the difference is significant.
- 5 of the 17 significant McNemar tests favour Q→P: SmolLM2-135M 50%+4, and Qwen2.5-0.5B 33%+4, 33%+3, 50%+4 and 60%+4. For example, Qwen 33%+4: 1,762 items right only under P→Q vs 2,176 only under Q→P, p = 4.5e−11.
- FINDINGS R4 reports the 15/18; the paper does not.
- Fix: add "by perplexity and KL" in the abstract and conclusion, and report the accuracy outcome of the Kim test in Results.

**4. [major] Abstract (`main.tex:182-184`), Conclusion (`:473-474`), Results (`:420-427`): "a mask chosen on rounded weights [explains it] only in part" is broader than the evidence, and the E3 evidence is partly within E3's own noise.**
- (a) Qualifiers lost. Results limits the claim to "on SmolLM2" (the mask helps in 0 of 2 Qwen2.5-0.5B cells) and states "per layer, neither reading explains the penalty". The abstract and conclusion drop both qualifiers.
- (b) Layer level (E2). The share of the penalty removed by the pruning-only mask is negative in 5 of 12 cells:

  | Model | Shares per cell |
  |---|---|
  | SmolLM2-135M | +6, +14, +22, +39% |
  | SmolLM2-360M | −19, −4, +6, +24% |
  | Qwen2.5-0.5B | −54, −6, +10, −19% |

- (c) E3 noise floor. E3 contains a control the paper never reports: P→Q with a stale grid. In full runs it changes P→Q perplexity by −1.3% to +14.1%. Examples: SmolLM2-135M 25%+3 +11.0%, 33%+3 +14.1%; Qwen 50%+3 +7.1%. The same change moves layer error by at most 0.4% in E2.
  - The quoted "+9%" (SmolLM2-135M 50%+4) is a 1.2% perplexity change (47.33 → 46.78), well inside that floor.
  - Qwen 50%+4's −194% is a share of a 0.21-ppl (0.9%) penalty, so it carries no information.
  - Read as signal instead of noise, the stale arm contradicts "a stale grid little" on SmolLM2-135M at 3 bits.
- Fix: qualify the abstract and conclusion ("in full runs of SmolLM2, not per layer, not on Qwen2.5-0.5B"). Report the stale arm as E3's noise floor. Interpret only the cells where the arm moves perplexity well beyond it (135M 50%+3 −24%, 360M 33%+3 −22%, 360M 50%+3 −24%).

**5. [major] Introduction (`main.tex:204-208`, `:214`) and Results (`:406-409`): Harma et al. are misrepresented by omission.**
- Their App. N ("Combining GPTQ with Sparsity", arXiv 2405.20935v2) already reports the paper's order result in a Hessian-compensated pipeline at 4 bits.
  - Table 11 (SparseGPT codebase + GPTQ-4b, OPT-350M/1.3B): S→Q beats Q→S in 17 of 18 entries. For example, OPT-350M 2:4 WikiText-2: 56.27 vs 67.96 (+21%).
  - Table 10 (magnitude 50% + GPTQ-4b): 30.93 vs 35.59.
  - They leave the mathematical study of the order for second-order methods to future work, with results pointing to S→Q.
- The intro mentions only the App. F "much smaller order effect" at 6- to 8-bit. That makes "yet the reverse order costs up to 77.3%" read as a contradiction of Harma et al., when it extends their App. N.
- "None of these uses a joint sweep": the SparseGPT reference code quantizes inside the pruning sweep (`sparsegpt.py` lines 58-60 and 111-115, https://github.com/IST-DASLab/sparsegpt/blob/master/sparsegpt.py). Harma's Table 11 S→Q runs were therefore probably joint sweeps (verify).
- Fix: cite App. N. Rephrase as "consistent with Harma et al.'s App. N (S→Q better for SparseGPT+GPTQ-4b on OPT-350M/1.3B); under our sweep the gap reaches 77%". Revise the novelty sentence.

**6. [major] `main.tex:215-216`: OBR [9] is cited for "the one-shot, Hessian-compensated, weight-only pipeline used in practice", but OBR is not weight-only.**
- OBR's abstract: "aggressive W4A4KV4 quantization with 50% sparsity" (https://arxiv.org/abs/2509.11177, https://iclr.cc/virtual/2026/poster/10009140). It quantizes activations and the KV cache, the very criterion the previous sentence uses to set prior work apart.
- Fix: cite [9] only for "one-shot, Hessian-compensated joint quantization and sparsification", and drop it from "weight-only".

**7. [major] `main.tex:308-309`: the single artifact URL points to the fork's default branch, which does not hold v2.**
- README.md says the v2 study is on branch `claude/busy-hopper-ghtynk` ("Everything below is on this branch"). DECISIONS #3: "Nothing is pushed to main". PR #1 is a draft (NEWS).
- A reader who opens github.com/martin-cousseau/esann2027-compose lands on v1. I did not check this online, to avoid sending account names to outside services.
- Fix: merge the PR before submission, or link a tag or commit permalink.

**8. [minor] Abstract `main.tex:180-181`: two wordings mislead about SmolLM2-1.7B.**
- "not at all at 1.7B" understates the result: plain 3-bit is 8.5% better there (13.01 vs 14.11).
- "collapses on the three smaller models" implies 1.7B was run at 2 bits; it was not (E6 has no 2-bit state).
- Fix: "…and loses at 1.7B; plain 2-bit collapses on the three models we ran it on".

**9. [minor] `main.tex:210-213`: the revised Bouriel et al. sentence over-generalizes.**
- Their bound is stated for Table 2 at s = 0.2 only: "Comparing Q4P and PQ4 ordering has relatively small influence compared to task and architecture: I differs by at most 1-2 points within each metric" (IJCAI-26 p. 5669, https://www.ijcai.org/proceedings/2026/631). At s = 0.3 they say "magnitudes generally grow".
- The earlier draft said "at 20% sparsity"; the current text dropped it.
- Fix: restore "at 20% sparsity".

**10. [minor] `main.tex:236-238`: "Joint [3,1]: … the grid is fitted on the kept weights" attributes this implementation's choice to SparseGPT.**
- SparseGPT's joint mode fits one per-row grid on the dense W before the sweep, with no groups (`sparsegpt.py:58-60`). That is a stale grid in this paper's own terms.
- The per-group grid fitted on kept weights is this repository's own code (`src/core.py:91-92`).
- Fix: "as in [1, 3], but with the grid fitted per group on the kept weights".

**11. [minor] Results: the cell sets behind the "N of M" counts are never defined.**

| Count | Cells | Source |
|---|---|---|
| 18 | grid cells | E1 |
| 14 | 4-bit cells | E1 + E4 60%+4 + E6 |
| 20 | Kim and Harma tests | E1 + E6, not E4 |
| 21 | McNemar tests | E1 + E4; E6 has no benchmarks |
| 10 | E3 cells | 4 per SmolLM2 model + 2 on Qwen |

- Adding E4's three 60%+4 cells to the Kim test gives three more failed Q→P predictions (CER_P ≈ 5.33 > 4), so 0 of 8.
- Fix: define each set in a clause.

**12. [minor] Several quantities are undefined or ambiguous.**
- `:432` "accuracy" is Sr over 3 tasks (ARC-E, PIQA, HellaSwag; `\resQTwoAgreeSrThree`), although 9 tasks are logged. With 9-task Sr the agreement is 7 of 18; with pooled 9-task accuracy, 8.
- `:440` "−1.4 accuracy points" is the unweighted mean over 9 tasks. The pooled item difference is −0.77 points; SciQ (−9.3 points) drives the result, and 4 of 9 tasks favour P→Q.
- `:392` "per-layer ratio" means I_E/(E_P+E_Q) of P→Q.
- "Penalty" means Q→P minus P→Q, not minus the best order (on SmolLM2-360M the best order is joint).
- `:456` "single methods" is 4-bit quantization alone; no prune-only seeds were run (DECISIONS #29).
- Fix: define each where it is used.

**13. [minor] Sec. 4 (`main.tex:289-309`) omits or contradicts details of the runs.**
- "E6 runs SmolLM2-1.7B" no longer lists its 11 states: 4 and 3 bits, 25% and 50%+4 in three orders, prune-only 25% and 50%, no benchmarks.
- The E3 cells are not given: 25/33/50%+3 and 50%+4 on SmolLM2, only 50%+3 and 50%+4 on Qwen.
- "All runs use fp32 on NVIDIA L4" is contradicted by the fp16 L4 and T4 runs used in Noise.
- The C4 evaluation windows come from the same validation shard as calibration (disjoint documents, DECISIONS #7); the paper does not say so.
- Fix: add one sentence on each.

**14. [minor] Noise paragraph (`main.tex:447-450`): the noise yardstick does not fit the study, and some order tallies rest on gaps below it.**
- "Weak evidence" rests on fp16 cross-GPU deltas, but the study is fp32 on one GPU type, where results are bit-exact. The relevant noise is the seed SD: at most 1.0%, measured on 360M only.
- Against that yardstick, 4 of the 20 E7 cells have a gap between Q→P and P→Q below 2%, all on unreplicated models: Qwen 25%+3 −0.4%, Qwen 33%+3 +0.9%, Qwen 50%+4 +0.9%, 1.7B 25%+4 +1.5%.
- One of the five "wrong" Kim predictions is a near-tie: Qwen 33%, CER 4.04 vs 4.00.
- Fix: state the seed SD as the yardstick and mark the cells below it in the tallies.

**15. [minor] Scale paragraph (`main.tex:462-467`): "the order matters less" is not monotone in size.**
- At 50%+4 the spread rises from 14.9% (135M) to 15.4% (360M) before falling at 1.7B.
- There are three single-seed points.
- Fix: say "falls at 1.7B".

**16. [minor] Fig. 1 (`paper/figures/frontier.tex`) is hard to read and its header comment is false.**
- The 1.7B panel has one y-tick label (10).
- There is no legend; the encoding is colour only.
- The caption does not explain the arrows or which cells are drawn: 25/33/50%+3 and 25/33/50/60%+4, and only 25% and 50%+4 at 1.7B.
- Header comment lines 4-5 say "No number is typed here: … axis ranges … come from those files", but `xmin`, `xmax` and `ytick` (lines 17-18) and the data filter `restrict x to domain=2.5:4.5` (line 41) are typed by hand.
- Fix: add a legend and y ticks, extend the caption, correct the comment.

**17. [minor] lm-eval batch configuration: four states ran with a single batch size, and the promised check is not recorded.**
- The four states: both dense references (SmolLM2-135M at 16, Qwen at 8), and SmolLM2-135M 4-bit and 50%+4 P→Q (16). All others ran at 48/16, or 24/8 for Qwen.
- DECISIONS #19 promises a re-evaluation of the dense models as a check; nothing records it.
- Every flip count depends on the dense references, including the printed 22.0% for 50%+4 P→Q.
- Fix: run and record the check, or state the caveat.

**18. [minor] The "best order per cell" is chosen on the test set.**
- Table 1's bold, Fig. 1 and every equal-storage gain use the minimum of 3 orders on WikiText-2 test.
- This favours pruning + quantization over plain quantization by about the seed SD (up to 1%). That matters for the 0.3% Qwen "tie" and the 0.5% gain on 360M.
- Fix: note the selection, or choose the order on C4 or KL.

**19. [minor] FINDINGS.md and NEWS.md disagree with the data or the paper.**
- R3's header says "explains part on SmolLM2 only, solid at layer level". E2 shows negative shares in 5 of 12 cells, and the paper says per layer neither reading explains the penalty.
- "Other findings" item 2 ("waits for the E1 3-bit states") and item 5 ("partial") are stale.
- NEWS says "no published rule predicts the order", but Harma's rule is right in 19 of 20 cells.
- R4 frames Harma et al. through App. F only (see 5).
- Fix: update both files.

**20. [minor] Integrity: tracked files keep traces of the private inputs.**
- `logs/env_check.txt:14-15` (tracked, public fork) holds the private Drive folder ID and the Google Doc IDs of "Review" and "Code". `:16` gives the review's title and structure.
- `QUESTIONS.md:4` repeats the folder ID.
- `review_code/` (the reviewer's scripts, from the private `context/review_code.zip`) is public.
- None of these are secrets, but DECISIONS #2 says private inputs stay out of git.
- Fix: remove the IDs. Confirm the reviewer's consent and a licence for `review_code/`.

**21. [minor] `src/core.py:49-57`: Cholesky failures fall back to larger damping without a record.**
- The code silently retries with damping × 10^k (k ≤ 4). The paper states 0.01·mean diag(H), and no result line records whether a retry happened.
- Fix: log the attempt per layer, or mention the fallback.

**22. [nit] Theory wording, `main.tex:228-229` and `:279-281`.**
- With H = 2XXᵀ, the objective is ½·tr(ΔHΔᵀ), not tr(ΔHΔᵀ).
- "P, R_M and R_M̄ have disjoint supports" is not right as written: P and R_M share the support M. Only R_M̄ is disjoint from both.
- "The guarantee … needs no compensation, a fixed grid and a diagonal H" states sufficient conditions as if they were necessary; write "holds without compensation, with a fixed grid and a diagonal H".
- `[H⁻¹]²_jj` needs a note that H⁻¹ is the Cholesky factor, as in SparseGPT's Alg. 1.

**23. [nit] `main.tex:197`: "SparseGPT stores such a bitmask" overstates.** SparseGPT's sentence is hypothetical ("if we … store only the non-zero weights and use a bitmask"). Use "counts".

**24. [nit] `refs.bib`: a stray `}` at line 253, and line 1 still carries the old title "Do Pruning and Quantization Compose?".**

**25. [nit] Notation collisions.** E is both the layer error and the family prefix E1-E6. B(s,b) collides with the matrix B in ⟨A,B⟩_H. P is both a matrix and part of the label P→Q.

**26. [nit] [15] follows the harness README.** Zenodo record 22214970 is titled "EleutherAI/lm-evaluation-harness: v0.4.13", with Sutawika as first creator. Either form is acceptable.

**27. [nit] The nine benchmarks are named but not cited.** Verified entries exist in `refs.bib`.

**28. [nit] Table 1's bold marks differences below seed noise** (360M 13.20 vs 13.27; Qwen 21.84 vs 21.91). Say so in the caption.

### Checks that passed

- **Macros:** all 95 text macros shown in the PDF and all 68 Table 1 macros recomputed from the raw files, with 0 mismatches.
- **Build:** `paper/` rebuilt from `d90af48` gives identical PDF text. 6 pages, A4, all 95 macros defined, none `--`. No BibTeX warnings; the only LaTeX warning is the template's class-name mismatch (`esannV2` vs `esann`).
- **Setup vs metadata:** all 161 v2 states (E1 and E3-E6) ran on NVIDIA L4 in float32, with TF32 off for matmul and cuDNN and lm_eval 0.4.13. All share n_calib 128 × 2048 tokens, group size 128 and one code hash (`cef8af3e…`), with one calibration hash per (model, seed). All have status ok and checks.all_ok. The 90 E1/E4 states have complete task splits (26,618 items each). The C4 shard is validation 00000 and the damping 0.01·mean diag (code).
- **Determinism:** DET reproduces (two L4 GPU UUIDs, identical NLL and all 210 rel_err). Both IMG runs give the same NLL as E1. REPRO deltas reproduce: +1.498%, +4.645%, and at most 0.373%.
- **Free-zero identity:** exact on all three core models (identical NLL sums and all 210/224/168 rel_err).
- **Secrets:** none in tracked files or in the 117-commit history (scanned for HF, GitHub, JWT, AWS, sk-, W&B-key, private-key and Colab-token patterns). The 40-hex strings in results are `repo_commit` SHAs. `context/` is ignored and was never committed.

## Check 1: re-derived numbers

Random sample: `random.Random(20261111).sample(sorted(M), 10)`, where M is the 151 macros displayed in the PDF at the
start of the audit (`2a3b170`): text macros plus Table 1 cells. All 10 are still displayed at `d90af48`. Rows 11-28
cover the required categories. Every other displayed macro was also recomputed, with no mismatch.

| # | Macro | macros.tex | Mine (own code) | Match | Definitional note |
|---|---|---|---|---|---|
| 1 | `\resPplSmolSmallSFiftyBFourJoint` | 44.47 | 44.4713 | yes | Table 1 |
| 2 | `\resKlSmolMediumSFiftyBFourPQ` | 0.631 | 0.63091 | yes | KL(dense‖model), nats/token |
| 3 | `\resPplQwenSFiftyBFourQP` | 22.39 | 22.3859 | yes | |
| 4 | `\resFrontierThreeGainSmolLargePct` | −8.5 | −8.455 | yes | best of 3 orders chosen on the test set (finding 18) |
| 5 | `\resPplSmolSmallSTwentyFiveBFourPQ` | 21.99 | 21.9923 | yes | |
| 6 | `\resKlSmolSmallSTwentyFiveBFourQP` | 0.449 | 0.44903 | yes | |
| 7 | `\resSeedSdMaxCombinedPct` | 1.0 | 0.995 (P→Q 33%+4) | yes | sample SD over 3 seeds / mean |
| 8 | `\resETwoSubJointMaxPct` | 100 | 100 | yes | share of layers with E_PQ < E_P + E_Q |
| 9 | `\resHwDeltaSmolSmallSFiftyBFourPQPct` | +4.6 | +4.645 (42.017 vs 40.152) | yes | fp16 L4 vs fp16 T4; not the study's precision (finding 14) |
| 10 | `\resEThreeRemovedPmaskSmolSmallSFiftyBFourPct` | +9 | +8.98 | yes | relative to P→Q; the arm changes ppl by only −1.2% (finding 4) |
| 11 | `\resFrontierFourGainMaxPct` | 0.5 | 0.504 (360M) | yes | floored at 0 |
| 12 | `\resFrontierThreeGainMinPct` / `MaxPct` | 6.8 / 28.4 | 6.794 / 28.403 | yes | 2 winners: 135M, 360M |
| 13 | `\resQTwoBestJointPpl` (of `\resQTwoCellsPpl`) | 11 (18) | 11 (18) | yes | E1 cells, WikiText-2 ppl |
| 14 | `\resQTwoQPWorstPplBFour` / `\resQTwoCellsPplBFour` | 14 / 14 | 14 / 14 | yes | set = 9 E1 + 3 E4 + 2 E6, not stated (finding 11) |
| 15 | `\resESevenKimCorrectPplAll` / `CellsPplAll` | 14 / 20 | 14 / 20 | yes | E1+E6 only; by ppl only (finding 3) |
| 16 | `\resESevenKimPredQPCorrectPplAll` / `PredQPPplAll` | 0 / 5 | 0 / 5 | yes | Qwen 33%+4 is a CER near-tie (4.04 vs 4.00) |
| 17 | `\resESevenHarmaPQBetterPplAll` / `OrderCellsPplAll` | 19 / 20 | 19 / 20 | yes | exception: Qwen 25%+3 (−0.37%) |
| 18 | `\resESevenHarmaMaxGapPplAllPct` | 77.3 | 77.27 | yes | 135M 50%+3 |
| 19 | `\resSeedPairJointPQSig` | 3 | 3 (p = 0.033, 0.037, 0.0075) | yes | paired t on log ppl, 2 df |
| 20 | `\resScaleSpreadSmolMediumSFiftyBFourPct` | 15.4 | 15.37 | yes | the other 5 spreads also match |
| 21 | `\resMcNemarSigPQQP` / `CellsPQQP` | 17 / 21 | 17 / 21 | yes | 5 of the 17 favour Q→P (finding 3) |
| 22 | `\resAccDiff…SFiftyBFourPQQP` (Lo, Hi) | −1.4 (−2.1, −0.7) | −1.40 (−2.1, −0.7), own bootstrap B = 10,000 | yes | unweighted 9-task mean; pooled −0.77 (finding 12) |
| 23 | `\resQTwoAgreeSrThree` | 10 | 10 | yes | 3-task Sr labelled "accuracy" (finding 12) |
| 24 | `\resIdentityQPThreeCount` / `Models` | 3 / 3 | 3 / 3 | yes | identical NLL and every rel_err |
| 25 | `\resETwoRhoMin` / `Max` / `PMax` | +0.41 / +0.60 / 5.43e−6 | +0.41 / +0.60 / 5.43e−6 | yes | Spearman, P→Q layer ratio vs h_corr_index |
| 26 | `\resEThreeCells` / `RefitHelps` / `PmaskHelps` / `PmaskHelpsQwen` | 10 / 2 / 7 / 0 (of 2) | 10 / 2 / 7 / 0 | yes | E3 control ignored (finding 4) |
| 27 | `\resQuantTwoPplMin` / `Max` | 7,399 / 94,280 | 7,398.8 / 94,279.7 | yes | |
| 28 | `\resFreeZerosPctSmolSmallQuantBThree` | 30.7 | 30.72 (size-weighted per-layer diags) | yes | |

## Check 2: numbers typed by hand

- **`main.tex`: no measured number is typed by hand.** The typed digits fall into these groups:
  - Configuration labels: 25/33.3/50/60%, 2/3/4/8-bit, 4.0/3.0/2.5/3.25/2.25 bits (all b(1−s)[+1]), model sizes.
  - Protocol constants: 128 windows, 2048 tokens, 0.01 damping, 128 columns, seeds 1-2, nine tasks, p < 0.05, 95%.
  - Analytic values: 0.19 = 1 − H2(0.25) = 0.189, H2(0.5) = 1, 1/128, (16+b)/128.
  - Numbers from cited papers: 2.7B (SparseGPT, verified), 1B-3.8B (Zhou, verified), 6- to 8-bit (Harma App. F, verified), about 1B and 1-2 points (Bouriel, verified at s = 0.2 only, finding 9).
  - Layout values: 2.8pt, 8pt, cmidrule column indices.
- **`frontier.tex`:** only presentation constants are typed, but the comment denying it is false, and the domain filter 2.5:4.5 is a typed data selection (finding 16).

## Check 3: claims in the abstract and conclusion

| # | Claim | Where supported | Verdict |
|---|---|---|---|
| A1 | Rates that ignore the mask make the combination look better | Fig. 1 (gray to blue); `:365-367` | Supported: TCr puts 25%+4 at 3.0 bits (21.99 vs 3-bit 57.55) |
| A2 | Joint, P→Q, Q→P on SmolLM2 and Qwen2.5 from 135M to 1.7B, one Hessian-compensated sweep | Secs 2 and 4 | Supported. Qwen2.5 is one model (0.5B); 1.7B has no benchmarks |
| A3 | With the mask, 25%+4 costs as much as plain 4-bit | Sec. 3 | Supported (analytic, 4.0 = 4.0) |
| A4 | …and beats it by at most 0.5% | Table 1; `:351-355` | Supported: +0.50 (360M, all 3 seeds), −0.26, −2.22, −3.36% |
| A5 | 50%+4 beats plain 3-bit at 3.0 bits on only 2 of 4 models | Table 1; `:356-365` | Supported: +28.4, +6.8, −0.3, −8.5% |
| A6 | …by less as SmolLM2 grows | Scale `:459-462` | Supported (3 points, 1 seed) |
| A7 | …and not at all at 1.7B | `:364-365` | Understated: loses by 8.5% (finding 8) |
| A8 | At 2 bits plain quantization collapses on the three smaller models | `:371-374` | Supported for the 3 models run (7,399-94,280 vs dense 13-15). Wording implies 1.7B was tested (finding 8) |
| A9 | Quantizing first is the worst order in most settings | `:396-401` | By ppl yes (21/23 cells). By 9-task accuracy no (worst 10/18, best 5/18). Overstated (finding 3) |
| A10 | …also where a weaker-first rule predicts it to be best | `:401-406` | By ppl yes (worst of the three orders in 5/5). By accuracy the rule is right in 3/4 (finding 3) |
| A11 | A stale quantization grid does not explain why | `:415-427` | Supported by E2 (refit worse in 12/12, stale ≤ 0.4%) and E3 refit (helps 2/10). E3's own stale arm (up to +14%) is unreported (finding 4) |
| A12 | A mask chosen on rounded weights explains it only in part | `:416-427` | Overstated: Results says "on SmolLM2" and "per layer, neither"; Qwen 0/2; E3 effects partly within noise (finding 4) |
| C1 | Quantizing first is worst in most settings | as A9 | Overstated (finding 3) |
| C2 | Stale grid does not explain why | as A11 | Supported, with the E3 caveat |
| C3 | Mask on rounded weights only in part | as A12 | Overstated (finding 4) |
| C4 | Mask-first orders are sub-additive in aggregate, not in every layer | E2 `:385-390` | Supported: 36/36 in aggregate; per layer P→Q 60-92%, joint 97-100%. Valid only with one dense Hessian on every 2nd block; say so |
| C5 | Counted honestly, pruning pays only when it replaces a bit of precision | `:349` heading | Contradicted by 33%+3 at 3.0 bits: −7.5% and −8.8% ppl on the two SmolLM2 models (finding 1) |
| C6 | 50%+4 beats 3-bit at 3.0 bits on 2 of 4 models, less as SmolLM2 grows | Table 1, Scale | Supported |
| C7 | 25%+4 does not beat 4-bit at 4.0 bits by more than 0.5% | Table 1 | Supported |
| C8 | Rounding zeros count for plain quantization too | Free zeros `:376-383`, `:368-371` | Supported: 30.7% zeros; 2.42 vs 2.62 bits under symbol entropy; Q→P 25%+3 identical to 3-bit |
| C9 | Recommend reporting mask-counted bits, KL, flips, paired tests | Evaluation `:430-441` | Reasonable: KL agrees with ppl in 17/18 cells, flips in 9/18 |

## Check 4: citations

| Key [#] | Bib fields vs primary source | Claims the paper makes about it | Verdict |
|---|---|---|---|
| frantar2023sparsegpt [1] | Correct: ICML 2023, PMLR 202:10323-10337 ([PMLR](https://proceedings.mlr.press/v202/frantar23a.html)) | Bitmask, and 50%+4 more accurate than 3-bit at the same memory from 2.7B up: correct ("store only the non-zero weights and use a bitmask… same overall memory consumption as 3-bit"; "50% + 4-bit models are more accurate than their respective 3-bit versions for 2.7B+", [arXiv 2301.00774](https://arxiv.org/abs/2301.00774) §4). Saliency w²/[H⁻¹]²_cc per block: correct (Alg. 1, with the Cholesky factor) | Joint grid detail misattributed (finding 10); "stores" is a nit (finding 23) |
| frantar2023gptq [2] | Correct: authors, ICLR 2023 (arXiv comment "ICLR 2023"); published title OPTQ (OpenReview not re-fetched) | One-shot PTQ | Correct |
| frantar2022obc [3] | Correct: 3 authors, NeurIPS 2022 ([arXiv v2](https://arxiv.org/abs/2208.11580): "Published at NeurIPS 2022") | One OBS framework for pruning and quantization | Correct; see finding 10 |
| laborde2025semantic [4] | Correct: authors, IJCNN 2025 ([arXiv](https://arxiv.org/abs/2505.07289) comment); DOI resolves to IEEE Xplore 11227279; pages not re-checked | TCr = 1 − q/16·(100−s)/100: correct (Eq. 1). Ignores masks: correct ("independent of non-informative storage requirements like pruning masks"). 25%+4 beats 3-bit at equal TCr: correct (about 20% higher Sr at 81.25% TCr) | Correct |
| zhou2025revisiting [5] | Correct: Findings EMNLP 2025, 12055-12070 ([ACL Anthology](https://aclanthology.org/2025.findings-emnlp.645/)) | "six SLMs from 0.5 to 3.8B"; "quantization consistently outperforms pruning" | Correct |
| harma2025interplay [6] | Correct: 11 authors ([arXiv](https://arxiv.org/abs/2405.20935)); ICLR 2025 (proceedings.iclr.cc listing) | Non-orthogonality proof and sparsity first for magnitude pruning with max-scaled quantization: correct (Thm 3.5/3.6). App. F SparseGPT/Wanda order effect "much lower" at 6-8 bits: correct (Table 7: INT8, MXFP8/6, HBFP8/6). Mask on rounded weights: correct (§3.2, "leading the sparsity step to prune different elements") | Misleading by omission of App. N (finding 5) |
| kim2026order [7] | Correct: title and authors ([iclr.cc poster](https://iclr.cc/virtual/2026/poster/10010118); [arXiv](https://arxiv.org/abs/2603.18426) "ICLR 2026") | Weaker first: correct ("weaker perturbations are applied first"). Activations and KV cache quantized: correct ("All quantization methods apply the same bit-width to weights, activations, and KV-cache", App. C). CER: correct (Definition 3, linear interpolation) | Correct; their rule wins on accuracy here (finding 3) |
| bouriel2026interaction [8] | Correct: IJCAI-26, 5666-5674, DOI 10.24963/ijcai.2026/631 ([IJCAI](https://www.ijcai.org/proceedings/2026/631)) | Magnitude pruning, bitsandbytes, models of about 1B: correct. "at most 1-2 points": correct for s = 0.2 only | Over-generalized (finding 9) |
| guo2026obr [9] | Correct: ICLR 2026, author order as on [iclr.cc](https://iclr.cc/virtual/2026/poster/10009140) (arXiv order differs) | Cited for a "weight-only pipeline used in practice" | Wrong: OBR is W4A4KV4 (finding 6) |
| benallal2025smollm2 [10] | COLM 2025 "Fully Open" title not re-verified (OpenReview refused automated access); a 2026 third-party citation uses it; the arXiv title differs | Model only | Plausible |
| qwen2024qwen25 [11] | Correct ([arXiv 2412.15115](https://arxiv.org/abs/2412.15115)) | Model only | Correct |
| dutta2024accuracy [12] | Correct: authors; NeurIPS 2024 (arXiv journal-ref to proceedings) | KL and flips: correct ("We propose two such metrics, KL-Divergence and flips"); flips definition correct | Correct |
| raffel2020t5 [13] | Correct: JMLR 21(140):1-67 ([JMLR](https://jmlr.org/papers/v21/20-074.html)) | C4 | Correct |
| merity2017pointer [14] | Correct: title and authors ([arXiv 1609.07843](https://arxiv.org/abs/1609.07843)); ICLR 2017 venue not re-fetched (OpenReview) | WikiText-2 | Correct |
| gao2026lmeval [15] | Correct: v0.4.13, DOI 10.5281/zenodo.22214970, 2026-08-31 ([Zenodo API](https://zenodo.org/api/records/22214970)); matches `metadata.lm_eval` 0.4.13 in all 161 states | lm-eval | Correct; see nit 26 |

## Check 5: consistency

- **FINDINGS vs paper:**
  - R1: the paper omits 33%+3 (finding 1).
  - R2, R5 and R7 match the paper number for number.
  - R3: header contradicts E2 and the paper (finding 19).
  - R4: the paper omits the accuracy result (finding 3) and frames Harma et al. through App. F only (finding 5).
  - R6 matches, except "single methods" vs "4-bit alone" (finding 12).
- **Captions:** Table 1 matches its content (bold equals the row minimum in all 16 groups). Fig. 1 matches its data files; see finding 16.
- **Setup vs metadata:** matches, except finding 13.
- **Counts vs CSVs:** all match.
  - `q2_order_ranking.csv`: 23 cells.
  - `e1_mcnemar.csv`: 21 pooled9 P→Q vs Q→P rows, 17 with p < 0.05.
  - `e7_summary.json`: wt_ppl 14/20 and 0/5; sr9_v2 15/18.
  - `frontier_equal_storage.csv`: gains as in check 1. It has no 33%+3 comparison.

## Check 6: integrity

- No secrets: see "Checks that passed".
- `context/` is untracked and was never committed.
- Exposure of private-input identifiers and the review code: finding 20.
