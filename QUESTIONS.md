# Questions only the author can decide

1. **Authors and affiliations (blocks submission).** The paper is authored "A. Yaacoub" (taken from README.md) with
   "[Affiliation to be confirmed]". ESANN reviewing is single-blind; confirm the author list and the affiliations.
2. **Where the paper's code link points.** The paper links the v2 branch (`.../tree/claude/busy-hopper-ghtynk`),
   because the default branch of the fork still holds v1. If PR #1 is merged before submission, the link can be
   shortened to the repository root.
3. **The reviewer's code in a public repository.** `review_code/` (unpacked from the review bundle, as the brief
   asked, and imported by the v2 pipeline) is public on the fork. Confirm that the reviewer agrees and add a licence,
   or say so and it can be replaced by an attribution and a pointer.
4. **Google Drive folder.** Its folder and document ids were written to `logs/env_check.txt` and this file during
   phase 0; they are now removed from the current files but remain in the public git history (history is not
   rewritten). Check that the folder is not shared by link.
5. **Explainer document.** It was not in the Drive folder. If it exists, its "corrections" section may change what is
   tested.
6. **LaTeX source.** The draft's LaTeX was not available; the paper is re-created from the PDF in the ESANN template
   (`paper/v1/`). If you have the original source, the rewrite can be ported onto it.
7. **Open check.** The lm-eval re-run of the two dense references at the later batch configuration (DECISIONS.md
   #19) was not done before the GPUs were released; it needs about 1 L4-hour (FINDINGS.md, caveats).
