# Questions only the authors can decide (updated 17 November 2026)

Open (none blocks the submission of V4.1; ESANN 2027 extended deadline: 25 November 2026):

1. **Submission form details.** V4.1 keeps the author list and affiliations of the first author's draft of 12
   November. The ESANN form asks for the full details of every author (institution, address, phone, e-mail); they
   are not in the package.
2. **The 7-8B rows of Table 2** (from the first author's draft). Both simulated reviews of V4.1 ask to drop them: they
   come from another toolchain and other tasks and have no per-item results (so no intervals), and LLaMA's value is
   driven by MATH. (Their worry that those models may not be sparse is most likely unfounded: DECISIONS.md #68.) Kept
   in V4.1; the caption names their source and why they have no intervals. A ready alternative without them is in the package (`paper/alternative_without_7_8B/`).
3. **Where the paper's link points.** V4.1 links the public results dataset on the Hub, which links the code (this
   branch) and the models. Alternatively merge the branch into the first author's repository and link that.
4. **Benchmark citations.** ARC-Easy, PIQA, HellaSwag and the six extra tasks are named, not cited (about four lines).
5. **The reviewer's code in a public repository.** `review_code/` (unpacked from the review bundle and imported by the
   v2 pipeline) is public on the fork. Confirm that the reviewer agrees and add a licence, or replace it by an
   attribution and a pointer.
6. **Google Drive folder.** Its folder and document ids were written to `logs/env_check.txt` during phase 0; they
   were removed from the current files but remain in the public git history. Check that the folder is not shared by
   link.
7. **Compute units.** The Colab balance is not visible from this session; the estimate is in `logs/budget.md` (about
   55 of the 80 left). Please read the real balance in Colab.

Resolved:

- *Authors and affiliations of the paper*: set by the first author's draft of 12 November (item 1 above remains).
- *Explainer document*: absent from the Drive folder; you chose to proceed without it (DECISIONS.md #1).
- *LaTeX source*: the first author's LaTeX of 12 November is the base of every revision since v2 of the package.
- *lm-eval batch check* (DECISIONS.md #19): done in v3 (BATCH family; identical perplexity and no changed item;
  FINDINGS.md, R6).
