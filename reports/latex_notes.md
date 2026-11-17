# LaTeX notes: phase L1 (template + faithful reconstruction of the draft)

Date: 2026-11-09. Author: `latex` subagent. Scope: `paper/esannV2.cls`, `paper/template/`, `paper/v1/`, this file.

## 1. Toolchain (installed with apt, Ubuntu 24.04)

`apt-get install --no-install-recommends latexmk texlive-latex-base texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended texlive-science texlive-pictures texlive-bibtex-extra lmodern cm-super`

| Component | Version |
|---|---|
| TeX Live | 2023/Debian (texlive-* 2023.20240207-1, binaries 2023.20230311.66589-9build3) |
| pdfTeX | 3.141592653-2.6-1.40.25 (the draft's producer is exactly this build: "pdfTeX 1.40.25, TeX Live 2023/Debian") |
| LaTeX kernel / L3 | LaTeX2e 2023-11-01 patch level 1 / L3 2024-01-22 |
| latexmk / BibTeX | 4.83 / 0.99d |
| Fonts | cm-super 0.3.4-17, lmodern 2.005-1 |

## 2. Official ESANN template

* Guidelines page: https://www.esann.org/node/5 ("Guidelines for authors | ESANN 2027"), read 2026-11-09.
* Files downloaded on 2026-11-09 at 23:26 UTC (server `Last-Modified`: 2022-03-09):

| URL | sha256 |
|---|---|
| https://www.esann.org/system/files/downloads/esannV2.cls | `36097ad5826ffbebfeea23f1675e63500e941fd273ef8fae765e8fca51639048` |
| https://www.esann.org/system/files/downloads/esannV2.tex | `2ee9ee3f42da88fce7a7b35705dee14467c025053424dfcae63ee7dedb51ff79` |
| https://www.esann.org/system/files/downloads/esannV2.pdf | `6f3329578ea6b1ff2014c364c6bbb5e5bd2a9de2dd1230dae2c5b88deb1da2f2` |
| https://www.esann.org/system/files/downloads/ESANN2005BW.eps | `35b9361eb50e52b2514eb84204319996d9558b7b3d13a0ce7415edad476adbb1` |

* `paper/esannV2.cls`: the official class, byte-identical (CRLF line endings kept). Do not modify it.
* `paper/template/`: the four files above, untouched (no build products in that folder).
* The Word templates (`esannV2.doc`, `.dot`) were not downloaded (LaTeX is "strongly encouraged").

Rules stated on the guidelines page: at most 6 pages including figures and references; no appendices or supplementary
material; single-blind review; the style files must be used; printing area 12.2 x 19.3 cm, centred on A4, with margins
left/right/top/bottom 4.4/4.4/5.2/5.2 cm ("mandatory"); no page numbers; PDF in A4 (not Letter); "PDF settings must be
set to 600 dpi".

## 3. The reconstruction: `paper/v1/`

| File | Content |
|---|---|
| `main.tex` | the draft in LaTeX (wording unchanged), ASCII only |
| `refs_v1.bib` | the 18 references transcribed exactly as printed (header says so; not a verified list) |
| `esannV2.cls` | byte-identical copy of `paper/esannV2.cls`, so the folder builds on its own |
| `fig1_interaction.pdf` | copy of `analysis/fig1_interaction.pdf` (sha256 `6543d247...`), the file used in the draft (same font subset tags) |
| `main.pdf`, `main.bbl` | build output |

Build: `cd paper/v1 && latexmk -pdf` (pdflatex, bibtex with `unsrt`, 3 passes). Log: no errors, no over/underfull
boxes, BibTeX 0 warnings; the only LaTeX warning is the template's own class-name mismatch (file `esannV2`, class
`esann`). I did not use TEXINPUTS to reach `../esannV2.cls`: with `..` on the search path a clean v1 build would read
a sibling `paper/main.aux` / `main.bbl` on its first pass.

### How close it is

Checked glyph by glyph against `context/esann2027_paper_draft.pdf` (pdfplumber, characters aligned per page) and by
pixel diff of renders:

* 6 pages, A4 (595.276 x 841.89 pt), same content on every page and the same line breaks.
* 13,437 of the 13,440 characters have the same font, size and position (max deviation 0.000 bp). The other 3 are
  the itemize bullets: same position, different font (see below).
* Even the font subset tags (e.g. `AVIMIY+CMR10`) are identical, so the glyph sets are the same.
* All rules (table rules, vertical rules, the overline, underbraces, fraction bar) have the same positions and widths;
  Fig. 1 is placed with the same matrix (scale 1.01118 = `width=\textwidth`, top of page 4).
* Pixel diff at 150 dpi (fuzz 10%): 0 differing pixels on pages 1 and 3-6, and 60 on page 2. Those 60 are the three
  itemize bullets.

Remaining differences (none visible at reading size):

1. Itemize bullets: the draft's `\textbullet` (TS1) was a Type 3 bitmap font (`F64`) because the machine that built it
   lacked cm-super; v1 uses the Type 1 cm-super font (`SFRM1000`) at the same position. I kept this on purpose: it
   is an improvement.
2. PDF metadata (creation date, document ID) differs. Neither PDF has hyperref links or an outline.

Draft source details recovered from the PDF (all reproduced in v1; useful if text is edited):

* There is a blank line before `\begin{itemize}` and none after it, which adds `\partopsep` (2 pt) above and below the list.
* `\mathrm{tr}` (no operator spacing after `/`), but `\operatorname{diag}` inside `0.01\,\overline{\operatorname{diag}(H)}`.
* Equation: `I_E \;\propto\; \underbrace{-\,\|R_M\|_H^2}_{\text{absorption}\;\le\;0} \;+\; \underbrace{...}_{\text{cross terms}}`.
* Table 1: `{\footnotesize\setlength{\tabcolsep}{3.5pt} llc|cc|ccc}` with booktabs, header `$s$ + $b$`; body rows equal
  `analysis/table1_rows.tex` exactly (verified value by value). Table 2: default `\tabcolsep`, `lcc|l`, booktabs.
* The Table 2 caption is `Sr_PSr_Q` (no thin space), while the body text on page 3 has `Sr_P\,Sr_Q`.
* `$1.7\times$`, `$2.0\times$` are in math mode; the Fig. 1 caption has `($s$/$b$)` with a text slash.
* The bibliography is `\bibliographystyle{unsrt}` inside `\begin{footnotesize}` (the template's scheme). Ref. [15] was
  written `{T}ry` (no T-r kern). URL: `\url{...}` (url package), DOI as plain text.
* Floats are `[t]`. The draft's float positions in the source are unknown. In v1, Fig. 1 is defined at the end of
  Section 4 and Tables 1-2 after the Q1 paragraph; this reproduces the draft's placement (Fig. 1 at the top of page 4,
  both tables at the top of page 5).

### Page budget (important for the revision)

* Page 1 has about 3 empty lines (40 bp) at the bottom, because the two-line Section 2 heading moved to page 2. Pages
  2-5 are full.
* That free space is all the room there is for the real author block. Two tests on copies of v1:
  * A compact one-institution block fits, and pages 2-6 stay bit-identical to v1: `\author{Names \vspace{.3cm}\\ School - Dept \\ Address - Country}`.
  * The template's two-institution block (names, then two address blocks separated by `\vspace{.1cm}\\`) pushes the
    paper to 7 pages: the last references land on page 7.
  * An optional `\thanks` footnote would also cost page-1 space. I did not test it.
* Page 6 fits only because TeX shrinks the reference list. The gap between references is 11.73 pt instead of the
  natural 13.5 pt, which squeezes about 30 pt in total. The slack left is 4 pt: adding `\vspace*{5pt}` before the
  bibliography produces a 7th page. So any net added line on pages 5-6 (or any text that flows into them) breaks the
  6-page limit unless something is removed.
* The abstract has 97 words; the template asks for "100 words".

## 4. Template report (esannV2.cls)

**What the class is.** `\ProvidesClass{esann}` dated 1999/01/07 v1.4a. It is article.cls (1999) with changes by
J. A. Lee. It loads no package, only `size10.clo` (10 pt). Main settings:

* Layout: oneside, onecolumn, `\pagestyle{empty}` (no page numbers or headers, as required), `\raggedbottom`.
* Front matter: `\maketitle` sets the title `\Large\bfseries` and the author in a `\small` tabular, with no date.
  The abstract is `\small` in a `quote`, run-in **Abstract.**
* Headings: `\section` is `\large\bfseries`; `\subsection` is bold; `\subsubsection` and `\paragraph` are italic.
* Captions: "Fig. n:" / "Table n:", with `\abovecaptionskip` 10 pt. Short captions are centred.
* Floats: topnumber 2, bottomnumber 1, totalnumber 3; topfraction .7, bottomfraction .3, textfraction .2,
  floatpagefraction .5.
* Bibliography: `thebibliography` with a `\section*` heading. The template wraps it in `footnotesize` and uses
  `\bibliographystyle{unsrt}`, which gives numeric citations in order of appearance.

**Packages loaded.** The template `.tex` loads `graphicx[dvips]`, `inputenc[latin1]`, `amssymb`, `amsmath` and `array`.
The draft and v1 load `graphicx` (no option), `amssymb`, `amsmath`, `array`, `booktabs` and `url`.

**Page geometry.** All values come from the class:

* `\textwidth` 12.2 cm (347.12 pt) and `\textheight` 19.2 cm (546.29 pt).
* `\oddsidemargin` = `\evensidemargin` = 2 cm and `\topmargin` 2.6 cm; `\headheight` = `\headsep` = 0.
* `\topskip` 10 pt, `\baselineskip` 12 pt (at most 45 lines per page) and `\parindent` 15 pt.
* `\paperwidth` = `\paperheight` = 0 pt (see item 2 below).

With the standard 1 in offset, the margins are 4.54 cm left, 4.26 cm right, 5.14 cm top and 5.36 cm bottom. I
measured these values on the draft, on v1 and on the official example `esannV2.pdf` (dvips, 2002).

**Package compatibility.** I tested in a scratch file under `paper/scratch/` (now deleted):

| Package | Version | Result under esannV2.cls |
|---|---|---|
| booktabs | 2020/01/12 v1.61803398 | OK. Rules are 0.8/0.5/0.8 pt. Vertical `\|` rules are interrupted at the booktabs rules (the draft's look). |
| siunitx | 2024-01-25 v3.3.10 | OK: `S` columns with `table-format`, `\num`, `\qty`, `\SI` (v2 name, still accepted) |
| tikz / pgf | 3.1.10 (2023-01-15) | OK |
| pgfplots | 1.18.1, `\pgfplotsset{compat=1.18}` | OK: compat 1.18 is the latest feature set for this version, so there is no warning |

* No errors and no warnings other than the class-name warning. All fonts are Type 1, and the page is A4.
* Loading siunitx, tikz and pgfplots (compat 1.18) into v1 leaves its layout bit-identical: 0.000 bp on all 6 pages.
* pgfplots gotcha (not class-related; it fails with `article` too): `\legend{Joint, P$\to$Q}` gives "Missing control
  sequence", because pgfplots list macros use `\to` as a delimiter. Use `$\rightarrow$` in legends.

**Things that can break the ESANN format rules.**

1. **Margins are not the ones in the rules.** The class gives 4.54/4.26/5.14/5.36 cm with a 19.2 cm text height, not
   4.4/4.4/5.2/5.2 cm and 19.3 cm. The text block is 1.4 mm right of centre. The official example PDF has the same
   offsets, so ESANN accepts the template's output. But the guidelines make authors responsible for the margins and
   say "if necessary, margins should be adapted".
   * Fix, tested: change the template's own offset block to
     `\voffset 0.06 cm \hoffset -0.14 cm \addtolength{\textwidth}{0cm}` and `\addtolength{\textheight}{0.1cm}`.
     This gives exactly 4.40/4.40/5.20/5.20 cm.
   * On v1 the fix keeps the same line breaks, page contents and 6 pages. Text only shifts, and page 6 gains 2.8 pt
     of glue.
   * This is a decision for the revision. v1 keeps the draft's geometry.
2. **Paper size is not set by the class.** Its default option list has `a4`, which is not a declared option (the
   declared one is `a4paper`). So `\paperwidth` = 0 and the PDF size comes from the local `pdftexconfig` (A4 here,
   from `/etc/papersize`; Letter on letter-configured installations).
   * Even `[letterpaper]` or `[a4paper]` does not change the PDF size, because nothing copies `\paperwidth` to
     `\pdfpagewidth`.
   * v1 sets `\pdfpagewidth=210mm \pdfpageheight=297mm`; keep this. After adding geometry or hyperref, re-check with
     `pdfinfo` that the page size is 595.276 x 841.89 pt.
3. **`\usepackage[dvips]{graphicx}` in the template.** Under pdflatex, every figure silently disappears ("Non-PDF
   special ignored!"). The template example built with pdflatex loses its EPS figure. Use `graphicx` without an option.
4. **`\usepackage[latin1]{inputenc}` in the template.** A UTF-8 source (→, ≈, é) would be mis-decoded. Keep sources in
   ASCII, or drop the option (UTF-8 has been the default since LaTeX 2018). v1 is ASCII and does not load inputenc.
5. **Bitmap (Type 3) fonts.** Without cm-super, TS1 symbols such as the itemize bullet become bitmaps, as in the draft.
   cm-super is installed now. Fig. 1 still embeds 5 Type 3 DejaVu fonts, because matplotlib's default is
   `pdf.fonttype` 3. The rules do not forbid this, but `pdf.fonttype: 42` in `analysis/reproduce_paper.py` (not my
   file) would avoid Type 3 fonts entirely.
6. **Page limit.** There is no slack: see "Page budget" above.
7. **Author block.** The author line is a placeholder. Review is single-blind, so the submission needs the real authors
   and affiliations in the template's author block: `\author{... \vspace{.3cm}\\ 1- School ...}`.
   * Tested: a one-institution block fits in page 1's free space, and the flow is unchanged.
   * Tested: the template's two-institution block gives 7 pages (see "Page budget").
8. **Harmless.** The warning "You have requested document class `esannV2', but the document class provides `esann'"
   comes from the official class itself.

# Phase L2: the revised paper (2026-11-10)

Author: `latex` subagent. Files written: `paper/main.tex`, `paper/figures/frontier.tex`, `paper/main.pdf`,
`reports/latex_macro_requests.md`, this section. Build: `cd paper && latexmk -pdf main.tex` (pdflatex +
BibTeX `unsrt`, 3 passes).

## Result

* 6 pages including references, A4 (595.276 x 841.89 pt). Margins measured on every page: 4.40 cm left and
  right; the text block starts at 5.20 cm.
* 0 errors, 0 overfull or underfull boxes, BibTeX 0 warnings, all fonts Type 1 and embedded. Remaining warnings:
  the class-name warning, and pgfplots "plot has no coordinates" for the panels whose data are still empty.
* Slack: about 26 pt (2 lines) at the bottom of page 6. Pages 1 to 5 are full.
* 37 `\TBD` placeholders print, R1 to R29 (see `reports/latex_macro_requests.md`). Macros that are still `--`
  print as an en dash (Table 1 rows of SmolLM2-360M, E3 shares, ...). Both are expected.
* Title: "Counting the Mask: When Does Pruning Pay in Quantized Small Language Models?" (12 words).
  Author: "A. Yaacoub" (README.md names A. Yaacoub as the author), affiliation "[Affiliation to be confirmed]".
  Any further authors must be added. A one-institution block costs nothing; the template's
  two-institution block cost a page in L1.
* 15 references, all keys of `paper/refs.bib`: SparseGPT, GPTQ/OPTQ, OBC, Laborde et al., Zhou et al., Harma et
  al., Kim et al., Bouriel et al., OBR, SmolLM2, Qwen2.5, Dutta et al., C4, WikiText-2, lm-eval. The nine
  tasks are named and lm-eval is cited, but the eight dataset papers in refs.bib (ARC, HellaSwag, PIQA,
  WinoGrande, OpenBookQA, BoolQ, SciQ, LAMBADA) are not: together they cost about 0.4 page. Restore them only
  if space appears.

## Structure and where the numbers come from

Abstract; 1 Introduction (two paragraphs, related work phrased as `reports/literature.md` supports);
2 One sweep, three orders; 3 Counting storage and interaction (TCr vs B(s,b), H2, symbol entropy; the v1
decomposition with the hedge); 4 Experimental setup; 5 Results (Storage, Free zeros, Sub-additivity with one
Hessian, Order + mechanism, Evaluation, Noise); 6 Conclusion.

* Fig. 1 (`figures/frontier.tex`): pgfplots groupplot, one panel per model, built only from
  `generated/fig_frontier_quant_<model>.dat` and `fig_frontier_sparse_<model>.dat`. Log y; black line:
  quantization only; gray circles: best order per cell at `bits_tcr`; blue dots: at `bits_mask`; gray arrows
  between the two (drawn as x error bars, because pgfplots quiver does not work on log axes). x is limited to
  1 to 4.5 bits, so the 8-bit point is left out. A panel whose two files have no rows is drawn with y = 10
  to 100 and the word "pending" (`\frontiercheck`). No number is typed in the figure.
* Table 1 (equal storage): built in `main.tex` from the per-state macros (`\resQcell`, `\resQrow`,
  `\resTabModel`). Per model, three rows (PPL, KL, flips) and 8 columns: 4-bit and the three orders of 25%+4
  (4.0 bits/w), then 3-bit and the three orders of 50%+4 (3.0 bits/w). Bold: smallest of the four values at equal
  storage, computed by TeX from the printed macro values (ties are both bold); no bold while one of the four is
  `--`. I did not use `generated/table_main_rows.tex`: at 43 rows it does not fit, and it shows only the best order.
* Table 2 (mechanism): built from the E2 per-cell macros (`\resMechRow`): 6 cells x 3 models x {penalty,
  refit, mask, both}. E2 is complete, so the table is final. `generated/table_mechanism_rows.tex` (18 rows,
  6 of them n/a) did not fit. E3 is quoted in the Order paragraph for one cell through
  `\resEThreeRemoved{Refit,Pmask,Both}SmolSmallSFiftyBFourPct`.
* Typed constants, all definitional: 128 windows, 2048 tokens, groups/blocks of 128, s and b of the grid,
  4.0/3.0/3.25/2.25 bits per weight, 1/128 bit, H2(0.5) = 1, H2(0.25) = 0.81, 0.19 bit (= 4 - 3.81),
  and the p < 0.05 threshold.
* Requested macros use their final names, with a fallback `\providecommand{\resX}{\TBD{Rn}}` placed after
  `\input{generated/macros}`. A macro added to make_tables.py under that name replaces the placeholder with
  no edit to `main.tex`.

## Switches for the families still running (preamble of main.tex)

`\Ethreetrue`, `\Efourtrue` (on), `\Efivefalse`, `\Esixfalse` (off).
* E3 off: drops the E3 clause in Sec. 4 and the E3 sentence in the Order paragraph.
* E4 off: drops "E4 adds 2-bit, 60% and 60%+4-bit" in Sec. 4; the frontier plots whatever the `.dat`
  files hold.
* E5 on: adds one sentence on seed stability (requests R30, R31) to the Noise paragraph.
* E6 on: SmolLM2-1.7B rows in Table 1, a fourth panel in Fig. 1 (panels narrowed so the four fit), "1.7B"
  in the abstract, "four models" in the Noise paragraph. To make room, it also drops the
  flips-vs-retention sentence of the Evaluation paragraph.
* Tested builds (copies in the scratchpad): all off, 6 pages; E5 on, 6 pages; E6 on, 6 pages with no slack;
  E5 and E6 on, 7 pages (about 40 pt over). All without errors or overfull boxes. With both on, cut first: the
  E3 sentence (if E3 is settled in Table 2 or not needed), the "With four models ... no scale trend" sentence,
  and then the free-zero paragraph's last sentence.
* The page break is tight. Filling the placeholders usually shortens lines ("[R13]" is wider than "60"), but
  values replacing `--` are wider. Check `pdfinfo` after every make_tables run.

## Claims that depend on pending results (check when the numbers arrive)

* Abstract and conclusion: "25%+4 ... does not beat [plain 4-bit]". This holds on SmolLM2-135M (21.99 vs
  21.94). The v1 16-bit numbers suggest that joint 25%+4 may beat 4-bit on SmolLM2-360M by about 1%. If R5
  turns out larger than the hardware noise, or a 25%+4 order wins on some model, rephrase as "is within R5% of
  it". The Storage paragraph is already worded that way.
* "50%+4 beats plain 3-bit at 3.0 bits on R1 of R2 models": in v1 the Qwen2.5-0.5B gap was only about 1.6%,
  within the measured cross-hardware noise. If R1 < R2, or the Qwen gain is that small, say so.
* "Quantizing first is the worst order in most settings" (abstract, conclusion): currently 3 of 4 cells of one
  model (R19/R20 restrict this to 4-bit cells).
* Storage paragraph: the SmolLM2-135M examples (P->Q best at 25%+4 and 50%+4) are final for that model.
* Evaluation: "Q->P answers more items correctly, mostly on SciQ" (SmolLM2-135M, 50%+4) comes from
  `e1_acc_orders.csv`: SciQ alone is -9.3 points, about 72% of the nine-task mean difference. It is final for
  that model.
* Mechanism wording follows E2 (complete): re-fitting the grid increases the penalty in every cell with a
  penalty; the pruning-only mask removes at most 39% and is negative in several cells. If E3 contradicts this at
  model level, the Order paragraph and the abstract need a sentence.
* Literature phrasing as checked in `reports/literature.md`: Harma (6-8-bit formats for the SparseGPT/Wanda
  order effect; "the analysis of [Harma] suggests" for the mask-selection reading, which the review attributes
  to them); Kim (weaker-first rule; sequential; weights, activations and KV cache; the "marginally positive"
  phrase is not used); Bouriel (about 1B, magnitude + bitsandbytes, at most 1-2 points for 4-bit hybrids at
  20%); SparseGPT ("from 2.7B parameters up"; the bitmask is described as implied, with no "1-bit" attributed
  to it); Zhou ("mostly of 1B to 3.8B parameters").

## Other changes from v1 and layout decisions

* Margins: the tested L1 fix is applied (`\voffset 0.06 cm \hoffset -0.14 cm`, `\textheight` + 0.1 cm), so the
  margins are exactly 4.4/4.4/5.2/5.2 cm, as the template requires.
* `\Urlmuskip=0mu plus 1mu` and `\emergencystretch=1em`: the three URLs in Sec. 4 overfilled a line in one switch
  combination. Neither setting changes a paragraph that already sets without an overfull line.
* Indicator in B(s,b) is `\mathbf{1}_{s>0}` (amssymb's `\mathbb{1}` prints a wrong glyph).
* Dropped from v1: Table 2 (I_Sr, including the 7-8B rows from another pipeline), the old Fig. 1, the Q3 scale
  claims (one disclaimer sentence remains), and Deep Compression/OBD/OBS/JSQ.
* v1's own fp32-vs-16-bit gap (`\resFpMaxAbsDeltaPct`, 12.98%) was cut for space. The noise paragraph keeps the
  cross-hardware 16-bit gaps (R27) and the bit-exact DET result.
* `generated/table_kl_flips.tex` was not used (KL and flips of the two equal-storage cells are in Table 1).
