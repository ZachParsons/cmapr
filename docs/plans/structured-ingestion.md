# Plan — Structured ingestion pipeline

**Status:** 🚧 Phase A in progress (2026-06-09)

## Problem

Ingest quality now caps extraction quality. The 2026-06-09 corpus tuning
(`docs/roadmap.md` Status) found running headers fused mid-sentence
("…Sextus Empiricus, **[16] SEMIOTICS AND THE PHILOSOPHY OF LANGUAGE** Adv.
Math. 7.245…") minting spurious typed edges; the extractor-side all-caps
guard is a band-aid. Beyond noise, every new text costs a manual two-file
workflow (raw text + hand-built TOC), and the input matrix cmapr claims to
support — `txt/pdf × clean/dirty × TOC/index/neither` — is mostly
unexercised (see the "QA coverage & product mapping" roadmap item).

Observed defects in `data/input/eco_spl1.txt` (pdftotext-style output):

- `[N]` page markers at line start; on even pages followed by the book-title
  running header (`[16] SEMIOTICS AND THE PHILOSOPHY OF LANGUAGE`), which
  splits a sentence in two; on odd pages glued to *real content*
  (`[23]1.5.3. The sign as difference` — a genuine section heading).
- Page numbers / bare markers on their own lines (`[14]`).
- Front-matter (title page, copyright, TOC) and back-matter (bibliography,
  index) currently handled only by the manual `--start-from-section` /
  `--exclude-sections` flags downstream.

## Decisions

- **Frequency-based header detection, not pattern lists.** A running header
  is any normalized line (page-marker stripped, whitespace collapsed,
  case-folded) recurring ≥3 times across the document. Content lines that
  happen to follow a page marker recur once and are kept. No hardcoded
  titles.
- **Cleaning stays opt-in via `--clean-ocr`** (existing flag) — header
  stripping joins it as a new pipeline step, on by default *within* that
  flag. No new flag until Phase B forces one.
- **Evaluation is empirical**: re-ingest eco_spl chapters, rebuild graphs,
  compare typed-edge counts/quality with the same diagnostics used for the
  2026-06-09 extraction tuning. A cleaning change that doesn't move those
  numbers (or visibly de-noise evidence sentences) doesn't ship.
- **Backend adoption (docling et al.) is Phase B and gated on a measurable
  win** over the hand-curated `eco_spl_toc.txt` baseline — accept/reject
  experiment, not a leap of faith.

## Implementation

### Phase A — header/page-marker stripping (no new dependencies) ✅ 2026-06-09

- [x] **A.1 `_remove_running_headers`** in `preprocessing/cleaning.py`:
  two-pass — (1) count normalized line bodies (leading `[N]` marker
  stripped, whitespace collapsed, case-folded); (2) drop bodies recurring
  ≥3 times (len ≥ 8), keep the content remainder of marker-glued lines
  (`[23]1.5.3. …` → `1.5.3. …`), drop bare `[N]` lines.
- [x] **A.2 Wired into `TextCleaner.clean`** as the *first* step — root
  cause was an ordering bug: `_remove_page_numbers` stripped `[N]` markers
  globally before its own header-line rules (keyed on those markers) could
  match, leaving orphaned header text to fuse into sentences. New flag
  `remove_running_headers=True`.
- [x] **A.3 Tests** — `tests/preprocessing/test_cleaning.py`
  `TestRunningHeaders` (6): repeated header dropped, sentence flow
  restored, marker-glued heading kept, bare marker dropped, unrepeated
  text unchanged, opt-out flag.
- [x] **A.4 Measured** on `eco_spl1.txt` (dependency engine, curated
  terms): edges with header fragments in their *evidence sentences*
  **5 → 0**; sentence count unchanged (625 — no content loss); graph
  shape stable (110 nodes, ~160 edges, 94 typed). The extractor's
  all-caps argument guard (which had already suppressed the worst
  header-minted *edges*) is now defense-in-depth.

### Phase B — PDF backend evaluation (docling first)

**B.0 trial (2026-06-09) — feasibility confirmed.** docling
(trial-installed, not yet in pyproject) on `Eco_1984_SPL.pdf` pages 1–40,
`do_ocr=False`, no table structure: **25s for 40 pages** (~0.6s/page CPU;
full ~270-page book ≈ 3 min). Detected 27 headings including the complete
chapter-1 skeleton — `[I] SIGNS`, 1.1–1.7, all 1.5.1–1.5.6 subsections —
matching the hand-curated `eco_spl_toc.txt`; also identified the CONTENTS
page and front-matter title pages (useful for C.1). Caveats: glyph noise
from the PDF's own text layer ("Ι.5.Ι" with Greek iota, "1.5*6",
"I . I .") needs section-number normalization, and some sub-subsection
headings (1.2.1–1.2.6) surface number-only. Verdict: proceed with B.1.

- [ ] **B.1 `--pdf-backend {pdfplumber|docling}`** on `cmapr ingest`;
  docling under a new `[ingest]` extra, lazy-loaded. Include a
  section-number normalizer (Greek-iota/`*`/spacing fixes seen in B.0).
- [ ] **B.2 Structure comparison harness**: extract all headings from
  `Eco_1984_SPL.pdf` via docling; score against hand-curated
  `eco_spl_toc.txt` (precision/recall on section titles + ordering).
- [ ] **B.3 Accept/reject**: if docling wins, it becomes the default PDF
  path and supplies structure directly (TOC file optional); if not,
  document why here and stop at Phase A cleaning.
- [ ] **B.4 nougat** only if scanned-OCR PDFs (no text layer) become a real
  use case — defer by default.

### Phase C — front/back-matter + structure unification

- [ ] **C.1 Front-matter classifier** (heuristic): title page, copyright,
  TOC pages, dedication — auto-skip with override flag.
- [ ] **C.2 Back-matter classifier**: bibliography/references/index/appendix
  detection (section-title + content-shape heuristics) — auto-truncate.
- [ ] **C.3 Auto structure labeling**: when the backend supplies headings,
  populate `structure_nodes`/`sentence_locations` without a `--toc` file;
  `--toc` stays as the manual override.
- [ ] **C.4 QA fixture matrix** (folds in the "Manual QA text samples"
  roadmap item): fixtures at sentence/paragraph/chapter/book scale ×
  clean/dirty × TOC/index/neither; `docs/qa/ingestion.md` walk-through.

## Verification

- Phase A: cleaning unit tests + eco_spl1 before/after edge metrics (A.4).
- Phase B: structure-comparison scores recorded here.
- Phase C: QA walk-through on the fixture matrix.
