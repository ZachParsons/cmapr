# Plan — Term-review cleanup

**Status:** ✅ Complete (2026-06-07)

## Problem

The web UI's **term review** step (`/review`) renders `terms.json` verbatim, and
that file (produced by the rarities filter chain in `terms/scoring.py`, shared by
`cmapr rarities` and the web flow) carried three classes of noise:

1. **Function words** — `since`, `whether`, `without`, `although`, `could`,
   `would`, `shall`, `else`, `onto`, `versa`, `upon`. The chain had no stopword
   filter, even though a shared `STOPWORDS` set already existed in
   `search/extract.py` (used by `graph/node_filter.py`).
2. **Broken non-words** — `semiosi`, `wherea`, `unles` (root cause: unguarded
   `inflect.singular_noun` in `preprocessing/lemmatize.py` strips a trailing `-s`
   off `-sis/-ss/-as` words via `lemma_and_derivational_merge`); `/man/` (only
   quote chars were stripped); OCR merges (`thesecase`, `mybody`, `intoplay`) and
   leading-char drops (`ictionary`).
3. **Un-deduplicated derivational variants** — `taxonomic`/`taxonomy`,
   `icon`/`iconic`. The existing suffix merge only collapses onto a *bare* stem
   that's itself a candidate; it can't span `-y ↔ -ic` or cross-POS pairs.

## Decisions

- Garbage cleanup = **root-cause fixes + safe heuristics** (no aggressive
  spell-check pass that would risk dropping legitimate neologisms).
- Derivational dedup canonical = **prefer the noun form** (fallback: highest score).

## Tasks

- [x] **Root-cause `-s` fix** — `preprocessing/lemmatize.py`: skip the inflect
  singular fallback for `-ss/-us/-is/-as/-os/-sis` endings (`_NON_PLURAL_S_ENDINGS`),
  in both `lemmatize()` and `lemmatize_tagged()`.
- [x] **Stopword filter** — extend `STOPWORDS` (`else`, `onto`, `versa`, `vice`,
  `upon`) in `search/extract.py`; add `filter_stopwords()` to `terms/scoring.py`.
- [x] **Edge-punctuation strip** — broaden `strip_stray_quotes()` to strip
  leading/trailing `/` and `\` (`_EDGE_CHARS`).
- [x] **Derivational dedup** — `merge_derivational_variants()` (union-find over
  WordNet `derivationally_related_forms`, noun-preferred), invoked as Pass 3 of
  `lemma_and_derivational_merge()` before the top-N trim.
- [x] **OCR-artifact filter** — `filter_ocr_artifacts()`: function-word merge
  (`_OCR_MERGE_PREFIXES`) + leading-char-drop heuristics, both gated on the term
  being absent from WordNet and STOPWORDS.
- [x] **Wire the chains** — `cli.py:rarities` and `terms/scoring.py:apply_run_pipeline`:
  `strip → proper-name → stopword → lemma/derivational merge (top-N) → fragment →
  OCR → POS → vetting`. Stopword filter runs before the top-N trim.
- [x] **Tests** — new `tests/test_scoring.py`; `-s` regression in
  `tests/test_preprocessing.py`; updated `tests/test_node_filter.py` (`upon` now a
  stopword). Full suite: 940 passed, 3 skipped.
- [x] **Docs** — `docs/architecture.md` (Stage-2 `S2filter` box + numbered list +
  module one-liner); `docs/roadmap.md` Status block; this plan.

## Verification

- `uv run pytest` — 940 passed, 3 skipped.
- Re-ran `cmapr rarities` on `eco_spl7`, `eco_spl1_pdf`, `eco_spl1_spacy`:
  function words, `semiosi`/`wherea`/`unles`, `/man/`, and OCR merges
  (`thesecase`, `intoplay`, `mybody`, …) are gone; corrected forms (`semiosis`,
  `prosthesis`) now surface; real terms retained.
