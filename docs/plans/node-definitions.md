# Plan — Guarantee every node a text-derived definition

**Status:** ✅ Complete (2026-06-08)

## Problem

Definitions were optional (`cmapr graph --definitions`, needs the `embeddings`
extra), single-signal (embedding similarity), and dropped anything below a 0.30
similarity floor — so many nodes had **no definition**. Definitions must be
mandatory, derived from the input text (never a dictionary), with the term's
**concordance** (all occurrence sentences) as the evidence pool.

## Decisions

- **Composite extractive, offline.** Rank a term's concordance sentences by a
  blended definitional score and attach the best *verbatim* sentence. Sentence
  embeddings stay an optional ranking booster, never required.
- **Coverage floor — never undefined.** No similarity floor; always take the
  top-ranked occurrence sentence, then fall back to an edge-derived gloss, then
  the first occurrence.

## Implementation

- [x] **`analysis/definitions.py`** (new) — `derive_definitions(graph, docs,
  ranker=None)`:
  - candidates from `search/concordance.build_concordance` (the concordance is
    the source; gives text + location);
  - `_composite_score` = pattern (rarity `DEFINITIONAL_PATTERNS` +
    `_EXTRA_DEFINITION_PATTERNS` for passive "X is defined as" / "X denotes")
    `·0.45` + embedding sim `·0.30` (folds into pattern when no ranker) +
    subject-position `·0.15` + intro-bonus `·0.10` − length penalty;
  - fallback chain `best concordance sentence → _edge_gloss
    (kind-of/property/relation) → _first_occurrence_sentence`;
  - whitespace-normalizes the stored `definition`; records `definition_source`;
    skips nodes that already have a definition.
- [x] **`cli.py:graph`** — derive by default; `--definitions` loads
  `DefinitionRanker` (embeddings extra) as the booster. **`cli.py:run`** — derive
  before export. Echoes `Derived definitions for N/N`.
- [x] **`export/d3.py`** — round-trip `definition_source` (so the web
  graph.json → export path keeps it). **`export/html.py`** — Relations drawer
  shows the definition (italic) + its source location (muted).
- [x] **`server/templates/options.html`** — relabel the checkbox to "Improve
  definitions with sentence embeddings (slower)"; definitions are always derived.
- [x] **`embeddings.py`** — `DefinitionRanker` kept and reused;
  `enrich_graph_with_definitions` left as an embeddings-only helper (still tested).
- [x] **Tests** — `tests/test_definitions.py` (pattern scoring, picks the
  definitional sentence, every-node coverage, edge-gloss fallback, source
  location, skip-existing). `tests/test_embeddings.py` unchanged.
- [x] **Docs** — `docs/architecture.md` (Stage-3 step + module tree + node
  schema), `docs/roadmap.md`, this plan.

## Verification

- `uv run pytest` — 986 passed, 3 skipped; ruff clean.
- `cmapr graph data/output/corpus/eco_spl1/corpus.json -t <terms> -o g.json` →
  **36/36** nodes carry a `definition` in the serialized graph.json, with
  `definition_source`. Strong terms (`semiotic`, …) get genuinely definitional
  sentences.
- Web: build → result graph → clicking any node shows a definition in the
  tooltip + Relations drawer; none blank.
