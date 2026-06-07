# Plan — Node concordance sidebar

**Status:** ✅ Complete (2026-06-07)

## Problem

The CLI `cmapr search TERM --lemma` lists every sentence a term's lemma appears
in, with structural location. We wanted that revived in the graph UI: clicking a
node opens a **second right sidebar** (beside the relationship-types detail
panel) listing all sentences containing the node's lemmatized term — document
order, independently scrollable, each with a chapter › section breadcrumb, the
term highlighted.

The graph viz is a standalone HTML page (`export/html.py`, inlined D3 + data),
embedded in an iframe on the web UI result page — so the concordance is
**precomputed and inlined** into the page (works standalone and under `serve`).

## Decisions

- Panels sit **side by side**: relations panel far-right (280px), concordance
  panel (380px) just to its left; one node click opens both; empty-canvas click
  closes both.
- Location uses **existing `SentenceLocation` fields only** (chapter/section/
  subsection/paragraph). Page numbers aren't tracked at ingest — out of scope.

## Tasks

- [x] **Concordance builder** — `search/concordance.py:build_concordance(docs,
  terms, per_term_cap=400)`. Pre-lemmatizes each sentence once (reused across
  terms); single-token terms match by lemma, phrases by case-insensitive
  substring; returns `{term: [{text, marks, loc}, …]}` in doc order with a
  `{truncated, total}` sentinel when capped. `_format_location` builds the
  breadcrumb (accepts `SentenceLocation` or raw dict).
- [x] **HTML sidebar** — `export/html.py`: `generate_html(…, docs=None)` inlines
  a `CONCORDANCE` const (built from node `term`/`label`/`id`); `#concordance-panel`
  CSS/HTML (right:280px, 380px wide, own scroll); `showConcordance`/
  `closeConcordance` JS with an escape + word-boundary `<mark>` highlighter;
  node click calls both panels, canvas click closes both. Degrades to `{}` when
  no docs.
- [x] **Thread the corpus** — `cli.py:export` gains `--corpus PATH` (loads docs,
  passes to `generate_html`); `cli.py:run` passes its in-process `docs`;
  `server/app.py:build` appends `--corpus <corpus.json>` to the export call.
- [x] **Tests** — `tests/test_concordance.py` (lemma/phrase match, doc order,
  location, marks, cap); `tests/test_export.py` (panel always present, `{}`
  without docs, inlined with docs); `tests/test_cli.py` (`export --corpus`).
- [x] **Docs** — `docs/architecture.md` (Stage-4 box + html flow + module tree);
  `docs/roadmap.md` Status; this plan.

## Verification

- `uv run pytest` green.
- CLI: `cmapr export <graph.json> --format html --corpus <corpus.json> -o
  /tmp/viz/index.html`; click a node → located, highlighted sentences scroll
  beside the relations panel; empty-canvas click closes both.
- `node --check` on the generated inline script passes; highlight verified
  (word boundaries respected, HTML escaped).
- UI: `cmapr serve` → walk a work to the result graph → click a node in the
  embedded iframe.
