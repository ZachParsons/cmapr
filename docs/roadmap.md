# Roadmap

This is the canonical entrypoint for *what's latest, what's next, what's blocked*. Update the **Status** block at the end of each session.

**Past:** completed phases, significant additions/pivots.
**Present:** planned work, WIP.
**Future:** unplanned features, ideas.

---

## Concept Mapper

A tool for extracting and visualizing an author's idiosyncratic conceptual vocabulary from primary texts. Identifies neologisms and terms with specialized technical meaning that are statistically distinctive compared to general English corpora. Maps concepts through co-occurrence and grammatical relations, exporting as D3 visualizations.

**Examples of target terms:** Aristotle's *eudaimonia*, Spinoza's *affect*, Hegel's *sublation*, Dennett's *intentional stance*, Deleuze & Guattari's *body without organs*.

---

## Status

_Last updated: 2026-06-09. Cold-re-entry checklist — keep glance-able._

- **Last completed:** **Corpus-driven extraction tuning** (2026-06-09, follow-up to the extraction batch) — measured the dep engine on the real `eco_spl1` corpus (625 sentences, 43 curated terms) and fixed the dominant failure modes the data exposed. Baseline → after: **typed edges 18 (regex) / 54 (dep v1) → 94**, property edges 2 → 44, seed terms with ≥1 composed-definition part 15/41 → **28/43**. Fixes, all in `graph/dep_extractor.py`: (1) **verb+prep mappings** mined from the corpus's own unmapped connecting verbs (`stands for`→definition, `belongs to`→kind-of, `based on`→dependence, `leads to`/`refers to`/`serves as`→production, plus `call`/`indicate` lemmas; `for` added to object preps); (2) **vacuous-nominal guard** (`something`, `one`, `thing`, …) — never arguments; when a copular target is vacuous but carries a relative clause, the relation is extracted *through* it ("a sign is something which produces X" → production(sign, X), not kind-of(sign, something)); (3) **all-caps argument guard** — running headers fused into sentences by the PDF ingest ("SEMIOTICS AND THE PHILOSOPHY OF LANGUAGE (b) There is…") were minting spurious edges (real fix remains the structured-ingestion item); (4) **amod property emission** — adjectival seed terms (`contextual`, `extensional`, `circumstantial`, …) never head an argument, so the modification itself is now the relation: "contextual selection" → property(selection, contextual), rendering as "Contextual — a quality of inference, instruction and element"; (5) **hyphenated-seed resolution** (`sign-function`, `co-text` tokenize as three tokens; rejoined during argument matching). Remaining sparseness floor: 65 cooccurrence edges are pairs Eco genuinely connects only discursively (across sentences) — that residue is the future coref-at-scale + cross-sentence question, not a parser gap.
- **Earlier:** **Extraction-quality batch** (3 features, 2026-06-09): (1) **Fallback suppression** — `PropositionExtractor._extract_from_sentence` no longer runs the generic `relation`/pos-verb catch-alls on sentences a typed pattern already claimed (was polluting edges with low-information types). (2) **Dependency-parse extraction engine** — new `graph/dep_extractor.py`: sentence-centric spaCy parsing replaces the pairwise regex scan; reads real argument structure (no more `.{0,80}` wildcard false positives), handles passives/copulars/appositives/prepositional objects/coordination expansion, and mines sentences containing only *one* seed term (new-node candidates gated by NodeFilter; generic relations never mint new nodes). Wired as `build_proposition_graph(engine=)` + `cmapr graph --engine [auto|dependency|regex]`; auto picks dependency when spaCy+model are importable, regex otherwise. (3) **Coreference resolution at ingest** — new `preprocessing/coref.py` + `[coref]` extra (fastcoref; `transformers<4.50` pin required) + `cmapr ingest --coref`: rewrites pronominal mentions with their antecedent's verbatim text (pronouns only — conservative, span-grounded, classification not generation) before tokenization, so concordance/definitions/extraction see every sentence a concept participates in. **Environment decision:** project pinned to **Python 3.13** (`.python-version`) — spaCy and the torch stack have no cp314 wheels yet, so 3.14 couldn't run the preferred engine; the old "spaCy 3.8 supports 3.14" claim was stale. Also removed the unused `pattern3` core dep — it shipped `pdfminer3k`, which file-clobbers pdfplumber's `pdfminer-six` module directory (install-order-dependent PDF breakage). Tests: `tests/test_dep_extractor.py` (15), `tests/test_coref.py` (6), fallback-suppression regression in `tests/test_graph.py`.
- **Earlier:** **Usage-based composed definitions** (+ richness fixes) — second definition layer per node: `analysis/definitions.py:compose_definitions` deterministically assembles a `composed_definition` sentence + a `composed_parts` scaffold (all 8 relation types — definition, kind-of, production, dependence, property, relation, component, opposition — with `None`/"—" marking parts that have no extracted relation; absence is stated, never fabricated). Generation without a model: every clause traces to an extracted edge (consistent with the REBEL-deferral policy). Coverage guarantee: nodes with only cooccurrence edges get a "co-occurs with …" fallback, so every exported node has one. **Root-cause fix for sparse compositions:** `build_proposition_graph` was keeping only the priority-winning proposition type per pair and discarding the rest — it now puts *all* extracted types on one edge via the additive multi-type schema (`relation_types`/`weight_by_type`/`verb_by_type`/`evidence_by_type`, same as `cmapr merge`). Drawer shows the full part scaffold first, then the verbatim sentence under an "Example:" label; tooltip mirrors the order. Remaining sparseness is extraction recall — the spaCy dependency-parsing initiative is the lever for that.
- **Earlier:** **Guaranteed node definitions** (`docs/plans/node-definitions.md`) — every graph node now gets a text-derived definition (no more blanks). `analysis/definitions.py:derive_definitions` ranks each term's **concordance** sentences by a composite score (definitional patterns + optional embedding similarity + subject-position + intro-bonus − length penalty) and attaches the best *verbatim* sentence (+ its `definition_source` location). The old 0.30 embedding floor is gone; coverage is guaranteed by the fallback chain *best concordance sentence → edge gloss → first occurrence*. Runs by default in `cmapr graph`/`run`/serve; `--definitions` keeps the sentence-embedding signal as an optional booster. Shows in the node tooltip + Relations drawer. New `tests/test_definitions.py`.
- **Earlier:** **Review-page UX + learned filters** — the web review step now shows real `corpus_count` (Freq column), exposes the full rarities knobs (min-score threshold, top-n, POS, keep-names/fragments, no-lemmatize, persisted per-work in `rarities_params.json`), sortable columns, and a single **Reason** selector per row. Reasons feed two cross-work learned-filter loops applied on every later rarities/run: "common & atopic" → grows `stopwords_extra.json` (consumed by `filter_stopwords`); "duplicate / lemma" → grows `aliases.json` ({alias: canonical}, canonical inferred via WordNet/lemma/stem; `apply_aliases` drops the variant and folds its frequency into the canonical). Graph viz also gained: half-tone drawer titles (Relations / Concordance), the legend repositioning left of open drawers, concordance reflow into the relations footprint on close, and draggable drawer widths. The node concordance now matches **all inflected surface forms** of a term and, via a per-work `variants.json` (provenance from the lemma/derivational merges + aliases, threaded through `cmapr export --variants`), also the **pipeline-merged derivational variants** (clicking `taxonomy` shows `taxonomic` sentences). Tests in `tests/test_scoring.py` + `tests/test_server.py` + `tests/test_concordance.py`.
- **Earlier:** **Node concordance sidebar** (`docs/plans/node-concordance.md`) — revived the CLI `search --lemma` concordance in the graph viz. Clicking a node now opens a second right sidebar (beside the relationship-types panel) listing every sentence the node's lemmatized term appears in, in document order, independently scrollable, each with a chapter › section breadcrumb and the term highlighted. New `search/concordance.py:build_concordance` (single-pass lemma index) is precomputed and inlined into the standalone HTML. Wired via `generate_html(docs=)`, a new `cmapr export --corpus PATH` option, `cmapr run` (in-process docs), and the `serve` build step. No page numbers (not tracked at ingest). New `tests/test_concordance.py` + export/CLI assertions.
- **Earlier:** **Term-review cleanup** (`docs/plans/term-review-cleanup.md`) — fixed three classes of noise surfacing in the web review step / `terms.json`: (1) function words (`since`, `whether`, `could`, …) via a new `filter_stopwords` step wired into both the rarities chain and `apply_run_pipeline`; (2) broken non-words — root-caused the `-s` corruption (`semiosis`→`semiosi`) to an unguarded `inflect.singular_noun` in `preprocessing/lemmatize.py`, plus broadened edge-punctuation stripping (`/man/`→`man`) and a conservative `filter_ocr_artifacts` for function-word merges (`thesecase`) and leading-char drops (`ictionary`); (3) derivational duplicates (`taxonomic`/`taxonomy`) via a WordNet `derivationally_related_forms` merge pass that prefers the noun form. New `tests/test_scoring.py`.
- **Earlier:** README now carries the canonical doc map (project-level docs + per-feature `research → specs → plans → qa` lifecycle). Replaces the prior three-pointer "Documentation" section. Also rules.md gained a doc-update-gate (table of every canonical doc, when it must update, what to update) plus the sentence-transformers (st.1–st.8) and REBEL (rb.1–rb.7) initiatives queued in Future Work. Modularity refactor + wall-scale `docs/architecture.md` diagram landed in commit `1197d50`.
- **Next up:** **sentence-transformers integration** (st.1–st.8 in Future Work) — recommended highest-payoff quality lift per the library survey (`docs/survey.md`). Start with st.1–st.3 (dep + embedding plumbing + `cmapr similar TERM`), generalizing the already-landed `[embeddings]` extra / `analysis/embeddings.py`. After that, the preferred relation-extraction lift is **spaCy dependency parsing** (see the SpaCy integration entry in Future Work), *not* REBEL — see next bullet.
- **Decision (2026-06-09): REBEL deferred.** All REBEL work (rb.1–rb.7) is postponed indefinitely. Rationale: REBEL is generative (seq2seq) — it can hallucinate triples and normalizes surface forms away from the verbatim source, conflicting with cmapr's span-grounded evidence model. Preference: simpler, deterministic, old-school methods — spaCy dependency parsing as the relation-extraction base layer. The same generative-extraction policy deprioritizes the LangExtract evaluation. Full rationale in `docs/survey.md` § "Technology generations & determinism". Revisit only if the spaCy-parse extractor plateaus with a measured recall gap.
- **Open issues:**
  - **Wizard reverted** — `cmapr wizard` was prototyped then reverted as redundant with the web UI. Don't re-propose without a distinct use case.
  - **`cmapr try-extract` scrapped** — debug command originally on the backlog, removed per user direction (couldn't justify the use case).
  - **Cluster feature lacks manual QA** — `docs/qa/cluster.md` is now ready; not yet exercised on the eco_spl1 corpus.
  - **`.claude/context.md` partly duplicates architecture.md** — the module tree and design-decision sections overlap. Candidate for slimming to a pointer.
- **Tests:** 1026 passed, 2 skipped (pydot-gated) with **all extras installed** on Python 3.13 (`uv sync --extra dev --extra spacy --extra coref --extra serve --extra embeddings` + `uv pip install <en_core_web_sm wheel URL>`; if PDF tests fail after a sync, `uv pip install --reinstall-package pdfminer-six pdfminer-six`). New: `tests/test_dep_extractor.py` (15 — verb constructions, copular/appositive, sentence-centric recall, builder integration), `tests/test_coref.py` (6 — pronoun/possessive rewriting, conservatism, pipeline integration), fallback-suppression + multi-type-edge regressions in `tests/test_graph.py`. `tests/test_definitions.py` covers composite ranking + the coverage guarantee + the relation-composed definitions (all-parts scaffold, cooccurrence fallback, stale-value recompute). Latest additions cover the learned-filter loops (`tests/test_scoring.py`) and the review-page reason column + file growth (`tests/test_server.py`). `tests/test_concordance.py` added for the node concordance; `tests/test_export.py` + `tests/test_cli.py` gained concordance-inlining assertions. (Prior batch: `tests/test_scoring.py` for the term-review filters; `-s`-corruption regression in `tests/test_preprocessing.py`.)

---

## v1 milestones

**🎉 Feature-complete** — all 11 phases implemented, tested, and documented.

| Phase | Description | Tests |
|-------|-------------|-------|
| 0 | Project scaffolding, storage layer, test corpus | 12 |
| 1 | Corpus loading, preprocessing pipeline (tokenization, POS, lemmas) | 46 |
| 2 | Frequency analysis, Brown corpus reference, TF-IDF | 21 |
| 3 | Philosophical term detection (multi-method rarity analysis) | 103 |
| 4 | Term list management (curation, import/export, auto-population) | 47 |
| 5 | Search & concordance (find, KWIC, context windows, dispersion) | 52 |
| 6 | Co-occurrence analysis (PMI, LLR, matrices) | 45 |
| 7 | Relation extraction (SVO, copular, prepositional) | 35 |
| 8 | Graph construction (networkx, builders, operations, metrics) | 62 |
| 9 | Export & visualization (D3 JSON, GraphML, DOT, CSV, HTML) | 30 |
| 10 | CLI interface (Click, unified command-line access) | 23 |
| 11 | Documentation & polish | — |

**718 tests passing, 2 skipped** (pydot-dependent DOT export tests).

### Post-completion additions (February 2026)

- [x] **OCR text cleaning** — `preprocessing/cleaning.py`, `--clean-ocr` flag. 21 tests.
- [x] **PDF input support** — `load_pdf()` via pdfplumber, auto-detected in `load_file()`. 6 tests.
- [x] **Paragraph segmentation** — `preprocessing/segment.py`, paragraph boundary detection. 21 tests.
- [x] **Synonym replacement** — inflection-preserving term replacement. `transformations/`, `cmapr replace`. 59 tests. See `docs/replacement.md`.
- [x] **Contextual relation extraction** — integrated SVO + co-occurrence workflow. `analysis/contextual_relations.py`, `cmapr analyze`. 38 tests.
- [x] **Remove legacy pos_tagger.py** — Deleted 490 lines of legacy code. Updated Makefile.
- [x] **Documentation consolidation** — Merged tutorial into README; deleted `docs/tutorial.md` and `scripts/workflow.sh`. Docs reduced to README + api-reference.
- [x] **Infrastructure cleanup** — Removed 77MB of old venv, cache files, and unused directories.
- [x] **Analyze window option** — `--window/-w` flag for `analyze`: shows significant terms in a sentence or paragraph window around each occurrence of the search term (e.g. `-w s0`, `-w s1`, `-w p0`). New `extract_terms_from_sentence_set()` in `search/extract.py`. 33 tests.
- [x] **Analyze `-g` shorthand** — `-g` shorthand for `--group-by` on the `analyze` command.
- [x] **Front/back-matter filters** — `--start-from-section N` (skip content before chapter N) and `--exclude-sections PATTERN` (regex exclusion by section title) on both `analyze` and `search`. Fixed `search` command to use `ProcessedDocument.from_dict()` for correct nested deserialization. 20 tests.
- [x] **Structure detection bug fix** — scaled sentence positions to match text coordinates; previously, structure nodes were not found for sentences mid-document.
- [x] **Analyze window performance** — reduced redundant path output; path header now only printed when it changes between matches.
- [x] **Dependency parse tree in analyze window** — `analyze --window` prints the matched sentence then renders its Stanza dependency parse tree below it.
- [x] **CLI module command index** — module docstring in `cli.py` lists all nine commands with a one-line description each.

### Project summary

11/11 phases complete (Jan 14–25, 2026, 12 days); 8,788 lines across 35 modules; 718 tests; ~107KB documentation across 5 guides; dual CLI/Python API covering the full pipeline from text loading through interactive D3 visualization — production-ready for digital humanities research.

---

## Existing Spike Implementations

Working implementations of tokenization, POS tagging, lemmatization, frequency analysis, and sentence search existed in `spike/tryout_nltk.py` and `spike/pos_tagger.py` and were refactored into the main codebase (Phases 1, 2, 5); graph construction, D3 export, and the CLI had no spike equivalents.

---

## Phase 0: Project Scaffolding ✅ COMPLETE

- [x] **0.1 Initialize project structure**
  - [x] Create directory layout: `src/concept_mapper/`, `tests/`, `data/sample/`, `data/output/`
  - [x] Initialize git repository
  - [x] Create `pyproject.toml`
  - [x] Initial dependencies: `nltk`, `pytest`, `click`, `ruff`, `ipython`

- [x] **0.2 Download NLTK data**
  - [x] Create setup script `scripts/download_nltk_data.py`
  - [x] Download: `punkt`, `averaged_perceptron_tagger`, `wordnet`, `brown`, `stopwords`
  - [x] Verify downloads succeed

- [x] **0.3 Create sample test corpus**
  - [x] Create 2-3 short `.txt` files in `data/sample/`
  - [x] Include invented "rare terms" with known frequencies
  - [x] Document expected values for verification

- [x] **0.4 Storage abstraction** (`src/concept_mapper/storage/`)
  - [x] Define `StorageBackend` ABC
  - [x] Implement `JSONBackend` as default
  - [x] Add filesystem utilities (create output dirs, check paths)
  - [x] Tests: round-trip save/load for each data type (12 tests passing)

### Storage architecture

JSON-backed `StorageBackend` ABC with `JSONBackend` as default, designed for future migration to SQLite (queryable intermediate data), Parquet (large matrices), or a database backend (web/multi-user access).

---

## Phase 1: Corpus Ingestion & Preprocessing ✅ COMPLETE

- [x] **1.1 File loader** (`src/concept_mapper/corpus/loader.py`)
  - [x] `load_file(path: Path) -> Document`
  - [x] `load_directory(path: Path, pattern: str = "*.txt") -> Corpus`
  - [x] Handle encoding (UTF-8 with Latin-1 fallback)
  - [x] Tests: load sample files, verify content (22 tests passing)

- [x] **1.2 Data structures** (`src/concept_mapper/corpus/models.py`)
  - [x] `Document` dataclass: text, metadata (title, author, date, source_path)
  - [x] `Corpus` class: collection of Documents
  - [x] `ProcessedDocument` dataclass: raw, sentences, tokens, pos_tags, lemmas

- [x] **1.3 Tokenization** (`src/concept_mapper/preprocessing/tokenize.py`)
  - [x] `tokenize_words(text: str) -> list[str]`
  - [x] `tokenize_sentences(text: str) -> list[str]`
  - [x] `tokenize_words_preserve_case()` - preserves original case
  - [x] Tests: verify token/sentence counts on sample (24 tests passing)

- [x] **1.4 POS tagging** (`src/concept_mapper/preprocessing/tagging.py`)
  - [x] `tag_tokens(tokens: list[str]) -> list[tuple[str, str]]`
  - [x] `tag_sentences(sentences: list[str]) -> list[list[tuple[str, str]]]`
  - [x] `filter_by_pos()` - extract tokens by POS tag
  - [x] Tests: spot-check known POS assignments (24 tests passing)

- [x] **1.5 Lemmatization** (`src/concept_mapper/preprocessing/lemmatize.py`)
  - [x] `get_wordnet_pos(treebank_tag: str) -> str`
  - [x] `lemmatize(word: str, pos: str) -> str`
  - [x] `lemmatize_tagged(tagged_tokens: list[tuple]) -> list[str]`
  - [x] `lemmatize_words()` - batch lemmatization
  - [x] Tests: "running" → "run", "better" → "good" (24 tests passing)

- [x] **1.6 Preprocessing pipeline** (`src/concept_mapper/preprocessing/pipeline.py`)
  - [x] `preprocess(document: Document) -> ProcessedDocument`
  - [x] `preprocess_corpus(corpus: Corpus) -> list[ProcessedDocument]`
  - [x] Single entry point: tokenize → tag → lemmatize
  - [x] Tests: round-trip load → preprocess → verify structure (24 tests passing)

- [x] **1.7 Paragraph segmentation** (`src/concept_mapper/preprocessing/segment.py`)
  - [x] `segment_paragraphs(text: str) -> list[str]`
  - [x] Handle various paragraph markers (double newline, indentation)
  - [x] Add paragraph indices to ProcessedDocument
  - [x] Tests: verify paragraph boundaries (21 tests passing)

---

## Phase 2: Term Extraction & Frequency Analysis ✅ COMPLETE

- [x] **2.1 Frequency distribution** (`src/concept_mapper/analysis/frequency.py`)
  - [x] `word_frequencies(doc: ProcessedDocument) -> Counter`
  - [x] `pos_filtered_frequencies(doc: ProcessedDocument, pos_tags: set) -> Counter`
  - [x] `get_vocabulary()` - extract unique terms
  - [x] Tests: manual count verification (21 tests passing)

- [x] **2.2 Corpus-level aggregation**
  - [x] `corpus_frequencies(docs: list[ProcessedDocument]) -> Counter`
  - [x] `document_frequencies(docs: list[ProcessedDocument]) -> Counter`
  - [x] Tests: term in 2 docs → doc_freq = 2

- [x] **2.3 Reference corpus** (`src/concept_mapper/analysis/reference.py`)
  - [x] `load_reference_corpus(name: str = "brown") -> Counter`
  - [x] Cache to disk after first computation (`data/output/cache/brown_corpus_freqs.json`)
  - [x] `get_reference_vocabulary()`, `get_reference_size()`
  - [x] Tests: verify Brown corpus loads, common words have high freq (21 tests passing)

- [x] **2.4 TF-IDF** (`src/concept_mapper/analysis/tfidf.py`)
  - [x] `tf(term, doc) -> float`, `idf(term, docs) -> float`, `tfidf(term, doc, docs) -> float`
  - [x] `corpus_tfidf_scores(docs) -> dict[str, float]`
  - [x] `document_tfidf_scores()` - per-document TF-IDF scores
  - [x] Tests: unique term scores high, common term scores low (21 tests passing)

---

## Phase 3: Philosophical Term Detection ✅ COMPLETE

- [x] **3.1 Corpus-comparative analysis** (`src/concept_mapper/analysis/rarity.py`)
  - [x] `compare_to_reference(docs, reference_corpus: Counter) -> dict[str, float]`
  - [x] `get_corpus_specific_terms(docs, reference, threshold) -> set[str]`
  - [x] `get_top_corpus_specific_terms()`, `get_neologism_candidates()`, `get_term_context_stats()`
  - [x] Tests: planted term with high author-freq/low reference-freq, verify detection

- [x] **3.2 TF-IDF against reference corpus**
  - [x] `tfidf_vs_reference(docs, reference: Counter) -> dict[str, float]`
  - [x] `get_top_tfidf_terms()`, `get_distinctive_by_tfidf()`, `get_combined_distinctive_terms()`
  - [x] Tests: author-specific term scores above generic vocabulary

- [x] **3.3 Neologism detection**
  - [x] `get_wordnet_neologisms(docs) -> set[str]` (not in WordNet's 117K word-sense pairs)
  - [x] `get_capitalized_technical_terms(docs) -> set[str]` (non-sentence-initial)
  - [x] `get_potential_neologisms()`, `get_all_neologism_signals()`
  - [x] Tests: planted neologism detected, common words excluded

- [x] **3.4 Definitional context extraction**
  - [x] `get_definitional_contexts(docs) -> list[tuple[str, str, str, str]]`
    - 8 patterns: copular, explicit_mean, metalinguistic, conceptual, appositive, explicit_define, referential, interpretive
  - [x] `score_by_definitional_context(terms, contexts) -> dict[str, int]`
  - [x] `get_definitional_sentences()`, `get_highly_defined_terms()`, `analyze_definitional_patterns()`
  - [x] Tests: pattern matching on planted definitional sentences

- [x] **3.5 POS-filtered candidate extraction**
  - [x] `filter_by_pos_tags(docs, include_tags, exclude_tags) -> set[str]`
  - [x] `get_philosophical_term_candidates()` with focus modes (nouns, verbs, adjectives, all_content)
  - [x] `get_compound_terms()` for hyphenated and noun phrases
  - [x] Tests: function words filtered out

- [x] **3.6 Hybrid philosophical term scorer**
  - [x] `PhilosophicalTermScorer` class with configurable weights
    - Weight 1: Corpus-comparative ratio (1.0)
    - Weight 2: TF-IDF vs reference (1.0)
    - Weight 3: Neologism detection (0.5)
    - Weight 4: Definitional context count (0.3)
    - Weight 5: Capitalization (0.2)
  - [x] `score_term(term: str) -> dict`
  - [x] `score_all(min_score, top_n) -> list[tuple[str, float, dict]]`
  - [x] `get_high_confidence_terms(min_signals) -> set[str]`
  - [x] `score_philosophical_terms()` convenience function
  - [x] Tests: known philosophical neologism scores high, common English words score low

---

## Phase 4: Term List Management ✅ COMPLETE

- [x] **4.1 Data structures** (`src/concept_mapper/terms/models.py`)
  - [x] `TermEntry` dataclass: term, lemma, pos, definition, notes, examples, metadata
  - [x] `TermList` class: collection with lookup by term
  - [x] Dictionary serialization (to_dict/from_dict)

- [x] **4.2 CRUD operations** (TermList methods)
  - [x] `add(entry)`, `remove(term)`, `update(term, **kwargs)`, `get(term) -> TermEntry | None`
  - [x] `list_terms() -> list[TermEntry]`, `list_term_names() -> list[str]`
  - [x] Tests: CRUD round-trip, error handling

- [x] **4.3 Persistence**
  - [x] `save(path: Path)` → JSON
  - [x] `load(path: Path) -> TermList` → JSON
  - [x] Tests: save → load preserves all data

- [x] **4.4 Bulk operations** (`src/concept_mapper/terms/manager.py`)
  - [x] `import_from_txt(path)`, `export_to_txt(path)`
  - [x] `import_from_csv(path)`, `export_to_csv(path)`
  - [x] `merge_from_file(path, format)`, `filter_by_pos(tags)`, `get_statistics()`
  - [x] Tests: import/export round-trip, formats

- [x] **4.5 Auto-populate from rarity** (`src/concept_mapper/terms/suggester.py`)
  - [x] `suggest_terms_from_analysis(docs, reference, min_score, top_n)`
  - [x] Populate examples from corpus automatically
  - [x] `suggest_terms_by_method(method)` — ratio, tfidf, neologism, definitional
  - [x] Tests: suggested list contains expected rare terms

---

## Phase 5: Search & Concordance ✅ COMPLETE

- [x] **5.1 Basic search** (`src/concept_mapper/search/find.py`)
  - [x] `SentenceMatch` dataclass: sentence, doc_id, sent_index, term_positions
  - [x] `find_sentences(term, docs) -> list[SentenceMatch]`
  - [x] `find_sentences_any()`, `find_sentences_all()`, `count_term_occurrences()`
  - [x] Case-sensitive and case-insensitive search; `match_lemma=True` option
  - [x] Tests: 12 tests

- [x] **5.2 KWIC concordance** (`src/concept_mapper/search/concordance.py`)
  - [x] `KWICLine` dataclass: left_context, keyword, right_context, doc_id
  - [x] `concordance(term, docs, width=50) -> list[KWICLine]`
  - [x] `concordance_sorted()`, `concordance_filtered()`
  - [x] Tests: 12 tests

- [x] **5.3 Context window** (`src/concept_mapper/search/context.py`)
  - [x] `ContextWindow` dataclass: before, match, after
  - [x] `get_context(term, docs, n_sentences=1) -> list[ContextWindow]`
  - [x] `get_context_by_match()`, `get_context_with_highlights()`, `format_context_windows()`
  - [x] Tests: 12 tests

- [x] **5.4 Dispersion** (`src/concept_mapper/search/dispersion.py`)
  - [x] `dispersion(term, docs) -> dict[str, list[int]]`
  - [x] `get_dispersion_summary()`, `compare_dispersion()`, `dispersion_plot_data()`
  - [x] `get_concentrated_regions()`
  - [x] Tests: 16 tests

---

## Phase 6: Co-occurrence Analysis ✅ COMPLETE

- [x] **6.1 Sentence-level co-occurrence** (`src/concept_mapper/analysis/cooccurrence.py`)
  - [x] `cooccurs_in_sentence(term, docs) -> Counter`
  - [x] Tests: 6 tests

- [x] **6.2 Filtered co-occurrence**
  - [x] `cooccurs_filtered(term, docs, term_list: TermList) -> Counter`
  - [x] Tests: 4 tests

- [x] **6.3 Paragraph-level co-occurrence**
  - [x] `cooccurs_in_paragraph(term, docs) -> Counter`
  - [x] Tests: 2 tests

- [x] **6.4 N-sentence window co-occurrence**
  - [x] `cooccurs_within_n(term, docs, n_sentences) -> Counter`
  - [x] Tests: 3 tests

- [x] **6.5 Statistical significance**
  - [x] `pmi(term1, term2, docs) -> float`
  - [x] `log_likelihood_ratio(term1, term2, docs) -> float`
  - [x] Tests: 12 tests

- [x] **6.6 Co-occurrence matrix**
  - [x] `build_cooccurrence_matrix(term_list, docs, method) -> Dict` — methods: count, pmi, llr
  - [x] `save_cooccurrence_matrix(matrix, path)` → CSV
  - [x] `get_top_cooccurrences()`
  - [x] Tests: 18 tests

---

## Phase 7: Relation Extraction ✅ COMPLETE

Note: Pattern-based implementation using NLTK POS tagging. spaCy dependency-parse extraction landed 2026-06-09 (`graph/dep_extractor.py`, used by `cmapr graph` when spaCy is available); the NLTK pattern chain remains as the no-extra fallback. spaCy requires Python ≤3.13 (no cp314 wheels) — project pinned accordingly.

- [x] **7.1 Parsing setup** (`src/concept_mapper/analysis/relations.py`)
  - [x] `parse_sentence(sentence: str) -> list[tuple[str, str]]`
  - [x] Tests: 2 tests

- [x] **7.2 SVO extraction**
  - [x] `SVOTriple` dataclass: subject, verb, object, sentence, doc_id
  - [x] `extract_svo(sentence, doc_id) -> list[SVOTriple]`
  - [x] `extract_svo_for_term(term, docs, case_sensitive) -> list[SVOTriple]`
  - [x] Tests: 8 tests

- [x] **7.3 Copular definitions**
  - [x] `CopularRelation` dataclass: subject, complement, copula, sentence, doc_id
  - [x] `extract_copular(term, docs, case_sensitive) -> list[CopularRelation]`
  - [x] Patterns: X {is|are|was|were|becomes|seems} Y
  - [x] Tests: 7 tests

- [x] **7.4 Prepositional relations**
  - [x] `PrepRelation` dataclass: head, prep, object, sentence, doc_id
  - [x] `extract_prepositional(term, docs, case_sensitive) -> list[PrepRelation]`
  - [x] Tests: 7 tests

- [x] **7.5 Relation aggregation**
  - [x] `Relation` dataclass: source, relation_type, target, evidence, metadata
  - [x] `get_relations(term, docs, types, case_sensitive) -> list[Relation]`
  - [x] Type filtering: ["svo", "copular", "prep"]
  - [x] Tests: 11 tests

---

## Phase 8: Graph Construction ✅ COMPLETE

- [x] **8.1 Graph data structure** (`src/concept_mapper/graph/model.py`)
  - [x] `ConceptGraph` class wrapping `nx.Graph` or `nx.DiGraph`
  - [x] Node attributes: label, frequency, pos, definition
  - [x] Edge attributes: weight, relation_type, evidence
  - [x] Tests: 19 tests

- [x] **8.2 Graph from co-occurrence** (`src/concept_mapper/graph/builders.py`)
  - [x] `graph_from_cooccurrence(matrix, threshold=0.0, directed=False) -> ConceptGraph`
  - [x] `graph_from_terms(terms, term_data=None) -> ConceptGraph`
  - [x] Tests: 12 tests

- [x] **8.3 Graph from relations**
  - [x] `graph_from_relations(relations, include_evidence=True) -> ConceptGraph`
  - [x] Directed edges labeled by relation type; evidence aggregation
  - [x] Tests: relation types preserved as edge labels

- [x] **8.4 Graph operations** (`src/concept_mapper/graph/operations.py`)
  - [x] `merge_graphs(g1, g2)`, `prune_edges(graph, min_weight)`, `prune_nodes(graph, min_degree)`
  - [x] `get_subgraph(graph, terms)`, `filter_by_relation_type(graph, relation_types)`
  - [x] Tests: 11 tests

- [x] **8.5 Graph metrics** (`src/concept_mapper/graph/metrics.py`)
  - [x] `centrality(graph, method="betweenness", normalized=True) -> dict[str, float]`
    - Methods: betweenness, degree, closeness, eigenvector, pagerank
  - [x] `detect_communities(graph) -> list[set[str]]`
  - [x] `assign_communities(graph, communities)`
  - [x] `get_connected_components(graph)`, `graph_density(graph)`, `get_shortest_path(graph, source, target)`
  - [x] Tests: 17 tests

---

## Phase 9: Export & Visualization ✅ COMPLETE

- [x] **9.1 D3 JSON export** (`src/concept_mapper/export/d3.py`)
  - [x] Schema: `{"nodes": [...], "links": [...]}`
  - [x] `export_d3_json(graph, path, include_evidence=False, size_by="degree", compute_communities=True, max_evidence=3)`
  - [x] Node size from centrality or frequency; node group from community detection
  - [x] Tests: 10 tests

- [x] **9.2 Alternative export formats** (`src/concept_mapper/export/formats.py`)
  - [x] `export_graphml(graph, path)` — for Gephi, yEd, Cytoscape
  - [x] `export_dot(graph, path)` — for Graphviz (requires pydot)
  - [x] `export_csv(graph, path)` → nodes.csv + edges.csv
  - [x] `export_gexf(graph, path)` — for Gephi
  - [x] Tests: 11 tests (2 skipped if pydot not installed)

- [x] **9.3 HTML visualization** (`src/concept_mapper/export/html.py`)
  - [x] `generate_html(graph, output_dir, title, width, height, include_evidence)`
  - [x] Standalone D3 force-directed graph; drag, zoom, pan, tooltips, community colors
  - [x] Tests: 6 tests

---

## Phase 10: CLI Interface ✅ COMPLETE

- [x] **10.1 CLI framework** (`src/concept_mapper/cli.py`)
  - [x] Click subcommand structure; main entry point: `cmapr`
  - [x] Global options: `--verbose`, `--output-dir`
  - [x] Tests: 23 comprehensive CLI tests

- [x] **10.2 Ingest command**
  - [x] `cmapr ingest <path> [--recursive] [--pattern "*.txt"] [--clean-ocr] --output corpus.json`
  - [x] Tests: 3 tests

- [x] **10.3 Rarities command**
  - [x] `cmapr rarities <corpus> --method tfidf --threshold 0.5 --output terms.json`
  - [x] Methods: ratio, tfidf, neologism, hybrid
  - [x] Tests: 3 tests

- [x] **10.4 Search commands**
  - [x] `cmapr search <corpus> --term "Begriff" --context 2`
  - [x] `cmapr concordance <corpus> --term "Begriff" --width 50`
  - [x] Tests: 7 tests

- [x] **10.5 Graph command**
  - [x] `cmapr graph <corpus> --terms terms.json --method cooccurrence --output graph.json`
  - [x] Methods: cooccurrence, relations
  - [x] Tests: 3 tests

- [x] **10.6 Export command**
  - [x] `cmapr export <graph> --format [d3|html|graphml|csv|gexf] --output <path>`
  - [x] Tests: 4 tests

- [x] **10.7 Replace command**
  - [x] `cmapr replace <corpus> --term "Begriff" --replacement "concept" [--preview]`
  - [x] Tests: 8 tests

- [x] **10.8 Analyze command**
  - [x] `cmapr analyze <corpus> "Begriff"`
  - [x] Integrated SVO + co-occurrence contextual analysis
  - [x] Tests included in contextual relations module

---

## Phase 11: Documentation & Polish ✅ COMPLETE

- [x] **11.1 README** — project overview, installation, quick start, links to docs
- [x] **11.2 Example workflow** — covered in README workflow section and `cmapr run` command
- [x] **11.3 API documentation** — `docs/api-reference.md`, docstrings, type hints throughout
- [x] **11.4 Tutorial** — merged into README
- [x] **11.5 Feature guides** — `docs/replacement.md` (synonym replacement)

---

## Notes for Development

Test each phase before proceeding; keep sample corpus small until Phase 4; curate terms before building graphs (garbage in → garbage out); add CLI subcommands incrementally; "rarity" means corpus-comparative (statistically improbable in general English), not merely infrequent within the primary text.

## Refactoring Strategy from Spike

Refactoring prioritized (1) extracting working spike implementations (tokenization, frequency, search), then (2) adding missing infrastructure (pytest, data structures, pipeline), then (3) implementing new functionality with no spike equivalent (graph construction, D3 export, CLI).

---

## Architecture

```
src/concept_mapper/
├── corpus/           # Document loading and models
├── preprocessing/    # Tokenization, POS tagging, lemmatization, cleaning, structure
├── analysis/         # Frequency, rarity, co-occurrence, relations, contextual
├── terms/            # Term list management
├── search/           # Search, concordance, context, dispersion, extraction
├── graph/            # Graph construction, operations, metrics
├── export/           # D3 JSON, GraphML, GEXF, DOT, CSV, HTML
├── transformations/  # Synonym replacement with inflection
├── storage/          # Storage abstraction and utilities
└── cli.py            # Click CLI (cmapr command)
```

**Dependencies:** NLTK (tokenization, POS, lemmas, WordNet), NetworkX (graphs), Click (CLI), pdfplumber (PDF input).

---

## Known Limitations

1. **English only** — NLTK resources are English-centric.
2. **Pattern-based relations (fallback only)** — spaCy dependency-parse extraction is now the default engine when the `[spacy]` extra + `en_core_web_sm` are installed; without them, the regex/POS chain is used. Note spaCy has no Python 3.14 wheels — the project is pinned to 3.13 (`.python-version`).
3. **Scale** — optimized for academic texts (10–100 documents), not massive corpora.
4. **Graph layout** — force-directed only.

---

## Manual QA tracker

Each `docs/qa/*.md` is an end-to-end manual walk-through against the live code. Tick the checkbox once the doc has been run on a representative corpus; re-open after structural changes.

| Verified | QA doc | Covers |
|---|---|---|
| [ ] | [`qa/graph.md`](qa/graph.md) | ingest → rarities → graph → export pipeline |
| [ ] | [`qa/cluster.md`](qa/cluster.md) | `cmapr cluster` — per-chapter sub-graphs + recurrence edges |
| [ ] | `qa/neural-similarity.md` *(planned, st.8)* | sentence-transformers `--neural` mode |
| — | `qa/neural-relations.md` *(deferred with REBEL, 2026-06-09)* | REBEL `--neural` mode |

Add a row when a new `docs/qa/*.md` is created. When a doc is run successfully, tick it; if a subsequent refactor invalidates the verification, un-tick.

---

## Future Work


- [x] **Implement concept graph** — see `docs/spec-graph.md` for full specification. Covers three layers:
  - [x] manually add extract source text for faster dev.
  - **Data extraction**: node filtering (POS, length, fragment, OCR artifact checks), typed directed proposition edges (definition, kind-of, property, opposition, production, dependence), co-occurrence as candidate-discovery mechanism not edge type, multi-word term support
  - **Graph workflows**: (A) threshold-driven from rarities list; (B) seed-word-driven with options for POS filter, count limit, section grouping (separate subgraphs per section), and depth limit (same-sentence or same-paragraph window); `graph` = batch `analyze` + merge
  - **Visualization**: tuned D3 force params, directed edges, typed edge labels, section subgraph navigation, interactive node expand/collapse, node detail panel (definition, frequency, location)
- [ ] **Structured ingestion pipeline** — a cleaner, isolated ingestion interface that produces richly labeled documents. Covers: (1) investigate whether an existing package (`docling`, `pymupdf4llm`, `unstructured`, `nougat`, `pdfplumber`) can replace the current two-file workflow (raw OCR text + manually cleaned TOC) by extracting structured text, TOC, and layout directly from PDF; (2) classify front-matter (title page, copyright, TOC, introduction) and back-matter (bibliography, references, index, appendix) sections by heuristic or model; (3) strip running headers and page numbers per page; (4) detect and label document structure (parts, chapters, sections) and paragraph boundaries automatically.
- [x] **Usage-based composed definitions** (2026-06-09) — extends the **Guaranteed node definitions** feature (`docs/plans/node-definitions.md`): each node now also gets a `composed_definition` assembled deterministically from its typed edges by template ("Semiosis — a kind of process; produced by sign; opposed to stasis.") plus a `composed_parts` scaffold carrying **all 8 relation types** (definition, kind-of, production, dependence, property, relation, component, opposition), with `None`/"—" for parts that have no extracted relation — absence is stated, never fabricated. This is old-school *generation* compatible with the generative-extraction policy: the sentence is new, but every clause is a template slot filled from an extracted edge, so nothing can be asserted that the graph doesn't evidence — no model, no hallucination. Coverage is guaranteed: nodes whose only edges are cooccurrence fall back to a "co-occurs with …" association clause (typed-assertion-free). Includes the **multi-type builder fix**: `build_proposition_graph` previously kept only the priority-winning proposition type per pair, silently discarding other extracted types — it now keeps them all on one edge via the additive `relation_types` schema (same as `cmapr merge`), which is the main feedstock for the part scaffold. `analysis/definitions.py:compose_definitions` / `compose_definition_parts`, runs by default next to `derive_definitions` in `cmapr graph`/`run`/serve; threaded through D3 export; the Relations drawer renders the part scaffold (with "—" rows) above the verbatim sentence labeled "Example:". Tests in `tests/test_definitions.py` + `tests/test_graph.py` (multi-type regression).
- [ ] **Test suite cleanup** — use `pytest-cov` to identify undercovered code paths (add tests) and overcovered ones (redundant tests testing the same path with different labels); target reducing test count from ~718 to ~510-540 (~25% reduction) while improving branch coverage; delete redundant tests rather than merging them to keep intent legible.
- [ ] **QA coverage & product mapping** — broaden manual-testing surface and document the product's user-story space. Two sub-deliverables:
  - [ ] **Manual QA text samples** — create fixture texts at four scale levels for end-to-end manual testing. Land under `data/sample/` (or similar) and cite from `docs/qa/*.md` walk-throughs.
    - sentence (single sentence)
    - paragraph (multi-sentence)
    - chapter (multi-paragraph)
    - book (multi-paragraph chapters)
  - [ ] **Product-level user-story diagram** — Mermaid graph (new `docs/product.md` or similar) enumerating the user-story dimensions cmapr must cover. Used to identify combinations not yet exercised by tests or QA samples.
    - **source text location:** in-project (`data/`) · external filepath
    - **source text file type:** `.txt` · `.pdf`
    - **source text file content:** messy (OCR artifacts) · clean
    - **source text guides:** with TOC · with index · neither
    - **source text structure:** parts · sections · chapters · paragraphs · (etc.)
- [x] **Library survey** — pipeline-keyed catalog of NLP / extraction / KG / viz libraries in `docs/survey.md`. Confirmed currently-integrated (per the survey's "Already integrated" verdicts): NLTK, NetworkX, pdfplumber, stanza, spaCy (`[spacy]` extra). Top-pick adoption tracked under the unticked item below.
- [x] **Graph merge command** — `cmapr merge ch1.json ch2.json [...] -o merged.json` aggregates per-chapter graphs. Frequencies sum; rarity scores are frequency-weighted means; same-pair-different-type edges keep both types via an additive multi-type schema (`relation_types`, `weight_by_type`, `evidence_by_type`, `verb_by_type`). HTML viz renders the per-type breakdown in tooltips and the detail panel. `aggregate_graphs()` in `graph/operations.py` is the underlying primitive; the legacy naïve `merge_graphs()` is kept as-is.
- [x] **Multi-chapter clustered visualization** (v1, option A) — `cmapr cluster CORPUS -t TERMS -o OUT [--by chapter|section]` builds per-chapter sub-graphs from `sentence_locations`, namespaces nodes as `<term>__<chapter>`, and adds `recurrence` edges between consecutive same-term occurrences (weight = span). HTML viz auto-detects cluster membership and adds a D3 cluster force that pulls each node toward its chapter's centroid; legend toggle hides recurrence edges. See `docs/plans/multi-chapter.md`. **Display options B (constellation) and C (timeline spine) remain deferred.**
- [x] **Session status doc** — resolved by the **Status** block at the top of this roadmap, which already records *last completed*, *next up*, *open issues*, and *tests* for cold re-entry. No separate `docs/status.md` needed.
- [x] **Pipeline architecture diagram** — `docs/architecture.md`: Mermaid flowchart with all 10 CLI commands and the processing modules grouped by stage (`corpus/`, `preprocessing/`, `analysis/`, `terms/`, `graph/`, `export/`, plus auxiliary `search/`, `transformations/`, `syntax/`, `server/`). README points at it.
- [ ] Database backend for large-scale corpora
- [ ] Temporal analysis across an author's career
- [ ] Web interface
- [ ] **SpaCy integration** (requires Python ≤3.13 — no cp314 wheels; project pinned via `.python-version`). **The preferred relation-extraction path (2026-06-09)**, replacing the deferred REBEL initiative: dependency parsing is deterministic, extractive, and span-grounded — no hallucination risk. **Landed so far:** noun chunks (`cmapr ingest --spacy`), and the **dependency-parse extraction engine** (`graph/dep_extractor.py`, `cmapr graph --engine`, default-on when spaCy is available — sentence-centric, handles passives/copulars/appositives/coordination, mines single-seed sentences). **Still open:** the full base-layer migration — replacing the NLTK `tokenize → tag → lemmatize` preprocessing pipeline with one spaCy pass (neural POS for neologisms, spaCy lemmatizer, NER), and retiring the regex chain once the dep engine has been QA'd on a real corpus (eco_spl1 before/after comparison).
  - *Upsides:* dependency parsing gives correct SVO assignments in complex sentences (passives, relative clauses, unusual word order — common in translated philosophical texts); neural POS models generalize better to out-of-vocabulary philosophical neologisms (*Dasein*, *différance*, *sublation*); noun-chunk spans handle multi-word terms as units rather than fragmenting them; NER distinguishes technical concept usage from proper-noun references
  - *Downsides:* heavy dependency with separately downloaded models (`python -m spacy download en_core_web_sm`); slower processing; requires refactoring the `tokenize → tag → lemmatize` pipeline that threads through most of the codebase — not a drop-in swap; highest-value target is relation extraction, which would benefit most from the dependency tree
- [ ] Multi-language support

---

## maybe won't do features.
- [ ] **Adopt survey top-picks** — work through the candidates flagged **Augment** / **Investigate** in `docs/survey.md` in roughly the recommended order: 
  - (1) sentence-transformers — tracked as st.1–st.8 below; 
  - (2) spaCy dependency parsing as relation-extraction base layer — see the SpaCy integration entry below; 
  - (3) GLiNER (zero-shot term typing on rarities); 
  - (4) KeyBERT (6th rarity signal); 
  - (5) ConceptNet lookup helper; 
  - (6) PyKEEN link prediction (defer until extraction quality peaks). 
  - Items (3)–(6) get their own roadmap blocks + plan files when they reach the top of the queue. *REBEL (rb.1–rb.7) and the LangExtract evaluation are deferred per the 2026-06-09 generative-extraction policy (see Status).*
- [ ] **sentence-transformers integration** — semantic embeddings for terms and sentences. See `docs/survey.md` § Stage 3 for rationale. Plan file: `docs/plans/neural-similarity.md` (to be drafted before code).
  - [ ] **st.1 `[neural]` optional dependency** — add `sentence-transformers>=3.0` (and the implicit `torch` + `transformers` transitive deps) under `[project.optional-dependencies]` in `pyproject.toml`. Mirror the existing `[spacy]` / `[serve]` / `[wizard]` pattern.
  - [ ] **st.2 Embedding plumbing module** — new `analysis/embeddings.py`: lazy-load `all-MiniLM-L6-v2` (or configurable), `embed_terms(terms) -> dict[str, np.ndarray]`, `embed_sentences(sentences) -> ndarray`, `cosine_similarity(a, b) -> float`. Disk-cache embeddings per corpus identifier under `data/output/cache/embeddings/<work>/` so re-runs skip re-embedding. Smallest meaningful entry point.
  - [ ] **st.3 `cmapr similar TERM [--top-n N]` command** — uses st.2 to find k-nearest terms by cosine in an existing rarities list. ~100 lines + tests. Shakes out the embedding plumbing end-to-end; useful on its own for vetting / seed discovery.
  - [ ] **st.4 Evidence-ranking augmentation** — extend `graph/proposition_extractor.py:_score_sentence` with a neural component (cosine between the candidate sentence and an `"X relates to Y"` probe). Gate behind `cmapr graph --neural`; pure-regex behaviour preserved by default. Catches good evidence that lacks surface markers.
  - [ ] **st.5 Semantic-similarity edges** — augment the PMI cooccurrence fallback in `graph/builders.py:build_proposition_graph` with semantic-similarity edges (new edge type `similarity`). Two terms with cosine ≥ threshold get linked even if they never co-occur in one sentence. Add to `EDGE_COLORS` / `LEGEND_TYPES` in `export/html.py`. Massively improves graph connectivity for related-but-distal concepts.
  - [ ] **st.6 Semantic term clustering** — alternative to Louvain in `graph/metrics.py:detect_communities`: k-means / HDBSCAN over term embeddings. Different (often more interpretable) partition. Expose as `--communities {louvain|semantic}` on `cmapr graph` or `cmapr export`.
  - [ ] **st.7 Spec/plan file** — `docs/plans/neural-similarity.md` with sub-task checklist mirroring this list, dev-loop snippet, pointer from roadmap Status.
  - [ ] **st.8 QA + tests** — `docs/qa/neural-similarity.md`; unit tests for each new function; CLI tests for `cmapr similar` + `cmapr graph --neural`; manual visual check that semantic-similarity edges render correctly.

- **~~REBEL integration~~ — DEFERRED (2026-06-09).** Neural relation extraction, postponed indefinitely per the generative-extraction policy: REBEL generates triples rather than extracting them, so it can hallucinate relations and drift from verbatim source spans — incompatible with cmapr's span-grounded evidence model. Preferred alternative: spaCy dependency parsing (see SpaCy integration entry). Sub-items rb.1–rb.7 retained below for the record; do not start without revisiting the policy. See `docs/survey.md` § "Technology generations & determinism".
  - [ ] **rb.1 Add REBEL to `[neural]` extra** — `transformers` already a transitive dep from sentence-transformers; add an explicit pin if needed. Model: `Babelscape/rebel-large` (~500MB; user downloads on first run).
  - [ ] **rb.2 REBEL wrapper module** — new `analysis/rebel.py`: lazy-load the model, `extract_triples(sentence) -> list[(subject, relation, object, score)]`. Wraps the seq2seq generation + Wikidata-relation parsing. Disk-cache per (sentence_hash) so re-runs of the same corpus skip re-inference.
  - [ ] **rb.3 Wikidata → cmapr type mapping** — table mapping REBEL's ~250 Wikidata relation labels (`subclass_of`, `instance_of`, `has_part`, `cause`, etc.) to cmapr's 9 types (`definition`, `kind-of`, `production`, `dependence`, `component`, `opposition`, `property`, `relation`, fallback). Unmapped relations land as generic `relation`. Document the mapping in `docs/plans/neural-relations.md` so it's auditable.
  - [ ] **rb.4 Integrate into PropositionExtractor** — new `_try_rebel` method in `graph/proposition_extractor.py`, runs *before* the regex chain when `--neural` is on. Or a parallel batch pass that merges with regex output (dedup on `(source, target, type)`, prefer higher-confidence proposition). Confidence threshold configurable.
  - [ ] **rb.5 `cmapr graph --neural` flag** — combines st.4 (evidence ranking) and rb.4 (relation extraction) under one user-facing toggle. Pure-regex behaviour preserved without the flag.
  - [ ] **rb.6 Spec/plan file** — `docs/plans/neural-relations.md`.
  - [ ] **rb.7 QA + tests** — `docs/qa/neural-relations.md`; per-relation-type unit tests (plant sentence → expect mapped cmapr type); CLI test for `cmapr graph --neural`; before/after comparison on `eco_spl1` to quantify typed-edge yield uplift.

---

## Acronym Reference

| Acronym | Meaning |
|---------|---------|
| POS | Part of Speech |
| TF-IDF | Term Frequency–Inverse Document Frequency |
| KWIC | Key Word In Context |
| PMI | Pointwise Mutual Information |
| LLR | Log-Likelihood Ratio |
| SVO | Subject-Verb-Object |
| CLI | Command Line Interface |

## References

- Lane 2019, *Natural Language Processing in Action*
- Rockwell & Sinclair 2016, *Hermeneutica*
- Moretti, *Graphs, Maps, Trees*
