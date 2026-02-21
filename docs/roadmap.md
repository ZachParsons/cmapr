# Roadmap

**Past:** completed phases, significant additions/pivots.
**Present:** planned work, WIP.
**Future:** unplanned features, ideas.

---

## Concept Mapper

A tool for extracting and visualizing an author's idiosyncratic conceptual vocabulary from primary texts. Identifies neologisms and terms with specialized technical meaning that are statistically distinctive compared to general English corpora. Maps concepts through co-occurrence and grammatical relations, exporting as D3 visualizations.

**Examples of target terms:** Aristotle's *eudaimonia*, Spinoza's *affect*, Hegel's *sublation*, Dennett's *intentional stance*, Deleuze & Guattari's *body without organs*.

---

## Status

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

**665 tests passing, 2 skipped** (pydot-dependent DOT export tests).

### Post-completion additions (February 2026)

- [x] **OCR text cleaning** — `preprocessing/cleaning.py`, `--clean-ocr` flag. 21 tests.
- [x] **PDF input support** — `load_pdf()` via pdfplumber, auto-detected in `load_file()`. 6 tests.
- [x] **Paragraph segmentation** — `preprocessing/segment.py`, paragraph boundary detection. 21 tests.
- [x] **Synonym replacement** — inflection-preserving term replacement. `transformations/`, `cmapr replace`. 59 tests. See `docs/replacement.md`.
- [x] **Contextual relation extraction** — integrated SVO + co-occurrence workflow. `analysis/contextual_relations.py`, `cmapr analyze`. 38 tests.
- [x] **Remove legacy pos_tagger.py** — Deleted 490 lines of legacy code. Updated Makefile.
- [x] **Documentation consolidation** — README reduced 68% (764→241 lines). Created `docs/tutorial.md`.
- [x] **Infrastructure cleanup** — Removed 77MB of old venv, cache files, and unused directories.

### Project summary

11/11 phases complete (Jan 14–25, 2026, 12 days); 8,788 lines across 35 modules; 665 tests; ~107KB documentation across 5 guides; dual CLI/Python API covering the full pipeline from text loading through interactive D3 visualization — production-ready for digital humanities research.

---

## Existing Spike Implementations

Working implementations of tokenization, POS tagging, lemmatization, frequency analysis, and sentence search existed in `spike/tryout_nltk.py` and `spike/pos_tagger.py` and were refactored into the main codebase (Phases 1, 2, 5); graph construction, D3 export, and the CLI had no spike equivalents.

---

## Phase 0: Project Scaffolding ✅ COMPLETE

- [x] **0.1 Initialize project structure**
  - [x] Create directory layout: `src/concept_mapper/`, `tests/`, `data/sample/`, `output/`
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
  - [x] Cache to disk after first computation (`output/cache/brown_corpus_freqs.json`)
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

Note: Pattern-based implementation using NLTK POS tagging. spaCy dependency parsing deferred due to Python 3.14 compatibility.

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
  - [x] `cmapr analyze <corpus> --term "Begriff"`
  - [x] Integrated SVO + co-occurrence contextual analysis
  - [x] Tests included in contextual relations module

---

## Phase 11: Documentation & Polish ✅ COMPLETE

- [x] **11.1 README** — project overview, installation, quick start, links to docs
- [x] **11.2 Example workflow** — `examples/workflow.sh` (CLI), `examples/workflow.py` (Python API)
- [x] **11.3 API documentation** — `docs/api-reference.md`, docstrings, type hints throughout
- [x] **11.4 Tutorial** — `docs/tutorial.md`, step-by-step workflow guide
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
2. **Pattern-based relations** — spaCy dependency parsing deferred (Python 3.14 incompatibility); NLTK pattern-matching works well for philosophical texts.
3. **Scale** — optimized for academic texts (10–100 documents), not massive corpora.
4. **Graph layout** — force-directed only.

---

## Future Work

- SpaCy integration when Python 3.14 compatible (richer dependency parsing)
- Multi-language support
- Usage-based definition generation (aggregate co-occurrences and relations into empirical definitions)
- Automatic document structure discovery (chapter/section segmentation for large texts)
- Temporal analysis across an author's career
- Web interface
- Database backend for large-scale corpora

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
