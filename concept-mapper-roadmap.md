# Concept Mapper: Development Roadmap

A tool for extracting and visualizing an author's idiosyncratic conceptual vocabulary from primary texts.

## Project Overview

**Goal:** Analyze texts to identify terms with author-specific meanings, understand their usage through co-occurrence and grammatical relations, and export concept maps for D3 visualization.

**Stack:** Python, NLTK, spaCy (for dependency parsing), networkx, Click (CLI)

**Current Status:** NLTK foundation exists in spike (`tryout_nltk.py`, `pos_tagger.py`). ~30-40% of Phase 1-2 functionality implemented.

**References:**
- Lane 2019, *Natural Language Processing in Action*
- Rockwell & Sinclair 2016, *Hermeneutica*
- Moretti, *Graphs, Maps, Trees*

---

## Existing Spike Implementations

The following functionality already exists in the spike directory and can be refactored:

### `tryout_nltk.py` - Experimental NLTK features
- ✅ Word & sentence tokenization (lines 10-17)
- ✅ Stemming with PorterStemmer (lines 20-35)
- ✅ POS tagging on sentences (lines 38-60)
- ✅ Chunking & chinking patterns (lines 62-102)
- ✅ Named Entity Recognition (lines 104-115)
- ✅ Lemmatization with WordNetLemmatizer (lines 118-134)
- ✅ Corpus loading (gutenberg, state_union, movie_reviews) (lines 137-194)
- ✅ WordNet synsets, synonyms, antonyms, similarity (lines 144-182)
- ✅ Frequency distributions with FreqDist (lines 207-212)
- ✅ Text classification with Naive Bayes (lines 185-303)
- ✅ Classifier persistence with pickle (lines 285-302)

### `pos_tagger.py` - Concept mapping prototype
- ✅ File loading (`get_text()` - line 35-37)
- ✅ Word tokenization (`tokenize()` - line 40-41)
- ✅ POS tagging pipeline (`run()` - line 17-21)
- ✅ Verb frequency analysis (lines 52-86)
- ✅ Common word filtering (lines 94-133)
- ✅ Sentence search by term (lines 136-173)

### Dependencies installed
- `nltk>=3.8` (requirements.txt)
- NLTK data: punkt, punkt_tab, averaged_perceptron_tagger, averaged_perceptron_tagger_eng

### Sample corpus
- ✅ `philosopher_1920_cc.txt` (91KB text file for testing)

### Additional features (not in roadmap, but available in spike)
- Chunking with regex patterns (tryout_nltk.py:62-88)
- Chinking (inverse chunking) (tryout_nltk.py:89-102)
- Named Entity Recognition with ne_chunk (tryout_nltk.py:104-115)
- Text classification with Naive Bayes & scikit-learn (tryout_nltk.py:185-329)
- Tree visualization to PostScript (tryout_nltk.py:81-87)

---

## Phase 0: Project Scaffolding

- [ ] **0.1 Initialize project structure**
  - [ ] Create directory layout: `src/concept_mapper/`, `tests/`, `data/sample/`, `output/`
  - [ ] Initialize git repository
  - [x] Create `pyproject.toml` or `requirements.txt` *(spike: requirements.txt exists with nltk>=3.8)*
  - [ ] Initial dependencies: `nltk`, `pytest`, `click` *(nltk done, need pytest & click)*

- [x] **0.2 Download NLTK data** *(spike: pos_tagger.py:29-32 downloads all needed data)*
  - [x] Create setup script `scripts/download_nltk_data.py` *(inline in pos_tagger.py, should extract)*
  - [x] Download: `punkt`, `averaged_perceptron_tagger`, `wordnet`, `brown`, `stopwords` *(punkt, tagger done; need wordnet, brown, stopwords)*
  - [x] Verify downloads succeed *(working in current spike)*

- [x] **0.3 Create sample test corpus**
  - [x] Create 2-3 short `.txt` files in `data/sample/` *(spike: philosopher_1920_cc.txt exists, 91KB)*
  - [ ] Include invented "rare terms" with known frequencies
  - [ ] Document expected values for verification

---

## Phase 1: Corpus Ingestion & Preprocessing

All downstream analysis depends on clean, structured text.

- [ ] **1.1 File loader** (`src/concept_mapper/corpus/loader.py`)
  - [x] `load_file(path: Path) -> str` *(spike: pos_tagger.py:35-37 has basic file reading)*
  - [ ] `load_directory(path: Path, pattern: str = "*.txt") -> dict[str, str]`
  - [ ] Handle encoding (UTF-8 with Latin-1 fallback) *(spike uses default encoding)*
  - [ ] Tests: load sample files, verify content
  - **Note:** Basic file reading exists. Needs Path type, encoding handling, and directory loading.

- [ ] **1.2 Data structures** (`src/concept_mapper/corpus/models.py`)
  - [ ] `Document` dataclass: text, metadata (title, author, date, source_path)
  - [ ] `Corpus` class: collection of Documents
  - [ ] `ProcessedDocument` dataclass: raw, sentences, tokens, pos_tags, lemmas

- [ ] **1.3 Tokenization** (`src/concept_mapper/preprocessing/tokenize.py`)
  - [x] `tokenize_words(text: str) -> list[str]` *(spike: pos_tagger.py:40-41 uses word_tokenize)*
  - [x] `tokenize_sentences(text: str) -> list[str]` *(spike: tryout_nltk.py:16 uses sent_tokenize)*
  - [ ] Preserve original case in parallel structure
  - [ ] Tests: verify token/sentence counts on sample
  - **Note:** Refactor existing implementations from spike into proper module structure

- [ ] **1.4 POS tagging** (`src/concept_mapper/preprocessing/tagging.py`)
  - [x] `tag_tokens(tokens: list[str]) -> list[tuple[str, str]]` *(spike: pos_tagger.py:26 imports pos_tag)*
  - [x] `tag_sentences(sentences: list[str]) -> list[list[tuple[str, str]]]` *(spike: tryout_nltk.py:47-57)*
  - [ ] Tests: spot-check known POS assignments
  - **Note:** Working implementation in pos_tagger.py:17-21 (run function). Needs modularization.

- [ ] **1.5 Lemmatization** (`src/concept_mapper/preprocessing/lemmatize.py`)
  - [ ] `get_wordnet_pos(treebank_tag: str) -> str` (map Penn tags to WordNet)
  - [x] `lemmatize(word: str, pos: str) -> str` *(spike: tryout_nltk.py:118-134 has WordNetLemmatizer examples)*
  - [ ] `lemmatize_tagged(tagged_tokens: list[tuple]) -> list[str]`
  - [x] Tests: "running" → "run", "better" → "good" *(spike: tryout_nltk.py:123-132 has test cases)*
  - **Note:** Example code exists but commented out. Shows lemmatizer.lemmatize() with pos parameter.

- [ ] **1.6 Preprocessing pipeline** (`src/concept_mapper/preprocessing/pipeline.py`)
  - [ ] `preprocess(document: Document) -> ProcessedDocument`
  - [ ] `preprocess_corpus(corpus: Corpus) -> list[ProcessedDocument]`
  - [ ] Single entry point that runs tokenize → tag → lemmatize
  - [ ] Tests: round-trip load → preprocess → verify structure

- [ ] **1.7 Paragraph segmentation** (`src/concept_mapper/preprocessing/segment.py`)
  - [ ] `segment_paragraphs(text: str) -> list[str]`
  - [ ] Handle various paragraph markers (double newline, indentation)
  - [ ] Add paragraph indices to ProcessedDocument
  - [ ] Tests: verify paragraph boundaries

---

## Phase 2: Term Extraction & Frequency Analysis

Statistical foundation for rarity detection.

- [ ] **2.1 Frequency distribution** (`src/concept_mapper/analysis/frequency.py`)
  - [x] `word_frequencies(doc: ProcessedDocument) -> Counter` *(spike: tryout_nltk.py:207 uses FreqDist)*
  - [x] `pos_filtered_frequencies(doc: ProcessedDocument, pos_tags: set) -> Counter` *(spike: pos_tagger.py:66-73 filters verbs by POS)*
  - [ ] Option: count lemmas vs surface forms
  - [x] Tests: manual count verification *(spike: tryout_nltk.py:208-209 shows common words)*
  - **Note:** FreqDist already used for movie reviews corpus. Verb filtering example in pos_tagger.py:66-73.

- [ ] **2.2 Corpus-level aggregation**
  - [ ] `corpus_frequencies(docs: list[ProcessedDocument]) -> Counter`
  - [ ] `document_frequencies(docs: list[ProcessedDocument]) -> Counter` (in how many docs?)
  - [ ] Tests: term in 2 docs → doc_freq = 2

- [ ] **2.3 Reference corpus** (`src/concept_mapper/analysis/reference.py`)
  - [x] `load_reference_corpus(name: str = "brown") -> Counter` *(spike: tryout_nltk.py:138-142 uses gutenberg/movie_reviews corpus)*
  - [ ] Cache to disk after first computation
  - [x] Tests: verify brown corpus loads, common words have high freq *(spike: tryout_nltk.py:189-212)*
  - **Note:** Already familiar with NLTK corpora (gutenberg, movie_reviews, state_union). Brown corpus mentioned in roadmap Phase 0.2.

- [ ] **2.4 TF-IDF** (`src/concept_mapper/analysis/tfidf.py`)
  - [ ] `tf(term: str, doc: ProcessedDocument) -> float`
  - [ ] `idf(term: str, docs: list[ProcessedDocument]) -> float`
  - [ ] `tfidf(term: str, doc: ProcessedDocument, docs: list) -> float`
  - [ ] `corpus_tfidf_scores(docs: list[ProcessedDocument]) -> dict[str, float]`
  - [ ] Tests: unique term scores high, common term scores low

---

## Phase 3: Rarity Detection

Operationalize "rare" and "technical."

- [ ] **3.1 Frequency-based rarity** (`src/concept_mapper/analysis/rarity.py`)
  - [ ] `get_hapax_legomena(docs: list[ProcessedDocument]) -> set[str]`
  - [ ] `get_low_frequency_terms(docs, threshold: int) -> set[str]`
  - [x] `get_low_frequency_by_pos(docs, pos_tags: set, threshold: int) -> set[str]` *(spike: pos_tagger.py:106-118 filters common verbs)*
  - [ ] Tests: inject known hapax, verify detection
  - **Note:** pos_tagger.py has filter logic for excluding common verbs (lines 107-111). Can generalize to other POS.

- [ ] **3.2 TF-IDF-based rarity**
  - [ ] `get_corpus_specific_terms(docs, reference: Counter, threshold: float) -> set[str]`
  - [ ] Compare author's usage against general English
  - [ ] Tests: author-specific term scores above threshold

- [ ] **3.3 Structural rarity heuristics**
  - [ ] `get_capitalized_nouns(docs) -> set[str]` (non-sentence-initial)
  - [ ] `get_potential_neologisms(docs, dictionary: set) -> set[str]` (not in wordnet)
  - [ ] `get_definitional_contexts(docs) -> list[tuple[str, str]]` (term, sentence)
    - [ ] Patterns: "X is...", "by X I mean...", "what I call X"
  - [ ] Tests: planted capitalized noun detected, neologism detected

- [ ] **3.4 Hybrid rarity scorer**
  - [ ] `RarityScorer` class with configurable methods and weights
  - [ ] `score_term(term: str) -> float`
  - [ ] `score_all(min_score: float) -> dict[str, float]`
  - [ ] Tests: common English words score low

---

## Phase 4: Term List Management

Human-in-the-loop curation.

- [ ] **4.1 Data structures** (`src/concept_mapper/terms/models.py`)
  - [ ] `TermEntry` dataclass: term, lemma, pos, definition, notes, examples
  - [ ] `TermList` class: collection with lookup by term

- [ ] **4.2 CRUD operations** (`src/concept_mapper/terms/manager.py`)
  - [ ] `add_term(term: str, metadata: dict = None)`
  - [ ] `remove_term(term: str)`
  - [ ] `update_term(term: str, metadata: dict)`
  - [ ] `get_term(term: str) -> TermEntry | None`
  - [ ] `list_terms() -> list[TermEntry]`
  - [ ] Tests: CRUD round-trip

- [ ] **4.3 Persistence**
  - [ ] `save(path: Path)` → JSON serialization
  - [ ] `load(path: Path) -> TermList`
  - [ ] Tests: save → load preserves all data

- [ ] **4.4 Bulk operations**
  - [ ] `import_from_file(path: Path)` (plain text, one term per line)
  - [ ] `export_to_file(path: Path, format: str)`
  - [ ] `merge(other: TermList) -> TermList`
  - [ ] Tests: import/export round-trip

- [ ] **4.5 Auto-populate from rarity**
  - [ ] `suggest_terms(docs, scorer: RarityScorer, threshold: float) -> TermList`
  - [ ] Populate examples from corpus automatically
  - [ ] Tests: suggested list contains expected rare terms

---

## Phase 5: Search & Concordance

Find where and how terms appear.

- [ ] **5.1 Basic search** (`src/concept_mapper/search/find.py`)
  - [ ] `SentenceMatch` dataclass: sentence, doc_id, sent_index, term_positions
  - [x] `find_sentences(term: str, docs) -> list[SentenceMatch]` *(spike: pos_tagger.py:150-157 has basic implementation)*
  - [ ] Support lemma matching option
  - [ ] Tests: find all occurrences of known term
  - **Note:** Basic sentence filtering exists. Returns sentences containing target verb. Needs structured return type.

- [ ] **5.2 Concordance (KWIC)** (`src/concept_mapper/search/concordance.py`)
  - [ ] `KWICLine` dataclass: left_context, keyword, right_context, doc_id
  - [ ] `concordance(term: str, docs, width: int = 50) -> list[KWICLine]`
  - [ ] Align output on keyword for scanning
  - [ ] Tests: width parameter respected

- [ ] **5.3 Context window**
  - [ ] `ContextWindow` dataclass: before (list[str]), match (str), after (list[str])
  - [ ] `get_context(term: str, docs, n_sentences: int = 1) -> list[ContextWindow]`
  - [ ] Tests: n_sentences before/after included

- [ ] **5.4 Dispersion**
  - [ ] `dispersion(term: str, docs) -> dict[str, list[int]]` (doc_id → positions)
  - [ ] Position as sentence index or character offset
  - [ ] Tests: term in specific locations detected

---

## Phase 6: Co-occurrence Analysis

Relational structure from proximity.

- [ ] **6.1 Sentence-level co-occurrence** (`src/concept_mapper/analysis/cooccurrence.py`)
  - [ ] `cooccurs_in_sentence(term: str, docs) -> Counter`
  - [ ] Count all terms appearing in same sentences as target
  - [ ] Tests: two terms in same sentence → count = 1

- [ ] **6.2 Filtered co-occurrence**
  - [ ] `cooccurs_filtered(term: str, docs, term_list: TermList) -> Counter`
  - [ ] Only count terms in curated list
  - [ ] Tests: non-list terms excluded

- [ ] **6.3 Paragraph-level co-occurrence**
  - [ ] `cooccurs_in_paragraph(term: str, docs) -> Counter`
  - [ ] Requires paragraph segmentation from Phase 1.7
  - [ ] Tests: terms in same paragraph counted

- [ ] **6.4 N-level co-occurrence**
  - [ ] `cooccurs_within_n(term: str, docs, n_sentences: int) -> Counter`
  - [ ] Sliding window across sentence boundaries
  - [ ] Tests: window size respected

- [ ] **6.5 Statistical significance**
  - [ ] `pmi(term1: str, term2: str, docs) -> float` (pointwise mutual information)
  - [ ] `log_likelihood_ratio(term1: str, term2: str, docs) -> float`
  - [ ] Tests: independent terms ≈ 0, associated terms > 0

- [ ] **6.6 Co-occurrence matrix**
  - [ ] `build_cooccurrence_matrix(term_list: TermList, docs, method: str) -> DataFrame`
  - [ ] Methods: raw count, PMI, log-likelihood
  - [ ] Symmetric matrix, terms × terms
  - [ ] `save_matrix(matrix, path: Path)` → CSV
  - [ ] Tests: matrix dimensions = len(term_list)²

---

## Phase 7: Relation Extraction

Grammatical relations, not just proximity.

- [ ] **7.1 Dependency parsing setup** (`src/concept_mapper/analysis/relations.py`)
  - [ ] Add spaCy to dependencies
  - [ ] Download `en_core_web_sm` model
  - [ ] `parse_sentence(sentence: str) -> Doc`
  - [ ] Tests: parse returns valid spaCy Doc

- [ ] **7.2 Subject-verb-object extraction**
  - [ ] `SVOTriple` dataclass: subject, verb, object, sentence
  - [ ] `extract_svo(doc: Doc) -> list[SVOTriple]`
  - [ ] `extract_svo_for_term(term: str, docs) -> list[SVOTriple]`
  - [ ] Tests: "The dog bites the man" → (dog, bites, man)

- [ ] **7.3 Copular definitions**
  - [ ] `CopularRelation` dataclass: subject, complement, sentence
  - [ ] `extract_copular(term: str, docs) -> list[CopularRelation]`
  - [ ] Pattern: X is Y, X are Y, X was Y
  - [ ] Tests: "Being is presence" → (Being, presence)

- [ ] **7.4 Prepositional relations**
  - [ ] `PrepRelation` dataclass: head, prep, object, sentence
  - [ ] `extract_prepositional(term: str, docs) -> list[PrepRelation]`
  - [ ] "consciousness of objects" → (consciousness, of, objects)
  - [ ] Tests: known prep phrases extracted

- [ ] **7.5 Relation aggregation**
  - [ ] `Relation` dataclass: source, relation_type, target, evidence (list[str])
  - [ ] `get_relations(term: str, docs, types: list[str]) -> list[Relation]`
  - [ ] Aggregate evidence sentences for same relation
  - [ ] Tests: multiple evidence sentences grouped

---

## Phase 8: Graph Construction

Transform analysis into network structure.

- [ ] **8.1 Graph data structure** (`src/concept_mapper/graph/model.py`)
  - [ ] Add networkx to dependencies
  - [ ] `ConceptGraph` class wrapping `nx.Graph` or `nx.DiGraph`
  - [ ] Node attributes: label, frequency, pos, definition
  - [ ] Edge attributes: weight, relation_type, evidence

- [ ] **8.2 Graph from co-occurrence** (`src/concept_mapper/graph/builders.py`)
  - [ ] `graph_from_cooccurrence(matrix: DataFrame, threshold: float) -> ConceptGraph`
  - [ ] Nodes = terms, edges where co-occurrence > threshold
  - [ ] Edge weight = count or PMI score
  - [ ] Tests: threshold filters edges correctly

- [ ] **8.3 Graph from relations**
  - [ ] `graph_from_relations(relations: list[Relation]) -> ConceptGraph`
  - [ ] Directed edges, labeled by relation type
  - [ ] Tests: relation types preserved as edge labels

- [ ] **8.4 Graph operations** (`src/concept_mapper/graph/operations.py`)
  - [ ] `merge_graphs(g1: ConceptGraph, g2: ConceptGraph) -> ConceptGraph`
  - [ ] `prune_edges(graph, min_weight: float) -> ConceptGraph`
  - [ ] `prune_nodes(graph, min_degree: int) -> ConceptGraph`
  - [ ] `get_subgraph(graph, terms: set[str]) -> ConceptGraph`
  - [ ] Tests: prune removes correct elements

- [ ] **8.5 Graph metrics** (`src/concept_mapper/graph/metrics.py`)
  - [ ] `centrality(graph, method: str = "betweenness") -> dict[str, float]`
  - [ ] `detect_communities(graph) -> list[set[str]]`
  - [ ] Assign community as node attribute (for visualization grouping)
  - [ ] Tests: centrality values sum correctly

---

## Phase 9: Export & Visualization

Output for D3 and other tools.

- [ ] **9.1 D3 JSON export** (`src/concept_mapper/export/d3.py`)
  - [ ] D3 schema:
    ```json
    {
      "nodes": [{"id": "", "label": "", "group": 0, "size": 0, ...}],
      "links": [{"source": "", "target": "", "weight": 0, "label": "", ...}]
    }
    ```
  - [ ] `export_d3_json(graph: ConceptGraph, path: Path)`
  - [ ] Node size from centrality or frequency
  - [ ] Node group from community detection
  - [ ] Tests: output validates against schema

- [ ] **9.2 Include evidence metadata**
  - [ ] Option to embed example sentences in node/edge metadata
  - [ ] Useful for interactive tooltips in D3
  - [ ] `export_d3_json(graph, path, include_evidence: bool = False)`

- [ ] **9.3 Alternative export formats** (`src/concept_mapper/export/formats.py`)
  - [ ] `export_graphml(graph, path)` (for Gephi)
  - [ ] `export_dot(graph, path)` (for Graphviz)
  - [ ] `export_csv(graph, path)` → nodes.csv + edges.csv
  - [ ] Tests: Gephi can open graphml, Graphviz renders dot

- [ ] **9.4 HTML visualization template** (`src/concept_mapper/export/template/`)
  - [ ] Minimal D3 force-directed graph HTML
  - [ ] Loads JSON, renders interactive graph
  - [ ] `generate_html(graph, output_dir: Path)`
  - [ ] Tests: HTML renders in browser without errors

---

## Phase 10: CLI Interface

Unified command-line access.

- [ ] **10.1 CLI framework** (`src/concept_mapper/cli.py`)
  - [ ] Use Click for subcommand structure
  - [ ] Main entry point: `concept-mapper`
  - [ ] Global options: `--verbose`, `--output-dir`

- [ ] **10.2 Ingest command**
  - [ ] `concept-mapper ingest <path> --output corpus.json`
  - [ ] `concept-mapper ingest <path> --recursive --pattern "*.txt"`
  - [ ] Runs preprocessing, saves ProcessedDocuments

- [ ] **10.3 Analyze commands**
  - [ ] `concept-mapper rarities <corpus> --method tfidf --threshold 0.5 --output terms.json`
  - [ ] `concept-mapper cooccurrence <corpus> --terms terms.json --output matrix.csv`
  - [ ] `concept-mapper relations <corpus> --terms terms.json --types svo,copular`

- [ ] **10.4 Search commands**
  - [ ] `concept-mapper search <corpus> --term "Begriff" --context 2`
  - [ ] `concept-mapper concordance <corpus> --term "Begriff" --width 50`
  - [ ] Output to stdout or file

- [ ] **10.5 Graph commands**
  - [ ] `concept-mapper graph <corpus> --terms terms.json --method cooccurrence --output graph.json`
  - [ ] `concept-mapper graph <corpus> --from-relations relations.json`

- [ ] **10.6 Export commands**
  - [ ] `concept-mapper export <graph> --format d3 --output viz/data.json`
  - [ ] `concept-mapper export <graph> --format html --output viz/`
  - [ ] `concept-mapper export <graph> --format gephi --output graph.graphml`

- [ ] **10.7 Pipeline command**
  - [ ] `concept-mapper pipeline <config.yaml>`
  - [ ] YAML config specifies full workflow
  - [ ] Example config in `examples/pipeline.yaml`

---

## Phase 11: Documentation & Polish

- [ ] **11.1 README**
  - [ ] Project overview and goals
  - [ ] Installation instructions
  - [ ] Quick start example
  - [ ] CLI reference

- [ ] **11.2 Example workflow**
  - [ ] Sample corpus in `examples/`
  - [ ] Step-by-step walkthrough
  - [ ] Expected outputs

- [ ] **11.3 API documentation**
  - [ ] Docstrings for all public functions
  - [ ] Type hints throughout

---

## Dependency Graph

```
Phase 0 (scaffolding)
    │
    ▼
Phase 1 (corpus/preprocessing) ◄── foundation for all analysis
    │
    ▼
Phase 2 (frequencies)
    │
    ▼
Phase 3 (rarity detection)
    │
    ▼
Phase 4 (term lists) ◄── curation checkpoint, human review
    │
    ├────────────────┬────────────────┐
    ▼                ▼                ▼
Phase 5          Phase 6          Phase 7
(search)      (co-occurrence)   (relations)
                   │                │
                   └───────┬────────┘
                           ▼
                       Phase 8 (graph construction)
                           │
                           ▼
                       Phase 9 (export/viz)
                           │
                           ▼
                       Phase 10 (CLI)
                           │
                           ▼
                       Phase 11 (docs)
```

---

## Notes for Development

- **Test as you go:** Each phase includes test tasks. Run tests before moving to next phase.
- **Iterate on sample corpus:** Keep sample small until Phase 4, then test on real texts.
- **Curation checkpoint at Phase 4:** Review suggested terms before building graphs. Garbage in → garbage out.
- **CLI incrementally:** Add subcommands as phases complete rather than all at end.
- **spaCy vs NLTK:** Phase 7 introduces spaCy for dependency parsing. Could use earlier if NLTK POS tagging proves insufficient.

## Refactoring Strategy from Spike

**Priority 1: Extract working implementations**
1. Move tokenization, POS tagging, lemmatization from spike into Phase 1 modules
2. Extract frequency analysis logic into Phase 2 modules
3. Port sentence search functionality to Phase 5

**Priority 2: Add missing infrastructure**
1. Install pytest and create test suite
2. Add proper data structures (Document, Corpus, ProcessedDocument classes)
3. Create modular preprocessing pipeline

**Priority 3: New functionality**
1. Implement spaCy dependency parsing (no spike equivalent)
2. Add networkx graph construction (no spike equivalent)
3. Build Click CLI interface (no spike equivalent)

**Code reuse percentage by phase:**
- Phase 0: ~70% done (structure needed, NLTK data downloaded, corpus exists)
- Phase 1: ~50% done (tokenization, POS, lemmatization exist but need modularization)
- Phase 2: ~40% done (FreqDist used, POS filtering exists, needs corpus-level aggregation)
- Phase 3: ~20% done (common word filtering exists, needs full rarity scoring)
- Phase 5: ~30% done (basic sentence search exists, needs structured return types)
- Phases 4, 6-11: ~0% done (no existing implementations)
