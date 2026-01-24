# Concept Mapper: Development Roadmap

A tool for extracting and visualizing an author's idiosyncratic conceptual vocabulary from primary texts.

## Project Overview

**Goal:** Analyze philosophical texts to identify author-specific conceptual vocabulary - neologisms and terms with specialized technical meaning that are statistically distinctive compared to general English corpora. Map these concepts through co-occurrence and grammatical relations, exporting as D3 visualizations.

**Examples of target terms:** Aristotle's 'eudaimonia', Spinoza's 'affect', Hegel's 'sublation', Philosopher' 'abstraction', Deleuze & Guattari's 'body without organs'

**Stack:** Python, NLTK, spaCy (for dependency parsing), networkx, Click (CLI)

**Current Status:**
- ✅ Phase 0 Complete: Storage layer, test corpus, NLTK data
- ✅ Phase 1 Complete: Corpus loading, preprocessing pipeline (tokenization, POS, lemmas)
- ✅ Phase 2 Complete: Frequency analysis, Brown corpus reference, TF-IDF
- 🚧 Phase 3 Next: Philosophical term detection (corpus-comparative analysis)
- 📊 125 tests passing, all green

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

## Phase 0: Project Scaffolding ✅ COMPLETE

- [x] **0.1 Initialize project structure**
  - [x] Create directory layout: `src/concept_mapper/`, `tests/`, `data/sample/`, `output/`
  - [x] Initialize git repository
  - [x] Create `pyproject.toml` or `requirements.txt` *(spike: requirements.txt exists with nltk>=3.8)*
  - [x] Initial dependencies: `nltk`, `pytest`, `click`, `black`, `ruff`, `ipython`

- [x] **0.2 Download NLTK data** *(spike: pos_tagger.py:29-32 downloads all needed data)*
  - [x] Create setup script `scripts/download_nltk_data.py` *(inline in pos_tagger.py, should extract)*
  - [x] Download: `punkt`, `averaged_perceptron_tagger`, `wordnet`, `brown`, `stopwords`
  - [x] Verify downloads succeed *(working in current spike)*

- [x] **0.3 Create sample test corpus**
  - [x] Create 2-3 short `.txt` files in `data/sample/` *(5 files total, 95KB)*
  - [x] Include invented "rare terms" with known frequencies *(test_philosophical_terms.txt with daseinology, temporalization, ekstatic)*
  - [x] Document expected values for verification *(test_corpus_manifest.json)*

---

## Storage Architecture

**Philosophy:** Start simple with human-readable formats, design for extensibility.

### Current Approach (Phases 1-6)

**File formats:**
- **JSON** for all structured data
  - Preprocessed corpus, term lists, relations, graphs
  - Human-readable, debuggable, language-agnostic
  - Easy to inspect and version control
- **CSV** for tabular/matrix data
  - Co-occurrence matrices, frequency distributions, TF-IDF scores
  - Opens in Excel, Pandas-native, easy inspection
- **Cache** reference corpora as JSON
  - Brown corpus frequencies (one-time computation)
  - Fast enough for development and moderate scale

**Filesystem organization:**
```
output/
├── corpus/
│   ├── preprocessed.json        # ProcessedDocument objects
│   └── metadata.json             # Corpus-level metadata
├── analysis/
│   ├── terms.json                # Curated TermList
│   ├── frequencies.csv           # Term frequency distributions
│   ├── cooccurrence.csv          # Co-occurrence matrix
│   └── tfidf_scores.csv          # TF-IDF scores vs reference
├── graphs/
│   ├── graph.json                # NetworkX graph (node-link format)
│   └── d3/
│       ├── data.json             # D3-formatted export
│       └── index.html            # Interactive visualization
└── cache/
    └── brown_corpus_freqs.json   # Reference corpus frequencies
```

### Extensibility Design

**Abstract storage backend** (`src/concept_mapper/storage/backend.py`):
```python
from abc import ABC, abstractmethod
from pathlib import Path

class StorageBackend(ABC):
    @abstractmethod
    def save_corpus(self, corpus: ProcessedDocument, path: Path) -> None: ...

    @abstractmethod
    def load_corpus(self, path: Path) -> ProcessedDocument: ...

    @abstractmethod
    def save_term_list(self, terms: TermList, path: Path) -> None: ...

    @abstractmethod
    def load_term_list(self, path: Path) -> TermList: ...

    # ... etc for other data types

class JSONBackend(StorageBackend):
    """Default: Human-readable, debuggable"""
    # Implementation using json.dump/load

class SQLiteBackend(StorageBackend):
    """Future: Searchable, queryable intermediate data"""
    # Implementation using sqlite3

class ParquetBackend(StorageBackend):
    """Future: Fast columnar storage for large matrices"""
    # Implementation using pandas.to_parquet
```

**Configuration-driven backend selection:**
```yaml
# pipeline.yaml
storage:
  backend: json  # Options: json (default), sqlite, parquet
  output_dir: ./output
  compression: false  # gzip for JSON when needed
```

### Migration Path

**When to switch backends:**
- **SQLite**: When you need to query intermediate data (search across documents, filter terms)
- **Parquet**: When matrices exceed ~100K terms (faster read/write than CSV)
- **HDF5**: When working with very large corpora (multi-GB preprocessed data)
- **Database (Postgres)**: When building web interface or multi-user access

**Always maintain:**
- Same API across all backends (`save_corpus()`, `load_corpus()`)
- Export to JSON/CSV for portability and archiving
- Human-readable reference formats in documentation

### Implementation Tasks

- [x] **Phase 0.4: Storage abstraction** (`src/concept_mapper/storage/`)
  - [x] Define `StorageBackend` ABC
  - [x] Implement `JSONBackend` as default
  - [x] Add filesystem utilities (create output dirs, check paths)
  - [x] Tests: round-trip save/load for each data type *(12 tests passing)*

---

## Phase 1: Corpus Ingestion & Preprocessing ✅ COMPLETE (except 1.7)

All downstream analysis depends on clean, structured text.

- [x] **1.1 File loader** (`src/concept_mapper/corpus/loader.py`)
  - [x] `load_file(path: Path) -> Document`
  - [x] `load_directory(path: Path, pattern: str = "*.txt") -> Corpus`
  - [x] Handle encoding (UTF-8 with Latin-1 fallback)
  - [x] Tests: load sample files, verify content *(22 tests passing)*

- [x] **1.2 Data structures** (`src/concept_mapper/corpus/models.py`)
  - [x] `Document` dataclass: text, metadata (title, author, date, source_path)
  - [x] `Corpus` class: collection of Documents
  - [x] `ProcessedDocument` dataclass: raw, sentences, tokens, pos_tags, lemmas

- [x] **1.3 Tokenization** (`src/concept_mapper/preprocessing/tokenize.py`)
  - [x] `tokenize_words(text: str) -> list[str]`
  - [x] `tokenize_sentences(text: str) -> list[str]`
  - [x] `tokenize_words_preserve_case()` - preserves original case
  - [x] Tests: verify token/sentence counts on sample *(24 tests passing)*

- [x] **1.4 POS tagging** (`src/concept_mapper/preprocessing/tagging.py`)
  - [x] `tag_tokens(tokens: list[str]) -> list[tuple[str, str]]`
  - [x] `tag_sentences(sentences: list[str]) -> list[list[tuple[str, str]]]`
  - [x] `filter_by_pos()` - extract tokens by POS tag
  - [x] Tests: spot-check known POS assignments *(24 tests passing)*

- [x] **1.5 Lemmatization** (`src/concept_mapper/preprocessing/lemmatize.py`)
  - [x] `get_wordnet_pos(treebank_tag: str) -> str` (map Penn tags to WordNet)
  - [x] `lemmatize(word: str, pos: str) -> str`
  - [x] `lemmatize_tagged(tagged_tokens: list[tuple]) -> list[str]`
  - [x] `lemmatize_words()` - batch lemmatization
  - [x] Tests: "running" → "run", "better" → "good" *(24 tests passing)*

- [x] **1.6 Preprocessing pipeline** (`src/concept_mapper/preprocessing/pipeline.py`)
  - [x] `preprocess(document: Document) -> ProcessedDocument`
  - [x] `preprocess_corpus(corpus: Corpus) -> list[ProcessedDocument]`
  - [x] Single entry point that runs tokenize → tag → lemmatize
  - [x] Tests: round-trip load → preprocess → verify structure *(24 tests passing)*

- [ ] **1.7 Paragraph segmentation** (`src/concept_mapper/preprocessing/segment.py`)
  - [ ] `segment_paragraphs(text: str) -> list[str]`
  - [ ] Handle various paragraph markers (double newline, indentation)
  - [ ] Add paragraph indices to ProcessedDocument
  - [ ] Tests: verify paragraph boundaries

---

## Phase 2: Term Extraction & Frequency Analysis ✅ COMPLETE

Statistical foundation for rarity detection.

- [x] **2.1 Frequency distribution** (`src/concept_mapper/analysis/frequency.py`)
  - [x] `word_frequencies(doc: ProcessedDocument) -> Counter`
  - [x] `pos_filtered_frequencies(doc: ProcessedDocument, pos_tags: set) -> Counter`
  - [x] Option: count lemmas vs surface forms (use_lemmas parameter)
  - [x] `get_vocabulary()` - extract unique terms
  - [x] Tests: manual count verification *(21 tests passing)*

- [x] **2.2 Corpus-level aggregation**
  - [x] `corpus_frequencies(docs: list[ProcessedDocument]) -> Counter`
  - [x] `document_frequencies(docs: list[ProcessedDocument]) -> Counter` (in how many docs?)
  - [x] Tests: term in 2 docs → doc_freq = 2 *(21 tests passing)*

- [x] **2.3 Reference corpus** (`src/concept_mapper/analysis/reference.py`)
  - [x] `load_reference_corpus(name: str = "brown") -> Counter`
  - [x] Cache to disk after first computation *(output/cache/brown_corpus_freqs.json)*
  - [x] `get_reference_vocabulary()` - all unique words in Brown
  - [x] `get_reference_size()` - total word count (1.16M words)
  - [x] Tests: verify brown corpus loads, common words have high freq *(21 tests passing)*

- [x] **2.4 TF-IDF** (`src/concept_mapper/analysis/tfidf.py`)
  - [x] `tf(term: str, doc: ProcessedDocument) -> float`
  - [x] `idf(term: str, docs: list[ProcessedDocument]) -> float`
  - [x] `tfidf(term: str, doc: ProcessedDocument, docs: list) -> float`
  - [x] `corpus_tfidf_scores(docs: list[ProcessedDocument]) -> dict[str, float]`
  - [x] `document_tfidf_scores()` - per-document TF-IDF
  - [x] Tests: unique term scores high, common term scores low *(21 tests passing)*

---

## Phase 3: Philosophical Term Detection

Identify author-specific conceptual vocabulary - terms with specialized meaning distinctive to this author's work, not merely terms rare within the primary text.

**Goal:** Find terms like Aristotle's 'eudaimonia', Spinoza's 'affect', Hegel's 'sublation', Philosopher' 'abstraction', or Deleuze & Guattari's 'body without organs' - philosophical neologisms and technical terminology statistically improbable in general English corpora.

- [ ] **3.1 Corpus-comparative analysis** (`src/concept_mapper/analysis/rarity.py`)
  - [ ] `compare_to_reference(docs, reference_corpus: Counter) -> dict[str, float]`
    - [ ] Calculate relative frequency: `(freq_in_author / total_author) / (freq_in_reference / total_reference)`
    - [ ] High ratio = term overused by author vs. general English
  - [ ] `get_corpus_specific_terms(docs, reference: Counter, threshold: float) -> set[str]`
    - [ ] Filter terms by minimum ratio threshold
    - [ ] Consider both absolute frequency in author and relative rarity
  - [ ] Tests: plant term with high author-freq/low reference-freq, verify detection
  - **Note:** This is PRIMARY method - terms distinctive to author's conceptual framework

- [ ] **3.2 TF-IDF against reference corpus**
  - [ ] `tfidf_vs_reference(docs, reference: Counter) -> dict[str, float]`
  - [ ] Treat author's corpus as single document, reference corpus as background
  - [ ] High TF-IDF = term characteristic of author's usage
  - [ ] Tests: author-specific philosophical term scores above generic vocabulary

- [ ] **3.3 Neologism detection**
  - [ ] `get_potential_neologisms(docs, dictionary: set) -> set[str]` (not in WordNet)
    - [ ] Load WordNet lemmas as baseline dictionary
    - [ ] Filter out proper nouns, OCR errors, typos with frequency threshold
  - [ ] `get_capitalized_technical_terms(docs) -> set[str]` (non-sentence-initial)
    - [ ] May indicate reified abstractions ("Being", "Concept", "Spirit")
  - [ ] Tests: planted neologism detected, common words excluded

- [ ] **3.4 Definitional context extraction**
  - [ ] `get_definitional_contexts(docs) -> list[tuple[str, str]]` (term, sentence)
    - [ ] Patterns: "X is...", "by X I mean...", "what I call X", "the concept of X"
    - [ ] Extract sentences where author explicitly defines terms
  - [ ] `score_by_definitional_context(terms: set[str], contexts: list) -> dict[str, int]`
    - [ ] Higher score = more authorial attention/definition
  - [ ] Tests: pattern matching on planted definitional sentences
  - **Note:** Direct signal of conceptual importance

- [ ] **3.5 POS-filtered candidate extraction**
  - [x] `filter_by_pos(terms: set[str], pos_tags: set[str], docs) -> set[str]` *(spike: pos_tagger.py:106-118 filters common verbs)*
    - [ ] Focus on nouns (NN, NNP, NNS), verbs (VB*), adjectives (JJ*)
    - [ ] Exclude function words, determiners, prepositions
  - [ ] Tests: function words filtered out
  - **Note:** pos_tagger.py has filter logic for excluding common verbs (lines 107-111). Generalize to other POS.

- [ ] **3.6 Hybrid philosophical term scorer**
  - [ ] `PhilosophicalTermScorer` class with configurable weights
    - [ ] Weight 1: Corpus-comparative ratio (primary signal)
    - [ ] Weight 2: TF-IDF vs reference
    - [ ] Weight 3: Neologism detection (boolean boost)
    - [ ] Weight 4: Definitional context count
    - [ ] Weight 5: Capitalization (reified abstractions)
  - [ ] `score_term(term: str) -> float`
  - [ ] `score_all(min_score: float) -> dict[str, float]`
  - [ ] Tests: known philosophical neologism scores high, common English words score low

**Explicitly deprioritized:**
- Hapax legomena within primary text (not useful - a term used 50x by author but rare in English is still a philosophical term)
- Within-text frequency thresholds (except for noise filtering)

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
- **Rarity = corpus-comparative:** "Rare" means statistically improbable in general English, not necessarily rare within the primary text. A term used 100 times by the author but nearly absent from Brown corpus is a philosophical term worth tracking.

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

**Implementation status by phase:**
- Phase 0: ✅ 100% COMPLETE (storage, test corpus, NLTK data)
- Phase 1: ✅ ~95% COMPLETE (all modules done except paragraph segmentation)
- Phase 2: ✅ 100% COMPLETE (frequency, reference corpus, TF-IDF)
- Phase 3: ~0% done (corpus-comparative analysis is new work)
- Phase 4: ~0% done (term list management)
- Phase 5: ~30% done (basic sentence search exists in spike, needs structured return types)
- Phases 6-11: ~0% done (no existing implementations)

**Test coverage:**
- 125 tests passing across all modules
- Phase 0: 12 tests (storage)
- Phase 1: 46 tests (corpus + preprocessing)
- Phase 2: 21 tests (analysis)
- Legacy: 46 tests (pos_tagger, sample corpus)
