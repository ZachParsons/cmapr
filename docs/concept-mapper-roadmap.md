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
- ✅ Phase 3 Complete: Philosophical term detection (multi-method rarity analysis)
- ✅ Phase 4 Complete: Term list management (curation, import/export, auto-population)
- ✅ Phase 5 Complete: Search & concordance (find, KWIC, context windows, dispersion)
- ✅ Phase 6 Complete: Co-occurrence analysis (PMI, LLR, matrices)
- 🚧 Phase 7 Next: Relation extraction (dependency parsing, SVO triples)
- 📊 368 tests passing, all green

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

## Phase 3: Philosophical Term Detection ✅ COMPLETE

Identify author-specific conceptual vocabulary - terms with specialized meaning distinctive to this author's work, not merely terms rare within the primary text.

**Goal:** Find terms like Aristotle's 'eudaimonia', Spinoza's 'affect', Hegel's 'sublation', Philosopher' 'abstraction', or Deleuze & Guattari's 'body without organs' - philosophical neologisms and technical terminology statistically improbable in general English corpora.

- [x] **3.1 Corpus-comparative analysis** (`src/concept_mapper/analysis/rarity.py`)
  - [x] `compare_to_reference(docs, reference_corpus: Counter) -> dict[str, float]`
    - [x] Calculate relative frequency: `(freq_in_author / total_author) / (freq_in_reference / total_reference)`
    - [x] High ratio = term overused by author vs. general English
  - [x] `get_corpus_specific_terms(docs, reference: Counter, threshold: float) -> set[str]`
    - [x] Filter terms by minimum ratio threshold
    - [x] Consider both absolute frequency in author and relative rarity
  - [x] `get_top_corpus_specific_terms()`, `get_neologism_candidates()`, `get_term_context_stats()`
  - [x] Tests: plant term with high author-freq/low reference-freq, verify detection
  - **Note:** This is PRIMARY method - terms distinctive to author's conceptual framework

- [x] **3.2 TF-IDF against reference corpus**
  - [x] `tfidf_vs_reference(docs, reference: Counter) -> dict[str, float]`
  - [x] Treat author's corpus as single document, reference corpus as background
  - [x] High TF-IDF = term characteristic of author's usage
  - [x] `get_top_tfidf_terms()`, `get_distinctive_by_tfidf()`, `get_combined_distinctive_terms()`
  - [x] Tests: author-specific philosophical term scores above generic vocabulary

- [x] **3.3 Neologism detection**
  - [x] `get_wordnet_neologisms(docs) -> set[str]` (not in WordNet's 117K word-sense pairs)
    - [x] Load WordNet lemmas as baseline dictionary
    - [x] Filter out proper nouns and stopwords with frequency threshold
  - [x] `get_capitalized_technical_terms(docs) -> set[str]` (non-sentence-initial)
    - [x] May indicate reified abstractions ("Being", "Concept", "Spirit")
    - [x] Position-based detection (avoids POS tagger issues)
  - [x] `get_potential_neologisms()`, `get_all_neologism_signals()` (multi-method with high-confidence subset)
  - [x] Tests: planted neologism detected, common words excluded

- [x] **3.4 Definitional context extraction**
  - [x] `get_definitional_contexts(docs) -> list[tuple[str, str, str, str]]` (term, sentence, pattern, doc_id)
    - [x] 8 patterns: copular, explicit_mean, metalinguistic, conceptual, appositive, explicit_define, referential, interpretive
    - [x] Extract sentences where author explicitly defines terms
  - [x] `score_by_definitional_context(terms: set[str], contexts: list) -> dict[str, int]`
    - [x] Higher score = more authorial attention/definition
  - [x] `get_definitional_sentences()`, `get_highly_defined_terms()`, `analyze_definitional_patterns()`
  - [x] Tests: pattern matching on planted definitional sentences
  - **Note:** Direct signal of conceptual importance

- [x] **3.5 POS-filtered candidate extraction**
  - [x] `filter_by_pos_tags(docs, include_tags, exclude_tags) -> set[str]`
    - [x] Focus on nouns (NN, NNP, NNS), verbs (VB*), adjectives (JJ*)
    - [x] Exclude function words, determiners, prepositions
  - [x] `get_philosophical_term_candidates()` with focus modes (nouns, verbs, adjectives, all_content)
  - [x] `get_compound_terms()` for hyphenated and noun phrases
  - [x] `get_filtered_candidates()` for comprehensive extraction
  - [x] Tests: function words filtered out

- [x] **3.6 Hybrid philosophical term scorer**
  - [x] `PhilosophicalTermScorer` class with configurable weights
    - [x] Weight 1: Corpus-comparative ratio (1.0, primary signal)
    - [x] Weight 2: TF-IDF vs reference (1.0)
    - [x] Weight 3: Neologism detection (0.5, boolean boost)
    - [x] Weight 4: Definitional context count (0.3)
    - [x] Weight 5: Capitalization (0.2, reified abstractions)
  - [x] `score_term(term: str) -> dict` (returns breakdown of all components)
  - [x] `score_all(min_score: float, top_n: int) -> list[tuple[str, float, dict]]`
  - [x] `get_high_confidence_terms(min_signals: int) -> set[str]` (multi-signal agreement)
  - [x] `score_philosophical_terms()` convenience function
  - [x] Tests: known philosophical neologism scores high, common English words score low

**Test Coverage:**
- 103 rarity detection tests (unit + integration)
- Tests on real philosophical corpus (sample1-3)
- All edge cases covered (empty docs, missing terms, etc.)

**Explicitly deprioritized:**
- Hapax legomena within primary text (not useful - a term used 50x by author but rare in English is still a philosophical term)
- Within-text frequency thresholds (except for noise filtering)

---

## Phase 4: Term List Management ✅ COMPLETE

Human-in-the-loop curation.

- [x] **4.1 Data structures** (`src/concept_mapper/terms/models.py`)
  - [x] `TermEntry` dataclass: term, lemma, pos, definition, notes, examples, metadata
  - [x] `TermList` class: collection with lookup by term
  - [x] Dictionary serialization (to_dict/from_dict)
  - [x] Iteration, length, containment support

- [x] **4.2 CRUD operations** (TermList methods)
  - [x] `add(entry: TermEntry)` - add term to list
  - [x] `remove(term: str)` - remove term from list
  - [x] `update(term: str, **kwargs)` - update term fields
  - [x] `get(term: str) -> TermEntry | None` - retrieve term
  - [x] `list_terms() -> list[TermEntry]` - get all terms (sorted)
  - [x] `list_term_names() -> list[str]` - get term names only
  - [x] Tests: CRUD round-trip, error handling

- [x] **4.3 Persistence**
  - [x] `save(path: Path)` → JSON serialization
  - [x] `load(path: Path) -> TermList` → JSON deserialization
  - [x] Creates parent directories automatically
  - [x] Tests: save → load preserves all data

- [x] **4.4 Bulk operations** (`src/concept_mapper/terms/manager.py`)
  - [x] TermManager class for bulk operations
  - [x] `import_from_txt(path)` - plain text, one term per line
  - [x] `export_to_txt(path)` - export to text
  - [x] `import_from_csv(path)` - CSV with term + metadata columns
  - [x] `export_to_csv(path)` - export to CSV with fields
  - [x] `merge_from_file(path, format)` - merge from file
  - [x] `filter_by_pos(tags)` - filter by POS tags
  - [x] `get_statistics()` - term list statistics
  - [x] Tests: import/export round-trip, formats

- [x] **4.5 Auto-populate from rarity** (`src/concept_mapper/terms/suggester.py`)
  - [x] `suggest_terms_from_analysis(docs, reference, min_score, top_n)` - use PhilosophicalTermScorer
  - [x] Populate examples from corpus automatically (max_examples parameter)
  - [x] Infer POS tags from documents
  - [x] Include score metadata for each term
  - [x] `suggest_terms_by_method(method)` - use specific detection method
  - [x] Support for ratio, tfidf, neologism, definitional methods
  - [x] Tests: suggested list contains expected rare terms

**Test Coverage:**
- 47 comprehensive tests for term list management
- Tests for data structures, CRUD, persistence, bulk operations, auto-population
- All edge cases covered (duplicates, missing fields, file errors, etc.)

---

## Phase 5: Search & Concordance ✅ COMPLETE

Find where and how terms appear.

- [x] **5.1 Basic search** (`src/concept_mapper/search/find.py`)
  - [x] `SentenceMatch` dataclass: sentence, doc_id, sent_index, term_positions
  - [x] `find_sentences(term: str, docs) -> list[SentenceMatch]`
  - [x] `find_sentences_any()`, `find_sentences_all()` for multiple terms
  - [x] `count_term_occurrences()` for frequency counting
  - [x] Case-sensitive and case-insensitive search
  - [ ] Support lemma matching option *(deferred: TODO in code)*
  - [x] Tests: 12 tests covering all search functionality

- [x] **5.2 Concordance (KWIC)** (`src/concept_mapper/search/concordance.py`)
  - [x] `KWICLine` dataclass: left_context, keyword, right_context, doc_id
  - [x] `concordance(term: str, docs, width: int = 50) -> list[KWICLine]`
  - [x] `concordance_sorted()` for sorting by left/right context
  - [x] `concordance_filtered()` for co-occurrence patterns
  - [x] Aligned output on keyword for scanning
  - [x] Tests: 12 tests covering KWIC display and formatting

- [x] **5.3 Context window** (`src/concept_mapper/search/context.py`)
  - [x] `ContextWindow` dataclass: before (list[str]), match (str), after (list[str])
  - [x] `get_context(term: str, docs, n_sentences: int = 1) -> list[ContextWindow]`
  - [x] `get_context_by_match()` for expanding existing search results
  - [x] `get_context_with_highlights()` for term highlighting
  - [x] `format_context_windows()` for display
  - [x] Tests: 12 tests covering context extraction and formatting

- [x] **5.4 Dispersion** (`src/concept_mapper/search/dispersion.py`)
  - [x] `dispersion(term: str, docs) -> dict[str, list[int]]` (doc_id → positions)
  - [x] `get_dispersion_summary()` with coverage statistics
  - [x] `compare_dispersion()` for multi-term comparison
  - [x] `dispersion_plot_data()` for visualization preparation
  - [x] `get_concentrated_regions()` for finding dense usage areas
  - [x] Position as sentence index or character offset
  - [x] Tests: 16 tests covering all dispersion functionality

---

## Phase 6: Co-occurrence Analysis ✅ COMPLETE

Relational structure from proximity.

- [x] **6.1 Sentence-level co-occurrence** (`src/concept_mapper/analysis/cooccurrence.py`)
  - [x] `cooccurs_in_sentence(term: str, docs) -> Counter`
  - [x] Count all terms appearing in same sentences as target
  - [x] Case-sensitive and case-insensitive modes
  - [x] Tests: 6 tests covering sentence co-occurrence

- [x] **6.2 Filtered co-occurrence**
  - [x] `cooccurs_filtered(term: str, docs, term_list: TermList) -> Counter`
  - [x] Only count terms in curated list
  - [x] Tests: 4 tests for filtering behavior

- [x] **6.3 Paragraph-level co-occurrence**
  - [x] `cooccurs_in_paragraph(term: str, docs) -> Counter`
  - [x] Currently treats documents as paragraphs (Phase 1.7 pending)
  - [x] Tests: 2 tests for paragraph-level analysis

- [x] **6.4 N-sentence window co-occurrence**
  - [x] `cooccurs_within_n(term: str, docs, n_sentences: int) -> Counter`
  - [x] Sliding window across sentence boundaries
  - [x] Configurable window size
  - [x] Tests: 3 tests covering window behavior

- [x] **6.5 Statistical significance**
  - [x] `pmi(term1: str, term2: str, docs) -> float` (pointwise mutual information)
  - [x] `log_likelihood_ratio(term1: str, term2: str, docs) -> float`
  - [x] PMI for measuring association strength
  - [x] G² (log-likelihood) with significance thresholds
  - [x] Symmetric measures
  - [x] Tests: 12 tests covering PMI and LLR

- [x] **6.6 Co-occurrence matrix**
  - [x] `build_cooccurrence_matrix(term_list: TermList, docs, method: str) -> Dict`
  - [x] Methods: "count", "pmi", "llr"
  - [x] Windows: "sentence", "n_sentences"
  - [x] Symmetric matrix, terms × terms
  - [x] `save_cooccurrence_matrix(matrix, path)` → CSV
  - [x] `get_top_cooccurrences()` for quick exploration
  - [x] Tests: 18 tests covering matrix building, saving, and top-N selection

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
