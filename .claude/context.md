# Concept Mapper: Project Context

## Project Purpose

Concept Mapper is a digital humanities tool for identifying and visualizing **author-specific philosophical terminology** in primary texts. It goes beyond simple frequency analysis to find terms that are:

- Statistically distinctive (rare in general English, frequent in the target text)
- Philosophically significant (neologisms, technical terms, reified abstractions)
- Relationally structured (co-occurring concepts, grammatical relationships)

**Target users:** Researchers, academics, and students analyzing philosophical texts, literary works, and specialized corpora.

## High-Level Architecture

### Processing Pipeline

```
Raw Text → Preprocessing → Analysis → Graph → Visualization
   ↓            ↓             ↓         ↓          ↓
 .txt      ProcessedDoc   TermList   ConceptGraph  HTML/GraphML
```

### Data Flow Stages

1. **Ingestion** (`corpus/`) - Load text files into Document objects
2. **Preprocessing** (`preprocessing/`) - Tokenize, tag POS, lemmatize
3. **Analysis** (`analysis/`) - Frequency, TF-IDF, rarity detection, co-occurrence
4. **Term Management** (`terms/`) - Curate and manage philosophical term lists
5. **Search** (`search/`) - Find terms in context, generate concordances
6. **Syntax** (`syntax/`) - Dependency parsing and sentence diagramming
7. **Relation Extraction** (`analysis/relations.py`) - Extract SVO triples, copular definitions
8. **Graph Construction** (`graph/`) - Build concept networks
9. **Export** (`export/`) - Generate visualizations and export formats
10. **CLI** (`cli.py`) - User-facing command-line interface

## Module Architecture

### Core Modules

```
src/concept_mapper/
├── corpus/              # Document loading and data models
│   ├── loader.py        # Load .txt files (single or directory)
│   └── models.py        # Document, ProcessedDocument classes
│
├── preprocessing/       # NLP preprocessing pipeline
│   ├── pipeline.py      # Main preprocess() function
│   ├── tokenizer.py     # Sentence and word tokenization
│   ├── pos_tagger.py    # Part-of-speech tagging
│   └── lemmatizer.py    # Lemmatization with WordNet
│
├── analysis/            # Statistical and linguistic analysis
│   ├── frequency.py     # Word frequency distributions
│   ├── tfidf.py         # TF-IDF scoring
│   ├── reference.py     # Brown corpus reference frequencies
│   ├── rarity.py        # Multi-method term detection (core algorithm)
│   ├── cooccurrence.py  # Co-occurrence matrices (PMI, LLR)
│   └── relations.py     # Grammatical relation extraction (SVO, copular)
│
├── terms/               # Term list management
│   ├── models.py        # TermList, Term data structures
│   ├── manager.py       # CRUD operations
│   └── suggester.py     # Auto-suggest terms from analysis
│
├── search/              # Text search and concordance
│   ├── find.py          # Sentence search with context windows
│   ├── concordance.py   # KWIC displays
│   └── dispersion.py    # Term distribution analysis
│
├── syntax/              # Syntactic analysis
│   ├── parser.py        # Stanza dependency parser
│   └── diagram.py       # Sentence diagramming
│
├── graph/               # Graph data structures and algorithms
│   ├── model.py         # ConceptGraph class
│   ├── builders.py      # graph_from_cooccurrence(), graph_from_relations()
│   ├── operations.py    # merge, prune, filter, subgraph
│   └── metrics.py       # centrality, communities, density
│
├── export/              # Visualization and export
│   ├── d3_json.py       # D3.js force-directed format
│   ├── graphml.py       # GraphML for Gephi/yEd
│   ├── csv.py           # CSV export
│   ├── gexf.py          # GEXF format
│   └── html.py          # Standalone HTML visualizations
│
├── storage/             # Persistence abstraction
│   ├── base.py          # Abstract storage interface
│   └── json_storage.py  # JSON file implementation
│
├── validation.py        # Output validation and error handling
└── cli.py               # Command-line interface (Click-based)
```

## Key Design Decisions

### 1. Philosophical Term Detection Strategy

The core algorithm (`analysis/rarity.py`) uses **five complementary signals**:

```python
# PhilosophicalTermScorer combines:
1. Corpus-comparative ratio (frequency in target vs. reference)
2. TF-IDF (term importance in document collection)
3. Neologism detection (not in WordNet dictionary)
4. Definitional context (appears in "X is..." sentences)
5. Capitalization (mid-sentence caps = reified abstractions)
```

**Why hybrid scoring?** Single metrics miss nuances:
- Frequency alone captures common words ("question", "thing")
- TF-IDF alone misses cross-document consistency
- Dictionary lookup alone misses technical uses of ordinary words ("being" as philosophical concept vs. verb)

### 2. Graph Construction Methods

**Two complementary approaches:**

**A. Co-occurrence** (`graph_from_cooccurrence`)
- Statistical: Terms appearing near each other
- Edges weighted by PMI (Pointwise Mutual Information)
- Good for: Discovering implicit conceptual clusters

**B. Relations** (`graph_from_relations`)
- Grammatical: Terms connected by syntax (SVO, copular, prepositional)
- Edges labeled with relation types ("defines", "modifies")
- Good for: Explicit conceptual definitions and hierarchies

### 3. Why NLTK + Stanza (not SpaCy)?

**NLTK:**
- Lightweight, stable on Python 3.14
- Excellent for basic NLP (tokenization, POS, lemmatization)
- WordNet integration for dictionary lookups

**Stanza:**
- Stanford NLP's neural models
- Superior dependency parsing for sentence diagrams
- Used only where needed (syntax module)

**SpaCy deferred:**
- Python 3.14 compatibility issues at project start
- Pattern-based relation extraction works well for philosophical texts
- Could be added later for advanced dependency parsing

### 4. Storage Strategy

**JSON as primary format:**
- Human-readable (inspect with text editor, git-friendly)
- Simple serialization (no database setup)
- Portable (works on any system)

**Trade-off:**
- Large corpora (>100 documents) would benefit from SQLite/PostgreSQL
- Current target: Academic texts (10-100 documents)

### 5. Lemmatization for Concept Matching

**All term matching uses lemmas, not surface forms:**
```python
"Being" → "be"
"entities" → "entity"
"dialectical" → "dialectic"
```

**Why?** Philosophical concepts appear in varied forms:
- "abstraction" (noun), "reify" (verb), "reified" (adjective)
- Matching lemmas captures conceptual unity

## Domain Concepts

### Philosophical Terminology

**Types of terms the tool targets:**

1. **Neologisms** - Coined terms (e.g., Heidegger's "Dasein", Derrida's "différance")
2. **Technical appropriations** - Ordinary words with specialized meaning (e.g., "substance" in Spinoza)
3. **Reified abstractions** - Abstract concepts treated as entities (e.g., "Being", "Truth")
4. **Theoretical vocabulary** - Discipline-specific jargon (e.g., "dialectical materialism")

### NLP Concepts

**POS (Part of Speech) Tags:**
- NN/NNS - Nouns (most philosophical terms)
- JJ - Adjectives (e.g., "dialectical")
- VB - Verbs (e.g., "reify")

**Lemmatization:**
- Reduces words to dictionary form
- "entities" → "entity", "was" → "be"

**Co-occurrence:**
- Terms appearing together more than chance would predict
- Measured by PMI (high = strong association)

**Dependency Parsing:**
- Grammatical structure (subject, object, modifier relationships)
- Used for relation extraction and sentence diagrams

### Graph Theory

**Network Metrics:**
- **Degree centrality** - How many connections a concept has
- **Betweenness centrality** - How often a concept bridges others
- **Community detection** - Clusters of related concepts

**Graph Types:**
- **Undirected** - Co-occurrence (symmetric relationships)
- **Directed** - Relations (asymmetric: X defines Y ≠ Y defines X)

## Critical Files and Entry Points

### Main Entry Points

1. **CLI** - `src/concept_mapper/cli.py`
   - `concept-mapper ingest` - Load and preprocess texts
   - `concept-mapper rarities` - Detect philosophical terms
   - `concept-mapper search/concordance` - Find terms in context
   - `concept-mapper diagram` - Parse sentence structure
   - `concept-mapper graph` - Build concept networks
   - `concept-mapper export` - Generate visualizations

2. **Python API** - `src/concept_mapper/__init__.py`
   - Import and use modules programmatically
   - See `examples/workflow.py` for full example

### Core Algorithms

1. **Philosophical term scoring** - `analysis/rarity.py:PhilosophicalTermScorer`
   - The "secret sauce" of the project
   - Implements hybrid scoring with five signals
   - Most complex algorithmic component

2. **Co-occurrence PMI** - `analysis/cooccurrence.py:build_cooccurrence_matrix`
   - Statistical co-occurrence detection
   - Uses pointwise mutual information (PMI)

3. **Relation extraction** - `analysis/relations.py:get_relations`
   - Pattern-based grammatical relation extraction
   - Handles SVO, copular, prepositional patterns

### Data Models

1. **ProcessedDocument** - `corpus/models.py`
   ```python
   {
     "text": "original text",
     "sentences": [
       {
         "text": "sentence text",
         "tokens": [
           {"text": "word", "pos": "NN", "lemma": "lemma"}
         ]
       }
     ]
   }
   ```

2. **TermList** - `terms/models.py`
   ```python
   {
     "terms": [
       {
         "term": "abstraction",
         "metadata": {
           "score": 4.87,
           "pos": "NN",
           "definition": "...",
           "examples": [...]
         }
       }
     ]
   }
   ```

3. **ConceptGraph** - `graph/model.py`
   - Wraps NetworkX DiGraph/Graph
   - Nodes: Terms with metadata
   - Edges: Relationships with weights

## Common Patterns and Conventions

### Naming Conventions

- **Functions:** `lowercase_with_underscores`
- **Classes:** `PascalCase`
- **Constants:** `UPPER_CASE`
- **Private methods:** `_leading_underscore`

### File Organization

- One main class per file
- Related utilities in same module
- `__init__.py` exports public API

### Error Handling

- Validate inputs early (see `validation.py`)
- Raise descriptive exceptions
- Handle missing files gracefully (check existence before loading)

### Testing Patterns

```python
# tests/ mirrors src/ structure
tests/
  test_corpus.py         → corpus/
  test_preprocessing.py  → preprocessing/
  test_analysis.py       → analysis/
  ...
```

- Use fixtures for sample data (`tests/data/`)
- Test edge cases (empty inputs, missing data)
- Integration tests for full workflows

### Type Hints

```python
def preprocess(doc: Document) -> ProcessedDocument:
    """Convert raw document to processed format."""
    ...
```

All public functions have type hints for IDE support.

## Technology Stack Rationale

### Core Dependencies

**NLTK (Natural Language Toolkit)**
- Mature, stable, well-documented
- Comprehensive resources (Brown corpus, WordNet)
- Sufficient for our use case

**Stanza (Stanford NLP)**
- State-of-art dependency parsing
- Used selectively (only syntax module)

**NetworkX**
- Standard for graph algorithms in Python
- Rich API, good documentation
- Performance adequate for our scale

**Click**
- Modern CLI framework
- Clean decorator-based API
- Automatic help text generation

### Development Tools

**pytest** - Standard testing framework
**Black** - Opinionated formatter (zero config)
**Ruff** - Fast linter (replaces flake8, isort, etc.)
**uv** - Fast package manager (replaces pip)

## Development Workflows

### Adding a New Feature

1. **Plan** - Determine which module(s) need changes
2. **Write tests** - Test-driven development preferred
3. **Implement** - Follow existing patterns
4. **Document** - Update docstrings and docs/
5. **Validate** - Run full test suite
6. **Format** - `make format` before committing

### Running the Test Suite

```bash
make test              # All tests
pytest tests/test_X.py # Specific module
pytest -v              # Verbose output
pytest -k "keyword"    # Filter by name
```

### Example Workflow (End-to-End)

```bash
# 1. Ingest text
concept-mapper ingest examples/sample_text.txt -o corpus.json

# 2. Detect terms
concept-mapper rarities corpus.json --top-n 20 -o terms.json

# 3. Build graph
concept-mapper graph corpus.json -t terms.json -o graph.json

# 4. Visualize
concept-mapper export graph.json --format html -o viz/
open viz/index.html
```

## Performance Characteristics

### Scalability Targets

- **Small corpus** (1-10 docs): <10 seconds
- **Medium corpus** (10-100 docs): <5 minutes
- **Large corpus** (100+ docs): May need optimization

### Bottlenecks

1. **Lemmatization** - WordNet lookup can be slow
   - Mitigated by caching in ProcessedDocument
2. **Reference corpus** - Brown corpus loading
   - Mitigated by disk caching after first load
3. **Co-occurrence calculation** - O(n²) for term pairs
   - Acceptable for typical term lists (<100 terms)

### Optimization Opportunities

- Parallel processing for multiple documents
- Database backend for large corpora
- GPU acceleration (if needed in future)

## Current Status

**Version:** 1.0.0 (Complete)
**Test Coverage:** 521 tests passing
**Documentation:** Comprehensive (README, Usage Guide, API Reference)
**Production Ready:** Yes

## Future Extension Points

### Planned Enhancements (Not Currently Implemented)

1. **Multi-language support** - Add NLTK resources for other languages
2. **Advanced dependency parsing** - Integrate SpaCy when Python 3.14 compatible
3. **Database backend** - SQLite/PostgreSQL for massive corpora
4. **Web interface** - Flask/Django app with interactive exploration
5. **Citation networks** - Track philosophical citations and influences
6. **Temporal analysis** - Track concept evolution across an author's career

### Extension Guidelines

- Maintain modular architecture
- Add comprehensive tests for new features
- Update documentation
- Follow existing code patterns
- Consider backwards compatibility

## Key Takeaways for AI Assistants

1. **Core innovation:** Multi-signal philosophical term detection (not just frequency)
2. **Domain focus:** Academic philosophy/literary analysis (not general NLP)
3. **Scale target:** 10-100 documents (not big data)
4. **Design philosophy:** Simple, tested, documented > cutting-edge but complex
5. **Current completeness:** All planned features implemented, focus on refinement not expansion
6. **Testing culture:** 521 tests, high coverage, don't break existing tests
7. **Documentation:** Keep README, usage guide, and API reference in sync
8. **Python version:** 3.14+ (use modern features, watch compatibility)
