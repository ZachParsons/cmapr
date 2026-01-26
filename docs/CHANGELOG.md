# Changelog

All notable changes to the Concept Mapper project.

## [1.0.0] - 2026-01-25

### Phase 11: Documentation & Polish ✅

**Added:**
- Complete example workflow in `examples/` directory
  - Sample philosophical text (Hegelian style)
  - Step-by-step walkthrough documentation
  - Bash workflow script (`workflow.sh`)
  - Python API workflow script (`workflow.py`)
- Comprehensive API reference documentation (`docs/api-reference.md`)
  - Complete coverage of all public APIs
  - Type signatures and parameter descriptions
  - Usage examples for each module
- Enhanced README with:
  - Links to all documentation
  - Complete roadmap status
  - Quick start guide

**Project Status:**
- ✅ All 11 phases complete
- ✅ 521 tests passing (2 skipped)
- ✅ Full pipeline: text → preprocessing → term detection → graph → visualization
- ✅ CLI and Python API fully documented
- ✅ Ready for production use

---

## [0.10.0] - 2026-01-24

### Phase 10: CLI Interface ✅

**Added:**
- Unified command-line interface using Click framework
- Commands: `ingest`, `rarities`, `search`, `concordance`, `graph`, `export`
- Global options: `--verbose`, `--output-dir`
- Progress bars for batch operations
- Package installation with entry point
- 23 comprehensive CLI tests
- setup.py for package distribution

---

## [0.9.0] - 2026-01-23

### Phase 9: Export & Visualization ✅

**Added:**
- D3.js JSON export for interactive visualizations
- GraphML export for Gephi, yEd, Cytoscape
- DOT export for Graphviz
- CSV export for spreadsheets
- GEXF export for Gephi
- Standalone HTML visualization generator
- Force-directed graph layout
- Interactive features: drag, zoom, pan, tooltips
- Community detection coloring
- Node sizing by centrality
- 30 comprehensive tests

---

## [0.8.0] - 2026-01-22

### Phase 8: Graph Construction ✅

**Added:**
- `ConceptGraph` class wrapping NetworkX
- Graph builders from co-occurrence matrices
- Graph builders from relation extraction
- Graph operations: merge, prune, filter, subgraph
- Graph metrics: centrality, communities, paths, density
- Support for directed and undirected graphs
- Node and edge attribute management
- 62 comprehensive tests

---

## [0.7.0] - 2026-01-21

### Phase 7: Relation Extraction ✅

**Added:**
- SVO (Subject-Verb-Object) triple extraction
- Copular definition extraction (X is Y)
- Prepositional relation extraction (X of Y)
- Pattern-based extraction using NLTK POS tagging
- Evidence aggregation for relations
- Relation filtering and type selection
- 35 comprehensive tests

**Note:** SpaCy dependency parsing deferred due to Python 3.14 compatibility. Pattern-based approach is effective for philosophical texts.

---

## [0.6.0] - 2026-01-20

### Phase 6: Co-occurrence Analysis ✅

**Added:**
- Sentence-level co-occurrence counting
- N-sentence window co-occurrence
- PMI (Pointwise Mutual Information) calculation
- LLR (Log-Likelihood Ratio) significance testing
- Co-occurrence matrix building (count, PMI, LLR methods)
- Filtered co-occurrence (curated term lists only)
- Matrix export to CSV
- 45 comprehensive tests

---

## [0.5.0] - 2026-01-19

### Phase 5: Search & Concordance ✅

**Added:**
- Basic sentence search
- Multi-term search (any/all)
- KWIC (Key Word In Context) concordance displays
- Context window extraction (N sentences before/after)
- Dispersion analysis across corpus
- Position tracking and coverage statistics
- 52 comprehensive tests

---

## [0.4.0] - 2026-01-18

### Phase 4: Term List Management ✅

**Added:**
- `TermEntry` and `TermList` data structures
- CRUD operations for term lists
- JSON, CSV, TXT import/export
- Bulk operations and filtering
- Auto-population from rarity analysis
- Term suggestion system
- Statistics and metadata management
- 47 comprehensive tests

---

## [0.3.0] - 2026-01-17

### Phase 3: Philosophical Term Detection ✅

**Added:**
- `PhilosophicalTermScorer` with multi-method detection
- Corpus-comparative rarity analysis
- TF-IDF scoring against reference corpus
- Neologism detection (WordNet lookup)
- Definitional context extraction (8 patterns)
- POS-filtered candidate extraction
- Hybrid scoring with weighted components
- High-confidence term filtering (multi-signal agreement)
- 103 comprehensive tests

**Methods:**
- Relative frequency ratio
- TF-IDF vs. reference
- Neologism detection
- Definitional context counting
- Capitalization (reified abstractions)

---

## [0.2.0] - 2026-01-16

### Phase 2: Frequency Analysis & TF-IDF ✅

**Added:**
- Word frequency distributions
- Corpus-level frequency aggregation
- Document frequency counting
- Brown corpus reference loading
- TF-IDF calculation
- POS-filtered frequencies
- Frequency caching
- 21 comprehensive tests

---

## [0.1.0] - 2026-01-15

### Phase 1: Corpus Preprocessing ✅

**Added:**
- Document and corpus loading
- Sentence tokenization
- Word tokenization
- POS (Part of Speech) tagging
- Lemmatization with WordNet
- Unified preprocessing pipeline
- `Document` and `ProcessedDocument` data models
- 46 comprehensive tests

---

## [0.0.1] - 2026-01-14

### Phase 0: Project Scaffolding ✅

**Added:**
- Project directory structure
- Git repository initialization
- NLTK data download scripts
- Sample test corpus (5 files, 95KB)
- Test corpus manifest with expected values
- Storage abstraction layer
- JSON backend implementation
- 12 storage tests
- requirements.txt with dependencies

---

## Development Metrics

### Test Coverage
- **Total tests:** 521 (2 skipped)
- **Test success rate:** 100%
- **Coverage:** Comprehensive across all modules

### Lines of Code (src/)
- **Python:** ~8,500 lines
- **Docstrings:** ~2,000 lines
- **Comments:** ~500 lines

### Modules
- **Core modules:** 35
- **Test modules:** 13
- **Documentation files:** 5

### Dependencies
- **Runtime:** nltk, networkx, click
- **Development:** pytest, black, ruff

---

## Future Enhancements (Optional)

### Possible Future Work
- SpaCy integration for dependency parsing
- GPU-accelerated text processing
- Web interface
- Database backend for large corpora
- Additional visualization styles
- Multi-language support
- PDF and EPUB corpus loading
- Citation network analysis
- Author comparison tools

**Note:** The current implementation is feature-complete for the intended use case of philosophical text analysis.
