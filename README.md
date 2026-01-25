# Concept Mapper

A tool for extracting and visualizing an author's idiosyncratic conceptual vocabulary from philosophical texts.

## Overview

Concept Mapper analyzes primary texts to identify author-specific philosophical terminology—neologisms and terms with specialized technical meaning that are statistically distinctive compared to general English. It then maps relationships between these concepts through co-occurrence analysis and grammatical extraction, producing interactive network visualizations.

**Examples of target terms:** Aristotle's *eudaimonia*, Spinoza's *affect*, Hegel's *sublation*, Philosopher' *abstraction*, Deleuze & Guattari's *body without organs*

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .

# Verify CLI installation
concept-mapper --help

# Run tests
pytest tests/ -v
```

## CLI Usage

Try the example workflow with included sample data:

```bash
# Process the sample philosophical text
concept-mapper ingest examples/sample_text.txt

# Detect philosophical terms (abstraction, praxis, separation, etc.)
concept-mapper rarities output/corpus/corpus.json --top-n 20

# Build concept graph from co-occurrence
concept-mapper graph output/corpus/corpus.json -t output/terms/terms.json -m cooccurrence

# Generate interactive visualization
concept-mapper export output/graphs/graph.json --format html

# Open visualization in browser
open output/exports/visualization/index.html
```

**Default output structure:**
```
output/
├── corpus/         # Preprocessed corpora
├── terms/          # Detected terms
├── graphs/         # Concept graphs
├── exports/        # Visualizations and export formats
└── cache/          # Reference corpus cache
```

Or run the complete example workflow (from project root):
```bash
bash examples/workflow.sh
# Or: python examples/workflow.py
```

For your own documents:
```bash
# Analyze your own text file (outputs to ./output/)
concept-mapper ingest your_document.txt
concept-mapper rarities output/corpus/corpus.json
concept-mapper graph output/corpus/corpus.json -t output/terms/terms.json
concept-mapper export output/graphs/graph.json --format html

# Or specify custom output locations with -o flag
concept-mapper ingest your_document.txt -o my_corpus.json
concept-mapper rarities my_corpus.json -o my_terms.json
concept-mapper graph my_corpus.json -t my_terms.json -o my_graph.json
concept-mapper export my_graph.json --format html -o my_viz/

# Or process a directory of texts
concept-mapper ingest path/to/corpus/ -r
```

See `concept-mapper --help` for full command reference and [examples/README.md](examples/README.md) for detailed walkthrough.

## Features (Complete: Phases 0-11)

### ✅ Phase 1: Corpus Preprocessing
- Load documents from text files
- Sentence and word tokenization
- POS (Part of Speech) tagging
- Lemmatization
- **[See examples →](docs/usage-guide.md#phase-1-corpus-preprocessing)**

### ✅ Phase 2: Frequency Analysis
- Word frequency distributions
- TF-IDF (Term Frequency-Inverse Document Frequency) scoring
- Comparison to Brown corpus reference
- **[See examples →](docs/usage-guide.md#phase-2-frequency-analysis--tf-idf)**

### ✅ Phase 3: Philosophical Term Detection
- Multi-method rarity detection (5 signals)
- Corpus-comparative analysis
- Neologism detection (WordNet lookup)
- Definitional context identification
- Hybrid scorer with weighted signals
- **[See examples →](docs/usage-guide.md#phase-3-philosophical-term-detection)**

### ✅ Phase 4: Term List Management
- CRUD operations for curated terms
- Import/export (JSON, CSV, TXT)
- Auto-population from statistical analysis
- Metadata: definitions, notes, examples, POS tags
- **[See examples →](docs/usage-guide.md#phase-4-term-list-management)**

### ✅ Phase 5: Search & Concordance
- Find sentences containing terms
- KWIC (Key Word In Context) concordance display
- Context windows (N sentences before/after)
- Dispersion analysis (where terms appear)
- **[See examples →](docs/usage-guide.md#phase-5-search--concordance)**

### ✅ Phase 6: Co-occurrence Analysis
- Sentence-level co-occurrence
- N-sentence window co-occurrence
- PMI (Pointwise Mutual Information)
- LLR (Log-Likelihood Ratio) significance testing
- Co-occurrence matrices (count/PMI/LLR)
- **[See examples →](docs/usage-guide.md#phase-6-co-occurrence-analysis)**

### ✅ Phase 7: Relation Extraction
- SVO (Subject-Verb-Object) triples
- Copular definitions (X is Y)
- Prepositional relations (X of Y)
- Evidence aggregation
- Pattern-based extraction using NLTK
- **[See examples →](docs/usage-guide.md#phase-7-relation-extraction)**

### ✅ Phase 8: Graph Construction
- ConceptGraph data structure (directed/undirected)
- Build graphs from co-occurrence matrices
- Build graphs from relation extraction
- Graph operations (merge, prune, filter, subgraph)
- Graph metrics (centrality, communities, paths, density)
- **[See examples →](docs/usage-guide.md#phase-8-graph-construction)**

### ✅ Phase 9: Export & Visualization
- D3.js JSON export for interactive web visualizations
- GraphML export for Gephi, yEd, Cytoscape
- DOT export for Graphviz
- CSV export for spreadsheets and databases
- Standalone HTML visualizations with force-directed layouts
- **[See examples →](docs/usage-guide.md#phase-9-export--visualization)**

### ✅ Phase 10: CLI Interface
- Unified command-line interface (`concept-mapper`)
- Commands: ingest, rarities, search, concordance, graph, export
- Global options: `--verbose`, `--output-dir`
- Progress bars for batch operations
- End-to-end workflow support
- **[See examples →](docs/usage-guide.md#phase-10-cli-interface)**

## Documentation

- **[Usage Guide](docs/usage-guide.md)** - Practical examples for each phase
- **[API Reference](docs/api-reference.md)** - Complete Python API reference
- **[Development Roadmap](docs/concept-mapper-roadmap.md)** - Complete project plan
- **[Examples](examples/)** - Complete workflow examples and sample data

## Example Workflow

```python
from concept_mapper.corpus.loader import load_document
from concept_mapper.preprocessing.pipeline import preprocess
from concept_mapper.analysis.reference import load_reference_corpus
from concept_mapper.analysis.rarity import PhilosophicalTermScorer

# Load and preprocess text
doc = load_document("data/sample/philosopher_1920_cc.txt")
processed = preprocess(doc)

# Detect philosophical terms
reference = load_reference_corpus()
scorer = PhilosophicalTermScorer([processed], reference)
candidates = scorer.score_all(min_score=2.0, top_n=20)

# Show results
for term, score, components in candidates:
    print(f"{term:20} {score:.2f}")
```

**Output:**
```
abstraction          4.87
proletariat          3.45
bourgeoisie          3.21
commodity            2.98
fetishism            2.76
...
```

**See [Usage Guide](docs/usage-guide.md#complete-workflow-example) for the complete workflow.**

## Sample Data

The `data/sample/` directory contains test corpora:

- `philosopher_1920_cc.txt` - Philosopher' *History and Class Consciousness* (93KB)
- `hegel_phenomenology_excerpt.txt` - Hegel's *Phenomenology of Spirit* excerpt
- `test_philosophical_terms.txt` - Synthetic test data with known terms

## Project Structure

```
.
├── src/concept_mapper/        # Main package
│   ├── corpus/                # Document loading and models
│   ├── preprocessing/         # Tokenization, POS, lemmatization
│   ├── analysis/              # Frequency, TF-IDF, rarity, co-occurrence, relations
│   ├── search/                # Search and concordance
│   └── terms/                 # Term list management
├── tests/                     # Test suite (406 tests)
├── data/sample/               # Sample corpus
├── docs/                      # Documentation
└── output/                    # Analysis results
```

## Technology Stack

- **Python 3.14**
- **NLTK** (Natural Language Toolkit) - tokenization, POS tagging, lemmatization
- **NetworkX** - graph construction and analysis
- **pytest** - testing framework
- **Black** - code formatting
- **Ruff** - linting

## Testing

All 521 tests passing (2 skipped):

```bash
# Run all tests
pytest tests/ -v

# Run specific module tests
pytest tests/test_corpus.py -v
pytest tests/test_analysis.py -v
pytest tests/test_search.py -v
pytest tests/test_cooccurrence.py -v
pytest tests/test_relations.py -v
pytest tests/test_graph.py -v
pytest tests/test_export.py -v
pytest tests/test_cli.py -v

# Run with coverage
pytest tests/ --cov=src/concept_mapper
```

## Roadmap

- ✅ **Phases 0-11:** Complete
  - ✅ Phase 0-1: Corpus loading and preprocessing
  - ✅ Phase 2-3: Frequency analysis and rarity detection
  - ✅ Phase 4: Term list management
  - ✅ Phase 5: Search and concordance
  - ✅ Phase 6: Co-occurrence analysis
  - ✅ Phase 7: Relation extraction
  - ✅ Phase 8: Graph construction
  - ✅ Phase 9: Export and visualization
  - ✅ Phase 10: CLI interface
  - ✅ Phase 11: Documentation and examples

**See [Development Roadmap](docs/concept-mapper-roadmap.md) for detailed phase breakdown.**

## Development

```bash
# Install development dependencies
pip install -r requirements.txt

# Install package in editable mode
pip install -e .

# Format code
black src/ tests/

# Lint
ruff check src/ tests/

# Run tests
pytest tests/ -v
```

## Use Cases

1. **Digital Humanities Research**
   - Identify key concepts in philosophical texts
   - Trace conceptual evolution across works
   - Compare authors' conceptual frameworks

2. **Literature Analysis**
   - Find author-specific terminology
   - Analyze concept relationships
   - Visualize conceptual networks

3. **Academic Writing**
   - Ensure consistent terminology usage
   - Identify central concepts for indexing
   - Generate concept maps for papers

## Examples by Phase

### Find Key Terms (Phase 3)
```python
scorer = PhilosophicalTermScorer(docs, reference)
terms = scorer.get_high_confidence_terms(min_signals=3)
# Returns: [(term, score, components), ...]
```

### Search in Context (Phase 5)
```python
from concept_mapper.search import get_context

windows = get_context("abstraction", docs, n_sentences=2)
for w in windows:
    print(w)  # Shows 2 sentences before/after
```

### Find Associations (Phase 6)
```python
from concept_mapper.analysis import get_top_cooccurrences

top = get_top_cooccurrences("consciousness", docs, n=10, method="pmi")
# Returns: [(term, pmi_score), ...]
```

### Extract Relations (Phase 7)
```python
from concept_mapper.analysis import get_relations

relations = get_relations("being", docs)
for r in relations:
    print(f"{r.source} --[{r.relation_type}]--> {r.target}")
    print(f"  Evidence: {len(r.evidence)} sentences")
```

**See [Usage Guide](docs/usage-guide.md) for complete examples.**

## License

MIT License - See LICENSE file for details

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{concept_mapper,
  title={Concept Mapper: Philosophical Term Extraction and Network Analysis},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/concept-mapper}
}
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Format code with Black
6. Submit a pull request

## Contact

Questions or suggestions? Open an issue on GitHub.
