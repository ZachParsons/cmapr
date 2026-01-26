# Concept Mapper

A tool for extracting and visualizing an author's idiosyncratic conceptual vocabulary from philosophical texts.

## Overview

Concept Mapper analyzes primary texts to identify author-specific philosophical terminology—neologisms and terms with specialized technical meaning that are statistically distinctive compared to general English. It then maps relationships between these concepts through co-occurrence analysis and grammatical extraction, producing interactive network visualizations.

**Examples of target terms:** Aristotle's *eudaimonia*, Spinoza's *affect*, Hegel's *sublation*, Philosopher' *abstraction*, Deleuze & Guattari's *body without organs*

## Quick Start

### Installation

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

### Try the Example Workflow

Process the included sample philosophical text:

```bash
# Run the complete workflow script
bash examples/workflow.sh

# Or run commands individually:
concept-mapper ingest examples/sample_text.txt -o examples/corpus.json
concept-mapper rarities examples/corpus.json --top-n 20 -o examples/terms.json
concept-mapper graph examples/corpus.json -t examples/terms.json -o examples/graph.json
concept-mapper export examples/graph.json --format html -o examples/visualization/

# Open the visualization
open examples/visualization/index.html
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

## Complete Tutorial

### Sample Corpus

The `examples/sample_text.txt` file contains a passage on critical social theory and Thinkerist philosophy, featuring characteristic philosophical terminology:

- **Abstraction** - Transformation of human relations into thing-like entities
- **Praxis** - Unity of theory and practice through transformative action
- **Dialectical negation** - Hegelian method of sublation and synthesis
- **Separation** - Separation of workers from products and species-being
- **Hegemony** - Cultural leadership and consent (Gramsci)
- **Recognition** - Mutual acknowledgment between subjects (Hegel/Honneth)

### Step 1: Ingest and Preprocess

Load the sample text and preprocess it (tokenization, POS tagging, lemmatization):

```bash
concept-mapper ingest examples/sample_text.txt -o examples/corpus.json
```

**Expected output:**
```
✓ Saved 1 processed document(s) to examples/corpus.json
```

**What happens:** The raw text is tokenized into sentences and words, each word is tagged with its part of speech (NN, VB, etc.), and lemmatized (e.g., "entities" → "entity").

### Step 2: Detect Philosophical Terms

Identify author-specific terminology using statistical rarity analysis:

```bash
concept-mapper rarities examples/corpus.json \
  --method hybrid \
  --threshold 1.5 \
  --top-n 20 \
  -o examples/terms.json
```

**Expected output:**
```
Top 20 rare terms:
------------------------------------------------------------
abstraction          4.87
praxis               3.45
dialectical          3.21
separation           2.98
hegemony             2.76
...

✓ Saved 20 terms to examples/terms.json
```

**What happens:** Each term is scored based on:
1. Relative frequency (how often it appears here vs. general English)
2. TF-IDF score vs. reference corpus
3. Neologism detection (not in WordNet dictionary)
4. Definitional context (appears in definitional sentences)
5. Capitalization (reified abstractions like "Being")

**Try different methods:**
```bash
# Corpus-comparative ratio only
concept-mapper rarities examples/corpus.json --method ratio --top-n 10

# TF-IDF only
concept-mapper rarities examples/corpus.json --method tfidf --top-n 10

# Neologisms only
concept-mapper rarities examples/corpus.json --method neologism --top-n 10
```

### Step 3: Search and Concordance

Find where specific terms appear:

```bash
# Basic search
concept-mapper search examples/corpus.json "abstraction"

# Search with context (2 sentences before/after)
concept-mapper search examples/corpus.json "dialectical" --context 2

# KWIC concordance
concept-mapper concordance examples/corpus.json "praxis" --width 40

# Create sentence diagram
concept-mapper diagram "Abstraction transforms social relations into things."
```

**Expected KWIC output:**
```
KWIC Concordance for 'praxis' (3 occurrences):
================================================================================
                    ... of theory and practice through  | praxis |  differs from mere contemplation...
```

### Step 4: Build Concept Graph

Create a network graph showing relationships between terms:

**Method A: Co-occurrence (proximity-based)**

```bash
concept-mapper graph examples/corpus.json \
  --terms examples/terms.json \
  --method cooccurrence \
  --threshold 0.3 \
  -o examples/graph.json
```

**Expected output:**
```
✓ Graph: 18 nodes, 42 edges
✓ Saved graph to examples/graph.json
```

Edges represent terms that frequently appear together (same sentence or nearby sentences), weighted by PMI (Pointwise Mutual Information).

**Method B: Relations (grammar-based)**

```bash
concept-mapper graph examples/corpus.json \
  --terms examples/terms.json \
  --method relations \
  -o examples/graph_relations.json
```

Edges represent grammatical relationships:
- **SVO triples**: "Praxis unifies theory"
- **Copular**: "Abstraction is transformation"
- **Prepositional**: "process of separation"

### Step 5: Export and Visualize

Generate an interactive HTML visualization:

```bash
concept-mapper export examples/graph.json \
  --format html \
  --title "Critical Theory Concept Network" \
  -o examples/visualization/
```

**Open the visualization:**
```bash
open examples/visualization/index.html
```

**Features of the visualization:**
- **Force-directed layout**: Nodes repel, connected nodes attract
- **Interactive**: Drag nodes, zoom/pan
- **Color-coded**: Nodes colored by community detection
- **Sized by importance**: Node size reflects centrality or frequency
- **Hover for details**: Tooltips show term information

**Export to other formats:**

```bash
# GraphML for Gephi
concept-mapper export examples/graph.json --format graphml -o examples/graph.graphml

# CSV for spreadsheets
concept-mapper export examples/graph.json --format csv -o examples/csv/

# GEXF for Gephi
concept-mapper export examples/graph.json --format gexf -o examples/graph.gexf
```

## CLI Reference

### Core Commands

```bash
# Ingest and preprocess text
concept-mapper ingest <path> [-o OUTPUT] [-r]

# Detect rare/philosophical terms
concept-mapper rarities <corpus> [--method METHOD] [--threshold N] [--top-n N] [-o OUTPUT]

# Search for terms in context
concept-mapper search <corpus> <query> [--context N] [-o OUTPUT]

# KWIC concordance
concept-mapper concordance <corpus> <term> [--width N] [-o OUTPUT]

# Create sentence diagram
concept-mapper diagram <sentence> [--format FORMAT] [-o OUTPUT]

# Build concept graph
concept-mapper graph <corpus> --terms <terms> [--method METHOD] [--threshold N] [-o OUTPUT]

# Export/visualize graph
concept-mapper export <graph> --format <FORMAT> [-o OUTPUT] [--title TITLE]
```

### Analyzing Your Own Texts

```bash
# Single file
concept-mapper ingest your_document.txt
concept-mapper rarities output/corpus/corpus.json
concept-mapper graph output/corpus/corpus.json -t output/terms/terms.json
concept-mapper export output/graphs/graph.json --format html

# Directory of files
concept-mapper ingest path/to/corpus/ -r -o my_corpus.json
concept-mapper rarities my_corpus.json -o my_terms.json
concept-mapper graph my_corpus.json -t my_terms.json -o my_graph.json
concept-mapper export my_graph.json --format html -o viz/

# Custom output locations
concept-mapper ingest doc.txt -o corpus.json
concept-mapper rarities corpus.json -o terms.json
concept-mapper graph corpus.json -t terms.json -o graph.json
concept-mapper export graph.json --format html -o viz/
```

### Tuning Parameters

**Rarity detection:**
- `--threshold`: Minimum score (lower = more terms, higher = only very distinctive)
- `--top-n`: Number of top terms to extract
- `--method`: Detection method (ratio, tfidf, neologism, hybrid)

**Graph construction:**
- `--threshold`: Minimum edge weight (higher = sparser graph, only strong connections)
- `--method`: Construction method (cooccurrence vs. relations)

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
- Sentence diagramming with dependency parsing
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
- Commands: ingest, rarities, search, concordance, diagram, graph, export
- Global options: `--verbose`, `--output-dir`
- Progress bars for batch operations
- End-to-end workflow support
- **[See examples →](docs/usage-guide.md#phase-10-cli-interface)**

## Python API Usage

For programmatic access, use the Python API directly:

```python
from concept_mapper.corpus.loader import load_file
from concept_mapper.preprocessing.pipeline import preprocess
from concept_mapper.analysis.reference import load_reference_corpus
from concept_mapper.analysis.rarity import PhilosophicalTermScorer
from concept_mapper.graph import graph_from_cooccurrence
from concept_mapper.analysis.cooccurrence import build_cooccurrence_matrix
from concept_mapper.export import export_d3_json, generate_html
from concept_mapper.terms.models import TermList

# Load and preprocess
doc = load_file("examples/sample_text.txt")
processed = preprocess(doc)

# Detect terms
reference = load_reference_corpus()
scorer = PhilosophicalTermScorer([processed], reference)
candidates = scorer.score_all(min_score=1.5, top_n=20)

# Create term list
term_list = TermList.from_dict({
    "terms": [{"term": term, "metadata": {"score": score}}
              for term, score, _ in candidates]
})

# Build graph
matrix = build_cooccurrence_matrix(term_list, [processed], method="pmi")
graph = graph_from_cooccurrence(matrix, threshold=0.3)

# Export
generate_html(graph, "examples/viz/", title="My Concept Network")
```

**Expected output:**
```
abstraction          4.87
proletariat          3.45
bourgeoisie          3.21
commodity            2.98
...
```

**See [Usage Guide](docs/usage-guide.md) for complete API documentation.**

## Understanding Results

### What Makes a Good Philosophical Term?

**High-scoring terms typically:**
1. Appear frequently in the text (absolute frequency)
2. Rarely appear in general English (low reference corpus frequency)
3. Don't appear in standard dictionaries (neologisms)
4. Appear in definitional contexts ("X is...", "we call this X")
5. Are capitalized mid-sentence (reified abstractions)

**Examples:**
- ✓ "abstraction" - Technical term, high frequency in critical theory
- ✓ "praxis" - Philosophical concept, rare in general English
- ✓ "Aufhebung" - Hegelian neologism
- ✗ "question" - Common word, even if frequent in philosophy

### Graph Interpretation

**Dense clusters** indicate:
- Closely related concepts
- Co-occurring conceptual families

**Bridge nodes** (high betweenness centrality):
- Concepts that connect different areas
- Central to the author's framework

**Peripheral nodes**:
- Specialized or less-integrated concepts
- May indicate tangential discussions

## Troubleshooting

**Issue: No terms detected**
- Try lowering `--threshold` value
- Check that your text has sufficient length (> 500 words recommended)
- Verify text is in English

**Issue: Graph has no edges**
- Lower the `--threshold` in graph command
- Use `--method relations` instead of cooccurrence
- Check that term list has multiple terms

**Issue: Visualization won't open**
- Use absolute path: `file:///full/path/to/index.html`
- Check browser console for JavaScript errors
- Verify graph data exists in visualization directory

## Sample Data

The `data/sample/` directory contains test corpora:

- `philosopher_1920_cc.txt` - Philosopher' *History and Class Consciousness* (93KB)
- `hegel_phenomenology_excerpt.txt` - Hegel's *Phenomenology of Spirit* excerpt
- `test_philosophical_terms.txt` - Synthetic test data with known terms

The `examples/` directory contains:
- `sample_text.txt` - Critical theory passage for quick testing
- `workflow.sh` - Complete bash workflow script
- `workflow.py` - Complete Python workflow script

## Project Structure

```
.
├── src/concept_mapper/        # Main package
│   ├── corpus/                # Document loading and models
│   ├── preprocessing/         # Tokenization, POS, lemmatization
│   ├── analysis/              # Frequency, TF-IDF, rarity, co-occurrence, relations
│   ├── search/                # Search and concordance
│   ├── syntax/                # Dependency parsing and sentence diagramming
│   ├── terms/                 # Term list management
│   ├── graph/                 # Graph construction and metrics
│   ├── export/                # Export to various formats
│   ├── storage/               # Persistence layer
│   └── cli.py                 # Command-line interface
├── tests/                     # Test suite (521 tests)
├── data/sample/               # Sample corpus
├── examples/                  # Example workflows and outputs
├── docs/                      # Documentation
└── output/                    # Analysis results
```

## Technology Stack

- **Python 3.14**
- **NLTK** - tokenization, POS tagging, lemmatization
- **Stanza** - dependency parsing and sentence diagramming
- **NetworkX** - graph construction and analysis
- **pytest** - testing framework (521 tests passing)
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
pytest tests/test_cli.py -v

# Run with coverage
pytest tests/ --cov=src/concept_mapper
```

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

## Documentation

- **[Usage Guide](docs/usage-guide.md)** - Practical examples for each phase
- **[API Reference](docs/api-reference.md)** - Complete Python API reference
- **[Development Roadmap](docs/concept-mapper-roadmap.md)** - Complete project plan
- **[Validation](VALIDATION.md)** - Output validation and error handling

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

## Roadmap

- ✅ **Phases 0-11:** Complete
  - ✅ Phase 0-1: Corpus loading and preprocessing
  - ✅ Phase 2-3: Frequency analysis and rarity detection
  - ✅ Phase 4: Term list management
  - ✅ Phase 5: Search, concordance, and sentence diagramming
  - ✅ Phase 6: Co-occurrence analysis
  - ✅ Phase 7: Relation extraction
  - ✅ Phase 8: Graph construction
  - ✅ Phase 9: Export and visualization
  - ✅ Phase 10: CLI interface
  - ✅ Phase 11: Documentation and examples

**See [Development Roadmap](docs/concept-mapper-roadmap.md) for detailed phase breakdown.**

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
