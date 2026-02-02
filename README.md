# Concept Mapper

A tool for extracting and visualizing an author's idiosyncratic conceptual vocabulary from philosophical texts.

## Overview

Concept Mapper analyzes primary texts to identify author-specific philosophical terminology—neologisms and terms with specialized technical meaning that are statistically distinctive compared to general English. It then maps relationships between these concepts through co-occurrence analysis and grammatical extraction, producing interactive network visualizations.

**Examples of target terms:** Aristotle's *eudaimonia*, Spinoza's *affect*, Hegel's *sublation*, Philosopher' *abstraction*, Deleuze & Guattari's *body without organs*

## Quick Start

### Installation

```bash
# Install package in development mode with all dependencies
uv pip install -e .

# Verify CLI installation
cmapr --help

# Run tests
pytest tests/ -v
```

### Try the Example Workflow

Process the included sample philosophical text:

```bash
# Run the complete workflow script
bash examples/workflow.sh

# Or run commands individually:
cmapr ingest samples/sample1_analytic_pragmatism.txt
cmapr rarities output/corpus/sample1_analytic_pragmatism.json --top-n 20
cmapr graph output/corpus/sample1_analytic_pragmatism.json -t output/terms/sample1_analytic_pragmatism.json
cmapr export output/graphs/sample1_analytic_pragmatism.json --format html

# Open the visualization
open output/exports/sample1_analytic_pragmatism/index.html
```

**Output structure:**
```
output/
├── corpus/         # Preprocessed corpora (named after source files)
│   ├── sample1_analytic_pragmatism.json
│   └── sample2_poststructural_political.json
├── terms/          # Detected terms (named after source files)
│   ├── sample1_analytic_pragmatism.json
│   └── sample2_poststructural_political.json
├── graphs/         # Concept graphs (named after source files)
│   ├── sample1_analytic_pragmatism.json
│   └── sample2_poststructural_political.json
├── exports/        # Visualizations (subdirectory per source)
│   ├── sample1_analytic_pragmatism/
│   │   └── index.html
│   └── sample2_poststructural_political/
│       └── index.html
└── cache/          # Session cache (reference corpus, etc.)
```

## Complete Tutorial

### Sample Corpus

The `samples/sample1_analytic_pragmatism.txt` file contains a passage on analytic philosophy and pragmatism, featuring characteristic philosophical terminology:

- **Meaning-variance** - Terms shift significance across theoretical frameworks
- **Instrumental-warranting** - Knowledge justified through practical success
- **Conceptual schemes** - Category-matrices that organize experience
- **Truth-aptness** - Whether sentences can bear truth values
- **Referential opacity** - Context where substitution of co-referring terms fails

### Step 1: Ingest and Preprocess

Load the sample text and preprocess it (tokenization, POS tagging, lemmatization):

```bash
cmapr ingest samples/sample1_analytic_pragmatism.txt
```

**Expected output:**
```
✓ Saved 1 processed document(s) to output/corpus/sample1_analytic_pragmatism.json
```

**What happens:** The raw text is tokenized into sentences and words, each word is tagged with its part of speech (NN, VB, etc.), and lemmatized (e.g., "entities" → "entity").

### Step 2: Detect Philosophical Terms

Identify author-specific terminology using statistical rarity analysis:

```bash
cmapr rarities output/corpus/sample1_analytic_pragmatism.json \
  --method hybrid \
  --threshold 1.5 \
  --top-n 20
```

**Expected output:**
```
Top 20 rare terms:
------------------------------------------------------------
meaning-variance        5.12
instrumental-warranting 4.89
truth-aptness          4.43
referential opacity    3.98
conceptual schemes     3.67
...

✓ Saved 20 terms to output/terms/sample1_analytic_pragmatism.json
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
cmapr rarities output/corpus/sample1_analytic_pragmatism.json --method ratio --top-n 10

# TF-IDF only
cmapr rarities output/corpus/sample1_analytic_pragmatism.json --method tfidf --top-n 10

# Neologisms only
cmapr rarities output/corpus/sample1_analytic_pragmatism.json --method neologism --top-n 10
```

### Step 3: Search and Concordance

Find where specific terms appear:

```bash
# Basic search
cmapr search output/corpus/sample1_analytic_pragmatism.json "abstraction"

# Search with context (2 sentences before/after)
cmapr search output/corpus/sample1_analytic_pragmatism.json "dialectical" --context 2

# Search with sentence diagrams for all matches
cmapr search output/corpus/sample1_analytic_pragmatism.json "abstraction" --diagram

# Search with diagrams in tree format
cmapr search output/corpus/sample1_analytic_pragmatism.json "dialectic" --diagram --diagram-format tree

# KWIC concordance
cmapr concordance output/corpus/sample1_analytic_pragmatism.json "praxis" --width 40

# Create sentence diagram for a single sentence
cmapr diagram "Abstraction transforms social relations into things."
```

**Expected KWIC output:**
```
KWIC Concordance for 'praxis' (3 occurrences):
================================================================================
                    ... of theory and practice through  | praxis |  differs from mere contemplation...
```

**Expected diagram output:**
```
Found 1 occurrence(s) of 'abstraction':
======================================================================

[1] document (sentence 5):
    Abstraction transforms social relations into things.

    Diagram:
    transforms (root)
      └─ Abstraction (nsubj)
      └─ relations (obj)
        └─ social (amod)
        └─ things (obl)
          └─ into (case)
      └─ . (punct)
```

### Step 4: Build Concept Graph

Create a network graph showing relationships between terms:

**Method A: Co-occurrence (proximity-based)**

```bash
cmapr graph output/corpus/sample1_analytic_pragmatism.json \
  --terms output/terms/sample1_analytic_pragmatism.json \
  --method cooccurrence \
  --threshold 0.3
```

**Expected output:**
```
✓ Graph: 18 nodes, 42 edges
✓ Saved graph to output/graphs/sample1_analytic_pragmatism.json
```

Edges represent terms that frequently appear together (same sentence or nearby sentences), weighted by PMI (Pointwise Mutual Information).

**Method B: Relations (grammar-based)**

```bash
cmapr graph output/corpus/sample1_analytic_pragmatism.json \
  --terms output/terms/sample1_analytic_pragmatism.json \
  --method relations
```

Edges represent grammatical relationships:
- **SVO triples**: "Praxis unifies theory"
- **Copular**: "Abstraction is transformation"
- **Prepositional**: "process of separation"

### Step 5: Export and Visualize

Generate an interactive HTML visualization:

```bash
cmapr export output/graphs/sample1_analytic_pragmatism.json \
  --format html \
  --title "Critical Theory Concept Network"
```

**Open the visualization:**
```bash
open output/exports/index.html
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
cmapr export output/graphs/sample1_analytic_pragmatism.json --format graphml -o output/exports/graph.graphml

# CSV for spreadsheets
cmapr export output/graphs/sample1_analytic_pragmatism.json --format csv -o output/exports/csv/

# GEXF for Gephi
cmapr export output/graphs/sample1_analytic_pragmatism.json --format gexf -o output/exports/graph.gexf
```

## CLI Reference

### Core Commands

```bash
# Ingest and preprocess text
cmapr ingest <path> [-o OUTPUT] [-r]

# Detect rare/philosophical terms
cmapr rarities <corpus> [--method METHOD] [--threshold N] [--top-n N] [-o OUTPUT]

# Search for terms in context
cmapr search <corpus> <query> [--context N] [-o OUTPUT]

# KWIC concordance
cmapr concordance <corpus> <term> [--width N] [-o OUTPUT]

# Create sentence diagram
cmapr diagram <sentence> [--format FORMAT] [-o OUTPUT]

# Build concept graph
cmapr graph <corpus> --terms <terms> [--method METHOD] [--threshold N] [-o OUTPUT]

# Export/visualize graph
cmapr export <graph> --format <FORMAT> [-o OUTPUT] [--title TITLE]
```

### Analyzing Your Own Texts

```bash
# Single file - outputs automatically named after source
cmapr ingest your_document.txt
cmapr rarities output/corpus/your_document.json
cmapr graph output/corpus/your_document.json -t output/terms/your_document.json
cmapr export output/graphs/your_document.json --format html
open output/exports/your_document/index.html

# Multiple files - each gets its own outputs, no overwrites!
cmapr ingest text1.txt
cmapr ingest text2.txt
cmapr ingest text3.txt
ls output/corpus/  # text1.json, text2.json, text3.json

# Directory of files - creates single merged corpus named after directory
cmapr ingest path/to/corpus/ -r -o my_corpus.json
cmapr rarities my_corpus.json -o my_terms.json
cmapr graph my_corpus.json -t my_terms.json -o my_graph.json
cmapr export my_graph.json --format html -o viz/

# Custom output locations (explicit -o always works)
cmapr ingest doc.txt -o corpus.json
cmapr rarities corpus.json -o terms.json
cmapr graph corpus.json -t terms.json -o graph.json
cmapr export graph.json --format html -o viz/
```

### Tuning Parameters

**Rarity detection:**
- `--threshold`: Minimum score (lower = more terms, higher = only very distinctive)
- `--top-n`: Number of top terms to extract
- `--method`: Detection method (ratio, tfidf, neologism, hybrid)

**Graph construction:**
- `--threshold`: Minimum edge weight (higher = sparser graph, only strong connections)
- `--method`: Construction method (cooccurrence vs. relations)

## Features

### Text Processing & Analysis
- **Document Loading** - Load single files or entire directories of texts
- **Preprocessing** - Tokenization, POS tagging, and lemmatization with NLTK
- **Frequency Analysis** - Word frequencies, TF-IDF scoring, corpus comparison
- **Search & Concordance** - Find terms in context with KWIC displays and context windows

### Term Detection
- **Multi-Method Detection** - 5 statistical signals for identifying philosophical terms:
  - Corpus-comparative frequency analysis (primary method)
  - TF-IDF scoring against reference corpus
  - Neologism detection via WordNet lookup
  - Definitional context identification
  - Capitalization analysis for reified abstractions
- **Hybrid Scoring** - Configurable weights combine all signals
- **Term Management** - CRUD operations with import/export (JSON, CSV, TXT)
- **Auto-Population** - Generate term lists automatically from analysis

### Relationship Extraction
- **Co-occurrence Analysis** - Sentence-level and windowed co-occurrence with PMI/LLR
- **Relation Extraction** - SVO triples, copular definitions, prepositional relations
- **Pattern-Based** - NLTK-powered grammatical extraction with evidence aggregation

### Graph Construction & Analysis
- **NetworkX Integration** - Build directed/undirected concept graphs
- **Multiple Methods** - Construct from co-occurrence or extracted relations
- **Graph Operations** - Merge, prune, filter, extract subgraphs
- **Graph Metrics** - Centrality, community detection, density, paths

### Visualization & Export
- **Interactive HTML** - D3.js force-directed visualizations (drag, zoom, tooltips)
- **Multiple Formats** - GraphML (Gephi/yEd), GEXF, DOT (Graphviz), CSV
- **Standalone** - Self-contained HTML files with inlined data (no CORS issues)
- **Customizable** - Node sizing, community coloring, evidence display

### User Interfaces
- **CLI** - Complete command-line interface with 6 commands
- **Python API** - Full programmatic access with comprehensive docstrings
- **Batch Processing** - Progress bars and multi-file support

**See [API Reference](docs/api-reference.md) for detailed examples and [Development Roadmap](docs/roadmap.md) for implementation phases.**

## Python API Usage

Complete workflow demonstrating all features:

```python
from pathlib import Path
from concept_mapper.corpus.loader import load_file
from concept_mapper.preprocessing.pipeline import preprocess
from concept_mapper.analysis.reference import load_reference_corpus
from concept_mapper.analysis.rarity import PhilosophicalTermScorer
from concept_mapper.analysis.cooccurrence import build_cooccurrence_matrix
from concept_mapper.graph import graph_from_cooccurrence, graph_from_relations
from concept_mapper.analysis.relations import get_relations
from concept_mapper.terms.models import TermList
from concept_mapper.terms.manager import TermManager
from concept_mapper.export import export_d3_json, export_graphml, export_csv, export_gexf, generate_html

# 1. Load and preprocess
doc = load_file("samples/sample1_analytic_pragmatism.txt")
processed = preprocess(doc)
print(f"✓ Processed {len(processed.sentences)} sentences")

# 2. Load reference corpus
reference = load_reference_corpus()

# 3. Detect philosophical terms
scorer = PhilosophicalTermScorer([processed], reference, use_lemmas=True)
candidates = scorer.score_all(min_score=1.5, top_n=20)
print(f"\n✓ Top terms:")
for term, score, _ in candidates[:5]:
    print(f"  {term:20} {score:.2f}")

# 4. Create term list and save
term_list = TermList.from_dict({"terms": [{"term": term, "metadata": {"score": score}} for term, score, _ in candidates]})
manager = TermManager(term_list)
manager.export_to_json(Path("output/terms.json"))

# 5. Build co-occurrence graph
matrix = build_cooccurrence_matrix(term_list, [processed], method="pmi", window="sentence")
graph_cooccur = graph_from_cooccurrence(matrix, threshold=0.3)
print(f"\n✓ Co-occurrence graph: {graph_cooccur.node_count()} nodes, {graph_cooccur.edge_count()} edges")

# 6. Build relations graph
all_relations = []
for term_data in term_list:
    relations = get_relations(term_data["term"], [processed])
    all_relations.extend(relations)
graph_relations = graph_from_relations(all_relations)
print(f"✓ Relations graph: {graph_relations.node_count()} nodes, {graph_relations.edge_count()} edges")

# 7. Export and visualize
viz_dir = Path("output/visualization")
html_path = generate_html(graph_cooccur, viz_dir, title="Concept Network")
export_graphml(graph_cooccur, Path("output/graph.graphml"))
export_csv(graph_cooccur, Path("output/csv"))
print(f"\n✓ Visualization ready: {html_path}")
```

**See [API Reference](docs/api-reference.md) for complete documentation.**

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

The `samples/` directory contains sample texts to use as inputs:

- `sample1_analytic_pragmatism.txt` - Analytic philosophy and pragmatism concepts (212 words)
- `sample2_poststructural_political.txt` - Poststructural and political philosophy concepts (203 words)
- `sample3_mind_consciousness.txt` - Philosophy of mind and consciousness concepts (290 words)

## Example Workflows

The `examples/` directory contains complete example workflows demonstrating the full pipeline.

**Quick Start:**
```bash
cd examples && bash workflow.sh
```

This runs the complete workflow:
1. Ingest and preprocess the sample text
2. Detect philosophical terms
3. Build concept graphs (co-occurrence and relations)
4. Generate HTML visualization
5. Export to multiple formats (GraphML, CSV, GEXF)

**Customization:**
To adapt these workflows for your own texts:
1. Place your text files in `samples/`
2. Edit `workflow.sh` to reference your files
3. Run the workflow
4. Find results in `output/`

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
├── tests/                     # Test suite (542 tests)
├── samples/                   # Sample texts (inputs)
├── examples/                  # Demo workflows (scripts)
├── docs/                      # Documentation
├── data/reference/            # Bundled reference datasets
└── output/                    # Generated results (gitignored)
    ├── corpus/                # Processed corpora
    ├── terms/                 # Extracted terms
    ├── graphs/                # Concept graphs
    ├── exports/               # Visualizations and export formats
    └── cache/                 # Session cache
```

**Directory organization rationale:**
- **`samples/`** - Three sample input texts (sample1, sample2, sample3) covering diverse philosophical traditions. For testing and learning.
- **`examples/`** - Workflow scripts only (no data files). Shows how to use the tool.
- **`data/reference/`** - Bundled reference datasets that ship with the tool (e.g., Brown corpus frequencies). Not user-generated.
- **`output/`** - All generated results. Fully gitignored and can be safely deleted. Organized into subdirectories by output type.

## Technology Stack

- **Python 3.14**
- **NLTK** - tokenization, POS tagging, lemmatization
- **Stanza** - dependency parsing and sentence diagramming
- **NetworkX** - graph construction and analysis
- **pytest** - testing framework (540 passing, 2 skipped)
- **Ruff** - formatting & linting

## Testing

All tests passing (540 passed, 2 skipped):

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
# Install package with development dependencies
uv pip install -e ".[dev]"

# Or use the Makefile shortcuts
make format    # Format code with Ruff
make lint      # Lint with Ruff
make test      # Run tests with pytest
make check     # Run all checks
```

## Documentation

- **[API Reference](docs/api-reference.md)** - Complete guide with examples and API documentation
- **[Development Roadmap](docs/roadmap.md)** - Complete project plan

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

## Project Status

✅ **All features complete** - Phases 0-11 fully implemented and tested

**See [Development Roadmap](docs/roadmap.md) for detailed development history and phase breakdown.**

## License

MIT License - See LICENSE file for details

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{concept_mapper,
  title={Concept Mapper: Philosophical Term Extraction and Network Analysis},
  author={Your Name},
  year={2026},
  url={https://github.com/yourusername/cmapr}
}
```

## Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass (`pytest tests/`)
5. Format code with Ruff
6. Submit a pull request

## Contact

Questions or suggestions? Open an issue on GitHub.
