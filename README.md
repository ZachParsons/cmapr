# Concept Mapper

A tool for extracting and visualizing an author's idiosyncratic conceptual vocabulary from philosophical texts.

## Overview

Concept Mapper analyzes primary texts to identify author-specific philosophical terminology—neologisms and terms with specialized technical meaning that are statistically distinctive compared to general English. It then maps relationships between these concepts through co-occurrence analysis and grammatical extraction, producing interactive network visualizations.

**Examples of target terms:** Aristotle's *eudaimonia*, Spinoza's *affect*, Hegel's *sublation*, Deleuze & Guattari's *body without organs*

## Quick Start

### Installation

```bash
# Install package in development mode
uv pip install -e ".[dev]"

# Verify installation
cmapr --help

# Download required NLTK data
python scripts/download_nltk_data.py
```

### Run the Example Workflow

Process Umberto Eco's "Semiotics and the Philosophy of Language" (~110K words):

```bash
# Complete automated workflow
bash examples/workflow.sh

# Or run individually:
cmapr ingest samples/eco_spl.txt
cmapr rarities output/corpus/eco_spl.json --method ratio --top-n 50
cmapr graph output/corpus/eco_spl.json -t output/terms/eco_spl.json
cmapr export output/graphs/eco_spl.json --format html
open output/exports/eco_spl/index.html
```

**Output:**
```
output/
├── corpus/         # Preprocessed texts with linguistic annotations
├── terms/          # Extracted distinctive terms
├── graphs/         # Relationship networks
└── exports/        # Visualizations (HTML, JSON, CSV)
```

## Documentation

- **[Complete Tutorial](docs/tutorial.md)** - Step-by-step walkthrough with explanations
- **[API Reference](docs/api-reference.md)** - Full CLI and Python API documentation
- **[Replacement Guide](docs/replacement.md)** - Synonym replacement for text transformation
- **[Roadmap](docs/roadmap.md)** - Development status and future plans

## Features

### Core Analysis
- **Term Extraction**: Statistical detection of distinctive vocabulary
  - Ratio method: Compare corpus vs reference (Brown corpus)
  - TF-IDF: Multi-document distinctiveness
  - Combined signals: Neologisms, definitional contexts, POS patterns
- **Relation Mining**: Discover conceptual relationships
  - Co-occurrence: Terms appearing together in context
  - SVO extraction: Subject-verb-object grammatical triples
- **Visualization**: Interactive network graphs with D3.js

### Text Processing
- **Preprocessing**: NLTK tokenization, POS tagging, lemmatization
- **Paragraph segmentation**: Structural analysis for context
- **OCR cleaning**: Automatic fix for scanned PDF artifacts
- **PDF support**: Direct ingestion from PDF files (via pdfplumber)

### Transformation
- **Synonym replacement**: Replace terms while preserving inflections
  - Maintains tense (ran → sprinted)
  - Preserves number (cats → felines)
  - Handles phrases ("body without organs" → "BwO")

### CLI Commands
- `ingest` - Load and preprocess texts
- `rarities` - Extract distinctive terms
- `graph` - Build concept relationship network
- `export` - Generate visualizations (HTML/JSON/CSV)
- `analyze` - Deep contextual analysis of specific terms
- `search` - Find sentences containing terms
- `frequencies` - Word frequency analysis
- `tfidf` - TF-IDF scoring
- `replace` - Synonym replacement with inflection preservation
- `analyze-context` - Extract contextual relations (SVO + co-occurrence)

## Sample Data

`samples/eco_spl.txt` - Umberto Eco's complete "Semiotics and the Philosophy of Language" (110K words)

Rich in semiotic terminology: *semiosis, interpretant, Porphyrian tree, isotopy, synecdoche, rhizome*

## Project Structure

```
cmapr/
├── src/concept_mapper/          # Main package
│   ├── analysis/                # Term extraction, frequency analysis
│   ├── corpus/                  # Document models and loading
│   ├── graph/                   # Network construction
│   ├── preprocessing/           # Tokenization, tagging, lemmatization
│   ├── relations/               # SVO extraction, co-occurrence
│   ├── search/                  # Term search and matching
│   ├── transformations/         # Text rewriting (synonym replacement)
│   ├── visualization/           # Export to HTML/JSON/CSV
│   └── cli.py                   # Command-line interface
├── tests/                       # Comprehensive test suite (665 tests)
├── docs/                        # Documentation
├── samples/                     # Example corpus
├── scripts/                     # Utility scripts
└── output/                      # Generated outputs (gitignored)
```

## Technology Stack

- **NLP**: NLTK (tokenization, POS tagging, lemmatization), Stanza (dependency parsing)
- **Graph**: NetworkX for network analysis and algorithms
- **CLI**: Click for command-line interface
- **Inflection**: Pattern3 and inflect for morphological transformations
- **PDF**: pdfplumber for text extraction
- **Testing**: pytest (665 passing tests)
- **Tooling**: Ruff (formatting/linting), uv (package management)

## Development

### Setup
```bash
# Install with dev dependencies
uv pip install -e ".[dev]"

# Run all checks (format, lint, test)
make check

# Individual commands
make format    # Format with Ruff
make lint      # Lint with Ruff
make test      # Run pytest
```

### Running Tests
```bash
# Full suite (665 tests)
pytest tests/ -v

# Specific modules
pytest tests/test_corpus.py -v
pytest tests/preprocessing/ -v
pytest tests/transformations/ -v
```

### Code Style
- Follow PEP 8 with type hints
- Prefer functional style (expressions over statements)
- Write tests for all new features
- Run `make check` before committing

## Use Cases

**Philosophical research:**
- Map an author's conceptual vocabulary
- Compare terminology across texts or authors
- Identify key concepts and their relationships

**Corpus linguistics:**
- Extract domain-specific terminology
- Analyze specialized vocabularies
- Discover semantic networks in texts

**Digital humanities:**
- Visualize conceptual structures in primary sources
- Explore evolution of ideas across documents
- Support close reading with computational analysis

## Project Status

**Current version**: 1.0.0

**Recently completed:**
- ✅ PDF input support (pdfplumber integration)
- ✅ OCR text cleaning for scanned documents
- ✅ Synonym replacement with inflection preservation
- ✅ Paragraph segmentation for structural analysis
- ✅ Contextual relation extraction (SVO + co-occurrence)

**See [docs/roadmap.md](docs/roadmap.md) for future plans and development status.**

## Testing

665 passing tests covering:
- Corpus loading (text and PDF)
- Preprocessing pipeline (tokenization, POS tagging, lemmatization)
- Term extraction (ratio, TF-IDF, combined methods)
- Graph construction (co-occurrence, SVO relations)
- Visualization export (HTML, JSON, CSV)
- CLI commands and workflows
- Synonym replacement and inflection

## License

MIT License

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{concept_mapper,
  author = {Zach},
  title = {Concept Mapper: Extract and Visualize Philosophical Vocabulary},
  year = {2025},
  url = {https://github.com/yourusername/concept-mapper}
}
```

## Contributing

Contributions welcome! See [.claude/rules.md](.claude/rules.md) for development guidelines.

**Development workflow:**
1. Fork the repository
2. Create a feature branch
3. Write tests for new functionality
4. Run `make check` to verify
5. Submit a pull request

**Key principles:**
- Always write tests
- Follow existing code patterns
- Update documentation
- Keep commits focused

---

**Questions or issues?** Open an issue on GitHub or consult the [complete documentation](docs/).
