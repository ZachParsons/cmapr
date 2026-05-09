# Concept Mapper

A tool for extracting and visualizing an author's idiosyncratic conceptual vocabulary from philosophical texts.

Concept Mapper analyzes primary texts to identify author-specific terminology — neologisms and terms with specialized technical meaning that are statistically distinctive compared to general English. It then maps relationships between these concepts through co-occurrence analysis and grammatical extraction, producing interactive network visualizations.

**Examples of target terms:** Aristotle's *eudaimonia*, Spinoza's *affect*, Hegel's *sublation*, Deleuze & Guattari's *body without organs*

<img width="1542" height="1162" alt="Screenshot 2026-04-12" src="https://github.com/user-attachments/assets/92eb0369-3cfc-4a5d-b672-aff6fb84725d" />
<img width="1231" height="1654" alt="Screenshot 2026-02-28" src="https://github.com/user-attachments/assets/8eade744-f0a1-426a-8c35-2e218fc71029" />


---

## Installation

```bash
uv pip install -e ".[dev]"
python scripts/download_nltk_data.py
cmapr --help
```

---

## Workflow

The four commands form a pipeline: **ingest → rarities → graph → export**. See [`docs/architecture.md`](docs/architecture.md) for the full pipeline diagram with all CLI commands and processing modules. Or run all at once:

```bash
cmapr run data/input/eco_spl.txt --toc data/input/eco_spl_toc.txt --top-n 50 --format html
```

### Step 1 — Ingest

Load and preprocess a text (tokenization, POS tagging, lemmatization):

```bash
cmapr ingest data/input/eco_spl.txt
# With table of contents for accurate structure detection:
cmapr ingest data/input/eco_spl.txt --toc data/input/eco_spl_toc.txt
# For scanned PDFs with OCR artifacts:
cmapr ingest data/input/eco_spl.pdf --clean-ocr
```

Produces `data/output/corpus/eco_spl/corpus.json` — sentences, tokens, POS tags, lemmas.

### Step 2 — Extract Rare Terms

Identify terms that appear frequently in the text but rarely in general English:

```bash
cmapr rarities data/output/corpus/eco_spl/corpus.json --top-n 50
```

Compares against the Brown corpus. Sample output from Eco:

```
semiosis      ratio: 245.0  freq: 98
porphyrian    ratio: 198.5  freq: 15
rhizome       ratio: 156.2  freq: 24
isotopy       ratio: 124.8  freq: 31
interpretant  ratio: 118.3  freq: 47
```

Produces `data/output/rarities/eco_spl/rarities.json`.

### Step 3 — Build Concept Graph

Map relationships between the distinctive terms via co-occurrence and SVO extraction:

```bash
cmapr graph data/output/corpus/eco_spl/corpus.json \
  -t data/output/rarities/eco_spl/rarities.json
```

Produces `data/output/graphs/eco_spl/graph.json`.

### Step 4 — Export Visualization

Generate an interactive D3.js force-directed graph:

```bash
cmapr export data/output/graphs/eco_spl/graph.json --format html
open data/output/exports/eco_spl/index.html
```

The visualization shows nodes sized by frequency, edges weighted by co-occurrence strength, and automatic community detection (clusters). Other formats: `--format graphml`, `--format csv`, `--format gexf`.

---

## Output Structure

```
data/output/
├── corpus/     # Preprocessed texts with linguistic annotations
├── rarities/   # Extracted distinctive terms
├── graphs/     # Relationship networks (JSON)
└── exports/    # Visualizations (HTML, CSV, GraphML)
```

Each subdirectory is namespaced by work: `corpus/<work>/corpus.json`, `graphs/<work>/graph.json`, etc. Per-term graphs live at `graphs/<work>/<term>.json` and their visualizations at `exports/<work>/<term>/`.

---

## Advanced Usage

### Deep analysis of a single term

```bash
cmapr analyze data/output/corpus/eco_spl/corpus.json "sign"
```

Extracts SVO triples ("sign → represents → object"), co-occurrence partners, and evidence sentences. Add `--format graph` to export a term-specific graph.

### Search

```bash
cmapr search data/output/corpus/eco_spl/corpus.json "semiosis" --context 2
cmapr search data/output/corpus/eco_spl/corpus.json "semiosis" --extract-significant
```

### Synonym replacement

Replace terms while preserving inflection (tense, number, case):

```bash
cmapr replace data/output/corpus/eco_spl/corpus.json "body,without,organs" "BwO" --preview
```

### Batch processing

```bash
cmapr ingest data/input/text1.txt
cmapr ingest data/input/text2.txt
# Then run rarities/graph/export on each corpus independently
```

---

## All Commands

| Command | Description |
|---|---|
| `ingest` | Load and preprocess text or PDF |
| `rarities` | Extract statistically distinctive terms |
| `graph` | Build concept relationship network |
| `export` | Generate visualization (HTML/CSV/GraphML/GEXF) |
| `analyze` | Deep contextual analysis of a specific term |
| `search` | Find sentences containing a term |
| `replace` | Synonym replacement with inflection preservation |
| `run` | Full pipeline in one command |
| `diagram` | Dependency parse diagram of a sentence |

Run `cmapr <command> --help` for full options.

---

## Sample Data

`data/input/eco_spl.txt` — Umberto Eco's "Semiotics and the Philosophy of Language" (~110K words). Rich in semiotic terminology: *semiosis, interpretant, Porphyrian tree, isotopy, synecdoche, rhizome*.

---

## Documentation

- **[API Reference](docs/api-reference.md)** — Python API: function signatures, parameters, return types
- **[Replacement Guide](docs/feat-replacement.md)** — Synonym replacement in depth
- **[Roadmap](docs/roadmap.md)** — Development status and future plans

---

## Project Structure

```
cmapr/
├── src/concept_mapper/   # Main package
│   ├── analysis/         # Term extraction, frequency, co-occurrence
│   ├── corpus/           # Document models and loading
│   ├── graph/            # Network construction
│   ├── preprocessing/    # Tokenization, tagging, lemmatization
│   ├── search/           # Term search and matching
│   ├── transformations/  # Synonym replacement
│   └── cli.py            # Command-line interface
├── tests/                # Test suite
├── docs/                 # Reference documentation
├── scripts/              # Utility scripts
└── data/                 # All input/output data (gitignored)
    ├── input/            # Source texts
    ├── output/           # Generated pipeline artifacts
    └── reference/        # Bundled reference data (Brown corpus freqs)
```

---

## Technology Stack

- **NLP**: NLTK (tokenization, POS tagging, lemmatization), Stanza (dependency parsing)
- **Graph**: NetworkX
- **Visualization**: D3.js (force-directed graphs)
- **CLI**: Click
- **Inflection**: Pattern3, inflect
- **PDF**: pdfplumber
- **Testing**: pytest
- **Tooling**: Ruff, uv

---

## Development

```bash
make check      # format + lint + test
make format     # Ruff format
make lint       # Ruff lint
make test       # pytest
```

Follow PEP 8 with type hints. Write tests for all new features. Run `make check` before committing.

---

## Use Cases

- Map an author's conceptual vocabulary across a text
- Compare terminology between authors or periods
- Identify key concepts and their relational structure
- Support close reading with computational analysis

---

## License

MIT

## Citation

```bibtex
@software{concept_mapper,
  author = {Zach},
  title = {Concept Mapper: Extract and Visualize Philosophical Vocabulary},
  year = {2025},
  url = {https://github.com/yourusername/concept-mapper}
}
```
