# Concept Mapper: Complete Workflow Example

This example demonstrates the complete workflow from raw text to interactive visualization.

## Sample Corpus

The `sample_text.txt` file contains a passage on critical social theory and Thinkerist philosophy, featuring characteristic philosophical terminology:

- **Abstraction** - Transformation of human relations into thing-like entities
- **Praxis** - Unity of theory and practice through transformative action
- **Dialectical negation** - Hegelian method of sublation and synthesis
- **Separation** - Separation of workers from products and species-being
- **Hegemony** - Cultural leadership and consent (Gramsci)
- **Recognition** - Mutual acknowledgment between subjects (Hegel/Honneth)
- And other critical theory concepts

## Step-by-Step Walkthrough

### Prerequisites

```bash
# Ensure you've installed the package
pip install -e .

# Verify CLI is available
concept-mapper --help
```

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

**Inspect the result:**
```bash
head -20 examples/corpus.json
```

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
Geist                          5.42
Aufhebung                      4.87
Sittlichkeit                   4.65
Selbstbewusstsein             4.23
dialectical                    3.98
Anerkennung                    3.76
sublation                      3.54
[...]

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
concept-mapper search examples/corpus.json "Geist"

# Search with context (2 sentences before/after)
concept-mapper search examples/corpus.json "dialectical" --context 2

# KWIC concordance
concept-mapper concordance examples/corpus.json "Geist" --width 40
```

**Expected KWIC output:**
```
KWIC Concordance for 'Geist' (3 occurrences):
================================================================================
                            ... degree it does so explicitly.  | Geist |  is the self-developing spirit...
                 ... totality is grounded in the dialectic.  | Geist | 's development as their object...
                          ... concrete universal and  | Geist | . Recognition actualizes Geist in...
```

### Step 4: Build Concept Graph

Create a network graph showing relationships between terms:

**Method A: Co-occurrence (proximity-based)**

```bash
concept-mapper graph examples/corpus.json \
  --terms examples/terms.json \
  --method cooccurrence \
  --threshold 0.3 \
  -o examples/graph_cooccur.json
```

**Expected output:**
```
✓ Graph: 18 nodes, 42 edges
✓ Saved graph to examples/graph_cooccur.json
```

Edges represent terms that frequently appear together (same sentence or nearby sentences), weighted by PMI (Pointwise Mutual Information).

**Method B: Relations (grammar-based)**

```bash
concept-mapper graph examples/corpus.json \
  --terms examples/terms.json \
  --method relations \
  -o examples/graph_relations.json
```

**Expected output:**
```
Extracting  [####################################]  20/20
Found 67 relations
Building graph...

✓ Graph: 16 nodes, 67 edges
✓ Saved graph to examples/graph_relations.json
```

Edges represent grammatical relationships:
- **SVO triples**: "Geist actualizes itself"
- **Copular**: "Geist is self-consciousness"
- **Prepositional**: "development of Geist"

### Step 5: Export and Visualize

Generate an interactive HTML visualization:

```bash
concept-mapper export examples/graph_cooccur.json \
  --format html \
  --title "Hegel Concept Network" \
  -o examples/visualization/
```

**Expected output:**
```
✓ Generated HTML visualization at examples/visualization/index.html
  Open in browser: file:///Users/you/concept-mapper/examples/visualization/index.html
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
concept-mapper export examples/graph_cooccur.json \
  --format graphml \
  -o examples/graph.graphml

# CSV for spreadsheets
concept-mapper export examples/graph_cooccur.json \
  --format csv \
  -o examples/csv/

# GEXF for Gephi
concept-mapper export examples/graph_cooccur.json \
  --format gexf \
  -o examples/graph.gexf
```

## Complete Workflow Script

The `workflow.sh` script runs the entire pipeline:

```bash
bash examples/workflow.sh
```

## Output Files

After running the complete workflow, you'll have:

```
examples/
├── sample_text.txt              # Input: Raw philosophical text
├── corpus.json                  # Preprocessed document (tokens, POS, lemmas)
├── terms.json                   # Detected philosophical terms
├── graph_cooccur.json          # Co-occurrence graph (D3 JSON)
├── graph_relations.json        # Relations graph (D3 JSON)
├── graph.graphml               # GraphML export for Gephi
├── graph.gexf                  # GEXF export for Gephi
├── csv/
│   ├── nodes.csv               # Nodes table
│   └── edges.csv               # Edges table
└── visualization/
    ├── index.html              # Interactive D3 visualization
    └── data.json               # Embedded graph data
```

## Next Steps

### Analyze Your Own Texts

1. **Prepare your corpus**: Place `.txt` files in a directory
2. **Ingest**: `concept-mapper ingest path/to/corpus/ -r -o my_corpus.json`
3. **Detect terms**: `concept-mapper rarities my_corpus.json -o my_terms.json`
4. **Build graph**: `concept-mapper graph my_corpus.json -t my_terms.json -o my_graph.json`
5. **Visualize**: `concept-mapper export my_graph.json --format html -o viz/`

### Tune the Parameters

**Rarity detection:**
- `--threshold`: Minimum score (lower = more terms, higher = only very distinctive)
- `--top-n`: Number of top terms to extract
- `--method`: Detection method (ratio, tfidf, neologism, hybrid)

**Graph construction:**
- `--threshold`: Minimum edge weight (higher = sparser graph, only strong connections)
- `--method`: Construction method (cooccurrence vs. relations)

**Co-occurrence window:**
Modify in code to use N-sentence windows instead of single sentences.

### Curate Terms Manually

Edit `terms.json` to:
- Remove false positives (common words incorrectly flagged)
- Add known philosophical terms missed by detection
- Add definitions and notes for each term

Then rebuild the graph with the curated list.

## Understanding the Results

### What makes a good philosophical term?

**High-scoring terms typically:**
1. Appear frequently in the text (absolute frequency)
2. Rarely appear in general English (low reference corpus frequency)
3. Don't appear in standard dictionaries (neologisms)
4. Appear in definitional contexts ("X is...", "we call this X")
5. Are capitalized mid-sentence (reified abstractions)

**Examples from this text:**
- ✓ "Geist" - Hegelian concept, high frequency
- ✓ "Aufhebung" - Philosophical neologism (sublation)
- ✓ "Sittlichkeit" - Technical term (ethical life)
- ✗ "question" - Common word, even if frequent in philosophy

### Graph interpretation

**Dense clusters** indicate:
- Closely related concepts
- Co-occurring conceptual families

**Bridge nodes** (high betweenness centrality):
- Concepts that connect different areas
- Central to the author's framework

**Peripheral nodes**:
- Specialized or less-integrated concepts
- May indicate tangential discussions

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
term_list = TermList([{"term": term, "score": score} for term, score, _ in candidates])

# Build graph
matrix = build_cooccurrence_matrix(term_list, [processed], method="pmi")
graph = graph_from_cooccurrence(matrix, threshold=0.3)

# Export
generate_html(graph, "examples/viz/", title="My Concept Network")
```

See the [Usage Guide](../docs/usage-guide.md) for comprehensive API documentation.

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
- Verify `data.json` exists in visualization directory

## Resources

- [Usage Guide](../docs/usage-guide.md) - Detailed API documentation
- [Development Roadmap](../docs/concept-mapper-roadmap.md) - Project architecture
- [README](../README.md) - Quick start and overview
