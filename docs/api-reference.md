# Concept Mapper: API Reference

Complete reference for the Concept Mapper Python API.

## Table of Contents

- [Corpus & Preprocessing](#corpus--preprocessing)
  - [Loading Documents](#loading-documents)
  - [Preprocessing Pipeline](#preprocessing-pipeline)
  - [Data Models](#data-models)
- [Analysis](#analysis)
  - [Frequency Analysis](#frequency-analysis)
  - [Rarity Detection](#rarity-detection)
  - [Co-occurrence Analysis](#co-occurrence-analysis)
  - [Relation Extraction](#relation-extraction)
- [Term Management](#term-management)
- [Search & Concordance](#search--concordance)
- [Graph Construction](#graph-construction)
- [Export & Visualization](#export--visualization)

---

## Corpus & Preprocessing

### Loading Documents

#### `concept_mapper.corpus.loader`

Load documents from files and directories.

```python
from concept_mapper.corpus.loader import load_file, load_directory

# Load single file
doc = load_file("path/to/document.txt")

# Load directory
docs = load_directory("path/to/corpus/", pattern="*.txt")
```

**Functions:**

- **`load_file(path: Path) -> Document`**
  Load a single text file.
  - `path`: Path to text file
  - Returns: `Document` object with text and metadata

- **`load_directory(path: Path, pattern: str = "*.txt") -> List[Document]`**
  Load all matching files from a directory.
  - `path`: Directory path
  - `pattern`: Glob pattern for file matching
  - Returns: List of `Document` objects

### Preprocessing Pipeline

#### `concept_mapper.preprocessing.pipeline`

Unified preprocessing pipeline.

```python
from concept_mapper.preprocessing.pipeline import preprocess

processed = preprocess(document)
```

**Functions:**

- **`preprocess(document: Document) -> ProcessedDocument`**
  Run full preprocessing pipeline: tokenization → POS tagging → lemmatization.
  - `document`: Input Document object
  - Returns: `ProcessedDocument` with linguistic annotations

- **`preprocess_corpus(documents: List[Document]) -> List[ProcessedDocument]`**
  Preprocess multiple documents.
  - `documents`: List of Document objects
  - Returns: List of ProcessedDocument objects

### Data Models

#### `concept_mapper.corpus.models`

Core data structures for documents.

```python
from concept_mapper.corpus.models import Document, ProcessedDocument

# Create a document
doc = Document(
    text="Your text here",
    metadata={"title": "Document Title", "author": "Author Name"}
)

# ProcessedDocument attributes
processed.raw_text        # Original text
processed.sentences       # List of sentence strings
processed.tokens          # List of word tokens
processed.lemmas          # List of lemmatized words
processed.pos_tags        # List of (word, POS) tuples
processed.metadata        # Metadata dict
```

**Classes:**

- **`Document`**
  Raw text document with metadata.
  - `text: str` - Document text
  - `metadata: Dict[str, Any]` - Metadata (title, author, date, etc.)

- **`ProcessedDocument`**
  Preprocessed document with linguistic annotations.
  - `raw_text: str` - Original text
  - `sentences: List[str]` - Sentence-segmented text
  - `tokens: List[str]` - Word tokens
  - `lemmas: List[str]` - Lemmatized words
  - `pos_tags: List[Tuple[str, str]]` - POS-tagged tokens
  - `metadata: Dict[str, Any]` - Metadata

---

## Analysis

### Frequency Analysis

#### `concept_mapper.analysis.frequency`

Word frequency distributions.

```python
from concept_mapper.analysis.frequency import (
    word_frequencies,
    corpus_frequencies,
    document_frequencies
)

# Single document frequencies
freq = word_frequencies(processed_doc, use_lemmas=True)

# Corpus-wide frequencies
corpus_freq = corpus_frequencies(processed_docs, use_lemmas=True)

# Document frequencies (in how many docs does term appear?)
doc_freq = document_frequencies(processed_docs)
```

**Functions:**

- **`word_frequencies(doc: ProcessedDocument, use_lemmas: bool = True) -> Counter`**
  Count word frequencies in a single document.

- **`corpus_frequencies(docs: List[ProcessedDocument], use_lemmas: bool = True) -> Counter`**
  Aggregate word frequencies across all documents.

- **`document_frequencies(docs: List[ProcessedDocument]) -> Counter`**
  Count in how many documents each term appears.

- **`pos_filtered_frequencies(doc: ProcessedDocument, pos_tags: Set[str]) -> Counter`**
  Count only words with specific POS tags.

### Rarity Detection

#### `concept_mapper.analysis.rarity`

Detect philosophical terms through statistical rarity analysis.

```python
from concept_mapper.analysis.rarity import PhilosophicalTermScorer
from concept_mapper.analysis.reference import load_reference_corpus

# Load reference corpus
reference = load_reference_corpus()

# Create scorer
scorer = PhilosophicalTermScorer(
    docs=processed_docs,
    reference_corpus=reference,
    use_lemmas=True
)

# Score all terms
candidates = scorer.score_all(min_score=2.0, top_n=50)

# Get high-confidence terms (multiple signals agree)
high_conf = scorer.get_high_confidence_terms(min_signals=3)
```

**Class: `PhilosophicalTermScorer`**

Multi-method term detection with weighted scoring.

**Methods:**

- **`score_term(term: str) -> Dict[str, float]`**
  Score a single term with breakdown of all components.
  - Returns: Dictionary with `total_score` and component scores

- **`score_all(min_score: float = 0, top_n: Optional[int] = None) -> List[Tuple[str, float, Dict]]`**
  Score all terms in corpus.
  - `min_score`: Minimum score threshold
  - `top_n`: Maximum number of results
  - Returns: List of (term, score, components) tuples, sorted by score

- **`get_high_confidence_terms(min_signals: int = 2) -> Set[str]`**
  Get terms detected by multiple methods.
  - `min_signals`: Minimum number of detection methods that must agree
  - Returns: Set of high-confidence terms

**Component Functions:**

- **`compare_to_reference(docs, reference_corpus, use_lemmas=True) -> Dict[str, float]`**
  Calculate relative frequency ratios.

- **`tfidf_vs_reference(docs, reference_corpus, use_lemmas=True) -> Dict[str, float]`**
  Calculate TF-IDF scores against reference.

- **`get_neologism_candidates(docs, use_lemmas=True) -> Set[str]`**
  Detect terms not in WordNet dictionary.

- **`get_definitional_contexts(docs) -> List[Tuple]`**
  Find sentences where terms are explicitly defined.

### Co-occurrence Analysis

#### `concept_mapper.analysis.cooccurrence`

Term co-occurrence and association strength.

```python
from concept_mapper.analysis.cooccurrence import (
    cooccurs_in_sentence,
    pmi,
    build_cooccurrence_matrix
)
from concept_mapper.terms.models import TermList

# Find terms co-occurring with target
cooccur = cooccurs_in_sentence("consciousness", docs)

# Calculate PMI (Pointwise Mutual Information)
association = pmi("consciousness", "intentionality", docs)

# Build full co-occurrence matrix
term_list = TermList([{"term": "being"}, {"term": "time"}, {"term": "dasein"}])
matrix = build_cooccurrence_matrix(
    term_list,
    docs,
    method="pmi",      # or "count", "llr"
    window="sentence"  # or "n_sentences"
)
```

**Functions:**

- **`cooccurs_in_sentence(term: str, docs: List[ProcessedDocument]) -> Counter`**
  Count terms appearing in same sentences.

- **`cooccurs_within_n(term: str, docs: List[ProcessedDocument], n_sentences: int = 3) -> Counter`**
  Count terms within N-sentence window.

- **`pmi(term1: str, term2: str, docs: List[ProcessedDocument]) -> float`**
  Calculate Pointwise Mutual Information (association strength).
  - Positive values = terms co-occur more than expected by chance
  - ~0 = independent
  - Negative = avoid each other

- **`log_likelihood_ratio(term1: str, term2: str, docs: List[ProcessedDocument]) -> float`**
  Calculate G² statistic for co-occurrence significance.
  - >3.84 = significant at p<0.05
  - >6.63 = significant at p<0.01
  - >10.83 = significant at p<0.001

- **`build_cooccurrence_matrix(term_list: TermList, docs: List[ProcessedDocument], method: str = "pmi", window: str = "sentence") -> Dict`**
  Build symmetric co-occurrence matrix.
  - `method`: "count", "pmi", or "llr"
  - `window`: "sentence" or "n_sentences"
  - Returns: Nested dict `{term1: {term2: score}}`

### Relation Extraction

#### `concept_mapper.analysis.relations`

Extract grammatical relationships between terms.

```python
from concept_mapper.analysis.relations import (
    extract_svo,
    extract_copular,
    extract_prepositional,
    get_relations
)

# Extract all relation types for a term
relations = get_relations("consciousness", docs, types=["svo", "copular", "prep"])

for rel in relations:
    print(f"{rel.source} --[{rel.relation_type}]--> {rel.target}")
    print(f"  Evidence: {rel.evidence[0]}")
```

**Classes:**

- **`Relation`**
  Aggregated relation between two concepts.
  - `source: str` - Source term
  - `relation_type: str` - Type of relation (svo, copular, prep)
  - `target: str` - Target term
  - `evidence: List[str]` - Example sentences
  - `metadata: Dict` - Additional info (verb, preposition, etc.)

**Functions:**

- **`extract_svo(sentence: str, doc_id: str = "") -> List[SVOTriple]`**
  Extract Subject-Verb-Object triples.

- **`extract_copular(term: str, docs: List[ProcessedDocument]) -> List[CopularRelation]`**
  Extract copular definitions (X is Y).

- **`extract_prepositional(term: str, docs: List[ProcessedDocument]) -> List[PrepRelation]`**
  Extract prepositional relations (X of Y, X through Y, etc.).

- **`get_relations(term: str, docs: List[ProcessedDocument], types: List[str] = None, case_sensitive: bool = False) -> List[Relation]`**
  Extract and aggregate all relations for a term.
  - `types`: List of relation types to extract (default: all)
  - Returns: List of Relation objects with evidence aggregation

---

## Term Management

### `concept_mapper.terms.models`

Data structures for term lists.

```python
from concept_mapper.terms.models import TermEntry, TermList

# Create term entry
entry = TermEntry(
    term="Geist",
    lemma="geist",
    pos="NN",
    definition="Spirit; self-developing rational principle (Hegel)",
    examples=["Geist actualizes itself through history."],
    notes="Central concept in Phenomenology of Spirit"
)

# Create term list
term_list = TermList([
    {"term": "dasein", "pos": "NN"},
    {"term": "being", "pos": "NN"},
    {"term": "temporality", "pos": "NN"}
])

# Access terms
entry = term_list.get("dasein")
all_terms = term_list.list_terms()
```

**Classes:**

- **`TermEntry`**
  Single philosophical term with metadata.
  - `term: str` - The term itself
  - `lemma: str` - Lemmatized form
  - `pos: str` - Part of speech tag
  - `definition: str` - Definition or description
  - `notes: str` - Additional notes
  - `examples: List[str]` - Example sentences
  - `metadata: Dict` - Custom metadata

- **`TermList`**
  Collection of terms with CRUD operations.

**TermList Methods:**

- **`add(entry: TermEntry)`** - Add a term
- **`remove(term: str)`** - Remove a term
- **`update(term: str, **kwargs)`** - Update term fields
- **`get(term: str) -> Optional[TermEntry]`** - Retrieve a term
- **`list_terms() -> List[TermEntry]`** - Get all terms (sorted)
- **`__contains__(term: str) -> bool`** - Check if term exists
- **`__len__() -> int`** - Number of terms

### `concept_mapper.terms.manager`

Import/export and bulk operations.

```python
from concept_mapper.terms.manager import TermManager

manager = TermManager(term_list)

# Export
manager.export_to_json("terms.json")
manager.export_to_csv("terms.csv")
manager.export_to_txt("terms.txt")

# Import
manager.import_from_json("terms.json")
manager.import_from_csv("terms.csv")

# Operations
stats = manager.get_statistics()
nouns = manager.filter_by_pos({"NN", "NNS"})
```

**Methods:**

- **`export_to_json(path: Path, indent: int = 2)`** - Export to JSON
- **`import_from_json(path: Path)`** - Import from JSON
- **`export_to_csv(path: Path)`** - Export to CSV
- **`import_from_csv(path: Path)`** - Import from CSV
- **`export_to_txt(path: Path)`** - Export term names to text file
- **`filter_by_pos(tags: Set[str]) -> TermList`** - Filter by POS tags
- **`get_statistics() -> Dict`** - Get term list statistics

### `concept_mapper.terms.suggester`

Auto-populate term lists from analysis.

```python
from concept_mapper.terms.suggester import suggest_terms_from_analysis
from concept_mapper.analysis.reference import load_reference_corpus

reference = load_reference_corpus()

# Automatically suggest terms
suggested = suggest_terms_from_analysis(
    docs=processed_docs,
    reference=reference,
    min_score=2.0,
    top_n=50,
    max_examples=3  # Include up to 3 example sentences per term
)
```

---

## Search & Concordance

### `concept_mapper.search.find`

Find term occurrences in corpus.

```python
from concept_mapper.search.find import (
    find_sentences,
    find_sentences_any,
    count_term_occurrences
)

# Find all sentences containing term
matches = find_sentences("consciousness", docs, case_sensitive=False)

for match in matches:
    print(f"[{match.doc_id}] {match.sentence}")

# Find sentences with any of multiple terms
matches = find_sentences_any(["being", "dasein", "time"], docs)

# Count occurrences
count = count_term_occurrences("intentionality", docs)
```

**Classes:**

- **`SentenceMatch`**
  A single match result.
  - `sentence: str` - Matching sentence
  - `doc_id: str` - Document identifier
  - `sent_index: int` - Sentence index in document
  - `term_positions: List[int]` - Word positions of matches

### `concept_mapper.search.concordance`

KWIC (Key Word In Context) displays.

```python
from concept_mapper.search.concordance import concordance

# Generate KWIC lines
lines = concordance("consciousness", docs, width=50)

for line in lines:
    print(f"{line.left_context:>50} | {line.keyword} | {line.right_context:<50}")
```

**Classes:**

- **`KWICLine`**
  A single concordance line.
  - `left_context: str` - Text before keyword
  - `keyword: str` - The search term
  - `right_context: str` - Text after keyword
  - `doc_id: str` - Document identifier

### `concept_mapper.search.context`

Context windows around matches.

```python
from concept_mapper.search.context import get_context

# Get N sentences before/after each match
windows = get_context("dasein", docs, n_sentences=2)

for window in windows:
    for sent in window.before:
        print(f"    {sent}")
    print(f">>> {window.match}")
    for sent in window.after:
        print(f"    {sent}")
```

### `concept_mapper.search.dispersion`

Analyze term distribution across corpus.

```python
from concept_mapper.search.dispersion import dispersion, get_dispersion_summary

# Get positions where term appears
positions = dispersion("being", docs)

# Get summary statistics
summary = get_dispersion_summary("being", docs)
print(f"Appears in {summary['num_docs']} documents")
print(f"Coverage: {summary['coverage']:.1%}")
```

---

## Graph Construction

### `concept_mapper.graph.model`

Graph data structure for concept networks.

```python
from concept_mapper.graph import ConceptGraph

# Create graph
graph = ConceptGraph(directed=True)

# Add nodes
graph.add_node("consciousness", label="Consciousness", frequency=42)
graph.add_node("intentionality", label="Intentionality", frequency=28)

# Add edges
graph.add_edge(
    "consciousness",
    "intentionality",
    weight=0.85,
    relation_type="copular",
    evidence=["Consciousness is intentional."]
)

# Query
print(f"Nodes: {graph.node_count()}")
print(f"Edges: {graph.edge_count()}")
neighbors = graph.neighbors("consciousness")
```

**Class: `ConceptGraph`**

**Methods:**

- **`add_node(node_id, label=None, frequency=None, pos=None, **attrs)`** - Add a node
- **`add_edge(source, target, weight=None, relation_type=None, **attrs)`** - Add an edge
- **`has_node(node_id) -> bool`** - Check if node exists
- **`has_edge(source, target) -> bool`** - Check if edge exists
- **`get_node_attrs(node_id) -> Dict`** - Get node attributes
- **`get_edge_attrs(source, target) -> Dict`** - Get edge attributes
- **`neighbors(node_id) -> List[str]`** - Get neighboring nodes
- **`node_count() -> int`** - Number of nodes
- **`edge_count() -> int`** - Number of edges

### `concept_mapper.graph.builders`

Build graphs from analysis results.

```python
from concept_mapper.graph import (
    graph_from_cooccurrence,
    graph_from_relations,
    graph_from_terms
)

# From co-occurrence matrix
graph = graph_from_cooccurrence(matrix, threshold=0.3)

# From relation extraction
graph = graph_from_relations(relations)

# From term list (nodes only)
graph = graph_from_terms(["being", "time", "dasein"])
```

### `concept_mapper.graph.operations`

Graph manipulation operations.

```python
from concept_mapper.graph.operations import (
    merge_graphs,
    prune_edges,
    prune_nodes,
    get_subgraph
)

# Merge two graphs
combined = merge_graphs(graph1, graph2)

# Prune weak edges
pruned = prune_edges(graph, min_weight=0.5)

# Prune isolated nodes
pruned = prune_nodes(graph, min_degree=1)

# Extract subgraph
subgraph = get_subgraph(graph, {"being", "time", "dasein"})
```

### `concept_mapper.graph.metrics`

Graph analysis metrics.

```python
from concept_mapper.graph.metrics import (
    centrality,
    detect_communities,
    graph_density
)

# Calculate centrality
scores = centrality(graph, method="betweenness")
top_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]

# Detect communities
communities = detect_communities(graph)
for i, community in enumerate(communities):
    print(f"Community {i}: {community}")

# Graph density
density = graph_density(graph)
print(f"Density: {density:.3f}")
```

**Centrality methods:**
- `"betweenness"` - Nodes on many shortest paths (bridges)
- `"degree"` - Number of connections
- `"closeness"` - Average distance to all other nodes
- `"eigenvector"` - Connected to highly-connected nodes
- `"pagerank"` - PageRank algorithm

---

## Export & Visualization

### `concept_mapper.export.d3`

Export graphs for D3.js visualization.

```python
from concept_mapper.export import export_d3_json, load_d3_json

# Export
export_d3_json(
    graph,
    "output/graph.json",
    size_by="betweenness",  # or "frequency", "degree"
    include_evidence=True,
    max_evidence=5
)

# Load
data = load_d3_json("output/graph.json")
```

### `concept_mapper.export.formats`

Export to various graph formats.

```python
from concept_mapper.export import (
    export_graphml,
    export_csv,
    export_gexf,
    export_dot
)

# GraphML for Gephi, yEd, Cytoscape
export_graphml(graph, "output/graph.graphml")

# CSV for spreadsheets
export_csv(graph, "output/csv/")  # Creates nodes.csv and edges.csv

# GEXF for Gephi
export_gexf(graph, "output/graph.gexf")

# DOT for Graphviz (requires pydot)
export_dot(graph, "output/graph.dot")
```

### `concept_mapper.export.html`

Generate standalone HTML visualizations.

```python
from concept_mapper.export import generate_html

# Generate interactive visualization
html_path = generate_html(
    graph,
    output_dir="output/visualization/",
    title="My Concept Network",
    width=1200,
    height=800,
    include_evidence=True
)

print(f"Open in browser: file://{html_path.absolute()}")
```

**Features:**
- Force-directed graph layout (D3 force simulation)
- Interactive: drag nodes, zoom/pan
- Color-coded by community detection
- Node size by centrality or frequency
- Tooltips with node/edge details
- Self-contained HTML file (no external dependencies)

---

## CLI Reference

See `concept-mapper --help` for full command-line interface documentation.

```bash
# Main commands
concept-mapper ingest <path>           # Load and preprocess
concept-mapper rarities <corpus>       # Detect terms
concept-mapper search <corpus> <term>  # Search
concept-mapper concordance <corpus> <term>  # KWIC display
concept-mapper graph <corpus> -t <terms>    # Build graph
concept-mapper export <graph> -f <format>   # Export/visualize
```

See [Usage Guide](usage-guide.md#phase-10-cli-interface) for detailed CLI examples.

---

## Type Hints

All public APIs include full type hints for IDE autocomplete and static type checking:

```python
from typing import List, Dict, Optional, Counter
from concept_mapper.corpus.models import ProcessedDocument

def my_function(
    docs: List[ProcessedDocument],
    threshold: float = 0.5,
    max_results: Optional[int] = None
) -> Dict[str, float]:
    ...
```

Use with mypy or pyright for static type checking:
```bash
mypy src/concept_mapper/
```

---

## Examples

See the [examples/](../examples/) directory for:
- Complete workflow examples (CLI and Python API)
- Sample philosophical text
- Step-by-step walkthrough

See the [Usage Guide](usage-guide.md) for detailed phase-by-phase examples.

---

## Further Reading

- [Usage Guide](usage-guide.md) - Practical examples for each feature
- [Development Roadmap](concept-mapper-roadmap.md) - Project architecture
- [README](../README.md) - Quick start and overview
