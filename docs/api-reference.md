# API Reference

**Belongs here:** Public function signatures, parameter types and defaults, return types, exceptions raised, one usage example per function, data class fields, conceptual notes for non-obvious design choices.

**Does not belong here:** Multi-step tutorials, expected output blocks, workflow narratives, installation instructions, CLI documentation (see `src/concept_mapper/cli.py` docstrings).

---

## Table of Contents

- [Data Loading](#data-loading) — `corpus.loader`
- [Preprocessing](#preprocessing) — `preprocessing.pipeline`
- [Reference Corpus](#reference-corpus) — `analysis.reference`
- [Frequency Analysis](#frequency-analysis) — `analysis.frequency`
- [Rarity & Term Detection](#rarity--term-detection) — `analysis.rarity`
- [Term List](#term-list) — `terms.models`
- [Search](#search) — `search.find`
- [Concordance](#concordance) — `search.concordance`
- [Co-occurrence](#co-occurrence) — `analysis.cooccurrence`
- [Relations](#relations) — `analysis.relations`
- [Graph](#graph) — `graph`
- [Export](#export) — `export`
- [Search vs. KWIC Concordance](#search-vs-kwic-concordance)
- [How Graph Links Are Determined](#how-graph-links-are-determined)
- [Complete Workflow Example](#complete-workflow-example)

---

## Data Loading

`from concept_mapper.corpus.loader import load_file, load_directory`

---

### `load_file(file_path, metadata=None)`

Load a single file into a `Document` object. Supports `.txt` and `.pdf`.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `file_path` | `str \| Path` | required | Path to text or PDF file |
| `metadata` | `dict \| None` | `None` | Custom metadata; if omitted, extracted from filename |

**Returns:** `Document`

**Raises:** `FileNotFoundError` if file doesn't exist. `ImportError` for PDF if `pdfplumber` not installed.

```python
doc = load_file("samples/eco_spl.txt")
```

---

### `load_directory(directory_path, pattern="*.txt", recursive=False)`

Load all matching files from a directory into a `Corpus`.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `directory_path` | `str \| Path` | required | Path to directory |
| `pattern` | `str` | `"*.txt"` | Glob pattern for file matching |
| `recursive` | `bool` | `False` | Search subdirectories recursively |

**Returns:** `Corpus` containing all loaded documents.

**Raises:** `FileNotFoundError`, `NotADirectoryError`.

```python
corpus = load_directory("data/sample", pattern="*.txt", recursive=True)
```

---

## Preprocessing

`from concept_mapper.preprocessing.pipeline import preprocess, preprocess_corpus`

---

### `preprocess(document, detect_structure=True, clean_ocr=False)`

Run a document through the full pipeline: tokenization → POS tagging → lemmatization → structure detection → paragraph segmentation.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `document` | `Document` | required | Raw document to process |
| `detect_structure` | `bool` | `True` | Detect chapters/sections (adds `structure_nodes`, `sentence_locations`) |
| `clean_ocr` | `bool` | `False` | Remove OCR/PDF artifacts before processing |

**Returns:** `ProcessedDocument` with `.sentences`, `.tokens`, `.pos_tags`, `.lemmas`, `.paragraph_indices`.

```python
processed = preprocess(doc, clean_ocr=True)
print(processed.num_sentences, processed.num_tokens)
```

---

### `preprocess_corpus(documents)`

Preprocess a list of documents.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `documents` | `list[Document]` | required | Documents to process |

**Returns:** `list[ProcessedDocument]`

```python
docs = preprocess_corpus(corpus.documents)
```

---

## Reference Corpus

`from concept_mapper.analysis.reference import load_reference_corpus`

---

### `load_reference_corpus(name="brown", cache=True, cache_dir=None)`

Load word frequencies from a reference corpus (Brown corpus). Checks bundled data first, then cache, then computes from NLTK.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | `"brown"` | Corpus name; only `"brown"` supported |
| `cache` | `bool` | `True` | Cache computed frequencies to disk |
| `cache_dir` | `Path \| None` | `None` | Cache directory (default: `output/cache/`) |

**Returns:** `Counter[str, int]` — word to frequency mapping (~1.16M words).

**Raises:** `ValueError` for unsupported corpus names.

```python
reference = load_reference_corpus()
```

---

## Frequency Analysis

`from concept_mapper.analysis.frequency import corpus_frequencies`

---

### `corpus_frequencies(docs, use_lemmas=True)`

Count term frequencies across all documents.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `docs` | `list[ProcessedDocument]` | required | Preprocessed documents |
| `use_lemmas` | `bool` | `True` | Count lemmas instead of surface forms |

**Returns:** `Counter[str, int]`

```python
freqs = corpus_frequencies(docs, use_lemmas=True)
freqs.most_common(10)
```

---

## Rarity & Term Detection

`from concept_mapper.analysis.rarity import ...`

---

### `compare_to_reference(docs, reference_corpus, use_lemmas=True, min_author_freq=3)`

Calculate how much more (or less) the author uses each term vs. general English.

`ratio = (author_freq / total_author) / (reference_freq / total_reference)`

A ratio of 100 means the author uses the term 100× more than in Brown corpus. Terms absent from the reference use a pseudocount of 0.5 to avoid division by zero.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `docs` | `list[ProcessedDocument]` | required | Author corpus |
| `reference_corpus` | `Counter` | required | Reference frequencies (e.g., Brown) |
| `use_lemmas` | `bool` | `True` | Match on lemmatized forms |
| `min_author_freq` | `int` | `3` | Minimum occurrences to include a term |

**Returns:** `dict[str, float]` — term → ratio.

```python
ratios = compare_to_reference(docs, reference)
top = sorted(ratios.items(), key=lambda x: x[1], reverse=True)[:10]
```

---

### `tfidf_vs_reference(docs, reference_corpus, use_lemmas=True, min_author_freq=3)`

Calculate TF-IDF treating the author corpus as document and Brown as background. High score = frequent in author AND rare in general English.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `docs` | `list[ProcessedDocument]` | required | Author corpus |
| `reference_corpus` | `Counter` | required | Reference frequencies |
| `use_lemmas` | `bool` | `True` | Match on lemmatized forms |
| `min_author_freq` | `int` | `3` | Minimum occurrences to include |

**Returns:** `dict[str, float]` — term → TF-IDF score.

```python
scores = tfidf_vs_reference(docs, reference)
```

---

### `get_combined_distinctive_terms(docs, reference_corpus, ratio_threshold=10.0, tfidf_threshold=0.001, use_lemmas=True, min_author_freq=3, method="union")`

Combine ratio and TF-IDF detection methods.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `docs` | `list[ProcessedDocument]` | required | Author corpus |
| `reference_corpus` | `Counter` | required | Reference frequencies |
| `ratio_threshold` | `float` | `10.0` | Minimum ratio to flag a term |
| `tfidf_threshold` | `float` | `0.001` | Minimum TF-IDF score to flag a term |
| `use_lemmas` | `bool` | `True` | Match on lemmatized forms |
| `min_author_freq` | `int` | `3` | Minimum occurrences to include |
| `method` | `str` | `"union"` | `"union"`, `"intersection"`, `"ratio_only"`, `"tfidf_only"` |

**Returns:** `set[str]`

```python
terms = get_combined_distinctive_terms(docs, reference, method="intersection")
```

---

### `score_philosophical_terms(docs, reference_corpus, use_lemmas=True, min_author_freq=3, top_n=50)`

Convenience wrapper around `PhilosophicalTermScorer` with default settings.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `docs` | `list[ProcessedDocument]` | required | Author corpus |
| `reference_corpus` | `Counter` | required | Reference frequencies |
| `use_lemmas` | `bool` | `True` | Match on lemmatized forms |
| `min_author_freq` | `int` | `3` | Minimum occurrences to include |
| `top_n` | `int` | `50` | Maximum terms to return |

**Returns:** `list[tuple[str, float]]` — `(term, score)` sorted descending.

```python
terms = score_philosophical_terms(docs, reference, top_n=20)
```

---

### `class PhilosophicalTermScorer`

Hybrid scorer combining five signals: relative frequency ratio, TF-IDF, neologism detection, definitional context, and mid-sentence capitalization.

#### `__init__(docs, reference_corpus, use_lemmas=True, min_author_freq=3, weights=None)`

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `docs` | `list[ProcessedDocument]` | required | Author corpus |
| `reference_corpus` | `Counter` | required | Reference frequencies |
| `use_lemmas` | `bool` | `True` | Match on lemmatized forms |
| `min_author_freq` | `int` | `3` | Minimum occurrences to include |
| `weights` | `dict \| None` | `None` | Per-signal weights: `ratio` (1.0), `tfidf` (1.0), `neologism` (0.5), `definitional` (0.3), `capitalized` (0.2) |

Signals are precomputed at init time.

#### `score_term(term, normalize=True)`

Score a single term with full component breakdown.

**Returns:** `dict` with keys: `total`, `ratio`, `tfidf`, `neologism`, `definitional`, `capitalized`, `raw_total`.

#### `score_all(min_score=0.0, top_n=None)`

Score all terms meeting the minimum frequency threshold.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `min_score` | `float` | `0.0` | Minimum total score to include |
| `top_n` | `int \| None` | `None` | Cap results at N terms |

**Returns:** `list[tuple[str, float, dict]]` — `(term, total_score, components)` sorted descending.

#### `get_high_confidence_terms(min_signals=3, min_score=1.0)`

Return terms where at least `min_signals` detection signals fire.

**Returns:** `set[str]`

```python
scorer = PhilosophicalTermScorer(docs, reference)
results = scorer.score_all(min_score=1.5, top_n=30)
for term, score, components in results:
    print(f"{term}: {score:.2f}")
```

---

## Term List

`from concept_mapper.terms.models import TermEntry, TermList`

---

### `class TermEntry`

A single curated term with metadata.

| Field | Type | Description |
|-------|------|-------------|
| `term` | `str` | Term as it appears in text |
| `lemma` | `str \| None` | Lemmatized form |
| `pos` | `str \| None` | POS tag (e.g., `"NN"`, `"VB"`) |
| `definition` | `str \| None` | Human-provided definition |
| `notes` | `str \| None` | Scholarly notes or context |
| `examples` | `list[str]` | Example sentences from corpus |
| `metadata` | `dict` | Additional custom metadata |

---

### `class TermList`

A curated collection of terms with lookup, CRUD, and persistence.

| Method | Signature | Description |
|--------|-----------|-------------|
| `add` | `(entry: TermEntry) -> None` | Add term; raises `ValueError` if duplicate |
| `remove` | `(term: str) -> None` | Remove term; raises `KeyError` if not found |
| `update` | `(term: str, **kwargs) -> None` | Update fields of existing term |
| `get` | `(term: str) -> TermEntry \| None` | Retrieve by name |
| `contains` | `(term: str) -> bool` | Check membership |
| `list_terms` | `() -> list[TermEntry]` | All entries sorted alphabetically |
| `list_term_names` | `() -> list[str]` | All term strings sorted alphabetically |
| `save` | `(path: Path) -> None` | Serialize to JSON |
| `load` | `(path: Path) -> TermList` | Class method; deserialize from JSON |
| `merge` | `(other: TermList, overwrite=False) -> TermList` | Merge two lists into a new one |
| `from_dict` | `(data: dict) -> TermList` | Class method; construct from dict |

```python
terms = TermList()
terms.add(TermEntry(term="praxis", pos="NN", definition="Practical action informed by theory"))
terms.save(Path("output/terms.json"))
loaded = TermList.load(Path("output/terms.json"))
```

---

## Search

`from concept_mapper.search.find import find_sentences, find_sentences_any, find_sentences_all, count_term_occurrences`

---

### `find_sentences(term, docs, case_sensitive=False, match_lemma=False)`

Find all sentences containing a term.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `term` | `str` | required | Term to search for |
| `docs` | `list[ProcessedDocument]` | required | Corpus to search |
| `case_sensitive` | `bool` | `False` | Exact case matching |
| `match_lemma` | `bool` | `False` | Match all inflected forms (`"run"` matches `"running"`, `"ran"`) |

**Returns:** `list[SentenceMatch]` in document order.

Each `SentenceMatch` has: `.sentence`, `.doc_id`, `.sent_index`, `.term_positions`, `.term`, `.location`.

```python
matches = find_sentences("intentionality", docs, match_lemma=True)
for m in matches:
    print(f"[{m.doc_id}:{m.sent_index}] {m.sentence}")
```

---

### `find_sentences_any(terms, docs, case_sensitive=False)`

Find sentences containing **any** of the given terms.

**Returns:** `list[SentenceMatch]` (may contain duplicates if multiple terms match).

---

### `find_sentences_all(terms, docs, case_sensitive=False)`

Find sentences containing **all** of the given terms.

**Returns:** `list[SentenceMatch]`

---

### `count_term_occurrences(term, docs, case_sensitive=False)`

Count total token-level occurrences across all documents.

**Returns:** `int`

```python
n = count_term_occurrences("consciousness", docs)
```

---

## Concordance

`from concept_mapper.search.concordance import concordance, concordance_sorted, concordance_filtered, format_kwic_lines`

---

### `concordance(term, docs, width=50, case_sensitive=False)`

Generate KWIC (Key Word In Context) lines with keyword centered.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `term` | `str` | required | Keyword to center |
| `docs` | `list[ProcessedDocument]` | required | Corpus to search |
| `width` | `int` | `50` | Characters of context on each side |
| `case_sensitive` | `bool` | `False` | Exact case matching |

**Returns:** `list[KWICLine]`

Each `KWICLine` has: `.left_context`, `.keyword`, `.right_context`, `.doc_id`, `.sent_index`.

```python
lines = concordance("consciousness", docs, width=40)
print(format_kwic_lines(lines[:10]))
```

---

### `concordance_sorted(term, docs, width=50, sort_by="left")`

Concordance sorted by context for pattern recognition.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `sort_by` | `str` | `"left"` | `"left"` (right-to-left sort from keyword) or `"right"` |

**Returns:** `list[KWICLine]`

---

### `concordance_filtered(term, docs, filter_terms, width=50)`

Concordance showing only lines where context also contains one of `filter_terms`.

**Returns:** `list[KWICLine]`

---

### `format_kwic_lines(lines, width=50, show_doc_id=False)`

Format KWIC lines as aligned text for display.

**Returns:** `str` — newline-separated aligned lines.

---

## Co-occurrence

`from concept_mapper.analysis.cooccurrence import ...`

---

### `cooccurs_in_sentence(term, docs, case_sensitive=False)`

Count all other words appearing in the same sentence as `term`.

**Returns:** `Counter[str, int]`

```python
cooccurs = cooccurs_in_sentence("consciousness", docs)
cooccurs.most_common(10)
```

---

### `cooccurs_within_n(term, docs, n_sentences=3, case_sensitive=False)`

Count words appearing within N sentences before or after `term`.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `n_sentences` | `int` | `3` | Window size in sentences on each side |

**Returns:** `Counter[str, int]`

---

### `pmi(term1, term2, docs, case_sensitive=False)`

Pointwise Mutual Information between two terms (sentence-level).

PMI > 0 = terms co-occur more than by chance. PMI = 0 = independent. PMI < 0 = negative association.

**Returns:** `float`

```python
score = pmi("consciousness", "intentionality", docs)
```

---

### `log_likelihood_ratio(term1, term2, docs, case_sensitive=False)`

Log-likelihood ratio (G²) for co-occurrence significance.

G² > 3.84 → p < 0.05. G² > 6.63 → p < 0.01. G² > 10.83 → p < 0.001.

**Returns:** `float`

---

### `build_cooccurrence_matrix(term_list, docs, method="count", window="sentence", n_sentences=None)`

Build a symmetric term × term association matrix.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `term_list` | `TermList` | required | Terms to include |
| `docs` | `list[ProcessedDocument]` | required | Corpus |
| `method` | `str` | `"count"` | `"count"`, `"pmi"`, or `"llr"` |
| `window` | `str` | `"sentence"` | `"sentence"` or `"n_sentences"` |
| `n_sentences` | `int \| None` | `None` | Window size if `window="n_sentences"` |

**Returns:** `dict[str, dict[str, float]]` — symmetric nested dict; `matrix[t1][t2] == matrix[t2][t1]`.

```python
matrix = build_cooccurrence_matrix(term_list, docs, method="pmi")
score = matrix["consciousness"]["intentionality"]
```

---

### `get_top_cooccurrences(term, docs, n=10, method="count", window="sentence", term_list=None)`

Convenience function: top N co-occurring terms for a single target term.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `term` | `str` | required | Target term |
| `n` | `int` | `10` | Number of results |
| `method` | `str` | `"count"` | `"count"`, `"pmi"`, or `"llr"` |
| `term_list` | `TermList \| None` | `None` | Restrict candidates to this list |

**Returns:** `list[tuple[str, float]]` sorted descending.

```python
top = get_top_cooccurrences("consciousness", docs, n=5, method="pmi")
```

---

## Relations

`from concept_mapper.analysis.relations import ...`

---

### `extract_svo_for_term(term, docs, case_sensitive=False)`

Extract Subject-Verb-Object triples where `term` appears as subject, verb, or object.

**Returns:** `list[SVOTriple]`

Each `SVOTriple` has: `.subject`, `.verb`, `.object`, `.sentence`, `.doc_id`.

```python
triples = extract_svo_for_term("consciousness", docs)
```

---

### `extract_copular(term, docs, case_sensitive=False)`

Extract copular relations (X is/are/was/becomes Y) involving `term`.

**Returns:** `list[CopularRelation]`

Each `CopularRelation` has: `.subject`, `.complement`, `.copula`, `.sentence`, `.doc_id`.

---

### `extract_prepositional(term, docs, case_sensitive=False)`

Extract prepositional phrases with `term` as head noun (e.g., "consciousness of objects").

**Returns:** `list[PrepRelation]`

Each `PrepRelation` has: `.head`, `.prep`, `.object`, `.sentence`, `.doc_id`.

---

### `get_relations(term, docs, types=None, case_sensitive=False)`

Extract and aggregate all relation types, grouping multiple occurrences of the same relation.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `term` | `str` | required | Term to find relations for |
| `docs` | `list[ProcessedDocument]` | required | Corpus |
| `types` | `list[str] \| None` | `None` | Relation types to extract: `"svo"`, `"copular"`, `"prep"`. None = all |
| `case_sensitive` | `bool` | `False` | Exact case matching |

**Returns:** `list[Relation]`

Each `Relation` has: `.source`, `.relation_type`, `.target`, `.evidence` (list of sentences), `.metadata`.

```python
relations = get_relations("consciousness", docs, types=["copular", "prep"])
for rel in relations:
    print(rel, f"  ({len(rel.evidence)} occurrences)")
```

---

## Graph

`from concept_mapper.graph import ...`

---

### `graph_from_cooccurrence(matrix, threshold=0.0, directed=False)`

Build a graph from a co-occurrence matrix. Creates edges for pairs above threshold.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `matrix` | `dict[str, dict[str, float]]` | required | Output of `build_cooccurrence_matrix()` |
| `threshold` | `float` | `0.0` | Minimum score to create an edge |
| `directed` | `bool` | `False` | Create directed graph |

**Returns:** `ConceptGraph` (undirected by default; edge weights = co-occurrence scores).

```python
graph = graph_from_cooccurrence(matrix, threshold=0.5)
```

---

### `graph_from_relations(relations, include_evidence=True)`

Build a directed graph from extracted relations.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `relations` | `list[Relation]` | required | Output of `get_relations()` |
| `include_evidence` | `bool` | `True` | Store evidence sentences on edges |

**Returns:** `ConceptGraph` (directed; edge weights = evidence count).

---

### `centrality(graph, method="betweenness", normalized=True)`

Compute node centrality scores.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `method` | `str` | `"betweenness"` | `"betweenness"`, `"degree"`, `"closeness"`, `"eigenvector"`, `"pagerank"` |
| `normalized` | `bool` | `True` | Normalize scores to [0, 1] |

**Returns:** `dict[str, float]` — node ID → score.

**Raises:** `ValueError` for unknown methods.

```python
scores = centrality(graph, method="betweenness")
top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
```

---

### `detect_communities(graph)`

Detect communities (clusters) in the graph using greedy modularity optimization.

**Returns:** `list[set[str]]` — list of node-ID sets, one per community.

---

### `assign_communities(graph, communities)`

Attach community index as a node attribute (`node["community"] = int`).

| Param | Type | Description |
|-------|------|-------------|
| `graph` | `ConceptGraph` | Graph to annotate (modified in place) |
| `communities` | `list[set[str]]` | Output of `detect_communities()` |

**Returns:** `None`

---

### `prune_edges(graph, min_weight)`

Return a new graph with edges below `min_weight` removed.

**Returns:** `ConceptGraph`

---

### `prune_nodes(graph, min_degree)`

Return a new graph with nodes below `min_degree` removed.

**Returns:** `ConceptGraph`

---

### `merge_graphs(g1, g2)`

Merge two graphs into a new graph. Both must have the same directedness.

**Returns:** `ConceptGraph`

---

### `get_subgraph(graph, nodes)`

Extract a subgraph containing only the given nodes and edges between them.

| Param | Type | Description |
|-------|------|-------------|
| `nodes` | `set[str]` | Node IDs to include |

**Returns:** `ConceptGraph`

---

### `graph_density(graph)`

Compute graph density: ratio of actual edges to maximum possible edges.

**Returns:** `float` in [0, 1].

---

## Export

`from concept_mapper.export import ...`

---

### `export_d3_json(graph, path, include_evidence=False, size_by="frequency", compute_communities=True, max_evidence=3)`

Export graph to D3.js JSON format.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `graph` | `ConceptGraph` | required | Graph to export |
| `path` | `Path` | required | Output `.json` file path |
| `include_evidence` | `bool` | `False` | Embed evidence sentences in link objects |
| `size_by` | `str` | `"frequency"` | Node size metric: `"frequency"`, `"degree"`, `"betweenness"` |
| `compute_communities` | `bool` | `True` | Auto-detect and assign community groups |
| `max_evidence` | `int` | `3` | Max evidence sentences per edge |

**Raises:** `EmptyOutputError` if graph has no nodes.

```python
export_d3_json(graph, Path("output/network.json"))
```

---

### `load_d3_json(path)`

Load a previously exported D3 JSON file.

**Returns:** `dict` with `"nodes"` and `"links"` keys.

---

### `generate_html(graph, output_dir, title="Concept Network", width=1200, height=800, include_evidence=False)`

Generate a standalone interactive HTML visualization.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `graph` | `ConceptGraph` | required | Graph to visualize |
| `output_dir` | `Path` | required | Output directory |
| `title` | `str` | `"Concept Network"` | Page title |
| `width` | `int` | `1200` | Canvas width in pixels |
| `height` | `int` | `800` | Canvas height in pixels |
| `include_evidence` | `bool` | `False` | Show evidence in tooltips |

**Returns:** `Path` to generated `index.html`.

Features: force-directed layout, node drag, zoom/pan, community color-coding, hover tooltips, edge width by weight.

```python
html_path = generate_html(graph, Path("output/viz"), title="Lukács Concepts")
```

---

### `export_graphml(graph, path)` · `export_csv(graph, path)` · `export_gexf(graph, path)` · `export_dot(graph, path)`

Export to alternative formats for external tools.

| Function | Format | Tool |
|----------|--------|------|
| `export_graphml` | GraphML | Gephi, yEd, Cytoscape |
| `export_csv` | CSV | Creates `nodes.csv` + `edges.csv` |
| `export_gexf` | GEXF | Gephi |
| `export_dot` | DOT | Graphviz (`dot -Tpng`) |

All take `(graph: ConceptGraph, path: Path)` and return `None`.

---

## Search vs. KWIC Concordance

These two tools serve different purposes and produce different output layouts.

### Search (`find_sentences`)

**Purpose:** Read how a term is used in its full sentential context.

- Returns complete sentences
- Shows N whole sentences before/after for context (`context` parameter in CLI)
- Vertical, paragraph-like layout
- Good for reading and understanding usage in natural flow

**Example output:**
```
[eco_spl.txt:44]   The proletariat must understand its historical position.
[eco_spl.txt:45] > The historical knowledge begins with abstraction.
[eco_spl.txt:46]   Abstraction is the transformation of social relations.
```

### KWIC Concordance (`concordance`)

**Purpose:** Analyze patterns in term usage across many occurrences at a glance.

- Shows a fixed-width snippet centered on the keyword, aligned in columns
- All instances visible simultaneously for scanning
- Sortable by left or right context to reveal collocational patterns
- Classic linguistic research format

**Example output:**
```
    historical knowledge begins with | abstraction | as the fundamental category
            social relations into    | abstraction | transforms them into thing-like
 consciousness can overcome this     | abstraction | by recognizing its own role
```

**When to use each:**

| Goal | Tool |
|------|------|
| Read how a term is used | `find_sentences` |
| Spot collocational patterns | `concordance` |
| Find sentences with multiple terms | `find_sentences_all` |
| Group similar usages | `concordance_sorted` |

---

## How Graph Links Are Determined

### Co-occurrence Graphs (`graph_from_cooccurrence`)

Links represent **terms that appear together** in the corpus.

- **Link creation:** Two terms are connected if their co-occurrence score exceeds `threshold`
- **Window:** Controlled by the `window` parameter in `build_cooccurrence_matrix` — `"sentence"` or `"n_sentences"`
- **Edge weight:** The association score (`count`, `pmi`, or `llr`)
- **Direction:** Undirected (co-occurrence is symmetric)

Example: "abstraction" and "consciousness" co-occurring in 12 sentences with PMI = 2.46 → undirected edge with `weight=2.46`.

### Relation Graphs (`graph_from_relations`)

Links represent **grammatical relationships** between terms.

- **Link creation:** Two terms are connected if a grammatical relation was extracted between them
- **Relation types:** `svo` (subject-verb-object), `copular` (X is Y), `prep` (X of Y)
- **Edge weight:** Number of evidence sentences supporting the relation
- **Edge attributes:** Include `relation_type` and optionally `evidence` sentences
- **Direction:** Directed (grammatical relations have direction)

Example: "consciousness is intentional" appearing 3 times → directed edge from "consciousness" to "intentional" with `relation_type="copular"`, `weight=3`.

---

## Complete Workflow Example

End-to-end pipeline from raw text to interactive visualization.

```python
from pathlib import Path
from concept_mapper.corpus.loader import load_file
from concept_mapper.preprocessing.pipeline import preprocess
from concept_mapper.analysis.reference import load_reference_corpus
from concept_mapper.analysis.rarity import PhilosophicalTermScorer
from concept_mapper.terms.models import TermList, TermEntry
from concept_mapper.analysis.cooccurrence import build_cooccurrence_matrix
from concept_mapper.graph import graph_from_cooccurrence, centrality, detect_communities, assign_communities
from concept_mapper.export import generate_html, export_csv

# 1. Load and preprocess
doc = load_file("samples/eco_spl.txt")
docs = [preprocess(doc)]

# 2. Detect philosophical terms
reference = load_reference_corpus()
scorer = PhilosophicalTermScorer(docs, reference)
candidates = scorer.score_all(min_score=2.0, top_n=30)
print(f"Found {len(candidates)} term candidates")

# 3. Build term list
terms = TermList()
for term, score, _ in candidates:
    terms.add(TermEntry(term=term, metadata={"score": score}))

# 4. Build co-occurrence graph
matrix = build_cooccurrence_matrix(terms, docs, method="pmi", window="sentence")
graph = graph_from_cooccurrence(matrix, threshold=0.3)
print(f"Graph: {graph.node_count()} nodes, {graph.edge_count()} edges")

# 5. Compute metrics
communities = detect_communities(graph)
assign_communities(graph, communities)
scores = centrality(graph, method="betweenness")
print(f"Most central: {max(scores, key=scores.get)}")

# 6. Export
output = Path("output/analysis")
html = generate_html(graph, output / "viz", title="Conceptual Network")
export_csv(graph, output / "csv")
print(f"Open {html} in a browser")
```
