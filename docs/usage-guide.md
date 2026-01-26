# Concept Mapper: Usage Guide

Practical examples demonstrating the functionality of each completed phase.

## Prerequisites

```bash
# Install package with all dependencies
uv pip install -e .
```

---

## Phase 0: Project Setup

**What it enables:** Basic project infrastructure, storage, and test data.

### Example: Verify Installation

```python
from concept_mapper.corpus.models import Document, Corpus
from pathlib import Path

# Check that sample corpus exists
sample_dir = Path("data/sample")
files = list(sample_dir.glob("*.txt"))
print(f"Found {len(files)} sample documents")
for f in files:
    print(f"  - {f.name}")
```

**Expected output:**
```
Found 5 sample documents
  - hegel_phenomenology_excerpt.txt
  - philosopher_1920_cc.txt
  - test_philosophical_terms.txt
  - ...
```

---

## Phase 1: Corpus Preprocessing

**What it enables:** Load and preprocess text documents (tokenization, POS tagging, lemmatization).

### Example: Process a Text File

```python
from concept_mapper.corpus.loader import load_document
from concept_mapper.preprocessing.pipeline import preprocess

# Load a document
doc = load_document("data/sample/philosopher_1920_cc.txt")
print(f"Loaded: {doc.metadata.get('title', 'untitled')}")
print(f"Length: {len(doc.text)} characters\n")

# Preprocess it
processed = preprocess(doc)

# Inspect results
print(f"Sentences: {processed.num_sentences}")
print(f"Tokens: {processed.num_tokens}\n")

# View first sentence with POS tags
print("First sentence:")
print(f"  Text: {processed.sentences[0]}\n")
print("  Tokens with POS tags:")
for token, pos in processed.pos_tags[:10]:  # First 10 tokens
    print(f"    {token:15} {pos}")
```

**Expected output:**
```
Loaded: History and Class Consciousness
Length: 93284 characters

Sentences: 347
Tokens: 19234

First sentence:
  Text: The historical knowledge of the proletariat...

  Tokens with POS tags:
    The             DT
    historical      JJ
    knowledge       NN
    of              IN
    the             DT
    proletariat     NN
    ...
```

### Example: Lemmatization

```python
from concept_mapper.preprocessing.lemmatize import lemmatize_tagged

# Get lemmas for the first sentence
first_sent_tokens = processed.pos_tags[:20]
lemmas = lemmatize_tagged(first_sent_tokens)

print("Word → Lemma:")
for (word, pos), lemma in zip(first_sent_tokens, lemmas):
    if word.lower() != lemma:  # Only show where lemma differs
        print(f"  {word:15} → {lemma}")
```

**Expected output:**
```
Word → Lemma:
  begins          → begin
  developing      → develop
  consciousness   → consciousness
```

---

## Phase 2: Frequency Analysis & TF-IDF

**What it enables:** Analyze term frequencies, compare to reference corpus, compute TF-IDF scores.

### Example: Word Frequency Distribution

```python
from concept_mapper.corpus.loader import load_directory
from concept_mapper.preprocessing.pipeline import preprocess_corpus
from concept_mapper.analysis.frequency import word_frequencies

# Load and preprocess corpus
corpus = load_directory("data/sample")
docs = preprocess_corpus(corpus)

# Get word frequencies (lemmatized)
freqs = word_frequencies(docs, use_lemmas=True)

# Top 20 most common words
print("Top 20 most common words:")
for word, count in freqs.most_common(20):
    print(f"  {word:20} {count:5}")
```

**Expected output:**
```
Top 20 most common words:
  the                  1247
  of                    856
  be                    623
  to                    589
  and                   512
  in                    487
  abstraction           156
  consciousness         142
  ...
```

### Example: TF-IDF Scores

```python
from concept_mapper.analysis.tfidf import corpus_tfidf_scores

# Compute TF-IDF across corpus
tfidf_scores = corpus_tfidf_scores(docs)

# Top distinctive terms
print("\nTop 10 terms by TF-IDF:")
sorted_terms = sorted(tfidf_scores.items(), key=lambda x: x[1], reverse=True)
for term, score in sorted_terms[:10]:
    print(f"  {term:20} {score:.4f}")
```

**Expected output:**
```
Top 10 terms by TF-IDF:
  abstraction          0.0156
  proletariat          0.0089
  bourgeoisie          0.0067
  commodity            0.0054
  ...
```

### Example: Compare to Brown Corpus

```python
from concept_mapper.analysis.reference import load_reference_corpus
from concept_mapper.analysis.rarity import compare_to_reference

# Load reference corpus
reference = load_reference_corpus()
print(f"Reference corpus size: {sum(reference.values()):,} words\n")

# Compare author's usage to general English
comparison = compare_to_reference(docs, reference)

# Terms overused compared to Brown corpus
print("Terms highly distinctive vs. general English:")
sorted_comp = sorted(comparison.items(), key=lambda x: x[1], reverse=True)
for term, ratio in sorted_comp[:15]:
    print(f"  {term:20} {ratio:8.2f}x more common")
```

**Expected output:**
```
Reference corpus size: 1,161,192 words

Terms highly distinctive vs. general English:
  abstraction            2847.56x more common
  proletariat             892.34x more common
  bourgeoisie             654.21x more common
  commodity               423.12x more common
  ...
```

---

## Phase 3: Philosophical Term Detection

**What it enables:** Automatically identify author-specific conceptual vocabulary using multiple detection methods.

### Example: Hybrid Philosophical Term Scorer

```python
from concept_mapper.analysis.rarity import PhilosophicalTermScorer

# Initialize scorer with multiple detection methods
scorer = PhilosophicalTermScorer(
    docs=docs,
    reference_corpus=reference,
    use_lemmas=True,
    min_author_freq=3,
    weights={
        'ratio': 1.0,        # Corpus comparison
        'tfidf': 1.0,        # TF-IDF
        'neologism': 0.5,    # Not in WordNet
        'definitional': 0.3, # Appears in definitions
        'capitalized': 0.2   # Capitalized technical terms
    }
)

# Score all terms
results = scorer.score_all(min_score=1.0, top_n=20)

print("Top 20 philosophical term candidates:")
print(f"{'Term':<25} {'Score':>8} {'Signals':>8}")
print("-" * 45)

for term, total_score, components in results:
    # Count how many signals detected this term
    signal_count = sum(1 for k, v in components.items()
                      if k != 'raw_total' and v > 0)
    print(f"{term:<25} {total_score:8.2f} {signal_count:8} signals")
```

**Expected output:**
```
Top 20 philosophical term candidates:
Term                      Score  Signals
---------------------------------------------
abstraction               4.87   5 signals
proletariat               3.45   4 signals
bourgeoisie               3.21   4 signals
commodification           2.98   3 signals
fetishism                 2.76   3 signals
praxis                    2.54   4 signals
...
```

### Example: High-Confidence Terms (Multiple Signal Agreement)

```python
# Get terms detected by multiple methods
high_confidence = scorer.get_high_confidence_terms(
    min_signals=3,  # At least 3 detection methods agree
    min_score=1.5
)

print("\nHigh-confidence philosophical terms (3+ signals):")
for term, total_score, components in high_confidence:
    print(f"\n{term} (score: {total_score:.2f})")
    print("  Detected by:")
    if components['ratio'] > 0:
        print(f"    ✓ Corpus comparison (ratio: {components['ratio']:.2f})")
    if components['tfidf'] > 0:
        print(f"    ✓ TF-IDF (score: {components['tfidf']:.4f})")
    if components['neologism'] > 0:
        print(f"    ✓ Neologism (not in WordNet)")
    if components['definitional'] > 0:
        print(f"    ✓ Definitional context ({components['definitional']:.0f} occurrences)")
    if components['capitalized'] > 0:
        print(f"    ✓ Capitalized term ({components['capitalized']:.0f} occurrences)")
```

**Expected output:**
```
High-confidence philosophical terms (3+ signals):

abstraction (score: 4.87)
  Detected by:
    ✓ Corpus comparison (ratio: 2847.56)
    ✓ TF-IDF (score: 0.0156)
    ✓ Neologism (not in WordNet)
    ✓ Definitional context (12 occurrences)
    ✓ Capitalized term (8 occurrences)

proletariat (score: 3.45)
  Detected by:
    ✓ Corpus comparison (ratio: 892.34)
    ✓ TF-IDF (score: 0.0089)
    ✓ Definitional context (7 occurrences)
    ✓ Capitalized term (15 occurrences)
...
```

---

## Phase 4: Term List Management

**What it enables:** Curate and manage a list of philosophical terms with metadata, import/export in multiple formats.

### Example: Create and Populate Term List

```python
from concept_mapper.terms.models import TermList, TermEntry
from concept_mapper.terms.suggester import suggest_terms_from_analysis

# Auto-populate from Phase 3 analysis
suggested_terms = suggest_terms_from_analysis(
    docs=docs,
    reference_corpus=reference,
    min_score=2.0,
    top_n=30,
    max_examples=3
)

print(f"Auto-suggested {len(suggested_terms.list_terms())} terms\n")

# Inspect suggested terms
for entry in suggested_terms.list_terms()[:5]:
    print(f"\nTerm: {entry.term}")
    print(f"  POS: {entry.pos}")
    if entry.examples:
        print(f"  Examples:")
        for ex in entry.examples[:2]:
            print(f"    - {ex[:80]}...")
```

**Expected output:**
```
Auto-suggested 30 terms

Term: abstraction
  POS: NN
  Examples:
    - The historical knowledge of the proletariat begins with abstraction...
    - Abstraction transforms social relations into things...

Term: proletariat
  POS: NN
  Examples:
    - The consciousness of the proletariat...
    - Only the proletariat can overcome abstraction...
...
```

### Example: Manual Curation

```python
# Add custom terms with definitions
term_list = TermList()

term_list.add(TermEntry(
    term="Geist",
    lemma="geist",
    pos="NN",
    definition="Spirit; the self-developing rational principle in Hegel's philosophy",
    notes="Central concept in Phenomenology of Spirit",
    examples=[
        "Geist is the process of becoming itself.",
        "Absolute Geist reconciles subject and substance."
    ]
))

term_list.add(TermEntry(
    term="praxis",
    lemma="praxis",
    pos="NN",
    definition="Practical action informed by theory",
    notes="Key Thinkerist concept"
))

# Save to file
term_list.save("output/philosophical_terms.json")
print(f"Saved {len(term_list.list_terms())} terms to file")

# Export to CSV for spreadsheet editing
from concept_mapper.terms.manager import TermManager
manager = TermManager(term_list)
manager.export_to_csv("output/terms.csv")
print("Also exported to CSV for editing in Excel")
```

### Example: Import and Merge

```python
# Load saved term list
loaded_terms = TermList.load("output/philosophical_terms.json")
print(f"Loaded {len(loaded_terms.list_terms())} terms")

# Merge with suggested terms (without overwriting)
merged = loaded_terms.merge(suggested_terms, overwrite=False)
print(f"After merge: {len(merged.list_terms())} terms")

# Get statistics
manager = TermManager(merged)
stats = manager.get_statistics()
print(f"\nTerm list statistics:")
print(f"  Total terms: {stats['total_terms']}")
print(f"  With definitions: {stats['with_definitions']}")
print(f"  With examples: {stats['with_examples']}")
print(f"  POS distribution: {stats['pos_distribution']}")
```

---

## Phase 5: Search & Concordance

**What it enables:** Find and view terms in context with various display formats.

### Example: Basic Search

```python
from concept_mapper.search import find_sentences

# Find all sentences containing a term
matches = find_sentences("abstraction", docs)

print(f"Found {len(matches)} sentences containing 'abstraction'\n")

# Display first 3 matches
for i, match in enumerate(matches[:3], 1):
    print(f"{i}. [{match.doc_id}:{match.sent_index}]")
    print(f"   {match.sentence}\n")
```

**Expected output:**
```
Found 23 sentences containing 'abstraction'

1. [philosopher_1920_cc.txt:45]
   The historical knowledge of the proletariat begins with abstraction.

2. [philosopher_1920_cc.txt:78]
   Abstraction transforms social relations into thing-like structures.

3. [philosopher_1920_cc.txt:112]
   Only by understanding abstraction can consciousness overcome it.
```

### Example: KWIC Concordance Display

```python
from concept_mapper.search import concordance, format_kwic_lines

# Generate KWIC display (aligned on keyword)
kwic_lines = concordance("abstraction", docs, width=40)

print("KWIC Concordance for 'abstraction':")
print(format_kwic_lines(kwic_lines[:10], width=40))
```

**Expected output:**
```
KWIC Concordance for 'abstraction':
    historical knowledge begins with [abstraction] as the fundamental category of
              social relations into [abstraction] transforms them into thing-like
     consciousness can overcome this [abstraction] by recognizing its own role in
...
```

### Example: Context Windows

```python
from concept_mapper.search import get_context, format_context_windows

# Get sentences before and after each match
windows = get_context("abstraction", docs, n_sentences=2)

print("Context windows (2 sentences before/after):\n")
print(format_context_windows(windows[:2], separator="---"))
```

**Expected output:**
```
Context windows (2 sentences before/after):

[philosopher_1920_cc.txt:45]

  The proletariat must understand its historical position.
  This understanding requires grasping fundamental categories.
> The historical knowledge begins with abstraction.
  Abstraction is the transformation of social relations.
  These relations appear as independent things.
---

[philosopher_1920_cc.txt:78]
...
```

### Example: Term Dispersion Analysis

```python
from concept_mapper.search import get_dispersion_summary, dispersion_plot_data

# Analyze where term appears across documents
summary = get_dispersion_summary("abstraction", docs)

print(f"Dispersion analysis for '{summary['term']}':")
print(f"  Appears in: {summary['docs_with_term']}/{summary['total_docs']} documents ({summary['coverage']:.1f}%)")
print(f"  Total occurrences: {summary['total_occurrences']}")
print(f"  Average per document: {summary['avg_occurrences_per_doc']:.1f}")
print("\nDocument distribution:")
for doc_id, positions in summary['positions'].items():
    print(f"  {doc_id}: {len(positions)} occurrences at sentences {positions[:5]}...")
```

**Expected output:**
```
Dispersion analysis for 'abstraction':
  Appears in: 3/5 documents (60.0%)
  Total occurrences: 23
  Average per document: 7.7

Document distribution:
  philosopher_1920_cc.txt: 18 occurrences at sentences [45, 78, 112, 156, 203]...
  hegel_excerpt.txt: 3 occurrences at sentences [12, 34, 67]...
  test_philosophical_terms.txt: 2 occurrences at sentences [5, 8]...
```

---

## Phase 6: Co-occurrence Analysis

**What it enables:** Discover which terms appear together and measure association strength.

### Example: Sentence-level Co-occurrence

```python
from concept_mapper.analysis import cooccurs_in_sentence

# Find terms that co-occur with "abstraction"
cooccurs = cooccurs_in_sentence("abstraction", docs)

print("Top 15 terms co-occurring with 'abstraction':")
for term, count in cooccurs.most_common(15):
    print(f"  {term:20} {count:3} times")
```

**Expected output:**
```
Top 15 terms co-occurring with 'abstraction':
  consciousness        12 times
  social                9 times
  relations             8 times
  commodity             7 times
  proletariat           6 times
  fetishism             5 times
  ...
```

### Example: Statistical Significance (PMI)

```python
from concept_mapper.analysis import pmi, log_likelihood_ratio

# Measure association between terms
term1 = "abstraction"
term2 = "consciousness"

pmi_score = pmi(term1, term2, docs)
llr_score = log_likelihood_ratio(term1, term2, docs)

print(f"Association between '{term1}' and '{term2}':")
print(f"  PMI score: {pmi_score:.3f}", end="")
if pmi_score > 0:
    print(" (positive association)")
elif pmi_score < 0:
    print(" (negative association)")
else:
    print(" (independent)")

print(f"  LLR (G²): {llr_score:.2f}", end="")
if llr_score > 10.83:
    print(" (highly significant, p < 0.001)")
elif llr_score > 6.63:
    print(" (significant, p < 0.01)")
elif llr_score > 3.84:
    print(" (significant, p < 0.05)")
else:
    print(" (not significant)")
```

**Expected output:**
```
Association between 'abstraction' and 'consciousness':
  PMI score: 2.456 (positive association)
  LLR (G²): 15.67 (highly significant, p < 0.001)
```

### Example: Co-occurrence Matrix

```python
from concept_mapper.analysis import (
    build_cooccurrence_matrix,
    save_cooccurrence_matrix,
    get_top_cooccurrences
)

# Build matrix for curated terms
matrix = build_cooccurrence_matrix(
    merged,  # Term list from Phase 4
    docs,
    method="pmi",       # Or "count" or "llr"
    window="sentence"
)

print(f"Built {len(matrix)}×{len(matrix)} co-occurrence matrix")

# Save to CSV
save_cooccurrence_matrix(matrix, "output/cooccurrence_pmi.csv")
print("Saved to output/cooccurrence_pmi.csv (open in Excel)\n")

# Quick exploration: top associations for a term
top = get_top_cooccurrences("abstraction", docs, n=10, method="pmi")
print("Top 10 terms associated with 'abstraction' (by PMI):")
for term, score in top:
    print(f"  {term:20} {score:6.3f}")
```

**Expected output:**
```
Built 30×30 co-occurrence matrix
Saved to output/cooccurrence_pmi.csv (open in Excel)

Top 10 terms associated with 'abstraction' (by PMI):
  fetishism            3.245
  commodity            2.987
  consciousness        2.456
  separation           2.234
  ...
```

---

## Phase 7: Relation Extraction

**What it enables:** Extract grammatical relationships (SVO, copular definitions, prepositional phrases).

### Example: Subject-Verb-Object Triples

```python
from concept_mapper.analysis import extract_svo_for_term

# Find all SVO triples involving "consciousness"
triples = extract_svo_for_term("consciousness", docs)

print(f"Found {len(triples)} SVO triples involving 'consciousness'\n")

print("Sample SVO triples:")
for triple in triples[:5]:
    print(f"  {triple}")
    print(f"    From: {triple.sentence[:80]}...\n")
```

**Expected output:**
```
Found 8 SVO triples involving 'consciousness'

Sample SVO triples:
  (consciousness, involves, intentionality)
    From: Consciousness involves intentionality in all its forms...

  (proletariat, develops, consciousness)
    From: The proletariat develops consciousness through struggle...

  (consciousness, overcomes, abstraction)
    From: Only consciousness can overcome abstraction...
```

### Example: Copular Definitions (X is Y)

```python
from concept_mapper.analysis import extract_copular

# Find definitional relationships
definitions = extract_copular("being", docs)

print("Definitional relations for 'being':")
for defn in definitions[:5]:
    print(f"\n  {defn.subject} {defn.copula} {defn.complement}")
    print(f"    Source: {defn.sentence[:80]}...")
```

**Expected output:**
```
Definitional relations for 'being':

  Being is presence
    Source: Being is presence in time and nothing else...

  being was conceived as
    Source: In the tradition, being was conceived as substance...

  Being becomes actual
    Source: Being becomes actual through consciousness...
```

### Example: Prepositional Relations

```python
from concept_mapper.analysis import extract_prepositional

# Find prepositional phrases
prep_relations = extract_prepositional("consciousness", docs)

print("Prepositional relations for 'consciousness':")
for rel in prep_relations[:5]:
    print(f"  {rel.head} {rel.prep} {rel.object}")
```

**Expected output:**
```
Prepositional relations for 'consciousness':
  consciousness of objects
  consciousness in time
  consciousness of self
  consciousness through praxis
  consciousness from abstraction
```

### Example: Aggregated Relations with Evidence

```python
from concept_mapper.analysis import get_relations

# Get all relations for a term with evidence sentences
relations = get_relations("abstraction", docs, types=["copular", "prep"])

print(f"Found {len(relations)} unique relations for 'abstraction'\n")

# Show relations with evidence
for rel in relations[:3]:
    print(f"\n{rel}")
    print(f"  Type: {rel.relation_type}")
    print(f"  Evidence ({len(rel.evidence)} sentences):")
    for evidence in rel.evidence[:2]:
        print(f"    - {evidence[:80]}...")
```

**Expected output:**
```
Found 7 unique relations for 'abstraction'

abstraction --[copular]--> transformation ({'copula': 'is'})
  Type: copular
  Evidence (3 sentences):
    - Abstraction is the transformation of social relations into things...
    - What is abstraction? It is the transformation of human activity...

abstraction --[prep]--> consciousness ({'preposition': 'of'})
  Type: prep
  Evidence (2 sentences):
    - The abstraction of consciousness prevents self-awareness...
    - Through abstraction of its own activity, consciousness becomes...
```

---

## Complete Workflow Example

Putting it all together to analyze a philosophical text:

```python
#!/usr/bin/env python3
"""
Complete workflow: Analyze a philosophical text to extract key concepts
and their relationships.
"""

from concept_mapper.corpus.loader import load_document
from concept_mapper.preprocessing.pipeline import preprocess
from concept_mapper.analysis.reference import load_reference_corpus
from concept_mapper.analysis.rarity import PhilosophicalTermScorer
from concept_mapper.terms.suggester import suggest_terms_from_analysis
from concept_mapper.analysis import (
    get_top_cooccurrences,
    get_relations
)

# 1. Load and preprocess
print("Loading document...")
doc = load_document("data/sample/philosopher_1920_cc.txt")
processed = preprocess(doc)
print(f"✓ Processed {processed.num_sentences} sentences\n")

# 2. Detect philosophical terms
print("Detecting philosophical terms...")
reference = load_reference_corpus()
scorer = PhilosophicalTermScorer([processed], reference)
candidates = scorer.score_all(min_score=2.0, top_n=10)
print(f"✓ Found {len(candidates)} term candidates\n")

# 3. Create term list
print("Top 10 philosophical terms:")
for term, score, _ in candidates:
    print(f"  {term:20} (score: {score:.2f})")
print()

# 4. Analyze relationships for top term
top_term = candidates[0][0]
print(f"Analyzing '{top_term}'...\n")

# Co-occurrence
print("Top associated terms:")
cooccurs = get_top_cooccurrences(top_term, [processed], n=5, method="pmi")
for term, pmi in cooccurs:
    print(f"  {term:20} (PMI: {pmi:.2f})")
print()

# Relations
print("Grammatical relations:")
relations = get_relations(top_term, [processed])
for rel in relations[:5]:
    print(f"  {rel.source} --[{rel.relation_type}]--> {rel.target}")
print()

print("✓ Analysis complete!")
```

**Run it:**
```bash
python workflow_example.py
```

**Expected output:**
```
Loading document...
✓ Processed 347 sentences

Detecting philosophical terms...
✓ Found 10 term candidates

Top 10 philosophical terms:
  abstraction          (score: 4.87)
  proletariat          (score: 3.45)
  bourgeoisie          (score: 3.21)
  commodity            (score: 2.98)
  fetishism            (score: 2.76)
  ...

Analyzing 'abstraction'...

Top associated terms:
  consciousness        (PMI: 2.46)
  fetishism            (PMI: 3.25)
  commodity            (PMI: 2.99)
  separation           (PMI: 2.23)
  proletariat          (PMI: 1.87)

Grammatical relations:
  abstraction --[copular]--> transformation
  consciousness --[svo]--> abstraction
  abstraction --[prep]--> consciousness
  proletariat --[svo]--> abstraction
  abstraction --[copular]--> process

✓ Analysis complete!
```

---

## Interactive Exploration with IPython

For interactive exploration, use IPython:

```bash
ipython
```

```python
# Quick setup
from concept_mapper.corpus.loader import load_directory
from concept_mapper.preprocessing.pipeline import preprocess_corpus

corpus = load_directory("data/sample")
docs = preprocess_corpus(corpus)

# Now explore with tab completion
from concept_mapper.search import *
from concept_mapper.analysis import *

# Find and view term in context
matches = find_sentences("consciousness", docs)
windows = get_context("consciousness", docs, n_sentences=2)
print(windows[0])

# Measure associations
pmi("consciousness", "being", docs)
cooccurs = cooccurs_in_sentence("consciousness", docs)
cooccurs.most_common(10)

# Extract relations
relations = get_relations("consciousness", docs)
for r in relations:
    print(r)
```

---

## Phase 8: Graph Construction

Build network graphs from co-occurrence and relation data.

### What It Enables

- **Graph Data Structure**: Represent concepts as nodes with relationships as edges
- **Build from Co-occurrence**: Create graphs from statistical co-occurrence matrices
- **Build from Relations**: Create graphs from extracted grammatical relations
- **Graph Operations**: Merge, prune, filter, and extract subgraphs
- **Graph Metrics**: Compute centrality, detect communities, find paths

### Example: Build Graph from Co-occurrence

```python
from concept_mapper.analysis.cooccurrence import build_cooccurrence_matrix
from concept_mapper.graph import graph_from_cooccurrence, centrality
from concept_mapper.terms.models import TermList

# Build co-occurrence matrix
terms = TermList([
    {"term": "consciousness", "pos": "NN"},
    {"term": "intentionality", "pos": "NN"},
    {"term": "being", "pos": "NN"},
    {"term": "presence", "pos": "NN"}
])

matrix = build_cooccurrence_matrix(
    term_list=terms,
    docs=docs,
    method="pmi",  # Use PMI scores as edge weights
    window="sentence"
)

# Create graph with threshold
graph = graph_from_cooccurrence(matrix, threshold=0.5)

print(graph)
# ConceptGraph(Undirected, nodes=4, edges=3)

# Check what's connected
print(graph.nodes())
# ['consciousness', 'intentionality', 'being', 'presence']

print(graph.edges())
# [('consciousness', 'intentionality'), ('consciousness', 'being'), ...]

# Get edge weight
edge = graph.get_edge("consciousness", "intentionality")
print(f"PMI: {edge['weight']:.2f}")
# PMI: 0.85
```

### Example: Build Graph from Relations

```python
from concept_mapper.analysis.relations import get_relations
from concept_mapper.graph import graph_from_relations

# Extract relations
relations = get_relations("consciousness", docs, types=["copular", "prep"])

# Build directed graph
graph = graph_from_relations(relations)

print(graph)
# ConceptGraph(Directed, nodes=5, edges=8)

# Examine a relation
edge = graph.get_edge("consciousness", "intentional")
print(f"Type: {edge['relation_type']}")
# Type: copular

print(f"Evidence count: {edge['weight']}")
# Evidence count: 3

print(f"Example: {edge['evidence'][0]}")
# Example: Consciousness is intentional.
```

### Example: Graph Operations

```python
from concept_mapper.graph import (
    merge_graphs,
    prune_edges,
    prune_nodes,
    get_subgraph,
    filter_by_relation_type
)

# Merge two graphs
cooccur_graph = graph_from_cooccurrence(matrix, threshold=0.3)
relation_graph = graph_from_relations(relations)
combined = merge_graphs(cooccur_graph, relation_graph.copy())  # Must have same directedness

# Prune weak edges
strong_graph = prune_edges(graph, min_weight=0.7)
print(f"Removed {graph.edge_count() - strong_graph.edge_count()} weak edges")
# Removed 12 weak edges

# Remove isolated nodes
connected_graph = prune_nodes(graph, min_degree=1)
print(f"Removed {graph.node_count() - connected_graph.node_count()} isolated nodes")
# Removed 3 isolated nodes

# Extract subgraph for specific terms
key_terms = {"consciousness", "being", "intentionality"}
subgraph = get_subgraph(graph, key_terms)
print(subgraph)
# ConceptGraph(Directed, nodes=3, edges=4)

# Filter to only copular relations
copular_graph = filter_by_relation_type(graph, {"copular"})
print(f"Copular relations: {copular_graph.edge_count()}")
# Copular relations: 15
```

### Example: Graph Metrics

```python
from concept_mapper.graph import (
    centrality,
    detect_communities,
    assign_communities,
    get_connected_components,
    graph_density
)

# Compute centrality (find most important concepts)
degree_scores = centrality(graph, method="degree")
betweenness_scores = centrality(graph, method="betweenness")

# Most central concepts
for term, score in sorted(degree_scores.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{term:20} {score:.3f}")
# consciousness        0.425
# being                0.380
# intentionality       0.315
# ...

# Detect communities (conceptual clusters)
communities = detect_communities(graph, method="greedy")
print(f"Found {len(communities)} communities")
# Found 3 communities

for i, community in enumerate(communities):
    print(f"Community {i}: {', '.join(sorted(community)[:5])}")
# Community 0: being, presence, time, consciousness
# Community 1: intentionality, object, awareness
# Community 2: abstraction, commodity, fetishism

# Assign communities as node attributes
assign_communities(graph, communities)
node = graph.get_node("consciousness")
print(f"Consciousness is in community {node['community']}")
# Consciousness is in community 0

# Check graph connectivity
components = get_connected_components(graph)
print(f"Graph has {len(components)} connected components")
# Graph has 2 connected components

# Compute graph density
density = graph_density(graph)
print(f"Graph density: {density:.3f}")
# Graph density: 0.147
```

### Example: Complete Workflow

```python
from pathlib import Path
from concept_mapper.corpus.loader import load_document
from concept_mapper.preprocessing.pipeline import preprocess
from concept_mapper.analysis.reference import load_reference_corpus
from concept_mapper.analysis.rarity import PhilosophicalTermScorer
from concept_mapper.terms.models import TermList
from concept_mapper.analysis.relations import get_relations
from concept_mapper.analysis.cooccurrence import build_cooccurrence_matrix
from concept_mapper.graph import (
    graph_from_relations,
    graph_from_cooccurrence,
    centrality,
    detect_communities,
    assign_communities,
)

# 1. Load and preprocess
doc = load_document("data/sample/philosopher_1920_cc.txt")
docs = [preprocess(doc)]

# 2. Detect key terms
reference = load_reference_corpus()
scorer = PhilosophicalTermScorer(docs, reference)
candidates = scorer.score_all(min_score=2.0, top_n=30)

# 3. Create term list
terms = TermList([{"term": term, "pos": "NN"} for term, _, _ in candidates])

# 4. Build co-occurrence graph
matrix = build_cooccurrence_matrix(terms, docs, method="pmi")
cooccur_graph = graph_from_cooccurrence(matrix, threshold=0.3)

# 5. Build relation graph
all_relations = []
for term_data in terms:
    relations = get_relations(term_data["term"], docs)
    all_relations.extend(relations)

relation_graph = graph_from_relations(all_relations)

# 6. Compute centrality
central = centrality(cooccur_graph, method="betweenness")
top_concepts = sorted(central.items(), key=lambda x: x[1], reverse=True)[:10]

print("Top 10 central concepts:")
for term, score in top_concepts:
    print(f"  {term:20} {score:.3f}")

# 7. Detect communities
communities = detect_communities(cooccur_graph)
assign_communities(cooccur_graph, communities)

print(f"\nFound {len(communities)} conceptual clusters")
for i, community in enumerate(communities[:3]):
    print(f"  Cluster {i}: {', '.join(sorted(community)[:5])}")
```

**Expected Output:**
```
Top 10 central concepts:
  abstraction          0.425
  consciousness        0.380
  commodity            0.315
  proletariat          0.298
  bourgeoisie          0.275
  fetishism            0.251
  totality             0.228
  praxis               0.210
  dialectic            0.195
  object               0.180

Found 3 conceptual clusters
  Cluster 0: commodity, fetishism, abstraction, value, exchange
  Cluster 1: consciousness, proletariat, bourgeoisie, class, totality
  Cluster 2: praxis, dialectic, subject, object, history
```

---

## Phase 9: Export & Visualization

Export graphs to various formats for visualization and analysis.

### What It Enables

- **D3.js JSON**: Interactive web visualizations with force-directed layouts
- **GraphML**: Import into Gephi, yEd, Cytoscape for advanced graph analysis
- **DOT**: Render with Graphviz for publication-quality diagrams
- **CSV**: Export to spreadsheets for manual inspection and editing
- **HTML**: Standalone interactive visualizations that run in any web browser

### Example: Export to D3 JSON

```python
from pathlib import Path
from concept_mapper.export import export_d3_json, load_d3_json

# Export graph to D3 JSON
export_d3_json(
    graph=cooccur_graph,
    path=Path("output/network.json"),
    include_evidence=False,  # Set True to include evidence sentences
    size_by="betweenness",   # Size nodes by centrality
    compute_communities=True  # Detect and color-code communities
)

# Load the exported data
data = load_d3_json(Path("output/network.json"))
print(f"Exported {len(data['nodes'])} nodes and {len(data['links'])} links")
# Exported 30 nodes and 45 links
```

### Example: Generate Interactive HTML Visualization

```python
from concept_mapper.export import generate_html

# Generate standalone HTML visualization
html_path = generate_html(
    graph=cooccur_graph,
    output_dir=Path("output/visualization"),
    title="Philosopher Conceptual Network",
    width=1200,
    height=800,
    include_evidence=True  # Show evidence in tooltips
)

print(f"Open {html_path} in your browser")
# Open output/visualization/index.html in your browser
```

**The generated HTML includes:**
- Force-directed graph layout
- Interactive node dragging
- Zoom and pan controls
- Tooltips showing node/edge information
- Color-coded communities
- Node size by centrality or frequency
- Edge width by weight

### Example: Export to Multiple Formats

```python
from concept_mapper.export import (
    export_graphml,
    export_dot,
    export_csv,
    export_gexf,
)

# For Gephi (GraphML)
export_graphml(cooccur_graph, Path("output/network.graphml"))

# For Graphviz (DOT) - requires pydot
try:
    export_dot(cooccur_graph, Path("output/network.dot"), layout="neato")
    # Then render: dot -Tpng output/network.dot -o output/network.png
except ImportError:
    print("Install pydot for DOT export: uv pip install pydot")

# For spreadsheets (CSV)
export_csv(cooccur_graph, Path("output/csv/"))
# Creates: output/csv/nodes.csv, output/csv/edges.csv

# For Gephi (GEXF)
export_gexf(cooccur_graph, Path("output/network.gexf"))
```

### Example: Export with Evidence

```python
from concept_mapper.export import export_d3_json

# Export relation graph with evidence sentences
export_d3_json(
    graph=relation_graph,
    path=Path("output/relations.json"),
    include_evidence=True,
    max_evidence=3  # Limit to 3 example sentences per edge
)

# The resulting JSON will have evidence in link objects:
# {
#   "nodes": [...],
#   "links": [
#     {
#       "source": "consciousness",
#       "target": "intentional",
#       "weight": 3,
#       "label": "copular",
#       "evidence": [
#         "Consciousness is intentional.",
#         "Consciousness is always intentional.",
#         "All consciousness is intentional."
#       ]
#     }
#   ]
# }
```

### Example: Complete Visualization Workflow

```python
from pathlib import Path
from concept_mapper.corpus.loader import load_document
from concept_mapper.preprocessing.pipeline import preprocess
from concept_mapper.analysis.reference import load_reference_corpus
from concept_mapper.analysis.rarity import PhilosophicalTermScorer
from concept_mapper.terms.models import TermList
from concept_mapper.analysis.cooccurrence import build_cooccurrence_matrix
from concept_mapper.graph import graph_from_cooccurrence, centrality, detect_communities, assign_communities
from concept_mapper.export import generate_html, export_csv

# 1. Load and analyze text
doc = load_document("data/sample/philosopher_1920_cc.txt")
docs = [preprocess(doc)]

reference = load_reference_corpus()
scorer = PhilosophicalTermScorer(docs, reference)
candidates = scorer.score_all(min_score=2.0, top_n=30)

# 2. Create term list
terms = TermList([{"term": term, "pos": "NN"} for term, _, _ in candidates])

# 3. Build co-occurrence graph
matrix = build_cooccurrence_matrix(terms, docs, method="pmi", window="sentence")
graph = graph_from_cooccurrence(matrix, threshold=0.3)

# 4. Compute metrics
communities = detect_communities(graph)
assign_communities(graph, communities)

central = centrality(graph, method="betweenness")
for node_id in graph.nodes():
    node_attrs = graph.get_node(node_id)
    node_attrs["centrality"] = central.get(node_id, 0.0)
    # Update node with centrality
    graph.graph.nodes[node_id].update(node_attrs)

# 5. Export to multiple formats
output_dir = Path("output/philosopher_network")

# Interactive HTML visualization
html_path = generate_html(
    graph,
    output_dir,
    title="Philosopher: History and Class Consciousness",
    width=1400,
    height=900
)

# CSV for spreadsheet analysis
export_csv(graph, output_dir / "csv")

print(f"""
Visualization complete!

Interactive HTML: {html_path}
CSV data: {output_dir}/csv/

Open the HTML file in a web browser to explore the network.
- Drag nodes to rearrange
- Hover for term information
- Use mouse wheel to zoom
- Colors show conceptual communities
""")
```

**Expected Output:**
```
Visualization complete!

Interactive HTML: output/philosopher_network/index.html
CSV data: output/philosopher_network/csv/

Open the HTML file in a web browser to explore the network.
- Drag nodes to rearrange
- Hover for term information
- Use mouse wheel to zoom
- Colors show conceptual communities
```

---

## Output Files

The analysis produces these output files:

```
output/
├── philosophical_terms.json       # Curated term list
├── terms.csv                      # CSV for Excel editing
├── cooccurrence_pmi.csv          # Co-occurrence matrix
└── analysis_results.txt          # Text report
```

**View co-occurrence matrix in Excel:**
```bash
open output/cooccurrence_pmi.csv
```

---

## Running Tests

Verify everything works:

```bash
# Run all tests
pytest tests/ -v

# Run specific phase tests
pytest tests/test_corpus.py -v          # Phase 1
pytest tests/test_analysis.py -v        # Phases 2-3
pytest tests/test_terms.py -v           # Phase 4
pytest tests/test_search.py -v          # Phase 5
pytest tests/test_cooccurrence.py -v    # Phase 6
pytest tests/test_relations.py -v       # Phase 7
pytest tests/test_graph.py -v           # Phase 8
pytest tests/test_export.py -v          # Phase 9
pytest tests/test_cli.py -v             # Phase 10
```

---

## Phase 10: CLI Interface

Unified command-line interface for all functionality.

### What It Enables

- **Batch Processing**: Process entire corpora from the command line
- **Workflow Automation**: Chain commands together in scripts
- **No Code Required**: Full functionality without writing Python
- **Progress Feedback**: Visual progress bars and verbose output
- **File-Based I/O**: Work with JSON, CSV, and other file formats

### Installation

After installing the package, the `concept-mapper` command is available:

```bash
# Install package
uv pip install -e .

# Verify installation
concept-mapper --help
```

### Example: Complete Workflow

```bash
# 1. Ingest and preprocess documents
concept-mapper ingest data/sample/philosopher_1920_cc.txt -o output/corpus.json

# 2. Detect philosophical terms
concept-mapper rarities output/corpus.json \
  --method hybrid \
  --threshold 2.0 \
  --top-n 30 \
  -o output/terms.json

# 3. Search for a specific term
concept-mapper search output/corpus.json "abstraction" \
  --context 2 \
  -o output/abstraction.txt

# 4. Generate concordance
concept-mapper concordance output/corpus.json "consciousness" \
  --width 60 \
  -o output/concordance.txt

# 5. Build concept graph
concept-mapper graph output/corpus.json \
  -t output/terms.json \
  --method cooccurrence \
  --threshold 0.3 \
  -o output/graph.json

# 6. Export to HTML visualization
concept-mapper export output/graph.json \
  --format html \
  --title "Philosopher Conceptual Network" \
  -o output/visualization/

# 7. Open in browser
open output/visualization/index.html
```

### Command Reference

#### Ingest Command

Load and preprocess documents:

```bash
# Single file
concept-mapper ingest document.txt -o corpus.json

# Directory (recursive)
concept-mapper ingest corpus/ \
  --recursive \
  --pattern "*.txt" \
  -o corpus.json

# With verbose output
concept-mapper --verbose ingest document.txt -o corpus.json
```

#### Rarities Command

Detect philosophical/rare terms:

```bash
# Basic usage
concept-mapper rarities corpus.json -o terms.json

# Specify method and threshold
concept-mapper rarities corpus.json \
  --method hybrid \
  --threshold 2.5 \
  --top-n 50 \
  -o terms.json

# Different methods: ratio, tfidf, neologism, hybrid
concept-mapper rarities corpus.json --method tfidf -o terms.json
```

#### Search Command

Search for term occurrences:

```bash
# Basic search
concept-mapper search corpus.json "consciousness"

# With context sentences
concept-mapper search corpus.json "being" --context 2

# Save to file
concept-mapper search corpus.json "abstraction" -o results.txt
```

#### Concordance Command

Display KWIC concordance:

```bash
# Basic concordance
concept-mapper concordance corpus.json "consciousness"

# Custom context width
concept-mapper concordance corpus.json "being" --width 80

# Save to file
concept-mapper concordance corpus.json "fetishism" -o concordance.txt
```

#### Graph Command

Build concept graphs:

```bash
# From co-occurrence
concept-mapper graph corpus.json \
  -t terms.json \
  --method cooccurrence \
  --threshold 0.3 \
  -o graph.json

# From relations
concept-mapper graph corpus.json \
  -t terms.json \
  --method relations \
  -o graph.json
```

#### Export Command

Export graphs to various formats:

```bash
# HTML visualization
concept-mapper export graph.json \
  --format html \
  --title "My Network" \
  -o viz/

# GraphML for Gephi
concept-mapper export graph.json \
  --format graphml \
  -o graph.graphml

# CSV for spreadsheets
concept-mapper export graph.json \
  --format csv \
  -o output/

# D3 JSON
concept-mapper export graph.json \
  --format d3 \
  -o network.json

# GEXF for Gephi
concept-mapper export graph.json \
  --format gexf \
  -o graph.gexf
```

### Global Options

Available for all commands:

```bash
# Verbose output
concept-mapper --verbose ingest document.txt

# Custom output directory
concept-mapper --output-dir /path/to/output ingest document.txt

# Combined
concept-mapper -v -o output/ rarities corpus.json
```

### Batch Processing Example

Process multiple documents:

```bash
#!/bin/bash

# Process entire corpus
for corpus_dir in data/corpora/*; do
    author=$(basename "$corpus_dir")

    echo "Processing $author..."

    # Ingest
    concept-mapper ingest "$corpus_dir" \
        --recursive \
        -o "output/$author/corpus.json"

    # Detect terms
    concept-mapper rarities "output/$author/corpus.json" \
        --top-n 30 \
        -o "output/$author/terms.json"

    # Build graph
    concept-mapper graph "output/$author/corpus.json" \
        -t "output/$author/terms.json" \
        -m cooccurrence \
        -o "output/$author/graph.json"

    # Export visualization
    concept-mapper export "output/$author/graph.json" \
        --format html \
        --title "$author Conceptual Network" \
        -o "output/$author/viz/"
done

echo "Done! Open output/*/viz/index.html to view networks"
```

---

## Next Steps

- **Phase 11:** Documentation and deployment
- **Phase 12:** Advanced features (temporal analysis, cross-corpus comparison)

See the [roadmap](concept-mapper-roadmap.md) for details on upcoming phases.
