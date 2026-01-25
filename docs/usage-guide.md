# Concept Mapper: Usage Guide

Practical examples demonstrating the functionality of each completed phase.

## Prerequisites

```bash
# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
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
    term="dasein",
    lemma="dasein",
    pos="NN",
    definition="Being-there; human existence in Heidegger's philosophy",
    notes="Central concept in Being and Time",
    examples=[
        "Dasein is always my Dasein.",
        "The being of Dasein is care."
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
```

---

## Next Steps

- **Phase 8:** Build concept graphs from co-occurrence and relations
- **Phase 9:** Export to D3.js for interactive visualization
- **Phase 10:** CLI interface for batch processing

See the [roadmap](concept-mapper-roadmap.md) for details on upcoming phases.
