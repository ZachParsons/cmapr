# Complete Tutorial

This tutorial walks through the complete workflow using Umberto Eco's "Semiotics and the Philosophy of Language" as an example.

## Sample Corpus

The `samples/eco_spl.txt` file contains Umberto Eco's complete book "Semiotics and the Philosophy of Language" (~110K words, 634KB). This rich philosophical text features extensive semiotic and linguistic terminology:

- **Semiotics/Semiosis** - The study of signs and meaning-making processes
- **Porphyrian tree** - Hierarchical classification system from Porphyry
- **Isotopy** - Semantic coherence across a text
- **Synecdoche/Metonymy** - Rhetorical tropes and meaning relations
- **Signifier/Interpretants** - Core Peircean semiotic concepts
- **Rhizome** - Deleuze & Guattari's non-hierarchical structure
- References to major theorists: Peirce, Hjelmslev, Greimas, Jakobson, Lacan, Lévi-Strauss

## Step 1: Ingest and Preprocess

Load the Eco text and preprocess it (tokenization, POS tagging, lemmatization):

```bash
cmapr ingest samples/eco_spl.txt
```

**Expected output:**
```
✓ Saved 1 processed document(s) to output/corpus/eco_spl.json
```

**What happens:** The raw text is tokenized into sentences and words, each word is tagged with its part of speech (NN, VB, etc.), and lemmatized (e.g., "entities" → "entity").

**OCR/PDF Cleaning:** For texts from scanned PDFs or with OCR errors, use the `--clean-ocr` flag to automatically fix common issues:

```bash
cmapr ingest scanned_text.pdf --clean-ocr
```

This fixes:
- Split words ("obsti nacy" → "obstinacy")
- Spaced numbers ("1 . 5" → "1.5")
- Page numbers at line breaks
- Common OCR character errors

**Inspect the output:**
```bash
jq '.documents[0] | keys' output/corpus/eco_spl.json
# Shows: raw_text, sentences, tokens, pos_tags, lemmas, metadata
```

### Step 2: Extract Rare/Distinctive Terms

Identify terms that appear frequently in Eco but rarely in general English:

```bash
cmapr rarities output/corpus/eco_spl.json \
  --method ratio \
  --top-n 50 \
  --threshold 0.3 \
  --output output/terms/eco_spl.json
```

**Method options:**
- `ratio` - Compare term frequency in corpus vs Brown reference corpus
- `tfidf` - TF-IDF scoring within a multi-document corpus (if you have multiple texts)
- `combined` - Combine multiple signals (ratio, tfidf, neologisms)

**Parameters:**
- `--top-n 50` - Return top 50 distinctive terms
- `--threshold 0.3` - Minimum distinctiveness score (higher = more distinctive)

**Expected output:**
```
📊 Analyzing term distinctiveness...
✓ Found 50 distinctive terms
✓ Saved to output/terms/eco_spl.json
```

**Sample distinctive terms** (from Eco):
```
semiosis (ratio: 245.0, freq: 98)
porphyrian (ratio: 198.5, freq: 15)
rhizome (ratio: 156.2, freq: 24)
isotopy (ratio: 124.8, freq: 31)
interpretant (ratio: 118.3, freq: 47)
```

### Step 3: Build Concept Graph

Analyze relationships between the distinctive terms:

```bash
cmapr graph output/corpus/eco_spl.json \
  --terms output/terms/eco_spl.json \
  --method cooccurrence \
  --threshold 0.3 \
  --output output/graphs/eco_spl.json
```

**Relationship methods:**
- `cooccurrence` - Terms appearing together in sentences (fastest, broadest)
- `svo` - Subject-verb-object grammatical triples (slowest, most precise)
- `all` - Both methods combined

**Expected output:**
```
🔍 Building concept graph...
✓ Found 142 co-occurrence relations
✓ Saved to output/graphs/eco_spl.json
```

**Sample relations:**
```json
{
  "source": "sign",
  "target": "interpretation",
  "relation_type": "cooccurrence",
  "score": 0.78,
  "evidence_count": 23
}
```

### Step 4: Export Visualization

Generate an interactive HTML network visualization:

```bash
cmapr export output/graphs/eco_spl.json \
  --format html \
  --title "Eco - Semiotics & Philosophy of Language" \
  --output output/exports/eco_spl
```

**Export formats:**
- `html` - Interactive D3.js force-directed graph (recommended)
- `json` - Raw graph data for custom visualization
- `csv` - Tabular format for analysis in Excel/R/Python

**Expected output:**
```
📦 Exporting visualization...
✓ Exported to output/exports/eco_spl/index.html
```

**Open the visualization:**
```bash
open output/exports/eco_spl/index.html
```

### Interpreting the Graph

The visualization shows:
- **Nodes**: Distinctive terms sized by frequency
- **Edges**: Relationships between terms, weighted by co-occurrence strength
- **Clusters**: Groups of related concepts (automatic community detection)
- **Interactive**: Click/drag nodes, hover for details

**Example insights from Eco graph:**
- **Semiotic cluster**: sign, signifier, interpretant, semiosis
- **Rhetorical cluster**: metaphor, synecdoche, metonymy
- **Structural cluster**: rhizome, tree, labyrinth, network
- **Reference cluster**: peirce, hjelmslev, jakobson (theorists)

## Advanced: Contextual Relations

For deeper analysis, extract full contextual relations (co-occurrence + SVO):

```bash
cmapr analyze output/corpus/eco_spl.json "sign" \
  --format text
```

This shows:
1. **SVO relations** - Grammatical triples like "sign → represents → object"
2. **Co-occurrence** - Terms that frequently appear with "sign"
3. **Evidence sentences** - Actual quotes showing each relation

**Sample output:**
```
Found 23 contextual relations for 'sign'

SVO Relations (8):
  1. sign → represents → object (score: 2.45, 12 occurrences)
     "The sign stands for the object..."

  2. sign → produces → interpretant (score: 1.89, 8 occurrences)
     "Every sign produces an interpretant..."

Co-occurrence Relations (15):
  1. sign ↔ interpretation (score: 3.21, 45 occurrences)
  2. sign ↔ referent (score: 2.87, 34 occurrences)
```

## Batch Processing Multiple Texts

Process multiple documents to compare vocabularies:

```bash
# Ingest multiple texts
cmapr ingest samples/text1.txt samples/text2.txt samples/text3.txt \
  -o output/corpus/multi_author.json

# Extract distinctive terms (TF-IDF works well for multi-document)
cmapr rarities output/corpus/multi_author.json \
  --method tfidf \
  --top-n 30 \
  --output output/terms/multi_author.json

# Build graph
cmapr graph output/corpus/multi_author.json \
  --terms output/terms/multi_author.json \
  --method cooccurrence \
  --output output/graphs/multi_author.json

# Export
cmapr export output/graphs/multi_author.json \
  --format html \
  --title "Comparative Philosophical Vocabularies" \
  --output output/exports/multi_author
```

## Quick Workflow Script

Use `examples/workflow.sh` to run all steps automatically:

```bash
bash examples/workflow.sh samples/eco_spl.txt
```

This executes:
1. Ingestion with preprocessing
2. Term extraction (ratio method)
3. Graph construction (cooccurrence)
4. HTML export
5. Opens result in browser

Perfect for quick exploration of new texts.
