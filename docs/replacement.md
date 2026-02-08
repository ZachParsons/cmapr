# Synonym Replacement with Inflection Preservation

Replace terms with synonyms throughout your corpus while automatically preserving grammatical inflections (tense, number, degree, person).

## Overview

The `cmapr replace` command performs intelligent synonym replacement that:
- Preserves verb tenses ("running" → "sprinting", "ran" → "sprinted")
- Maintains noun number ("cats" → "dogs")
- Keeps adjective/adverb degrees ("quicker" → "faster", "quickly" → "swiftly")
- Matches capitalization patterns
- Handles both single words and multi-word phrases

## Basic Usage

```bash
# Single word replacement
cmapr replace corpus.json "source_term" "target_term"

# Save to file
cmapr replace corpus.json "source_term" "target_term" -o output.txt

# Preview changes before saving
cmapr replace corpus.json "source_term" "target_term" --preview
```

## Examples

### Single Word Replacement

Replace "run" with "sprint" while preserving all inflections:

```bash
cmapr replace corpus.json "run" "sprint" -o replaced.txt
```

**Input text:**
```
The cat runs quickly. Yesterday, the cats ran fast. They are running now.
```

**Output:**
```
The cat sprints quickly. Yesterday, the cats sprinted fast. They are sprinting now.
```

**What happened:**
- "runs" (present 3rd singular) → "sprints"
- "ran" (past) → "sprinted"
- "running" (present progressive) → "sprinting"

### Phrase to Single Word

Replace multi-word philosophical terms with simpler equivalents:

```bash
cmapr replace corpus.json "body,without,organ" "medium" -o replaced.txt
```

**Input:**
```
The body without organs is a concept. Deleuze's bodies without organs resist.
```

**Output:**
```
The medium is a concept. Deleuze's mediums resist.
```

**Note:** Comma-separated lemmas ("body,without,organ") match any inflected form ("body without organs", "bodies without organs").

### Phrase to Phrase

Replace one multi-word term with another:

```bash
cmapr replace corpus.json "body,without,organ" "blank,resistant,field" -o replaced.txt
```

**Input:**
```
The body without organs is a concept.
```

**Output:**
```
The blank resistant fields is a concept.
```

## How It Works

### 1. Lemma Matching

The replacement system works with **lemmas** (base forms) rather than surface forms:
- Searching for "run" matches "runs", "ran", "running"
- Searching for "cat" matches "cats"
- Searching for "quick" matches "quicker", "quickest", "quickly"

### 2. Inflection Generation

When a match is found, the target term is inflected to match the source:

| Source Form | Source Lemma | Target Lemma | Target Form |
|-------------|--------------|--------------|-------------|
| runs        | run          | sprint       | sprints     |
| ran         | run          | sprint       | sprinted    |
| running     | run          | sprint       | sprinting   |
| cats        | cat          | dog          | dogs        |
| quicker     | quick        | fast         | faster      |

### 3. Capitalization Matching

Capitalization patterns are preserved:
- Sentence-initial: "Running" → "Sprinting"
- All-caps: "RUN" → "SPRINT"
- Title case: "The Cat" → "The Dog"

### 4. Phrase Matching

Multi-word phrases are matched via lemmatized n-grams:
- Matches any inflected variant
- Identifies head word (rightmost noun/verb)
- Inflects target based on head word's grammar

## Advanced Usage

### Preview Mode

Check changes before committing:

```bash
cmapr replace corpus.json "dialectical" "dynamic" --preview
```

Output:
```
Preview of changes:
======================================================================
The dynamic method involves the negation of negation. Dynamic logic
applies to concrete thought. The dynamics differ from formal logic.
======================================================================

Total length: 1245 characters across 3 document(s)
```

### Multi-Word Phrase Syntax

Use comma-separated **lemmas** (not inflected forms):

| Phrase in Text            | Command Syntax           |
|---------------------------|--------------------------|
| "body without organs"     | "body,without,organ"     |
| "bodies without organs"   | "body,without,organ"     |
| "runs quickly"            | "run,quickly"            |
| "ran quickly"             | "run,quickly"            |

**Always use the lemma (base form) in the command.**

## Limitations

### 1. Grammatical Context

The inflector matches the morphology of the source word but not broader grammatical context:

```
Input:  "The bodies without organs is important"  (note: "organs" is plural)
Output: "The mediums is important"                 (note: "medium" → "mediums")
```

The replacement preserves the plurality of "organs" (plural noun) even though it creates an awkward sentence. Manual post-editing may be needed for complex cases.

### 2. Ambiguous POS Tagging

The POS tagger sometimes misidentifies words in ambiguous contexts:

```
"The body without organs resists"  →  "organs" tagged as adjective (wrong)
"The body without organs is a concept"  →  "organs" tagged as noun (correct)
```

Use clearer sentence structures for better results.

### 3. Semantic Appropriateness

The tool preserves **form** but not **meaning**. Replacing "dialectical" with "dynamic" may be semantically inappropriate depending on context. Always review output.

## Python API

For programmatic access:

```python
from concept_mapper.corpus.models import ProcessedDocument
from concept_mapper.transformations.replacement import ReplacementSpec, SynonymReplacer

# Load preprocessed document
doc = ProcessedDocument.from_dict(corpus_data[0])

# Single word replacement
spec = ReplacementSpec("run", "sprint")
replacer = SynonymReplacer()
result = replacer.replace_in_document(spec, doc)

# Phrase to single word
spec = ReplacementSpec(["body", "without", "organ"], "medium")
result = replacer.replace_in_document(spec, doc)

# Phrase to phrase
spec = ReplacementSpec(
    ["body", "without", "organ"],
    ["blank", "resistant", "field"]
)
result = replacer.replace_in_document(spec, doc)
```

## Use Cases

### 1. Stylistic Revision

Replace technical jargon with accessible synonyms:
```bash
cmapr replace corpus.json "ontological" "existential" -o accessible.txt
```

### 2. Theoretical Translation

Convert between philosophical vocabularies:
```bash
cmapr replace corpus.json "subject" "dasein" -o heideggerian.txt
```

### 3. Corpus Normalization

Standardize variant terminology:
```bash
cmapr replace corpus.json "phenomenological,reduction" "epoché" -o normalized.txt
```

### 4. Exploratory Analysis

Test how changing key terms affects text meaning:
```bash
cmapr replace corpus.json "being" "becoming" --preview
```

## Technical Details

### Modules

- `transformations/inflection.py`: POS-aware inflection generation
- `transformations/phrase_matcher.py`: Multi-word phrase matching
- `transformations/replacement.py`: Synonym replacement orchestration
- `transformations/text_reconstruction.py`: Smart spacing and punctuation

### Inflection Rules

- **Verbs**: Handles 6 forms (base, 3rd singular, past, progressive, perfect, gerund)
- **Nouns**: Singular/plural inflection
- **Adjectives**: Positive/comparative/superlative degrees
- **Adverbs**: Handles -ly suffixation and degree

### Dependencies

- **NLTK**: POS tagging and lemmatization
- **Custom inflection rules**: Python 3.14 compatible (no lemminflect dependency issues)

## Troubleshooting

### Replacement Not Working

**Problem:** Term not being replaced.

**Solution:** Ensure you're using the **lemma** (base form), not an inflected form:
- Use "run" not "running"
- Use "body,without,organ" not "bodies,without,organs"

### Awkward Grammar

**Problem:** Replacement creates grammatically awkward sentences.

**Solution:** The tool preserves morphology but not syntax. Edit manually or use simpler sentence structures.

### Missing Inflections

**Problem:** Some inflected forms aren't generated correctly.

**Solution:** Check that the POS tag is being identified correctly. Complex or rare words may need manual inflection rules added to `inflection.py`.

## See Also

- [API Reference](api-reference.md): Programmatic usage examples
- [Development Roadmap](roadmap.md): Implementation details and future enhancements
- [Text Cleaning](../README.md#ocr-pdf-cleaning): Preprocess OCR artifacts before replacement
