# Sample Test Corpus Specification

## Overview

This directory contains a small test corpus designed to verify the concept-mapper's ability to detect and analyze rare, domain-specific terminology. The corpus consists of three philosophical texts with invented technical terms that have known frequencies.

## Files

1. `sample1_dialectics.txt` - Focuses on dialectical concepts (175 words)
2. `sample2_epistemology.txt` - Focuses on epistemological concepts (180 words)
3. `sample3_ontology.txt` - Focuses on ontological concepts (172 words)

**Total corpus size:** 527 words

## Invented Rare Terms (Hyphenated Compounds)

These are the technical terms invented for testing. They should be detected as rare/significant terms by the rarity detection algorithms.

**Note:** Some compound terms like "categorial synthesis" and "eidetic reduction" are tokenized as separate words. Both the compound and individual components are counted below.

### Cross-File Terms (appear in multiple files)

| Term | File 1 | File 2 | File 3 | Total | Notes |
|------|--------|--------|--------|-------|-------|
| `dasein-flux` | 6 | 0 | 1 | 7 | Appears in both dialectics and ontology |
| `geist-praxis` | 7 | 0 | 1 | 8 | Appears in both dialectics and ontology |

### File-Specific Terms (Actual Frequencies)

#### sample1_dialectics.txt (170 tokens)
| Term | Count | POS | Notes |
|------|-------|-----|-------|
| `dasein-flux` | 6 | NN | Hyphenated, kept together |
| `geist-praxis` | 7 | NN | Hyphenated, kept together |
| `abstraction` | 5 | NN | Single word |
| `totality-consciousness` | 5 | NN | Hyphenated, kept together |

#### sample2_epistemology.txt (172 tokens)
| Term | Count | POS | Notes |
|------|-------|-----|-------|
| `noetic-intuition` | 4 | NN | Hyphenated, kept together |
| `categorial` | 4 | NN/JJ | From "categorial synthesis" |
| `synthesis` | 4 | NN | From "categorial synthesis" |
| `intentionality-vectors` | 5 | NN | Hyphenated, kept together |
| `eidetic` | 5 | JJ | From "eidetic reduction" |
| `reduction` | 5 | NN | From "eidetic reduction" |
| `lifeworld-horizons` | 4 | NN | Hyphenated, kept together |

#### sample3_ontology.txt (182 tokens)
| Term | Count | POS | Notes |
|------|-------|-----|-------|
| `being-toward-finitude` | 6 | NN | Hyphenated, kept together |
| `existential-thrownness` | 6 | NN | Hyphenated, kept together |
| `worldhood-disclosure` | 5 | NN | Hyphenated, kept together |
| `hermeneutic-circle` | 5 | NN | Hyphenated, kept together |
| `resolute` | 4 | JJ/NN | Adjective or noun depending on context |
| `dasein-flux` | 1 | NN | Hyphenated, kept together |
| `geist-praxis` | 1 | NN | Hyphenated, kept together |

## Expected Behavior

### Phase 2: Frequency Analysis
- All hyphenated terms should have frequency >= 1
- Cross-file terms (`Dasein-flux`, `Geist-praxis`) should have higher total frequency (5)
- Most single-file terms appear exactly 4 times in their respective file

### Phase 3: Rarity Detection

#### Hapax Legomena (appear exactly once in entire corpus)
None of the invented technical terms are hapax legomena, but these single-occurrence terms exist:
- Various common words appearing only once

#### Low Frequency Terms (actual counts across corpus)
Should detect all invented technical terms as low-frequency:
- `geist-praxis`: 8 (7 in sample1, 1 in sample3)
- `dasein-flux`: 7 (6 in sample1, 1 in sample3)
- `being-toward-finitude`: 6
- `existential-thrownness`: 6
- `abstraction`: 5
- `totality-consciousness`: 5
- `intentionality-vectors`: 5
- `eidetic`: 5 (component of "eidetic reduction")
- `reduction`: 5 (component of "eidetic reduction")
- `worldhood-disclosure`: 5
- `hermeneutic-circle`: 5
- `noetic-intuition`: 4
- `categorial`: 4 (component of "categorial synthesis")
- `synthesis`: 4 (component of "categorial synthesis")
- `lifeworld-horizons`: 4
- `resolute`: 4

#### POS-Filtered Low Frequency Terms
When filtering for nouns (NN, NNP, NNS), should detect all hyphenated terms listed above.

### Phase 4: Term List
Suggested terms to auto-populate:
1. `Dasein-flux`
2. `Geist-praxis`
3. `totality-consciousness`
4. `noetic-intuition`
5. `intentionality-vectors`
6. `eidetic reduction`
7. `lifeworld-horizons`
8. `Being-toward-finitude`
9. `existential-thrownness`
10. `worldhood-disclosure`
11. `hermeneutic-circle`
12. `abstraction`
13. `categorial synthesis`

### Phase 5: Search & Concordance
- Searching for "Dasein-flux" should return 7 sentences (6 from sample1, 1 from sample3)
- Searching for "Geist-praxis" should return 8 sentences (7 from sample1, 1 from sample3)
- Searching for "consciousness" should return sentences from sample1 (includes totality-consciousness)
- Searching for "reduction" should return 5 sentences from sample2 (eidetic reduction)

### Phase 6: Co-occurrence Analysis

#### Expected Co-occurrences (Sentence Level)
- `Dasein-flux` co-occurs with:
  - `abstraction` (1 sentence in sample1)
  - `totality-consciousness` (1 sentence in sample1)
  - `Geist-praxis` (1 sentence in sample1 and 1 in sample3)
  - `worldhood-disclosure` (1 sentence in sample3)

- `Geist-praxis` co-occurs with:
  - `Dasein-flux` (2 sentences)
  - `totality-consciousness` (1 sentence)
  - `abstraction` (1 sentence)
  - `resolute` (1 sentence in sample3)

#### Expected Co-occurrences (Paragraph Level)
Each file is essentially one large paragraph, so paragraph-level co-occurrence should capture all terms within each file.

## Usage for Testing

```python
import pos_tagger as pt

# Load and analyze the sample corpus
files = ['data/sample/sample1_dialectics.txt',
         'data/sample/sample2_epistemology.txt',
         'data/sample/sample3_ontology.txt']

# Test frequency detection
for file in files:
    result = pt.run(file)
    print(f"\nFile: {file}")
    print(f"Content verbs: {result['content_verbs'][:5]}")
    print(f"Nouns: {result['nouns'][:10]}")

# Test search
sentences = pt.search_term_in_file('data/sample/sample1_dialectics.txt', 'Dasein-flux')
print(f"\nFound {len(sentences)} sentences with 'Dasein-flux' in sample1")
# Expected: 4 sentences

# Cross-file search (once directory loading is implemented)
# Should find 5 total occurrences across files
```

## Validation Checklist

- [ ] All three sample files created
- [ ] Each file contains 4 unique invented technical terms
- [ ] Cross-file terms (`Dasein-flux`, `Geist-praxis`) appear in exactly 2 files
- [ ] Frequencies documented match actual counts
- [ ] Files are readable and grammatically correct
- [ ] Terms are detectable as rare/significant
- [ ] POS tagging correctly identifies hyphenated terms as nouns

## Notes

- Hyphenated compound terms may be tokenized in different ways by NLTK (kept together or split)
- If tokenization splits hyphenated terms, expected frequencies will differ
- This corpus is intentionally artificial to provide ground truth for testing
- Real philosophical texts would have more variation and less predictable frequencies
