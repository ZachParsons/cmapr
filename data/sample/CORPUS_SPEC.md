# Sample Test Corpus Specification

## Overview

This directory contains a diverse philosophical test corpus designed to verify the concept-mapper's ability to detect and analyze rare, domain-specific terminology across multiple philosophical traditions. The corpus samples from:

1. **Analytic Philosophy & Pragmatism** - Quine, Davidson, Dewey, Peirce
2. **Post-structuralism & Political Philosophy** - Foucault, Derrida, Deleuze, Rawls, Nozick, Sen
3. **Philosophy of Mind & Consciousness Studies** - Dennett, Chalmers, Searle, Levine

## Files

### Real Philosophical Texts
1. `philosopher_1920_cc.txt` - Philosopher' "Class Consciousness" excerpt (90KB, real text with terms like "abstraction")

### Test Files with Invented Terms
2. `test_philosophical_terms.txt` - Simple test file with pure neologisms (dialectology, sublation, concrete universality)
3. `sample1_analytic_pragmatism.txt` - Analytic philosophy & pragmatism (~1500 words)
4. `sample2_poststructural_political.txt` - Post-structuralism & political philosophy (~1600 words)
5. `sample3_mind_consciousness.txt` - Philosophy of mind & consciousness studies (~1800 words)

**Total test corpus size:** ~5000 words across samples

## Invented Technical Terms by Tradition

### Sample 1: Analytic Philosophy & Pragmatism

**Pure neologisms (should detect as not in Brown corpus):**
- `meaning-variance` - Quine's semantic holism
- `instrumental-warranting` - Pragmatist epistemology
- `category-matrices` - Conceptual scheme organization

**Established but rare:**
- `truth-aptness` - Metaethics
- `referential-opacity` - Intensional logic (Frege)
- `pragmatic-maxim` - Peirce's criterion

### Sample 2: Post-structuralism & Political Philosophy

**Pure neologisms:**
- `bio-regulation` - Foucauldian biopolitics
- `différance` - Derrida (French spelling, definitely not in Brown)
- `entitlement-transfer` - Nozick's libertarianism
- `rhizomatic-becoming` - Deleuze & Guattari
- `deterritorialization` / `reterritorialization` - D&G

**Established but specialized:**
- `original-position` - Rawls (may appear in political philosophy texts)
- `capability-approach` - Sen (may appear in development economics)

### Sample 3: Philosophy of Mind & Consciousness

**Pure neologisms:**
- `quale-inversion` - Inverted spectrum arguments
- `zombie-conceivability` - Chalmers' zombie argument
- `modal-reconfiguration` - Modal metaphysics of content

**Established technical terms:**
- `phenomenal-character` - Standard in consciousness studies
- `functional-architecture` - Cognitive science
- `explanatory-gap` - Levine's problem
- `Chinese-room` - Searle's thought experiment
- `intentional-stance` - Dennett

## Expected Detection Behavior

### Phase 2: Frequency Analysis

**High-frequency terms in samples (4+ occurrences):**
- Sample 1: meaning-variance, instrumental-warranting, category-matrices
- Sample 2: bio-regulation, différance, rhizomatic-becoming, deterritorialization
- Sample 3: phenomenal-character, functional-architecture, quale-inversion, zombie-conceivability

**Cross-tradition commonalities:**
- "consciousness" appears across all samples
- "concept" appears frequently
- Methodological terms vary by tradition

### Phase 3: Philosophical Term Detection

**Expected neologisms (not in Brown corpus):**
- meaning-variance, instrumental-warranting, category-matrices
- bio-regulation, différance, rhizomatic-becoming, deterritorialization
- quale-inversion, zombie-conceivability, modal-reconfiguration

**Corpus-specific terms (rare in general English):**
- truth-aptness, referential-opacity
- entitlement-transfer, capability-approach
- phenomenal-character, functional-architecture, explanatory-gap

**Should distinguish from common terms:**
- "the", "of", "and" (function words)
- "theory", "argument", "concept" (meta-vocabulary)

### Phase 4: Term List Auto-Population

Suggested terms to populate (by tradition):

**Analytic/Pragmatist:**
1. meaning-variance
2. instrumental-warranting
3. category-matrices
4. referential-opacity
5. truth-aptness

**Post-structural/Political:**
1. bio-regulation
2. différance
3. rhizomatic-becoming
4. deterritorialization
5. entitlement-transfer
6. capability-approach

**Philosophy of Mind:**
1. phenomenal-character
2. functional-architecture
3. explanatory-gap
4. quale-inversion
5. zombie-conceivability
6. modal-reconfiguration

### Phase 5: Search & Concordance

**Test searches:**
- "meaning-variance" → should find 3-4 sentences in sample1
- "différance" → should find 2-3 sentences in sample2
- "zombie-conceivability" → should find 2-3 sentences in sample3
- "consciousness" → should find sentences across multiple files

### Phase 6: Co-occurrence Analysis

**Expected co-occurrences (sentence level):**

**Sample 1 (Analytic):**
- meaning-variance + referential-opacity
- instrumental-warranting + pragmatic-maxim
- category-matrices + conceptual-scheme

**Sample 2 (Post-structural/Political):**
- bio-regulation + disciplinary (institutions)
- différance + presence
- rhizomatic-becoming + deterritorialization
- entitlement-transfer + taxation

**Sample 3 (Philosophy of Mind):**
- phenomenal-character + explanatory-gap
- functional-architecture + multiple-realizability
- zombie-conceivability + materialism
- quale-inversion + functionalism

## Validation Checklist

- [x] Three diverse sample files created
- [x] Diverse philosophical traditions represented
- [x] Mix of 5 philosophical traditions
- [x] Each file contains 6-8 technical/neologistic terms
- [x] Terms span pure neologisms to established-but-rare
- [x] Frequencies documented in test_corpus_manifest.json
- [x] Files are grammatically correct
- [x] Terms should be detectable as rare/significant

## Usage for Testing

```python
from src.concept_mapper.corpus import load_file, load_directory
from src.concept_mapper.preprocessing import preprocess
from src.concept_mapper.analysis import (
    word_frequencies,
    pos_filtered_frequencies,
    load_reference_corpus
)

# Load sample corpus
corpus = load_directory('data/sample', pattern='sample*.txt')
docs = [preprocess(load_file(f'data/sample/{f}'))
        for f in ['sample1_analytic_pragmatism.txt',
                  'sample2_poststructural_political.txt',
                  'sample3_mind_consciousness.txt']]

# Test frequency detection
for doc in docs:
    nouns = pos_filtered_frequencies(doc, {'NN', 'NNS'}, use_lemmas=True)
    print(f"\nTop nouns: {nouns.most_common(10)}")

# Compare to Brown corpus
brown = load_reference_corpus("brown")
for term in ['meaning-variance', 'différance', 'zombie-conceivability']:
    print(f"{term}: Brown={brown.get(term, 0)}")
```

## Notes

- These samples represent diverse philosophical methodologies and traditions
- Terms are selected to span analytic, continental, and political philosophy
- Some hyphenated terms may be tokenized differently by NLTK
- This corpus provides ground truth for testing term detection algorithms
- Real philosophical texts would have more variation and context
