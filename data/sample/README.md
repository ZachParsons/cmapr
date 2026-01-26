# Sample Corpus Files

This directory contains sample philosophical texts for testing and demonstration.

## Files

### Main Sample

**`philosopher_1920_cc.txt`** (89KB)
- Excerpt from György Philosopher' *History and Class Consciousness* (1920)
- Large sample for realistic testing
- Contains terms like: abstraction, proletariat, bourgeoisie, commodity, consciousness
- Use for: Full pipeline testing, performance benchmarks

### Test Files

**`test_philosophical_terms.txt`** (961B)
- Synthetic test file with planted rare terms
- Contains deliberately unusual terms for detection testing
- Used in unit tests to verify term detection algorithms

**`sample1_analytic_pragmatism.txt`** (1.9KB)
- Analytic philosophy and pragmatism concepts
- Terms: meaning-variance, instrumental-warranting, pragmatic maxim, truth-aptness, referential opacity
- Used in: Corpus analysis tests (test_sample_corpus.py)

**`sample2_poststructural_political.txt`** (2.1KB)
- Poststructural and political philosophy concepts
- Terms: bio-regulation, différance, rhizomatic-becoming, deterritorialization, capability-approach
- Used in: Corpus analysis tests (test_sample_corpus.py)

**`sample3_mind_consciousness.txt`** (2.6KB)
- Philosophy of mind concepts
- Terms: phenomenal-character, functional-architecture, explanatory gap, quale-inversion, zombie-conceivability
- Used in: Corpus analysis tests (test_sample_corpus.py)

### Documentation

**`CORPUS_SPEC.md`**
- Specification for test corpus structure
- Documents expected properties of test files

**`test_corpus_manifest.json`**
- Machine-readable manifest of test corpus
- Expected values for verification

## Usage

### Quick Test
```bash
# Process main sample
concept-mapper ingest data/sample/philosopher_1920_cc.txt
```

### Full Corpus Test
```bash
# Process all samples
concept-mapper ingest data/sample/ -r -p "*.txt"
```

### Individual Samples
```bash
# Test analytic philosophy detection
concept-mapper ingest data/sample/sample1_analytic_pragmatism.txt
concept-mapper rarities output/corpus/corpus.json --top-n 10

# Test poststructural detection
concept-mapper ingest data/sample/sample2_poststructural_political.txt
concept-mapper rarities output/corpus/corpus.json --top-n 10

# Test philosophy of mind detection
concept-mapper ingest data/sample/sample3_mind_consciousness.txt
concept-mapper rarities output/corpus/corpus.json --top-n 10
```

## Design Notes

- **Diverse traditions**: Samples include continental & analytic, philosophy of mind & society.
- **Test-driven**: Each sample has expected terms for validation
- **Realistic vs. Synthetic**: philosopher_1920_cc.txt is real text; others are constructed for testing specific terms
