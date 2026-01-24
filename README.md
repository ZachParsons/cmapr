# Concept Mapper

A tool for extracting and visualizing an author's idiosyncratic conceptual vocabulary from primary texts.

## Overview

Concept Mapper analyzes texts to identify terms with author-specific meanings, understand their usage through co-occurrence and grammatical relations, and export concept maps for D3 visualization.


## Installation

### 1. Clone and Setup Virtual Environment

```bash
cd /Users/zach/se/projects/nlp/spike
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download NLTK Data

```bash
python scripts/download_nltk_data.py
```

This downloads all required NLTK datasets (punkt, POS taggers, wordnet, brown corpus, etc.)

## Development Commands

### Code Formatting

Format all Python files with black:

```bash
# Format all files
venv/bin/black *.py

# Check formatting without making changes
venv/bin/black *.py --check

# Format specific file
venv/bin/black pos_tagger.py
```

### Linting

Check code quality with ruff:

```bash
# Check all files
venv/bin/ruff check *.py

# Auto-fix issues
venv/bin/ruff check *.py --fix

# Check specific files
venv/bin/ruff check pos_tagger.py test_pos_tagger.py
```

### Running Tests

Run tests with pytest:

```bash
# Run all tests
venv/bin/pytest

# Run with verbose output
venv/bin/pytest -v

# Run specific test file
venv/bin/pytest test_pos_tagger.py -v

# Run tests in a directory
venv/bin/pytest tests/ -v

# Run specific test
venv/bin/pytest tests/test_sample_corpus.py::TestSampleCorpusFrequencies::test_sample1_token_count -v

# Show test coverage
venv/bin/pytest --cov=src
```

### Combined Check (Format, Lint, Test)

Run all checks before committing:

```bash
venv/bin/black *.py && \
venv/bin/ruff check *.py && \
venv/bin/pytest tests/ -v
```

## Usage

### Basic Analysis

Analyze a text file:

```python
import pos_tagger as pt

# Analyze a file
result = pt.run('data/sample/philosopher_1920_cc.txt')

print(f"Total tokens: {result['token_count']}")
print(f"Top content verbs: {result['content_verbs'][:5]}")
print(f"Top nouns: {result['nouns'][:5]}")
```

### Custom Pipeline

Build a custom analysis pipeline:

```python
import pos_tagger as pt

# Load and process step by step
text = pt.load_text('your_file.txt')
tokens = pt.tokenize_words(text)
tagged = pt.tag_parts_of_speech(tokens)

# Extract specific POS categories
verbs = pt.extract_words_by_pos(tagged, 'V')
nouns = pt.extract_words_by_pos(tagged, 'N')

# Filter and count
common_verbs = pt.get_common_verbs()
stop_words = pt.get_stopwords_set()
filtered = pt.filter_common_words(verbs, common_verbs, stop_words)
freq_dist = pt.calculate_frequency_distribution(filtered)
top_verbs = pt.get_most_common(freq_dist, 20)
```

### Search Functionality

Find sentences containing specific terms:

```python
import pos_tagger as pt

# Search for a term in a file
sentences = pt.search_term_in_file('data/sample/philosopher_1920_cc.txt', 'consciousness')
print(f"Found {len(sentences)} sentences")

# Display results
for sentence in sentences[:5]:
    print(f"  - {sentence}")
```

### Sample Corpus

Test with the provided sample corpus:

```python
import pos_tagger as pt

files = [
    'data/sample/sample1_analytic_pragmatism.txt',
    'data/sample/sample2_poststructural_political.txt',
    'data/sample/sample3_mind_consciousness.txt'
]

for file in files:
    result = pt.run(file)
    print(f"\nFile: {file}")
    print(f"Top nouns: {result['nouns'][:5]}")
```

## Interactive Shell

Use the enhanced IPython REPL for development:

```bash
venv/bin/ipython
```

```python
import pos_tagger as pt

# Auto-reload modules when they change
%load_ext autoreload
%autoreload 2

# Run analysis
result = pt.run('data/sample/sample1_analytic_pragmatism.txt')
result['content_verbs'][:10]

# Reload module after changes
import importlib
importlib.reload(pt)
```

## Project Structure

```
spike/
├── src/
│   └── concept_mapper/          # Main package (Phase 1+)
│       └── __init__.py
├── tests/                       # Test suite
│   ├── test_pos_tagger.py      # Core functionality tests (17 tests)
│   └── test_sample_corpus.py   # Corpus validation tests (29 tests)
├── data/
│   └── sample/                  # Sample test corpus
│       ├── sample1_analytic_pragmatism.txt
│       ├── sample2_poststructural_political.txt
│       ├── sample3_mind_consciousness.txt
│       ├── philosopher_1920_cc.txt  # Philosopher test file
│       └── CORPUS_SPEC.md      # Detailed corpus specification
├── docs/                        # Project documentation
│   ├── concept-mapper-roadmap.md  # Full development roadmap
├── scripts/
│   └── download_nltk_data.py   # NLTK data downloader
├── output/                      # Generated output files
├── pos_tagger.py               # Refactored analysis functions
├── requirements.txt            # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Dependencies

### Core NLP
- `nltk==3.9.2` - Natural Language Processing toolkit
- `click==8.3.1` - CLI framework

### Development Tools
- `pytest==9.0.2` - Testing framework
- `black==26.1.0` - Code formatter
- `ruff==0.14.14` - Fast Python linter
- `ipython==9.9.0` - Enhanced REPL

See `requirements.txt` for full dependency list.

## Testing

### Test Suites

- **`test_pos_tagger.py`** (17 tests) - Tests for core NLP functions
  - Tokenization, POS tagging, lemmatization
  - Filtering, frequency analysis
  - Search and pipeline functions

- **`tests/test_sample_corpus.py`** (29 tests) - Validates sample corpus
  - Token counts
  - Term frequencies
  - Cross-file term detection
  - Search functionality
  - Analysis pipeline integration

### Run All Tests

```bash
venv/bin/pytest tests/ -v
# 29 passed
```

### Test Coverage

Current test coverage:
- Core functions: 17 tests ✓
- Corpus validation: 29 tests ✓
- **Total: 46/46 tests passing** ✓

## Development Workflow

1. **Activate virtual environment**
   ```bash
   source venv/bin/activate
   ```

2. **Make changes to code**

3. **Format code**
   ```bash
   venv/bin/black *.py
   ```

4. **Check linting**
   ```bash
   venv/bin/ruff check *.py --fix
   ```

5. **Run tests**
   ```bash
   venv/bin/pytest tests/ -v
   ```

6. **Commit changes**
   ```bash
   git add .
   git commit -m "Description of changes"
   ```

## Roadmap

See `docs/concept-mapper-roadmap.md` for the complete development roadmap.


## Documentation

- **`docs/concept-mapper-roadmap.md`** - Full project roadmap
- **`data/sample/CORPUS_SPEC.md`** - Sample corpus specification

## References

- Lane 2019, *Natural Language Processing in Action*
- Rockwell & Sinclair 2016, *Hermeneutica*

## License

[License information to be added]

## Contributing

[Contributing guidelines to be added]
