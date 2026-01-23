# Environment Setup

## Quick Start

```bash
# Activate virtual environment
source venv/bin/activate

# Install dependencies (already done)
pip install -r requirements.txt

# Verify installation
python -c "import nltk; print('NLTK:', nltk.__version__)"
```

## Environment Details

- **Python Version:** 3.14.0
- **Virtual Environment:** `venv/` (fresh install on 2026-01-22)
- **Package Manager:** pip 25.3

## Installed Packages

### Core NLP
- `nltk==3.9.2` - Natural Language Processing toolkit
- `click==8.3.1` - CLI framework (bundled with nltk, also needed for Phase 10)

### Development Tools
- `pytest==9.0.2` - Testing framework
- `black==26.1.0` - Code formatter
- `ruff==0.14.14` - Fast linter
- `ipython==9.9.0` - Enhanced REPL with autoreload

### NLTK Data
The following NLTK datasets are already downloaded:
- punkt (tokenizer)
- punkt_tab
- averaged_perceptron_tagger (POS tagger)
- averaged_perceptron_tagger_eng
- wordnet (lemmatization)
- brown (reference corpus)
- stopwords (common words)
- movie_reviews (sample corpus for classification)

## Future Dependencies

When you reach these phases, uncomment these in `requirements.txt`:

```bash
# Phase 7: Dependency parsing
pip install spacy>=3.7.0
python -m spacy download en_core_web_sm

# Phase 8: Graph construction
pip install networkx>=3.2

# Phase 2-6: Data manipulation
pip install pandas>=2.1.0
```

## Troubleshooting

### Old venv backup
The previous virtual environment (with broken Python 3.13 path) was backed up to `venv.old/` and can be safely deleted:
```bash
rm -rf venv.old
```

### Recreate environment
If you need to recreate the environment from scratch:
```bash
rm -rf venv
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
