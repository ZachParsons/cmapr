.PHONY: format lint test check shell help

# Detect the correct Python/venv path
VENV := venv/bin
PYTHON := $(VENV)/python
BLACK := $(VENV)/black
RUFF := $(VENV)/ruff
PYTEST := $(VENV)/pytest
IPYTHON := $(VENV)/ipython

help:
	@echo "Available commands:"
	@echo "  make format    - Format code with Black"
	@echo "  make lint      - Lint code with Ruff (auto-fix)"
	@echo "  make test      - Run tests with pytest"
	@echo "  make check     - Run all checks (format + lint + test)"
	@echo "  make shell     - Start IPython interactive shell"

format:
	@echo "🔧 Formatting code with Black..."
	@$(BLACK) *.py src/ tests/ scripts/

lint:
	@echo "🔍 Linting code with Ruff..."
	@$(RUFF) check *.py src/ tests/ scripts/ --fix

test:
	@echo "🧪 Running tests..."
	@$(PYTEST) tests/ -v

check: format lint test
	@echo "✅ All checks passed! Ready to commit."

shell:
	@$(IPYTHON) -i -c "\
import pos_tagger; \
from src.concept_mapper.corpus import load_file, load_directory, Document, Corpus; \
from src.concept_mapper.preprocessing import preprocess, preprocess_corpus, filter_by_pos; \
from src.concept_mapper.analysis import word_frequencies, pos_filtered_frequencies, corpus_frequencies, load_reference_corpus, corpus_tfidf_scores, tfidf; \
from collections import Counter; \
print('📦 Auto-imported:'); \
print('  Corpus: load_file, load_directory, Document, Corpus'); \
print('  Preprocessing: preprocess, preprocess_corpus, filter_by_pos'); \
print('  Analysis: word_frequencies, pos_filtered_frequencies, corpus_frequencies'); \
print('  Analysis: load_reference_corpus, corpus_tfidf_scores, tfidf'); \
print('  Utils: Counter, pos_tagger'); \
"
