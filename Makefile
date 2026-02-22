.PHONY: install format lint test coverage check shell help

help:
	@echo "Available commands:"
	@echo "  make install   - Install dependencies with uv"
	@echo "  make format    - Format code with Ruff"
	@echo "  make lint      - Lint code with Ruff (auto-fix)"
	@echo "  make test      - Run tests with pytest"
	@echo "  make coverage  - Run tests with coverage report"
	@echo "  make check     - Run all checks (format + lint + test)"
	@echo "  make shell     - Start IPython interactive shell"

install:
	@echo "📦 Installing dependencies with uv..."
	@uv pip install -e ".[dev]"

format:
	@echo "🔧 Formatting code with Ruff..."
	@uv run ruff format src/ tests/ scripts/

lint:
	@echo "🔍 Linting code with Ruff..."
	@uv run ruff check src/ tests/ scripts/ --fix

test:
	@echo "🧪 Running tests..."
	@uv run pytest tests/ -v

coverage:
	@echo "📊 Running tests with coverage..."
	@uv run pytest --cov=concept_mapper --cov-report=term-missing --cov-report=html tests/
	@echo "✅ Coverage report generated: htmlcov/index.html"

check: format lint test
	@echo "✅ All checks passed! Ready to commit."

shell:
	@uv run ipython -i -c "\
from src.concept_mapper.corpus import load_file, load_directory, Document, Corpus; \
from src.concept_mapper.preprocessing import preprocess, preprocess_corpus, filter_by_pos; \
from src.concept_mapper.analysis import word_frequencies, pos_filtered_frequencies, corpus_frequencies, load_reference_corpus, corpus_tfidf_scores, tfidf; \
from src.concept_mapper.analysis.rarity import compare_to_reference, get_corpus_specific_terms, get_top_corpus_specific_terms, get_neologism_candidates, get_term_context_stats, tfidf_vs_reference, get_top_tfidf_terms, get_distinctive_by_tfidf, get_combined_distinctive_terms, get_wordnet_neologisms, get_all_neologism_signals, get_definitional_contexts, score_by_definitional_context, get_definitional_sentences, get_highly_defined_terms, filter_by_pos_tags, get_philosophical_term_candidates, get_compound_terms, get_filtered_candidates, PhilosophicalTermScorer, score_philosophical_terms; \
from collections import Counter; \
print('📦 Auto-imported:'); \
print('  Corpus: load_file, load_directory, Document, Corpus'); \
print('  Preprocessing: preprocess, preprocess_corpus, filter_by_pos'); \
print('  Analysis: word_frequencies, pos_filtered_frequencies, corpus_frequencies'); \
print('  Analysis: load_reference_corpus, corpus_tfidf_scores, tfidf'); \
print('  Rarity (Ratio): compare_to_reference, get_corpus_specific_terms, get_top_corpus_specific_terms'); \
print('  Rarity (Neologisms): get_neologism_candidates, get_wordnet_neologisms, get_all_neologism_signals'); \
print('  Rarity (TF-IDF): tfidf_vs_reference, get_top_tfidf_terms, get_distinctive_by_tfidf'); \
print('  Rarity (Combined): get_combined_distinctive_terms, get_term_context_stats'); \
print('  Definitional: get_definitional_contexts, score_by_definitional_context'); \
print('  Definitional: get_definitional_sentences, get_highly_defined_terms'); \
print('  POS Filtering: filter_by_pos_tags, get_philosophical_term_candidates'); \
print('  POS Filtering: get_compound_terms, get_filtered_candidates'); \
print('  Hybrid Scorer: PhilosophicalTermScorer, score_philosophical_terms'); \
print('  Utils: Counter'); \
"
