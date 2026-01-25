"""
Analysis module for frequency and statistical analysis.

Provides functions for:
- Frequency distributions (word/lemma counts)
- Corpus-level aggregation
- Reference corpus loading (Brown corpus)
- TF-IDF calculations
- Rarity detection and corpus-comparative analysis
- Co-occurrence analysis and relationship discovery
"""

from .frequency import (
    corpus_frequencies,
    document_frequencies,
    get_vocabulary,
    pos_filtered_frequencies,
    word_frequencies,
)
from .reference import (
    get_reference_size,
    get_reference_vocabulary,
    load_reference_corpus,
)
from .tfidf import (
    corpus_tfidf_scores,
    document_tfidf_scores,
    idf,
    tf,
    tfidf,
)
from .rarity import (
    compare_to_reference,
    get_combined_distinctive_terms,
    get_corpus_specific_terms,
    get_distinctive_by_tfidf,
    get_neologism_candidates,
    get_term_context_stats,
    get_top_corpus_specific_terms,
    get_top_tfidf_terms,
    tfidf_vs_reference,
    get_wordnet_neologisms,
    get_capitalized_technical_terms,
    get_all_neologism_signals,
    get_definitional_contexts,
    score_by_definitional_context,
    get_definitional_sentences,
    get_highly_defined_terms,
    filter_by_pos_tags,
    get_philosophical_term_candidates,
    get_compound_terms,
    get_filtered_candidates,
    PhilosophicalTermScorer,
    score_philosophical_terms,
)
from .cooccurrence import (
    cooccurs_in_sentence,
    cooccurs_filtered,
    cooccurs_in_paragraph,
    cooccurs_within_n,
    pmi,
    log_likelihood_ratio,
    build_cooccurrence_matrix,
    save_cooccurrence_matrix,
    get_top_cooccurrences,
)

__all__ = [
    # Frequency
    "word_frequencies",
    "pos_filtered_frequencies",
    "corpus_frequencies",
    "document_frequencies",
    "get_vocabulary",
    # Reference corpus
    "load_reference_corpus",
    "get_reference_vocabulary",
    "get_reference_size",
    # TF-IDF
    "tf",
    "idf",
    "tfidf",
    "corpus_tfidf_scores",
    "document_tfidf_scores",
    # Rarity detection - Corpus comparison
    "compare_to_reference",
    "get_corpus_specific_terms",
    "get_top_corpus_specific_terms",
    "get_term_context_stats",
    "tfidf_vs_reference",
    "get_top_tfidf_terms",
    "get_distinctive_by_tfidf",
    "get_combined_distinctive_terms",
    # Rarity detection - Neologisms
    "get_neologism_candidates",
    "get_wordnet_neologisms",
    "get_capitalized_technical_terms",
    "get_all_neologism_signals",
    # Definitional contexts
    "get_definitional_contexts",
    "score_by_definitional_context",
    "get_definitional_sentences",
    "get_highly_defined_terms",
    # POS filtering
    "filter_by_pos_tags",
    "get_philosophical_term_candidates",
    "get_compound_terms",
    "get_filtered_candidates",
    # Hybrid scorer (Phase 3.6)
    "PhilosophicalTermScorer",
    "score_philosophical_terms",
    # Co-occurrence analysis (Phase 6)
    "cooccurs_in_sentence",
    "cooccurs_filtered",
    "cooccurs_in_paragraph",
    "cooccurs_within_n",
    "pmi",
    "log_likelihood_ratio",
    "build_cooccurrence_matrix",
    "save_cooccurrence_matrix",
    "get_top_cooccurrences",
]
