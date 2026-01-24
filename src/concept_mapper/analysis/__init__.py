"""
Analysis module for frequency and statistical analysis.

Provides functions for:
- Frequency distributions (word/lemma counts)
- Corpus-level aggregation
- Reference corpus loading (Brown corpus)
- TF-IDF calculations
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
]
