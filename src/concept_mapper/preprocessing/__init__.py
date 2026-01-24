"""
Text preprocessing module.

Provides tokenization, POS tagging, and lemmatization functions
for linguistic analysis.
"""

from .lemmatize import (
    get_wordnet_pos,
    lemmatize,
    lemmatize_tagged,
    lemmatize_words,
)
from .pipeline import preprocess, preprocess_corpus
from .tagging import filter_by_pos, tag_sentences, tag_tokens
from .tokenize import (
    tokenize_sentences,
    tokenize_words,
    tokenize_words_preserve_case,
)

__all__ = [
    # Tokenization
    "tokenize_words",
    "tokenize_sentences",
    "tokenize_words_preserve_case",
    # POS tagging
    "tag_tokens",
    "tag_sentences",
    "filter_by_pos",
    # Lemmatization
    "lemmatize",
    "lemmatize_words",
    "lemmatize_tagged",
    "get_wordnet_pos",
    # Pipeline
    "preprocess",
    "preprocess_corpus",
]
