"""Tokenization utilities for text preprocessing."""

from typing import List
from nltk import word_tokenize, sent_tokenize


def tokenize_words(text: str) -> List[str]:
    """Tokenize text into words."""
    return word_tokenize(text)


def tokenize_sentences(text: str) -> List[str]:
    """Tokenize text into sentences."""
    return sent_tokenize(text)
