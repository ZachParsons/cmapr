"""
Tokenization module.

Provides functions for word and sentence tokenization using NLTK.
"""

from typing import List

from nltk import sent_tokenize, word_tokenize


def tokenize_words(text: str) -> List[str]:
    """
    Tokenize text into words.

    Uses NLTK's word_tokenize which handles punctuation,
    contractions, and special characters appropriately.

    Args:
        text: Input text string

    Returns:
        List of word tokens

    Example:
        >>> tokenize_words("The cat sat. It's here!")
        ['The', 'cat', 'sat', '.', 'It', "'s", 'here', '!']
    """
    return word_tokenize(text)


def tokenize_sentences(text: str) -> List[str]:
    """
    Tokenize text into sentences.

    Uses NLTK's sent_tokenize which handles abbreviations,
    decimal points, and other sentence boundary cases.

    Args:
        text: Input text string

    Returns:
        List of sentence strings

    Example:
        >>> tokenize_sentences("The cat sat. It's here!")
        ['The cat sat.', "It's here!"]
    """
    return sent_tokenize(text)


def tokenize_words_preserve_case(text: str) -> tuple[List[str], List[str]]:
    """
    Tokenize words while preserving original case.

    Returns both original tokens and lowercased versions for
    case-insensitive analysis while maintaining original text.

    Args:
        text: Input text string

    Returns:
        Tuple of (original_tokens, lowercased_tokens)

    Example:
        >>> orig, lower = tokenize_words_preserve_case("The Cat")
        >>> orig
        ['The', 'Cat']
        >>> lower
        ['the', 'cat']
    """
    tokens = tokenize_words(text)
    lowercased = [token.lower() for token in tokens]
    return tokens, lowercased
