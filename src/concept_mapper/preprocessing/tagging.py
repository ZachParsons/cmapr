"""
Part-of-speech tagging module.

Provides functions for POS tagging using NLTK's averaged perceptron tagger.
"""

from typing import List

from nltk import pos_tag


def tag_tokens(tokens: List[str]) -> List[tuple[str, str]]:
    """
    Tag tokens with part-of-speech labels.

    Uses NLTK's pos_tag with the Penn Treebank tagset.

    Args:
        tokens: List of word tokens

    Returns:
        List of (word, POS_tag) tuples

    Example:
        >>> tag_tokens(['The', 'cat', 'sat'])
        [('The', 'DT'), ('cat', 'NN'), ('sat', 'VBD')]
    """
    return pos_tag(tokens)


def tag_sentences(sentences: List[str]) -> List[List[tuple[str, str]]]:
    """
    Tag each sentence with POS labels.

    Tokenizes and tags each sentence separately to maintain
    sentence boundaries in the output.

    Args:
        sentences: List of sentence strings

    Returns:
        List of tagged sentence (each sentence is a list of (word, POS) tuples)

    Example:
        >>> tag_sentences(['The cat sat.', 'It ran.'])
        [[('The', 'DT'), ('cat', 'NN'), ('sat', 'VBD'), ('.', '.')],
         [('It', 'PRP'), ('ran', 'VBD'), ('.', '.')]]
    """
    from .tokenize import tokenize_words

    return [tag_tokens(tokenize_words(sent)) for sent in sentences]


def filter_by_pos(
    tagged_tokens: List[tuple[str, str]], pos_tags: set[str]
) -> List[str]:
    """
    Filter tokens by POS tag.

    Args:
        tagged_tokens: List of (word, POS_tag) tuples
        pos_tags: Set of POS tags to keep (e.g., {'NN', 'NNS', 'VB'})

    Returns:
        List of words matching the specified POS tags

    Example:
        >>> tagged = [('The', 'DT'), ('cat', 'NN'), ('sat', 'VBD')]
        >>> filter_by_pos(tagged, {'NN', 'NNS'})
        ['cat']
    """
    return [word for word, pos in tagged_tokens if pos in pos_tags]
