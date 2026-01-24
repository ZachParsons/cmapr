"""
Lemmatization module.

Provides functions for lemmatizing words using NLTK's WordNet lemmatizer.
"""

from typing import List

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer (singleton)
_lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag: str) -> str:
    """
    Map Penn Treebank POS tag to WordNet POS tag.

    WordNet lemmatizer requires POS tags in its own format:
    - 'n' for noun
    - 'v' for verb
    - 'a' for adjective
    - 'r' for adverb

    Args:
        treebank_tag: Penn Treebank POS tag (e.g., 'NN', 'VBD', 'JJ')

    Returns:
        WordNet POS tag ('n', 'v', 'a', 'r', or 'n' as default)

    Example:
        >>> get_wordnet_pos('VBD')
        'v'
        >>> get_wordnet_pos('NN')
        'n'
        >>> get_wordnet_pos('JJ')
        'a'
    """
    if treebank_tag.startswith("J"):
        return wordnet.ADJ
    elif treebank_tag.startswith("V"):
        return wordnet.VERB
    elif treebank_tag.startswith("N"):
        return wordnet.NOUN
    elif treebank_tag.startswith("R"):
        return wordnet.ADV
    else:
        # Default to noun
        return wordnet.NOUN


def lemmatize(word: str, pos: str = wordnet.NOUN) -> str:
    """
    Lemmatize a single word.

    Args:
        word: Word to lemmatize
        pos: WordNet POS tag ('n', 'v', 'a', 'r')

    Returns:
        Lemmatized word

    Example:
        >>> lemmatize('running', wordnet.VERB)
        'run'
        >>> lemmatize('better', wordnet.ADJ)
        'good'
    """
    return _lemmatizer.lemmatize(word.lower(), pos=pos)


def lemmatize_tagged(tagged_tokens: List[tuple[str, str]]) -> List[str]:
    """
    Lemmatize tokens using their POS tags.

    Args:
        tagged_tokens: List of (word, POS_tag) tuples from POS tagger

    Returns:
        List of lemmatized words

    Example:
        >>> tagged = [('The', 'DT'), ('cats', 'NNS'), ('were', 'VBD'), ('running', 'VBG')]
        >>> lemmatize_tagged(tagged)
        ['the', 'cat', 'be', 'run']
    """
    lemmas = []
    for word, pos_tag in tagged_tokens:
        wordnet_pos = get_wordnet_pos(pos_tag)
        lemma = lemmatize(word, wordnet_pos)
        lemmas.append(lemma)
    return lemmas


def lemmatize_words(words: List[str], default_pos: str = wordnet.NOUN) -> List[str]:
    """
    Lemmatize words without POS tags.

    Uses a default POS tag (typically noun) for all words.
    For better results, use lemmatize_tagged() with POS-tagged input.

    Args:
        words: List of words
        default_pos: Default WordNet POS tag to use

    Returns:
        List of lemmatized words

    Example:
        >>> lemmatize_words(['cats', 'dogs'])
        ['cat', 'dog']
    """
    return [lemmatize(word, default_pos) for word in words]
