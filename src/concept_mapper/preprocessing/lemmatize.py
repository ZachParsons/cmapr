"""
Lemmatization module.

Provides functions for lemmatizing words using NLTK's WordNet lemmatizer,
with inflect as a fallback for specialized terms not in WordNet.
"""

from typing import List

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
import inflect

# Initialize lemmatizer (singleton)
_lemmatizer = WordNetLemmatizer()
_inflect_engine = inflect.engine()


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
    Lemmatize a single word using WordNet, with inflect fallback for plurals.

    WordNet may not recognize specialized philosophical/technical terms.
    For plural nouns that WordNet doesn't lemmatize, use inflect to
    normalize to singular form (e.g., "semiotics" -> "semiotic").

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
        >>> lemmatize('semiotics', wordnet.NOUN)
        'semiotic'
    """
    word_lower = word.lower()
    lemma = _lemmatizer.lemmatize(word_lower, pos=pos)

    # If WordNet didn't change the word and it's a noun, try inflect
    # to handle specialized terms (e.g., "semiotics" -> "semiotic")
    if lemma == word_lower and pos == wordnet.NOUN:
        singular = _inflect_engine.singular_noun(word_lower)
        # singular_noun returns False if already singular, or the singular form
        if singular:
            lemma = singular

    return lemma


def lemmatize_tagged(tagged_tokens: List[tuple[str, str]]) -> List[str]:
    """
    Lemmatize tokens using their POS tags, with inflect fallback for plurals.

    For specialized terms not in WordNet (e.g., "semiotics", "isotopies"),
    uses inflect to normalize plural nouns to singular forms. Only applies
    inflect fallback to terms explicitly tagged as plural (NNS, NNPS) to
    avoid incorrectly "singularizing" already-singular terms.

    Args:
        tagged_tokens: List of (word, POS_tag) tuples from POS tagger

    Returns:
        List of lemmatized words

    Example:
        >>> tagged = [('The', 'DT'), ('cats', 'NNS'), ('were', 'VBD'), ('running', 'VBG')]
        >>> lemmatize_tagged(tagged)
        ['the', 'cat', 'be', 'run']
        >>> tagged = [('semiotics', 'NNS'), ('isotopies', 'NNS')]
        >>> lemmatize_tagged(tagged)
        ['semiotic', 'isotopy']
    """
    lemmas = []
    for word, pos_tag in tagged_tokens:
        wordnet_pos = get_wordnet_pos(pos_tag)
        word_lower = word.lower()
        lemma = _lemmatizer.lemmatize(word_lower, pos=wordnet_pos)

        # If WordNet didn't change the word and it's explicitly tagged as plural,
        # try inflect to handle specialized terms not in WordNet
        # Only apply to NNS (plural noun) and NNPS (proper plural noun) to avoid
        # incorrectly singularizing words like "semiosis", "process", "Paris"
        if lemma == word_lower and pos_tag in ('NNS', 'NNPS'):
            singular = _inflect_engine.singular_noun(word_lower)
            if singular:
                lemma = singular

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
