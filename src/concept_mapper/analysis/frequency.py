"""
Frequency distribution analysis.

Provides functions for counting word/lemma frequencies in documents
and filtering by part-of-speech tags.
"""

from collections import Counter
from typing import List, Set

from ..corpus.models import ProcessedDocument


def word_frequencies(
    doc: ProcessedDocument, use_lemmas: bool = False, lowercase: bool = True
) -> Counter:
    """
    Count word or lemma frequencies in a document.

    Args:
        doc: ProcessedDocument with tokens/lemmas
        use_lemmas: If True, count lemmas instead of surface forms
        lowercase: If True, normalize to lowercase (only affects tokens, not lemmas)

    Returns:
        Counter mapping words/lemmas to their frequencies

    Example:
        >>> doc = ProcessedDocument(
        ...     raw_text="The cat sat. The cat ran.",
        ...     tokens=["The", "cat", "sat", ".", "The", "cat", "ran", "."],
        ...     lemmas=["the", "cat", "sit", ".", "the", "cat", "run", "."]
        ... )
        >>> word_frequencies(doc)
        Counter({'the': 2, 'cat': 2, 'sat': 1, 'ran': 1, '.': 2})
    """
    if use_lemmas:
        # Lemmas are already lowercased by WordNet lemmatizer
        return Counter(doc.lemmas)
    else:
        tokens = doc.tokens
        if lowercase:
            tokens = [t.lower() for t in tokens]
        return Counter(tokens)


def pos_filtered_frequencies(
    doc: ProcessedDocument,
    pos_tags: Set[str],
    use_lemmas: bool = False,
    lowercase: bool = True,
) -> Counter:
    """
    Count frequencies of words with specific POS tags.

    Args:
        doc: ProcessedDocument with POS tags
        pos_tags: Set of POS tags to include (e.g., {'NN', 'NNS', 'VB'})
        use_lemmas: If True, count lemmas instead of surface forms
        lowercase: If True, normalize to lowercase

    Returns:
        Counter of words/lemmas matching the POS tags

    Example:
        >>> doc = ProcessedDocument(
        ...     raw_text="The cat sat.",
        ...     tokens=["The", "cat", "sat", "."],
        ...     pos_tags=[("The", "DT"), ("cat", "NN"), ("sat", "VBD"), (".", ".")],
        ...     lemmas=["the", "cat", "sit", "."]
        ... )
        >>> pos_filtered_frequencies(doc, {'NN', 'NNS'})
        Counter({'cat': 1})
    """
    filtered_items = []

    for i, (word, pos) in enumerate(doc.pos_tags):
        if pos in pos_tags:
            if use_lemmas:
                # Use corresponding lemma
                filtered_items.append(doc.lemmas[i])
            else:
                item = word.lower() if lowercase else word
                filtered_items.append(item)

    return Counter(filtered_items)


def corpus_frequencies(
    docs: List[ProcessedDocument], use_lemmas: bool = False, lowercase: bool = True
) -> Counter:
    """
    Aggregate word frequencies across multiple documents.

    Args:
        docs: List of ProcessedDocument objects
        use_lemmas: If True, count lemmas instead of surface forms
        lowercase: If True, normalize to lowercase

    Returns:
        Counter with frequencies across entire corpus

    Example:
        >>> docs = [
        ...     ProcessedDocument(raw_text="The cat.", tokens=["The", "cat"], lemmas=["the", "cat"]),
        ...     ProcessedDocument(raw_text="The dog.", tokens=["The", "dog"], lemmas=["the", "dog"])
        ... ]
        >>> corpus_frequencies(docs)
        Counter({'the': 2, 'cat': 1, 'dog': 1})
    """
    total_freq = Counter()
    for doc in docs:
        doc_freq = word_frequencies(doc, use_lemmas=use_lemmas, lowercase=lowercase)
        total_freq.update(doc_freq)
    return total_freq


def document_frequencies(
    docs: List[ProcessedDocument], use_lemmas: bool = False, lowercase: bool = True
) -> Counter:
    """
    Count in how many documents each term appears.

    This is different from term frequency - it counts documents containing
    the term, not total occurrences. Used for IDF calculation.

    Args:
        docs: List of ProcessedDocument objects
        use_lemmas: If True, count lemmas instead of surface forms
        lowercase: If True, normalize to lowercase

    Returns:
        Counter mapping terms to number of documents containing them

    Example:
        >>> docs = [
        ...     ProcessedDocument(raw_text="cat cat", tokens=["cat", "cat"], lemmas=["cat", "cat"]),
        ...     ProcessedDocument(raw_text="cat dog", tokens=["cat", "dog"], lemmas=["cat", "dog"]),
        ...     ProcessedDocument(raw_text="dog bird", tokens=["dog", "bird"], lemmas=["dog", "bird"])
        ... ]
        >>> document_frequencies(docs)
        Counter({'cat': 2, 'dog': 2, 'bird': 1})
    """
    doc_freq = Counter()

    for doc in docs:
        # Get unique terms in this document
        if use_lemmas:
            unique_terms = set(doc.lemmas)
        else:
            terms = doc.tokens
            if lowercase:
                terms = [t.lower() for t in terms]
            unique_terms = set(terms)

        # Increment document count for each unique term
        for term in unique_terms:
            doc_freq[term] += 1

    return doc_freq


def get_vocabulary(
    docs: List[ProcessedDocument], use_lemmas: bool = False, lowercase: bool = True
) -> Set[str]:
    """
    Get all unique terms across corpus.

    Args:
        docs: List of ProcessedDocument objects
        use_lemmas: If True, use lemmas instead of surface forms
        lowercase: If True, normalize to lowercase

    Returns:
        Set of all unique terms in corpus

    Example:
        >>> docs = [
        ...     ProcessedDocument(raw_text="The cat", tokens=["The", "cat"], lemmas=["the", "cat"]),
        ...     ProcessedDocument(raw_text="The dog", tokens=["The", "dog"], lemmas=["the", "dog"])
        ... ]
        >>> sorted(get_vocabulary(docs))
        ['cat', 'dog', 'the']
    """
    vocab = set()
    for doc in docs:
        if use_lemmas:
            vocab.update(doc.lemmas)
        else:
            terms = doc.tokens
            if lowercase:
                terms = [t.lower() for t in terms]
            vocab.update(terms)
    return vocab
