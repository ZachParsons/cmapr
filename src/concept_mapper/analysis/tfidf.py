"""
TF-IDF (Term Frequency - Inverse Document Frequency) analysis.

Provides functions for calculating TF-IDF scores to identify
terms that are characteristic of specific documents or corpora.
"""

import math
from collections import Counter
from typing import Dict, List

from ..corpus.models import ProcessedDocument
from .frequency import corpus_frequencies, document_frequencies, word_frequencies


def tf(term: str, doc: ProcessedDocument, use_lemmas: bool = False) -> float:
    """
    Calculate term frequency (TF) for a term in a document.

    TF = (number of times term appears in doc) / (total terms in doc)

    Args:
        term: Term to calculate TF for
        doc: ProcessedDocument
        use_lemmas: If True, count lemmas instead of tokens

    Returns:
        Term frequency (0.0 to 1.0)

    Example:
        >>> doc = ProcessedDocument(
        ...     raw_text="cat cat dog",
        ...     tokens=["cat", "cat", "dog"],
        ...     lemmas=["cat", "cat", "dog"]
        ... )
        >>> tf("cat", doc)
        0.6666666666666666
    """
    freq = word_frequencies(doc, use_lemmas=use_lemmas, lowercase=True)
    total = sum(freq.values())

    if total == 0:
        return 0.0

    return freq[term.lower()] / total


def idf(term: str, docs: List[ProcessedDocument], use_lemmas: bool = False) -> float:
    """
    Calculate inverse document frequency (IDF) for a term.

    IDF = log(total documents / documents containing term)

    Higher IDF means term is more distinctive (appears in fewer documents).

    Args:
        term: Term to calculate IDF for
        docs: List of ProcessedDocument objects
        use_lemmas: If True, use lemmas instead of tokens

    Returns:
        Inverse document frequency

    Example:
        >>> docs = [
        ...     ProcessedDocument(raw_text="cat dog", tokens=["cat", "dog"], lemmas=["cat", "dog"]),
        ...     ProcessedDocument(raw_text="cat bird", tokens=["cat", "bird"], lemmas=["cat", "bird"]),
        ...     ProcessedDocument(raw_text="dog bird", tokens=["dog", "bird"], lemmas=["dog", "bird"])
        ... ]
        >>> idf("cat", docs)  # cat appears in 2/3 docs
        0.4054651081081644
        >>> idf("rare", docs)  # rare appears in 0 docs
        0.0
    """
    doc_freq = document_frequencies(docs, use_lemmas=use_lemmas, lowercase=True)
    n_docs = len(docs)
    n_docs_with_term = doc_freq[term.lower()]

    if n_docs_with_term == 0:
        return 0.0

    return math.log(n_docs / n_docs_with_term)


def tfidf(
    term: str,
    doc: ProcessedDocument,
    docs: List[ProcessedDocument],
    use_lemmas: bool = False,
) -> float:
    """
    Calculate TF-IDF score for a term in a document.

    TF-IDF = TF * IDF

    High TF-IDF means term is frequent in this document but rare across corpus.

    Args:
        term: Term to calculate TF-IDF for
        doc: Document containing the term
        docs: Full corpus (including doc)
        use_lemmas: If True, use lemmas instead of tokens

    Returns:
        TF-IDF score

    Example:
        >>> docs = [
        ...     ProcessedDocument(raw_text="cat cat dog", tokens=["cat", "cat", "dog"], lemmas=["cat", "cat", "dog"]),
        ...     ProcessedDocument(raw_text="dog bird", tokens=["dog", "bird"], lemmas=["dog", "bird"])
        ... ]
        >>> tfidf("cat", docs[0], docs)  # cat is frequent in doc 0, rare in corpus
        0.4621171572600098
    """
    tf_score = tf(term, doc, use_lemmas=use_lemmas)
    idf_score = idf(term, docs, use_lemmas=use_lemmas)
    return tf_score * idf_score


def corpus_tfidf_scores(
    docs: List[ProcessedDocument], use_lemmas: bool = False, min_score: float = 0.0
) -> Dict[str, float]:
    """
    Calculate TF-IDF scores for all terms across corpus.

    For each term, uses the maximum TF-IDF score across all documents.

    Args:
        docs: List of ProcessedDocument objects
        use_lemmas: If True, use lemmas instead of tokens
        min_score: Only return terms with score >= min_score

    Returns:
        Dictionary mapping terms to their max TF-IDF scores

    Example:
        >>> docs = [
        ...     ProcessedDocument(raw_text="cat cat", tokens=["cat", "cat"], lemmas=["cat", "cat"]),
        ...     ProcessedDocument(raw_text="dog", tokens=["dog"], lemmas=["dog"])
        ... ]
        >>> scores = corpus_tfidf_scores(docs)
        >>> scores["cat"] > scores["dog"]  # cat appears more in its document
        False
    """
    # Get all unique terms
    freq = corpus_frequencies(docs, use_lemmas=use_lemmas, lowercase=True)
    terms = freq.keys()

    # Calculate max TF-IDF for each term
    scores = {}
    for term in terms:
        max_score = 0.0
        for doc in docs:
            score = tfidf(term, doc, docs, use_lemmas=use_lemmas)
            max_score = max(max_score, score)

        if max_score >= min_score:
            scores[term] = max_score

    return scores


def document_tfidf_scores(
    doc: ProcessedDocument, docs: List[ProcessedDocument], use_lemmas: bool = False
) -> Dict[str, float]:
    """
    Calculate TF-IDF scores for all terms in a specific document.

    Args:
        doc: Document to analyze
        docs: Full corpus (including doc)
        use_lemmas: If True, use lemmas instead of tokens

    Returns:
        Dictionary mapping terms to their TF-IDF scores in this document

    Example:
        >>> docs = [
        ...     ProcessedDocument(raw_text="cat cat dog", tokens=["cat", "cat", "dog"], lemmas=["cat", "cat", "dog"]),
        ...     ProcessedDocument(raw_text="bird", tokens=["bird"], lemmas=["bird"])
        ... ]
        >>> scores = document_tfidf_scores(docs[0], docs)
        >>> "cat" in scores and "dog" in scores
        True
    """
    freq = word_frequencies(doc, use_lemmas=use_lemmas, lowercase=True)
    terms = freq.keys()

    scores = {}
    for term in terms:
        scores[term] = tfidf(term, doc, docs, use_lemmas=use_lemmas)

    return scores
