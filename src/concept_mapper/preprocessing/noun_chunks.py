"""
spaCy noun-chunk extraction.

Optional preprocessing step that adds multi-word phrase candidates to a
``ProcessedDocument``'s metadata. Loaded lazily so the spaCy dependency
is only required when the user opts in (``cmapr ingest --spacy``).

The model is cached at module level so repeated calls within one process
reuse the same loaded ``en_core_web_sm`` instance.
"""

from typing import List

_SPACY_NLP = None

_LEADING_DETS = frozenset(
    {
        "the",
        "a",
        "an",
        "this",
        "that",
        "these",
        "those",
        "its",
        "their",
        "our",
        "your",
        "my",
        "his",
        "her",
    }
)


def _get_spacy_nlp():
    """Load and cache the spaCy en_core_web_sm model."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy  # noqa: PLC0415

        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _SPACY_NLP


def extract_noun_chunks(text: str) -> List[str]:
    """
    Extract multi-word noun phrases from *text* using spaCy.

    Returns a deduplicated list of lowercased multi-word phrases with
    leading determiners stripped (e.g. 'the sign vehicle' → 'sign vehicle').
    Single-token chunks and chunks containing 1-char tokens (likely OCR
    artifacts) are dropped.
    """
    nlp = _get_spacy_nlp()
    max_len = nlp.max_length - 100
    seen: set = set()
    for start in range(0, len(text), max_len):
        doc = nlp(text[start : start + max_len])
        for chunk in doc.noun_chunks:
            words = chunk.text.lower().split()
            # Strip leading determiners/articles
            while words and words[0] in _LEADING_DETS:
                words = words[1:]
            if len(words) < 2:
                continue
            # Reject if any token is a single character (OCR artifact)
            if any(len(w) < 2 for w in words):
                continue
            seen.add(" ".join(words))
    return list(seen)
