"""
Basic search functionality for finding sentences containing terms.

Provides structured search results with document metadata and positions.
"""

from dataclasses import dataclass
from typing import List, Optional
from ..corpus.models import ProcessedDocument, SentenceLocation


@dataclass
class SentenceMatch:
    """
    A sentence containing a search term.

    Attributes:
        sentence: The matching sentence text
        doc_id: Document identifier (from metadata or index)
        sent_index: Sentence index within document (0-based)
        term_positions: List of character positions where term appears in sentence
        term: The search term that matched
        location: Optional structural location in document (chapter, section, etc.)
    """

    sentence: str
    doc_id: str
    sent_index: int
    term_positions: List[int]
    term: str
    location: Optional[SentenceLocation] = None

    def __str__(self) -> str:
        """String representation showing document and sentence."""
        return f"[{self.doc_id}:{self.sent_index}] {self.sentence}"


def find_sentences(
    term: str,
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
    match_lemma: bool = False,
) -> List[SentenceMatch]:
    """
    Find all sentences containing a term.

    Args:
        term: Term to search for
        docs: List of preprocessed documents
        case_sensitive: Whether search is case-sensitive (default: False)
        match_lemma: Whether to match lemmatized forms (default: False).
                    When True, searches for lemma matches (e.g., "run" matches
                    "running", "ran", "runs")

    Returns:
        List of SentenceMatch objects, in document order

    Example:
        >>> # Exact word search
        >>> matches = find_sentences("intentionality", docs)
        >>> for match in matches:
        ...     print(f"{match.doc_id}: {match.sentence}")

        >>> # Lemma-based search (matches all forms)
        >>> matches = find_sentences("run", docs, match_lemma=True)
        >>> # Will match "run", "running", "ran", "runs", etc.
    """
    matches = []

    # If doing lemma search, get the lemma of the search term
    search_lemma = None
    if match_lemma:
        from ..preprocessing.tokenize import tokenize_words
        from ..preprocessing.tagging import tag_tokens
        from ..preprocessing.lemmatize import lemmatize_tagged

        # Get lemma of search term
        tokens = tokenize_words(term)
        if tokens:
            tagged = tag_tokens(tokens)
            lemmas = lemmatize_tagged(tagged)
            search_lemma = lemmas[0] if lemmas else term.lower()
        else:
            search_lemma = term.lower()

    # Normalize search term for exact matching
    search_term = term if case_sensitive else term.lower()

    for doc_idx, doc in enumerate(docs):
        # Get document ID from metadata or use index
        doc_id = doc.metadata.get("source_path", f"doc_{doc_idx}")

        for sent_idx, sentence in enumerate(doc.sentences):
            found = False
            positions = []

            if match_lemma:
                # Lemma-based search: tokenize sentence and check lemmas
                from ..preprocessing.tokenize import tokenize_words
                from ..preprocessing.tagging import tag_tokens
                from ..preprocessing.lemmatize import lemmatize_tagged

                sent_tokens = tokenize_words(sentence)
                if sent_tokens:
                    sent_tagged = tag_tokens(sent_tokens)
                    sent_lemmas = lemmatize_tagged(sent_tagged)

                    # Check if any lemma matches
                    if search_lemma in sent_lemmas:
                        found = True
                        # Find character positions of matching tokens
                        # This is approximate since we're matching lemmas
                        sentence_lower = sentence.lower()
                        for i, lemma in enumerate(sent_lemmas):
                            if lemma == search_lemma and i < len(sent_tokens):
                                # Try to find this token in the sentence
                                token = sent_tokens[i]
                                pos = sentence_lower.find(token.lower())
                                if pos != -1:
                                    positions.append(pos)
            else:
                # Exact text search
                sentence_compare = sentence if case_sensitive else sentence.lower()

                if search_term in sentence_compare:
                    found = True
                    # Find all positions of the term in the sentence
                    start = 0
                    while True:
                        pos = sentence_compare.find(search_term, start)
                        if pos == -1:
                            break
                        positions.append(pos)
                        start = pos + 1

            if found:
                # Get location info if available
                location = None
                if sent_idx < len(doc.sentence_locations):
                    location = doc.sentence_locations[sent_idx]

                # Create match object
                match = SentenceMatch(
                    sentence=sentence.strip(),
                    doc_id=doc_id,
                    sent_index=sent_idx,
                    term_positions=positions if positions else [0],
                    term=term,
                    location=location,
                )
                matches.append(match)

    return matches


def find_sentences_any(
    terms: List[str],
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
) -> List[SentenceMatch]:
    """
    Find sentences containing any of the given terms.

    Args:
        terms: List of terms to search for
        docs: List of preprocessed documents
        case_sensitive: Whether search is case-sensitive (default: False)

    Returns:
        List of SentenceMatch objects (may contain duplicates if multiple terms match)
    """
    all_matches = []

    for term in terms:
        matches = find_sentences(term, docs, case_sensitive=case_sensitive)
        all_matches.extend(matches)

    return all_matches


def find_sentences_all(
    terms: List[str],
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
) -> List[SentenceMatch]:
    """
    Find sentences containing all of the given terms.

    Args:
        terms: List of terms that must all appear
        docs: List of preprocessed documents
        case_sensitive: Whether search is case-sensitive (default: False)

    Returns:
        List of SentenceMatch objects
    """
    if not terms:
        return []

    # Start with matches for first term
    candidates = find_sentences(terms[0], docs, case_sensitive=case_sensitive)

    # Filter to only sentences containing all other terms
    matches = []
    for match in candidates:
        sentence_compare = match.sentence if case_sensitive else match.sentence.lower()

        # Check if all other terms appear
        if all(
            (t if case_sensitive else t.lower()) in sentence_compare for t in terms[1:]
        ):
            matches.append(match)

    return matches


def count_term_occurrences(
    term: str,
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
) -> int:
    """
    Count total occurrences of a term across all documents.

    Args:
        term: Term to count
        docs: List of preprocessed documents
        case_sensitive: Whether to match case (default: False)

    Returns:
        Total count of term occurrences
    """
    matches = find_sentences(term, docs, case_sensitive=case_sensitive)

    # Sum up positions (each position is one occurrence)
    total = sum(len(match.term_positions) for match in matches)

    return total


def find_in_document(
    term: str,
    doc: ProcessedDocument,
    case_sensitive: bool = False,
) -> List[SentenceMatch]:
    """
    Find all occurrences of term in a single document.

    Args:
        term: Term to search for
        doc: Single preprocessed document
        case_sensitive: Whether search is case-sensitive (default: False)

    Returns:
        List of SentenceMatch objects from this document
    """
    return find_sentences(term, [doc], case_sensitive=case_sensitive)
