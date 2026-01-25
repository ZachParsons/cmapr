"""
Context window extraction for viewing terms with surrounding sentences.

Provides N sentences before and after each match for deeper contextual analysis.
"""

from dataclasses import dataclass
from typing import List
from ..corpus.models import ProcessedDocument


@dataclass
class ContextWindow:
    """
    A match with surrounding sentence context.

    Attributes:
        before: List of sentences before the match
        match: The matching sentence
        after: List of sentences after the match
        doc_id: Document identifier
        sent_index: Index of matching sentence in document
    """

    before: List[str]
    match: str
    after: List[str]
    doc_id: str
    sent_index: int

    def __str__(self) -> str:
        """Format as readable context."""
        lines = []
        lines.append(f"[{self.doc_id}:{self.sent_index}]")
        lines.append("")

        # Before context
        for sent in self.before:
            lines.append(f"  {sent}")

        # Match (highlighted)
        lines.append(f"> {self.match}")

        # After context
        for sent in self.after:
            lines.append(f"  {sent}")

        return "\n".join(lines)


def get_context(
    term: str,
    docs: List[ProcessedDocument],
    n_sentences: int = 1,
    case_sensitive: bool = False,
) -> List[ContextWindow]:
    """
    Get context windows showing sentences before/after each match.

    Args:
        term: Term to search for
        docs: List of preprocessed documents
        n_sentences: Number of sentences before and after (default: 1)
        case_sensitive: Whether search is case-sensitive (default: False)

    Returns:
        List of ContextWindow objects

    Example:
        >>> windows = get_context("abstraction", docs, n_sentences=2)
        >>> for window in windows:
        ...     print(window)
        ...     print("---")
    """
    windows = []

    # Normalize search term
    search_term = term if case_sensitive else term.lower()

    for doc_idx, doc in enumerate(docs):
        doc_id = doc.metadata.get("source_path", f"doc_{doc_idx}")
        sentences = doc.sentences

        for sent_idx, sentence in enumerate(sentences):
            # Normalize for comparison
            sentence_compare = sentence if case_sensitive else sentence.lower()

            # Check if term appears
            if search_term in sentence_compare:
                # Get before context
                before_start = max(0, sent_idx - n_sentences)
                before = sentences[before_start:sent_idx]

                # Get after context
                after_end = min(len(sentences), sent_idx + 1 + n_sentences)
                after = sentences[sent_idx + 1 : after_end]

                # Create window
                window = ContextWindow(
                    before=[s.strip() for s in before],
                    match=sentence.strip(),
                    after=[s.strip() for s in after],
                    doc_id=doc_id,
                    sent_index=sent_idx,
                )
                windows.append(window)

    return windows


def get_context_by_match(
    matches: List,  # SentenceMatch objects (avoid circular import)
    docs: List[ProcessedDocument],
    n_sentences: int = 1,
) -> List[ContextWindow]:
    """
    Get context windows for specific matches.

    Useful when you already have search results and want to expand context.

    Args:
        matches: List of SentenceMatch objects from find_sentences()
        docs: List of preprocessed documents (same as used for search)
        n_sentences: Number of sentences before and after

    Returns:
        List of ContextWindow objects
    """
    # Build document lookup for efficient access
    doc_lookup = {}
    for doc_idx, doc in enumerate(docs):
        doc_id = doc.metadata.get("source_path", f"doc_{doc_idx}")
        doc_lookup[doc_id] = doc

    windows = []

    for match in matches:
        # Get the document
        doc = doc_lookup.get(match.doc_id)
        if not doc:
            continue

        sent_idx = match.sent_index
        sentences = doc.sentences

        # Get context
        before_start = max(0, sent_idx - n_sentences)
        before = sentences[before_start:sent_idx]

        after_end = min(len(sentences), sent_idx + 1 + n_sentences)
        after = sentences[sent_idx + 1 : after_end]

        # Create window
        window = ContextWindow(
            before=[s.strip() for s in before],
            match=match.sentence,
            after=[s.strip() for s in after],
            doc_id=match.doc_id,
            sent_index=sent_idx,
        )
        windows.append(window)

    return windows


def format_context_windows(windows: List[ContextWindow], separator: str = "---") -> str:
    """
    Format context windows as readable text.

    Args:
        windows: List of ContextWindow objects
        separator: Separator between windows (default: "---")

    Returns:
        Formatted string
    """
    output = []

    for window in windows:
        output.append(str(window))
        output.append(separator)
        output.append("")

    return "\n".join(output)


def get_context_with_highlights(
    term: str,
    docs: List[ProcessedDocument],
    n_sentences: int = 1,
    highlight_start: str = "**",
    highlight_end: str = "**",
) -> List[ContextWindow]:
    """
    Get context windows with the search term highlighted in match sentence.

    Args:
        term: Term to search for
        docs: List of preprocessed documents
        n_sentences: Number of sentences before/after
        highlight_start: String to insert before term (default: "**")
        highlight_end: String to insert after term (default: "**")

    Returns:
        List of ContextWindow objects with highlighted matches
    """
    windows = get_context(term, docs, n_sentences=n_sentences, case_sensitive=False)

    # Add highlighting to match sentences
    for window in windows:
        # Simple case-insensitive replacement
        term_lower = term.lower()
        match_lower = window.match.lower()

        # Find and replace all occurrences
        highlighted = window.match
        start = 0
        while True:
            pos = match_lower.find(term_lower, start)
            if pos == -1:
                break

            # Get actual term with original case
            actual_term = window.match[pos : pos + len(term)]

            # Replace with highlighted version
            highlighted = (
                highlighted[:pos]
                + highlight_start
                + actual_term
                + highlight_end
                + highlighted[pos + len(term) :]
            )

            # Adjust position for inserted highlight markers
            start = pos + len(highlight_start) + len(term) + len(highlight_end)
            match_lower = highlighted.lower()

        window.match = highlighted

    return windows
