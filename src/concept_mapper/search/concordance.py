"""
Concordance (KWIC - Key Word In Context) display.

Shows search terms aligned with surrounding context for easy scanning.
"""

from dataclasses import dataclass
from typing import List
from ..corpus.models import ProcessedDocument


@dataclass
class KWICLine:
    """
    A Key Word In Context line showing term with surrounding text.

    Attributes:
        left_context: Text before the keyword
        keyword: The matched keyword
        right_context: Text after the keyword
        doc_id: Document identifier
        sent_index: Sentence index within document
    """

    left_context: str
    keyword: str
    right_context: str
    doc_id: str
    sent_index: int

    def __str__(self) -> str:
        """Format as aligned KWIC line."""
        return f"{self.left_context:>50} [{self.keyword}] {self.right_context:<50}"


def concordance(
    term: str,
    docs: List[ProcessedDocument],
    width: int = 50,
    case_sensitive: bool = False,
) -> List[KWICLine]:
    """
    Generate KWIC (Key Word In Context) concordance for a term.

    Creates aligned display with keyword centered and context on both sides.

    Args:
        term: Term to create concordance for
        docs: List of preprocessed documents
        width: Character width for context on each side (default: 50)
        case_sensitive: Whether matching is case-sensitive (default: False)

    Returns:
        List of KWICLine objects

    Example:
        >>> lines = concordance("intentionality", docs, width=30)
        >>> for line in lines:
        ...     print(line)
    """
    kwic_lines = []

    # Normalize search term
    search_term = term if case_sensitive else term.lower()

    for doc_idx, doc in enumerate(docs):
        doc_id = doc.metadata.get("source_path", f"doc_{doc_idx}")

        for sent_idx, sentence in enumerate(doc.sentences):
            # Normalize for comparison
            sentence_compare = sentence if case_sensitive else sentence.lower()

            # Find all occurrences in this sentence
            start = 0
            while True:
                pos = sentence_compare.find(search_term, start)
                if pos == -1:
                    break

                # Extract context
                left_start = max(0, pos - width)
                left_context = sentence[left_start:pos]

                # Get the actual keyword as it appears (preserving case)
                keyword = sentence[pos : pos + len(term)]

                # Right context
                right_end = min(len(sentence), pos + len(term) + width)
                right_context = sentence[pos + len(term) : right_end]

                # Trim to word boundaries for cleaner display
                left_context = _trim_to_word_boundary(left_context, left=True)
                right_context = _trim_to_word_boundary(right_context, left=False)

                # Create KWIC line
                kwic_line = KWICLine(
                    left_context=left_context.strip(),
                    keyword=keyword,
                    right_context=right_context.strip(),
                    doc_id=doc_id,
                    sent_index=sent_idx,
                )
                kwic_lines.append(kwic_line)

                # Move past this occurrence
                start = pos + 1

    return kwic_lines


def _trim_to_word_boundary(text: str, left: bool = True) -> str:
    """
    Trim text to word boundary for cleaner display.

    Args:
        text: Text to trim
        left: If True, trim from left (keep right words whole)
              If False, trim from right (keep left words whole)

    Returns:
        Trimmed text
    """
    if not text:
        return text

    if left:
        # Find first space and remove everything before it
        first_space = text.find(" ")
        if first_space > 0 and first_space < len(text) - 1:
            return text[first_space + 1 :]
    else:
        # Find last space and remove everything after it
        last_space = text.rfind(" ")
        if last_space > 0:
            return text[:last_space]

    return text


def format_kwic_lines(
    lines: List[KWICLine],
    width: int = 50,
    show_doc_id: bool = False,
) -> str:
    """
    Format KWIC lines as aligned text for display.

    Args:
        lines: List of KWICLine objects
        width: Width for padding (should match concordance width)
        show_doc_id: Whether to include document ID in output

    Returns:
        Formatted string with aligned KWIC lines
    """
    output = []

    for line in lines:
        if show_doc_id:
            prefix = f"[{line.doc_id}:{line.sent_index}] "
        else:
            prefix = ""

        # Right-align left context, left-align right context
        formatted = (
            f"{prefix}{line.left_context:>{width}} "
            f"[{line.keyword}] "
            f"{line.right_context:<{width}}"
        )
        output.append(formatted)

    return "\n".join(output)


def concordance_sorted(
    term: str,
    docs: List[ProcessedDocument],
    width: int = 50,
    sort_by: str = "left",
) -> List[KWICLine]:
    """
    Generate concordance sorted by context.

    Useful for identifying patterns in term usage.

    Args:
        term: Term to create concordance for
        docs: List of preprocessed documents
        width: Character width for context (default: 50)
        sort_by: Sort by 'left' or 'right' context (default: 'left')

    Returns:
        Sorted list of KWICLine objects
    """
    lines = concordance(term, docs, width=width)

    if sort_by == "left":
        # Sort by left context (reversed for right-to-left sort from keyword)
        lines.sort(key=lambda x: x.left_context[::-1].lower())
    elif sort_by == "right":
        # Sort by right context
        lines.sort(key=lambda x: x.right_context.lower())
    else:
        raise ValueError(f"Invalid sort_by: {sort_by}. Must be 'left' or 'right'")

    return lines


def concordance_filtered(
    term: str,
    docs: List[ProcessedDocument],
    filter_terms: List[str],
    width: int = 50,
) -> List[KWICLine]:
    """
    Generate concordance showing only lines containing additional terms.

    Useful for finding co-occurrence patterns.

    Args:
        term: Primary term to create concordance for
        docs: List of preprocessed documents
        filter_terms: Additional terms that must appear in context
        width: Character width for context (default: 50)

    Returns:
        Filtered list of KWICLine objects
    """
    all_lines = concordance(term, docs, width=width)

    # Filter to lines where context contains any filter term
    filtered = []
    for line in all_lines:
        full_context = (
            line.left_context + " " + line.keyword + " " + line.right_context
        ).lower()

        if any(ft.lower() in full_context for ft in filter_terms):
            filtered.append(line)

    return filtered
