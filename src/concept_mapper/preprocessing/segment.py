"""
Paragraph segmentation for document structure analysis.

Identifies paragraph boundaries using various markers:
- Double newlines (most common)
- Indentation patterns
- Line breaks with spacing patterns
"""

import re
from typing import List, Tuple


def segment_paragraphs(text: str, preserve_empty_lines: bool = False) -> List[str]:
    """
    Segment text into paragraphs.

    Identifies paragraph boundaries using:
    1. Double newlines (blank line separation)
    2. Indentation changes (indented first line)
    3. Line breaks with significant spacing

    Args:
        text: Raw text to segment
        preserve_empty_lines: Keep empty lines as separate paragraphs

    Returns:
        List of paragraph strings (whitespace-stripped)

    Example:
        >>> text = "First paragraph.\\n\\nSecond paragraph."
        >>> segment_paragraphs(text)
        ['First paragraph.', 'Second paragraph.']
    """
    if not text or not text.strip():
        return []

    # Method 1: Split on double newlines (most common)
    paragraphs = _split_on_blank_lines(text)

    # Method 2: Further split on indentation changes if needed
    paragraphs = _split_on_indentation(paragraphs)

    # Clean up paragraphs
    paragraphs = [p.strip() for p in paragraphs]

    # Filter empty paragraphs unless preserve_empty_lines is True
    if not preserve_empty_lines:
        paragraphs = [p for p in paragraphs if p]

    return paragraphs


def _split_on_blank_lines(text: str) -> List[str]:
    """
    Split text on blank lines (double newlines).

    Args:
        text: Text to split

    Returns:
        List of text chunks
    """
    # Split on 2+ newlines (handles \\n\\n, \\n\\n\\n, etc.)
    return re.split(r"\n\s*\n", text)


def _split_on_indentation(paragraphs: List[str]) -> List[str]:
    """
    Further split paragraphs on indentation changes.

    Detects when lines start with indentation (spaces/tabs) after
    non-indented lines, indicating a new paragraph.

    Args:
        paragraphs: List of paragraph candidates

    Returns:
        List of refined paragraphs
    """
    refined = []

    for para in paragraphs:
        lines = para.split("\n")
        if len(lines) <= 1:
            # Single line paragraph, keep as-is
            refined.append(para)
            continue

        # Check for indentation pattern changes
        current_chunk = []
        prev_indented = False

        for line in lines:
            # Check if line starts with whitespace (excluding empty lines)
            is_indented = bool(line) and (line[0] in " \t")

            # If indentation changes, start new chunk
            if current_chunk and is_indented and not prev_indented:
                # Save previous chunk
                refined.append("\n".join(current_chunk))
                current_chunk = [line]
            else:
                current_chunk.append(line)

            prev_indented = is_indented

        # Add remaining chunk
        if current_chunk:
            refined.append("\n".join(current_chunk))

    return refined


def get_paragraph_indices(text: str, sentences: List[str]) -> List[int]:
    """
    Map each sentence to its containing paragraph index.

    Args:
        text: Original text
        sentences: List of sentences (from tokenize_sentences)

    Returns:
        List of paragraph indices (one per sentence)

    Example:
        >>> text = "Sent 1. Sent 2.\\n\\nSent 3."
        >>> sentences = ["Sent 1.", "Sent 2.", "Sent 3."]
        >>> get_paragraph_indices(text, sentences)
        [0, 0, 1]
    """
    if not sentences:
        return []

    # Get paragraph boundaries
    paragraphs = segment_paragraphs(text, preserve_empty_lines=False)

    if not paragraphs:
        # No paragraphs detected, all sentences in paragraph 0
        return [0] * len(sentences)

    # Build map from sentence to paragraph
    sentence_to_para = []

    for sent in sentences:
        # Find which paragraph contains this sentence
        para_idx = _find_containing_paragraph(sent, paragraphs)
        sentence_to_para.append(para_idx)

    return sentence_to_para


def _find_containing_paragraph(sentence: str, paragraphs: List[str]) -> int:
    """
    Find which paragraph contains the given sentence.

    Args:
        sentence: Sentence to locate
        paragraphs: List of paragraphs

    Returns:
        Paragraph index (0-based), or 0 if not found
    """
    # Clean sentence for matching (strip whitespace)
    sent_clean = sentence.strip()

    for i, para in enumerate(paragraphs):
        if sent_clean in para:
            return i

    # If not found, default to paragraph 0
    return 0


def get_paragraph_spans(text: str) -> List[Tuple[int, int]]:
    """
    Get character spans of paragraphs in original text.

    Args:
        text: Original text

    Returns:
        List of (start_pos, end_pos) tuples

    Example:
        >>> text = "Para 1.\\n\\nPara 2."
        >>> get_paragraph_spans(text)
        [(0, 7), (9, 16)]
    """
    if not text or not text.strip():
        return []

    paragraphs = segment_paragraphs(text, preserve_empty_lines=False)
    spans = []

    search_pos = 0
    for para in paragraphs:
        # Find paragraph in original text
        start = text.find(para, search_pos)
        if start != -1:
            end = start + len(para)
            spans.append((start, end))
            search_pos = end
        else:
            # Fallback: approximate position
            spans.append((search_pos, search_pos + len(para)))
            search_pos += len(para)

    return spans
