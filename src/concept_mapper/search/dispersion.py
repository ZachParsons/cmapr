"""
Term dispersion analysis - visualizing where terms appear across documents.

Provides tools for understanding the distribution of terms throughout a corpus.
"""

from typing import Dict, List, Tuple
from collections import defaultdict
from ..corpus.models import ProcessedDocument


def dispersion(
    term: str,
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
    by_char: bool = False,
) -> Dict[str, List[int]]:
    """
    Analyze where a term appears across documents.

    Returns a mapping of document IDs to lists of positions where the term appears.

    Args:
        term: Term to analyze
        docs: List of preprocessed documents
        case_sensitive: Whether search is case-sensitive (default: False)
        by_char: If True, return character offsets; if False, return sentence indices

    Returns:
        Dictionary mapping doc_id -> list of positions (sentence indices or char offsets)

    Example:
        >>> disp = dispersion("abstraction", docs)
        >>> for doc_id, positions in disp.items():
        ...     print(f"{doc_id}: appears in sentences {positions}")
    """
    results = defaultdict(list)

    # Normalize search term
    search_term = term if case_sensitive else term.lower()

    for doc_idx, doc in enumerate(docs):
        doc_id = doc.metadata.get("source_path", f"doc_{doc_idx}")

        if by_char:
            # Character-level dispersion
            full_text = " ".join(doc.sentences)
            text_compare = full_text if case_sensitive else full_text.lower()

            start = 0
            while True:
                pos = text_compare.find(search_term, start)
                if pos == -1:
                    break
                results[doc_id].append(pos)
                start = pos + 1

        else:
            # Sentence-level dispersion (more common)
            for sent_idx, sentence in enumerate(doc.sentences):
                sentence_compare = sentence if case_sensitive else sentence.lower()

                if search_term in sentence_compare:
                    results[doc_id].append(sent_idx)

    return dict(results)


def get_dispersion_summary(
    term: str,
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
) -> Dict[str, any]:
    """
    Get summary statistics about term dispersion.

    Args:
        term: Term to analyze
        docs: List of preprocessed documents
        case_sensitive: Whether search is case-sensitive

    Returns:
        Dictionary with dispersion statistics:
        - term: The analyzed term
        - total_docs: Total number of documents in corpus
        - docs_with_term: Number of documents containing the term
        - coverage: Percentage of documents containing term
        - total_occurrences: Total times term appears
        - positions: Dict mapping doc_id -> list of sentence indices
        - avg_occurrences_per_doc: Average occurrences in docs that have the term

    Example:
        >>> summary = get_dispersion_summary("abstraction", docs)
        >>> print(f"Appears in {summary['docs_with_term']}/{summary['total_docs']} documents")
        >>> print(f"Coverage: {summary['coverage']:.1f}%")
    """
    positions = dispersion(term, docs, case_sensitive=case_sensitive)

    total_docs = len(docs)
    docs_with_term = len(positions)
    total_occurrences = sum(len(pos_list) for pos_list in positions.values())

    coverage = (docs_with_term / total_docs * 100) if total_docs > 0 else 0.0

    avg_per_doc = total_occurrences / docs_with_term if docs_with_term > 0 else 0.0

    return {
        "term": term,
        "total_docs": total_docs,
        "docs_with_term": docs_with_term,
        "coverage": coverage,
        "total_occurrences": total_occurrences,
        "positions": positions,
        "avg_occurrences_per_doc": avg_per_doc,
    }


def compare_dispersion(
    terms: List[str],
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
) -> Dict[str, Dict[str, any]]:
    """
    Compare dispersion patterns across multiple terms.

    Useful for understanding which terms are more widely distributed vs concentrated.

    Args:
        terms: List of terms to compare
        docs: List of preprocessed documents
        case_sensitive: Whether search is case-sensitive

    Returns:
        Dictionary mapping term -> dispersion summary

    Example:
        >>> comparison = compare_dispersion(["abstraction", "ontology", "epistemology"], docs)
        >>> for term, summary in comparison.items():
        ...     print(f"{term}: {summary['coverage']:.1f}% coverage")
    """
    results = {}

    for term in terms:
        results[term] = get_dispersion_summary(
            term, docs, case_sensitive=case_sensitive
        )

    return results


def dispersion_plot_data(
    term: str,
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
) -> List[Tuple[str, int, int]]:
    """
    Generate data suitable for plotting term dispersion.

    Returns list of (doc_id, doc_length, positions) tuples that can be
    used to create dispersion plots showing term distribution.

    Args:
        term: Term to analyze
        docs: List of preprocessed documents
        case_sensitive: Whether search is case-sensitive

    Returns:
        List of tuples: (doc_id, num_sentences, list_of_positions)

    Example:
        >>> plot_data = dispersion_plot_data("abstraction", docs)
        >>> for doc_id, length, positions in plot_data:
        ...     # Plot positions as vertical lines at their sentence indices
        ...     print(f"{doc_id} ({length} sentences): {positions}")
    """
    positions = dispersion(term, docs, case_sensitive=case_sensitive)

    plot_data = []

    for doc_idx, doc in enumerate(docs):
        doc_id = doc.metadata.get("source_path", f"doc_{doc_idx}")
        doc_length = len(doc.sentences)
        doc_positions = positions.get(doc_id, [])

        plot_data.append((doc_id, doc_length, doc_positions))

    return plot_data


def get_concentrated_regions(
    term: str,
    docs: List[ProcessedDocument],
    window_size: int = 10,
    min_occurrences: int = 3,
    case_sensitive: bool = False,
) -> List[Dict[str, any]]:
    """
    Find regions where a term appears frequently (concentrated usage).

    Useful for identifying passages with heavy usage of particular terminology.

    Args:
        term: Term to analyze
        docs: List of preprocessed documents
        window_size: Size of sliding window in sentences (default: 10)
        min_occurrences: Minimum occurrences in window to be considered concentrated
        case_sensitive: Whether search is case-sensitive

    Returns:
        List of dictionaries describing concentrated regions:
        - doc_id: Document identifier
        - start_sent: Starting sentence index
        - end_sent: Ending sentence index
        - occurrences: Number of occurrences in this window
        - density: Occurrences per sentence in window

    Example:
        >>> regions = get_concentrated_regions("abstraction", docs, window_size=10, min_occurrences=3)
        >>> for region in regions:
        ...     print(f"{region['doc_id']} sentences {region['start_sent']}-{region['end_sent']}: "
        ...           f"{region['occurrences']} occurrences")
    """
    positions = dispersion(term, docs, case_sensitive=case_sensitive)
    concentrated = []

    for doc_idx, doc in enumerate(docs):
        doc_id = doc.metadata.get("source_path", f"doc_{doc_idx}")
        doc_positions = positions.get(doc_id, [])

        if not doc_positions:
            continue

        # Slide window across document
        num_sentences = len(doc.sentences)

        for start in range(num_sentences - window_size + 1):
            end = start + window_size

            # Count occurrences in this window
            occurrences_in_window = sum(
                1 for pos in doc_positions if start <= pos < end
            )

            if occurrences_in_window >= min_occurrences:
                density = occurrences_in_window / window_size

                concentrated.append(
                    {
                        "doc_id": doc_id,
                        "start_sent": start,
                        "end_sent": end,
                        "occurrences": occurrences_in_window,
                        "density": density,
                    }
                )

    # Sort by density (highest first)
    concentrated.sort(key=lambda x: x["density"], reverse=True)

    return concentrated
