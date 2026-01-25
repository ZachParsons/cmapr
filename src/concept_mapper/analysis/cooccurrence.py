"""
Co-occurrence analysis for discovering term relationships.

Analyzes which terms appear together in various contexts (sentences, paragraphs,
windows) to identify conceptual relationships and associations.
"""

from collections import Counter
from typing import List, Optional, Dict, Tuple
import math
from ..corpus.models import ProcessedDocument
from ..terms.models import TermList


def cooccurs_in_sentence(
    term: str,
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
) -> Counter:
    """
    Count terms that co-occur with target term in the same sentence.

    Args:
        term: Target term to find co-occurrences for
        docs: List of preprocessed documents
        case_sensitive: Whether matching is case-sensitive (default: False)

    Returns:
        Counter of terms appearing in same sentences as target term

    Example:
        >>> cooccurs = cooccurs_in_sentence("abstraction", docs)
        >>> print(cooccurs.most_common(10))
        [('philosophy', 5), ('concept', 4), ('process', 3), ...]
    """
    cooccurrences = Counter()

    # Normalize search term
    search_term = term if case_sensitive else term.lower()

    for doc in docs:
        for sentence in doc.sentences:
            sentence_compare = sentence if case_sensitive else sentence.lower()

            # Check if target term appears in this sentence
            if search_term in sentence_compare:
                # Tokenize and count all words in this sentence
                words = sentence.split()
                for word in words:
                    # Clean and normalize word
                    cleaned = word.strip('.,;:!?"()[]{}').lower()
                    if cleaned and cleaned != search_term.lower():
                        cooccurrences[cleaned] += 1

    return cooccurrences


def cooccurs_filtered(
    term: str,
    docs: List[ProcessedDocument],
    term_list: TermList,
    case_sensitive: bool = False,
) -> Counter:
    """
    Count co-occurrences only for terms in the curated term list.

    Useful for focusing analysis on philosophical vocabulary rather than
    common words.

    Args:
        term: Target term to find co-occurrences for
        docs: List of preprocessed documents
        term_list: Curated list of terms to count (others ignored)
        case_sensitive: Whether matching is case-sensitive (default: False)

    Returns:
        Counter of curated terms appearing with target term

    Example:
        >>> cooccurs = cooccurs_filtered("abstraction", docs, philosophical_terms)
        >>> # Only counts terms in philosophical_terms list
    """
    # Get all co-occurrences
    all_cooccurs = cooccurs_in_sentence(term, docs, case_sensitive=case_sensitive)

    # Build set of terms in list for fast lookup
    list_terms = {entry.term.lower() for entry in term_list.list_terms()}

    # Filter to only terms in the list
    filtered = Counter()
    for cooccur_term, count in all_cooccurs.items():
        if cooccur_term.lower() in list_terms:
            filtered[cooccur_term] = count

    return filtered


def cooccurs_in_paragraph(
    term: str,
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
) -> Counter:
    """
    Count terms that co-occur with target term in the same paragraph.

    Note: Requires paragraph segmentation. Currently treats each document
    as a single paragraph since Phase 1.7 (paragraph segmentation) is not
    yet implemented.

    Args:
        term: Target term to find co-occurrences for
        docs: List of preprocessed documents
        case_sensitive: Whether matching is case-sensitive (default: False)

    Returns:
        Counter of terms appearing in same paragraphs as target term

    Example:
        >>> cooccurs = cooccurs_in_paragraph("abstraction", docs)
    """
    # TODO: Implement proper paragraph segmentation (Phase 1.7)
    # For now, treat each document as a single paragraph
    cooccurrences = Counter()

    search_term = term if case_sensitive else term.lower()

    for doc in docs:
        # Check if term appears anywhere in document
        full_text = " ".join(doc.sentences)
        text_compare = full_text if case_sensitive else full_text.lower()

        if search_term in text_compare:
            # Count all words in the entire document
            words = full_text.split()
            for word in words:
                cleaned = word.strip('.,;:!?"()[]{}').lower()
                if cleaned and cleaned != search_term.lower():
                    cooccurrences[cleaned] += 1

    return cooccurrences


def cooccurs_within_n(
    term: str,
    docs: List[ProcessedDocument],
    n_sentences: int = 3,
    case_sensitive: bool = False,
) -> Counter:
    """
    Count terms appearing within N sentences of the target term.

    Uses a sliding window that extends N sentences before and after
    each occurrence of the target term.

    Args:
        term: Target term to find co-occurrences for
        docs: List of preprocessed documents
        n_sentences: Window size in sentences (default: 3)
        case_sensitive: Whether matching is case-sensitive (default: False)

    Returns:
        Counter of terms appearing within N sentences of target

    Example:
        >>> cooccurs = cooccurs_within_n("abstraction", docs, n_sentences=2)
        >>> # Counts terms within 2 sentences before/after "abstraction"
    """
    cooccurrences = Counter()

    search_term = term if case_sensitive else term.lower()

    for doc in docs:
        sentences = doc.sentences

        # Find all sentences containing the target term
        for sent_idx, sentence in enumerate(sentences):
            sentence_compare = sentence if case_sensitive else sentence.lower()

            if search_term in sentence_compare:
                # Define window boundaries
                start_idx = max(0, sent_idx - n_sentences)
                end_idx = min(len(sentences), sent_idx + n_sentences + 1)

                # Count words in all sentences within window
                for window_sent in sentences[start_idx:end_idx]:
                    words = window_sent.split()
                    for word in words:
                        cleaned = word.strip('.,;:!?"()[]{}').lower()
                        if cleaned and cleaned != search_term.lower():
                            cooccurrences[cleaned] += 1

    return cooccurrences


def pmi(
    term1: str,
    term2: str,
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
) -> float:
    """
    Calculate Pointwise Mutual Information between two terms.

    PMI measures how much more likely two terms are to appear together
    than would be expected by chance:
    - PMI > 0: Terms appear together more than expected (positive association)
    - PMI ≈ 0: Terms appear independently
    - PMI < 0: Terms appear together less than expected (negative association)

    Args:
        term1: First term
        term2: Second term
        docs: List of preprocessed documents
        case_sensitive: Whether matching is case-sensitive (default: False)

    Returns:
        PMI score (can be negative)

    Example:
        >>> pmi_score = pmi("abstraction", "separation", docs)
        >>> if pmi_score > 2.0:
        ...     print("Strong association")
    """
    # Count sentences
    total_sentences = sum(len(doc.sentences) for doc in docs)

    if total_sentences == 0:
        return 0.0

    # Normalize terms
    t1 = term1 if case_sensitive else term1.lower()
    t2 = term2 if case_sensitive else term2.lower()

    # Count occurrences
    count_t1 = 0
    count_t2 = 0
    count_both = 0

    for doc in docs:
        for sentence in doc.sentences:
            s = sentence if case_sensitive else sentence.lower()

            has_t1 = t1 in s
            has_t2 = t2 in s

            if has_t1:
                count_t1 += 1
            if has_t2:
                count_t2 += 1
            if has_t1 and has_t2:
                count_both += 1

    # Avoid division by zero
    if count_t1 == 0 or count_t2 == 0 or count_both == 0:
        return 0.0

    # Calculate probabilities
    p_t1 = count_t1 / total_sentences
    p_t2 = count_t2 / total_sentences
    p_both = count_both / total_sentences

    # PMI = log2(P(both) / (P(t1) * P(t2)))
    pmi_score = math.log2(p_both / (p_t1 * p_t2))

    return pmi_score


def log_likelihood_ratio(
    term1: str,
    term2: str,
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
) -> float:
    """
    Calculate log-likelihood ratio for term co-occurrence.

    G² (log-likelihood ratio) is a statistical test for association:
    - Higher values indicate stronger evidence of association
    - G² > 3.84 is significant at p < 0.05
    - G² > 6.63 is significant at p < 0.01
    - G² > 10.83 is significant at p < 0.001

    Args:
        term1: First term
        term2: Second term
        docs: List of preprocessed documents
        case_sensitive: Whether matching is case-sensitive (default: False)

    Returns:
        Log-likelihood ratio (G²)

    Example:
        >>> llr = log_likelihood_ratio("abstraction", "separation", docs)
        >>> if llr > 10.83:
        ...     print("Highly significant association (p < 0.001)")
    """
    # Count sentences
    total_sentences = sum(len(doc.sentences) for doc in docs)

    if total_sentences == 0:
        return 0.0

    # Normalize terms
    t1 = term1 if case_sensitive else term1.lower()
    t2 = term2 if case_sensitive else term2.lower()

    # Build contingency table:
    # a = both terms present
    # b = term1 present, term2 absent
    # c = term1 absent, term2 present
    # d = neither term present

    a = 0  # both
    b = 0  # only t1
    c = 0  # only t2
    d = 0  # neither

    for doc in docs:
        for sentence in doc.sentences:
            s = sentence if case_sensitive else sentence.lower()

            has_t1 = t1 in s
            has_t2 = t2 in s

            if has_t1 and has_t2:
                a += 1
            elif has_t1 and not has_t2:
                b += 1
            elif not has_t1 and has_t2:
                c += 1
            else:
                d += 1

    # Avoid division by zero
    if a == 0 or (a + b) == 0 or (a + c) == 0:
        return 0.0

    # Calculate expected frequencies
    n = a + b + c + d
    e_a = ((a + b) * (a + c)) / n
    e_b = ((a + b) * (b + d)) / n
    e_c = ((c + d) * (a + c)) / n
    e_d = ((c + d) * (b + d)) / n

    # Calculate G² using the log-likelihood formula
    # G² = 2 * sum(observed * log(observed / expected))
    g_squared = 0.0

    if a > 0 and e_a > 0:
        g_squared += a * math.log(a / e_a)
    if b > 0 and e_b > 0:
        g_squared += b * math.log(b / e_b)
    if c > 0 and e_c > 0:
        g_squared += c * math.log(c / e_c)
    if d > 0 and e_d > 0:
        g_squared += d * math.log(d / e_d)

    g_squared *= 2

    return g_squared


def build_cooccurrence_matrix(
    term_list: TermList,
    docs: List[ProcessedDocument],
    method: str = "count",
    window: str = "sentence",
    n_sentences: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    """
    Build a symmetric co-occurrence matrix for all terms in the list.

    Creates a term × term matrix showing association strength between
    all pairs of terms.

    Args:
        term_list: List of terms to analyze
        docs: List of preprocessed documents
        method: Scoring method - "count", "pmi", or "llr" (default: "count")
        window: Co-occurrence window - "sentence" or "n_sentences" (default: "sentence")
        n_sentences: Window size if window="n_sentences" (default: None)

    Returns:
        Nested dictionary: {term1: {term2: score, ...}, ...}
        Matrix is symmetric: matrix[t1][t2] == matrix[t2][t1]

    Example:
        >>> matrix = build_cooccurrence_matrix(
        ...     philosophical_terms, docs, method="pmi", window="sentence"
        ... )
        >>> # Access association between two terms
        >>> score = matrix["abstraction"]["separation"]
    """
    terms = [entry.term for entry in term_list.list_terms()]

    # Initialize matrix
    matrix = {term: {} for term in terms}

    # Fill matrix
    for i, term1 in enumerate(terms):
        for j, term2 in enumerate(terms):
            if i == j:
                # Diagonal: self-association is 0
                matrix[term1][term2] = 0.0
            elif term2 in matrix[term1]:
                # Already computed (symmetric)
                continue
            else:
                # Compute association score
                if method == "count":
                    # Raw co-occurrence count
                    if window == "sentence":
                        cooccurs = cooccurs_in_sentence(term1, docs)
                    elif window == "n_sentences":
                        if n_sentences is None:
                            raise ValueError(
                                "n_sentences must be specified for window='n_sentences'"
                            )
                        cooccurs = cooccurs_within_n(term1, docs, n_sentences)
                    else:
                        raise ValueError(f"Invalid window: {window}")

                    score = float(cooccurs.get(term2.lower(), 0))

                elif method == "pmi":
                    # Pointwise Mutual Information
                    score = pmi(term1, term2, docs)

                elif method == "llr":
                    # Log-likelihood ratio
                    score = log_likelihood_ratio(term1, term2, docs)

                else:
                    raise ValueError(f"Invalid method: {method}")

                # Fill both directions (symmetric)
                matrix[term1][term2] = score
                matrix[term2][term1] = score

    return matrix


def save_cooccurrence_matrix(
    matrix: Dict[str, Dict[str, float]],
    output_path: str,
) -> None:
    """
    Save co-occurrence matrix as CSV file.

    Args:
        matrix: Co-occurrence matrix from build_cooccurrence_matrix()
        output_path: Path to output CSV file

    Example:
        >>> matrix = build_cooccurrence_matrix(terms, docs, method="pmi")
        >>> save_cooccurrence_matrix(matrix, "output/cooccurrence_pmi.csv")
    """
    import csv

    # Get sorted list of terms
    terms = sorted(matrix.keys())

    # Write CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header row
        writer.writerow([""] + terms)

        # Data rows
        for term1 in terms:
            row = [term1]
            for term2 in terms:
                score = matrix[term1].get(term2, 0.0)
                row.append(f"{score:.4f}")
            writer.writerow(row)


def get_top_cooccurrences(
    term: str,
    docs: List[ProcessedDocument],
    n: int = 10,
    method: str = "count",
    window: str = "sentence",
    term_list: Optional[TermList] = None,
) -> List[Tuple[str, float]]:
    """
    Get top N terms that co-occur with the target term.

    Convenience function for quick exploration of term associations.

    Args:
        term: Target term
        docs: List of preprocessed documents
        n: Number of top co-occurrences to return (default: 10)
        method: "count", "pmi", or "llr" (default: "count")
        window: "sentence" or other window type (default: "sentence")
        term_list: Optional term list to filter results

    Returns:
        List of (term, score) tuples, sorted by score descending

    Example:
        >>> top = get_top_cooccurrences("abstraction", docs, n=5, method="pmi")
        >>> for cooccur_term, score in top:
        ...     print(f"{cooccur_term}: {score:.2f}")
    """
    if method == "count":
        if window == "sentence":
            cooccurs = cooccurs_in_sentence(term, docs)
        else:
            raise ValueError(f"Window '{window}' not yet implemented for count method")

        if term_list:
            cooccurs = Counter(
                {
                    t: c
                    for t, c in cooccurs.items()
                    if any(e.term.lower() == t.lower() for e in term_list.list_terms())
                }
            )

        return [(t, float(c)) for t, c in cooccurs.most_common(n)]

    elif method == "pmi" or method == "llr":
        # For PMI/LLR, we need to compute for all candidate terms
        # Get all terms that co-occur at least once
        cooccurs = cooccurs_in_sentence(term, docs)

        if term_list:
            candidates = [
                e.term
                for e in term_list.list_terms()
                if e.term.lower() in cooccurs and e.term.lower() != term.lower()
            ]
        else:
            candidates = [t for t in cooccurs.keys() if t != term.lower()]

        # Compute scores
        scores = []
        for candidate in candidates:
            if method == "pmi":
                score = pmi(term, candidate, docs)
            else:  # llr
                score = log_likelihood_ratio(term, candidate, docs)
            scores.append((candidate, score))

        # Sort by score and return top N
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:n]

    else:
        raise ValueError(f"Invalid method: {method}")
