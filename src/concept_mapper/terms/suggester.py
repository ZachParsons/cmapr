"""
Auto-population of term lists from statistical analysis.

Uses Phase 3's philosophical term detection to suggest initial term lists
with automatic example extraction from corpus.
"""

from typing import List, Optional
from collections import Counter
from ..corpus.models import ProcessedDocument
from ..analysis.rarity import PhilosophicalTermScorer, score_philosophical_terms
from .models import TermList, TermEntry


def suggest_terms_from_analysis(
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    min_score: float = 1.0,
    top_n: int = 50,
    max_examples: int = 3,
    use_lemmas: bool = True,
    min_author_freq: int = 3,
    scorer_weights: Optional[dict] = None,
) -> TermList:
    """
    Generate term list from statistical analysis of corpus.

    Uses PhilosophicalTermScorer to identify distinctive terms and
    automatically populates examples from the corpus.

    Args:
        docs: List of preprocessed documents
        reference_corpus: Reference corpus (e.g., Brown)
        min_score: Minimum score threshold (default: 1.0)
        top_n: Maximum number of terms to suggest (default: 50)
        max_examples: Maximum example sentences per term (default: 3)
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_author_freq: Minimum occurrences in author corpus (default: 3)
        scorer_weights: Optional custom weights for scorer

    Returns:
        TermList with suggested terms and examples

    Example:
        >>> from src.concept_mapper.analysis import load_reference_corpus
        >>> brown = load_reference_corpus("brown")
        >>> terms = suggest_terms_from_analysis(docs, brown, min_score=1.5, top_n=30)
        >>> print(f"Found {len(terms)} suggested terms")
    """
    # Use scorer to get top philosophical terms
    if scorer_weights:
        scorer = PhilosophicalTermScorer(
            docs,
            reference_corpus,
            use_lemmas=use_lemmas,
            min_author_freq=min_author_freq,
            weights=scorer_weights,
        )
        results = scorer.score_all(min_score=min_score, top_n=top_n)
    else:
        # Use convenience function with default weights
        results_simple = score_philosophical_terms(
            docs,
            reference_corpus,
            use_lemmas=use_lemmas,
            min_author_freq=min_author_freq,
            top_n=top_n,
        )
        # Filter by min_score
        results = [
            (term, score, {}) for term, score in results_simple if score >= min_score
        ]

    # Create term list
    term_list = TermList(
        name="Suggested Philosophical Terms",
        description=f"Auto-generated from corpus analysis (min_score={min_score}, top_n={top_n})",
    )

    # Populate terms with examples
    for term, score, components in results:
        # Find example sentences containing this term
        examples = _extract_examples(term, docs, max_examples=max_examples)

        # Try to determine POS from documents
        pos = _infer_pos(term, docs)

        # Create term entry
        entry = TermEntry(
            term=term,
            lemma=term if use_lemmas else None,
            pos=pos,
            definition=None,  # Human should fill this in
            notes=f"Score: {score:.2f}",
            examples=examples,
            metadata={"score": score, "components": components if components else {}},
        )

        term_list.add(entry)

    return term_list


def _extract_examples(
    term: str, docs: List[ProcessedDocument], max_examples: int = 3
) -> List[str]:
    """
    Extract example sentences containing the term.

    Args:
        term: Term to find examples for
        docs: List of preprocessed documents
        max_examples: Maximum number of examples to extract

    Returns:
        List of example sentences
    """
    examples = []

    for doc in docs:
        if len(examples) >= max_examples:
            break

        # Check each sentence for the term
        for sentence in doc.sentences:
            # Case-insensitive match
            if term.lower() in sentence.lower():
                examples.append(sentence.strip())

                if len(examples) >= max_examples:
                    break

    return examples


def _infer_pos(term: str, docs: List[ProcessedDocument]) -> Optional[str]:
    """
    Infer the most common POS tag for a term from documents.

    Args:
        term: Term to find POS for
        docs: List of preprocessed documents

    Returns:
        Most common POS tag, or None if not found
    """
    pos_counts = Counter()

    for doc in docs:
        # Look for term in POS tags
        for token, pos in doc.pos_tags:
            if token.lower() == term.lower():
                pos_counts[pos] += 1

    # Return most common POS tag
    if pos_counts:
        return pos_counts.most_common(1)[0][0]

    return None


def suggest_terms_by_method(
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    method: str = "ratio",
    threshold: float = 10.0,
    top_n: int = 50,
    max_examples: int = 3,
) -> TermList:
    """
    Generate term list using a specific detection method.

    Useful for comparing results from different methods.

    Args:
        docs: List of preprocessed documents
        reference_corpus: Reference corpus
        method: Detection method ('ratio', 'tfidf', 'neologism', 'definitional')
        threshold: Method-specific threshold
        top_n: Maximum number of terms
        max_examples: Maximum examples per term

    Returns:
        TermList with terms from specified method
    """
    from ..analysis.rarity import (
        get_top_corpus_specific_terms,
        get_top_tfidf_terms,
        get_all_neologism_signals,
        get_highly_defined_terms,
    )

    # Get terms using specified method
    if method == "ratio":
        results = get_top_corpus_specific_terms(
            docs, reference_corpus, n=top_n, min_author_freq=3
        )
        terms = [term for term, ratio in results]
        name = f"Corpus-Comparative Ratio (threshold={threshold})"

    elif method == "tfidf":
        results = get_top_tfidf_terms(
            docs, reference_corpus, n=top_n, min_author_freq=3
        )
        terms = [term for term, score in results]
        name = "TF-IDF vs Reference"

    elif method == "neologism":
        signals = get_all_neologism_signals(docs, reference_corpus, min_author_freq=3)
        terms = list(signals["all_neologisms"])[:top_n]
        name = "Neologism Detection"

    elif method == "definitional":
        terms = list(get_highly_defined_terms(docs, min_definitions=2, terms=None))[
            :top_n
        ]
        name = "Definitional Contexts"

    else:
        raise ValueError(
            f"Unknown method: {method}. Must be 'ratio', 'tfidf', 'neologism', or 'definitional'"
        )

    # Create term list
    term_list = TermList(
        name=f"Suggested Terms ({name})",
        description=f"Auto-generated using {method} method",
    )

    # Populate with examples
    for term in terms:
        examples = _extract_examples(term, docs, max_examples=max_examples)
        pos = _infer_pos(term, docs)

        entry = TermEntry(
            term=term,
            lemma=term,
            pos=pos,
            examples=examples,
            metadata={"method": method},
        )

        try:
            term_list.add(entry)
        except ValueError:
            # Skip duplicates
            pass

    return term_list
