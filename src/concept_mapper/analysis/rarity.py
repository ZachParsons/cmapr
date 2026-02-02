"""
Rarity and corpus-comparative analysis for philosophical term detection.

Identifies author-specific conceptual vocabulary through statistical comparison
with reference corpora. Focuses on terms distinctive to the author's work,
not just rare within the primary text.

## Detection Methods

This module implements three complementary approaches to identify philosophical terms:

### 1. Relative Frequency Ratio (compare_to_reference)
Compares how much more (or less) the author uses a term vs. general English:
    ratio = (freq_in_author / total_author) / (freq_in_reference / total_reference)

High ratio (e.g., 50x) means the author overuses the term compared to Brown corpus.
Example: If "abstraction" appears 10 times in a 1000-word author corpus but only
1 time per million words in Brown, the ratio would be ~10,000x.

### 2. TF-IDF (Term Frequency - Inverse Document Frequency)
TF-IDF measures how characteristic a term is to a specific document/corpus by
combining two signals:

**TF (Term Frequency):** How often the term appears in the author's work
    TF = count_in_author / total_words_in_author

**IDF (Inverse Document Frequency):** How rare the term is across all corpora
    IDF = log(total_corpora / corpora_containing_term)

**TF-IDF Score = TF × IDF**

- High TF-IDF: Term is both frequent in author AND rare in general English
- Low TF-IDF: Term is either rare in author OR common everywhere
- Zero TF-IDF: Term appears in all corpora (e.g., "the", "is")

Example: "différance" in Derrida's work:
- High TF (appears frequently in Derrida)
- High IDF (absent from Brown corpus)
- High TF-IDF → Strong signal of Derrida-specific term

Example: "the" in any text:
- High TF (appears frequently)
- Low IDF (appears in all corpora)
- Low TF-IDF → Common word, not distinctive

### 3. Neologism Detection
Identifies terms completely absent from reference corpus - strong candidates for
author-invented technical terminology.

### Combined Method
Uses union or intersection of multiple methods for robust detection. Intersection
provides high-confidence terms (agree across multiple signals).
"""

import math
import re
from collections import Counter
from typing import Dict, Set, List, Tuple, Optional
import string
from ..corpus.models import ProcessedDocument
from .frequency import corpus_frequencies

try:
    from nltk.corpus import wordnet
except ImportError:
    wordnet = None


def compare_to_reference(
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    use_lemmas: bool = True,
    min_author_freq: int = 3,
) -> Dict[str, float]:
    """
    Calculate relative frequency ratios comparing author corpus to reference.

    The ratio indicates how much more (or less) the author uses a term
    compared to general English:
        ratio = (freq_author / total_author) / (freq_reference / total_reference)

    High ratio (>1) = author overuses term vs. general English
    Low ratio (<1) = author underuses term vs. general English

    Args:
        docs: List of preprocessed documents
        reference_corpus: Reference corpus frequency distribution (e.g., Brown)
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_author_freq: Minimum occurrences in author corpus (filters noise)

    Returns:
        Dictionary mapping terms to their relative frequency ratios
    """
    # Get author corpus frequencies
    author_freqs = corpus_frequencies(docs, use_lemmas=use_lemmas)
    total_author = sum(author_freqs.values())
    total_reference = sum(reference_corpus.values())

    ratios = {}
    for term, author_count in author_freqs.items():
        # Skip terms below minimum frequency threshold
        if author_count < min_author_freq:
            continue

        # Get reference frequency (0 if term not in reference)
        ref_count = reference_corpus.get(term, 0)

        # Calculate normalized frequencies
        author_norm = author_count / total_author
        ref_norm = ref_count / total_reference if ref_count > 0 else 0

        # Calculate ratio
        # If term not in reference, use pseudocount to avoid division by zero
        if ref_norm == 0:
            # Use pseudocount of 0.5 occurrences in reference corpus
            ref_norm = 0.5 / total_reference
            ratios[term] = author_norm / ref_norm
        else:
            ratios[term] = author_norm / ref_norm

    return ratios


def get_corpus_specific_terms(
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    threshold: float = 10.0,
    use_lemmas: bool = True,
    min_author_freq: int = 3,
    min_reference_freq: int = None,
) -> Set[str]:
    """
    Extract terms distinctive to the author corpus based on ratio threshold.

    Filters for terms that are:
    1. Used frequently enough by author (>= min_author_freq)
    2. Significantly overrepresented vs. reference (ratio >= threshold)
    3. Optionally: rare or absent in reference corpus

    Args:
        docs: List of preprocessed documents
        reference_corpus: Reference corpus frequency distribution
        threshold: Minimum ratio to consider term corpus-specific (default: 10.0)
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_author_freq: Minimum occurrences in author corpus (default: 3)
        min_reference_freq: Maximum occurrences in reference (None = no limit)

    Returns:
        Set of corpus-specific terms
    """
    ratios = compare_to_reference(
        docs, reference_corpus, use_lemmas=use_lemmas, min_author_freq=min_author_freq
    )

    specific_terms = set()
    for term, ratio in ratios.items():
        # Check ratio threshold
        if ratio >= threshold:
            # Optionally filter by reference frequency
            if min_reference_freq is not None:
                ref_count = reference_corpus.get(term, 0)
                if ref_count <= min_reference_freq:
                    specific_terms.add(term)
            else:
                specific_terms.add(term)

    return specific_terms


def get_top_corpus_specific_terms(
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    n: int = 50,
    use_lemmas: bool = True,
    min_author_freq: int = 3,
) -> List[Tuple[str, float]]:
    """
    Get top N corpus-specific terms ranked by relative frequency ratio.

    Args:
        docs: List of preprocessed documents
        reference_corpus: Reference corpus frequency distribution
        n: Number of top terms to return (default: 50)
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_author_freq: Minimum occurrences in author corpus (default: 3)

    Returns:
        List of (term, ratio) tuples, sorted by ratio descending
    """
    ratios = compare_to_reference(
        docs, reference_corpus, use_lemmas=use_lemmas, min_author_freq=min_author_freq
    )

    # Sort by ratio descending and take top N
    sorted_terms = sorted(ratios.items(), key=lambda x: x[1], reverse=True)
    return sorted_terms[:n]


def get_neologism_candidates(
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    use_lemmas: bool = True,
    min_author_freq: int = 3,
) -> Set[str]:
    """
    Find potential neologisms: terms in author corpus but not in reference.

    These are strong candidates for author-invented technical terminology.

    Args:
        docs: List of preprocessed documents
        reference_corpus: Reference corpus frequency distribution
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_author_freq: Minimum occurrences in author corpus (default: 3)

    Returns:
        Set of potential neologism terms
    """
    author_freqs = corpus_frequencies(docs, use_lemmas=use_lemmas)

    neologisms = set()
    for term, count in author_freqs.items():
        if count >= min_author_freq and term not in reference_corpus:
            neologisms.add(term)

    return neologisms


def get_term_context_stats(
    term: str,
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    use_lemmas: bool = True,
) -> Dict[str, any]:
    """
    Get comprehensive statistics for a specific term.

    Useful for understanding why a term was flagged as corpus-specific.

    Args:
        term: The term to analyze
        docs: List of preprocessed documents
        reference_corpus: Reference corpus frequency distribution
        use_lemmas: Whether to use lemmatized forms (default: True)

    Returns:
        Dictionary with statistics:
            - author_count: Raw count in author corpus
            - author_freq: Normalized frequency in author corpus
            - reference_count: Raw count in reference corpus
            - reference_freq: Normalized frequency in reference corpus
            - ratio: Relative frequency ratio
            - documents_containing: Number of documents containing term
    """
    author_freqs = corpus_frequencies(docs, use_lemmas=use_lemmas)
    total_author = sum(author_freqs.values())
    total_reference = sum(reference_corpus.values())

    author_count = author_freqs.get(term, 0)
    reference_count = reference_corpus.get(term, 0)

    author_freq = author_count / total_author if total_author > 0 else 0
    reference_freq = reference_count / total_reference if total_reference > 0 else 0

    # Calculate ratio
    if reference_freq > 0:
        ratio = author_freq / reference_freq
    elif author_count > 0:
        # Pseudocount for terms not in reference
        pseudo_ref_freq = 0.5 / total_reference
        ratio = author_freq / pseudo_ref_freq
    else:
        ratio = 0.0

    # Count documents containing term
    docs_containing = 0
    for doc in docs:
        tokens = doc.lemmas if use_lemmas else doc.tokens
        if term in tokens:
            docs_containing += 1

    return {
        "author_count": author_count,
        "author_freq": author_freq,
        "reference_count": reference_count,
        "reference_freq": reference_freq,
        "ratio": ratio,
        "documents_containing": docs_containing,
        "in_reference": reference_count > 0,
    }


def tfidf_vs_reference(
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    use_lemmas: bool = True,
    min_author_freq: int = 3,
) -> Dict[str, float]:
    """
    Calculate TF-IDF scores treating author corpus as document, reference as background.

    This is an alternative measure of distinctiveness that combines:
    - TF: How much the author uses the term (normalized)
    - IDF: How rare the term is in general English

    High TF-IDF = term characteristic of author's usage pattern.

    Args:
        docs: List of preprocessed documents
        reference_corpus: Reference corpus frequency distribution
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_author_freq: Minimum occurrences in author corpus (default: 3)

    Returns:
        Dictionary mapping terms to TF-IDF scores
    """
    author_freqs = corpus_frequencies(docs, use_lemmas=use_lemmas)
    total_author = sum(author_freqs.values())

    # Number of "documents" = 2 (author corpus + reference corpus)
    num_documents = 2

    tfidf_scores = {}
    for term, author_count in author_freqs.items():
        if author_count < min_author_freq:
            continue

        # TF: term frequency in author corpus (normalized)
        tf = author_count / total_author

        # IDF: inverse document frequency
        # Count how many "documents" contain the term
        docs_containing = 1  # Always in author corpus (we're iterating author terms)
        if reference_corpus.get(term, 0) > 0:
            docs_containing += 1  # Also in reference

        # IDF = log(total_documents / documents_containing_term)
        idf = math.log(num_documents / docs_containing)

        # TF-IDF score
        tfidf_scores[term] = tf * idf

    return tfidf_scores


def get_top_tfidf_terms(
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    n: int = 50,
    use_lemmas: bool = True,
    min_author_freq: int = 3,
) -> List[Tuple[str, float]]:
    """
    Get top N terms ranked by TF-IDF against reference corpus.

    Terms with high TF-IDF are both:
    1. Frequently used by the author (high TF)
    2. Rare or absent in reference corpus (high IDF)

    Args:
        docs: List of preprocessed documents
        reference_corpus: Reference corpus frequency distribution
        n: Number of top terms to return (default: 50)
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_author_freq: Minimum occurrences in author corpus (default: 3)

    Returns:
        List of (term, tfidf_score) tuples, sorted by score descending
    """
    scores = tfidf_vs_reference(
        docs, reference_corpus, use_lemmas=use_lemmas, min_author_freq=min_author_freq
    )

    # Sort by score descending and take top N
    sorted_terms = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_terms[:n]


def get_distinctive_by_tfidf(
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    threshold: float = 0.001,
    use_lemmas: bool = True,
    min_author_freq: int = 3,
) -> Set[str]:
    """
    Extract distinctive terms using TF-IDF threshold.

    Args:
        docs: List of preprocessed documents
        reference_corpus: Reference corpus frequency distribution
        threshold: Minimum TF-IDF score to consider distinctive (default: 0.001)
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_author_freq: Minimum occurrences in author corpus (default: 3)

    Returns:
        Set of distinctive terms
    """
    scores = tfidf_vs_reference(
        docs, reference_corpus, use_lemmas=use_lemmas, min_author_freq=min_author_freq
    )

    distinctive = {term for term, score in scores.items() if score >= threshold}
    return distinctive


def get_combined_distinctive_terms(
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    ratio_threshold: float = 10.0,
    tfidf_threshold: float = 0.001,
    use_lemmas: bool = True,
    min_author_freq: int = 3,
    method: str = "union",
) -> Set[str]:
    """
    Combine multiple rarity detection methods for robust term extraction.

    Args:
        docs: List of preprocessed documents
        reference_corpus: Reference corpus frequency distribution
        ratio_threshold: Minimum relative frequency ratio (default: 10.0)
        tfidf_threshold: Minimum TF-IDF score (default: 0.001)
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_author_freq: Minimum occurrences in author corpus (default: 3)
        method: How to combine results:
            - "union": Return terms found by ANY method
            - "intersection": Return terms found by ALL methods
            - "ratio_only": Use only relative frequency ratio
            - "tfidf_only": Use only TF-IDF

    Returns:
        Set of distinctive terms
    """
    if method == "ratio_only":
        return get_corpus_specific_terms(
            docs,
            reference_corpus,
            threshold=ratio_threshold,
            use_lemmas=use_lemmas,
            min_author_freq=min_author_freq,
        )

    if method == "tfidf_only":
        return get_distinctive_by_tfidf(
            docs,
            reference_corpus,
            threshold=tfidf_threshold,
            use_lemmas=use_lemmas,
            min_author_freq=min_author_freq,
        )

    # Get terms from both methods
    ratio_terms = get_corpus_specific_terms(
        docs,
        reference_corpus,
        threshold=ratio_threshold,
        use_lemmas=use_lemmas,
        min_author_freq=min_author_freq,
    )

    tfidf_terms = get_distinctive_by_tfidf(
        docs,
        reference_corpus,
        threshold=tfidf_threshold,
        use_lemmas=use_lemmas,
        min_author_freq=min_author_freq,
    )

    if method == "union":
        return ratio_terms | tfidf_terms
    elif method == "intersection":
        return ratio_terms & tfidf_terms
    else:
        raise ValueError(
            f"Invalid method: {method}. Must be 'union', 'intersection', 'ratio_only', or 'tfidf_only'"
        )


def _load_wordnet_vocabulary() -> Set[str]:
    """
    Load all lemmas from WordNet as a baseline English dictionary.

    Returns:
        Set of all lemma names in WordNet (lowercase)

    Raises:
        ImportError: If WordNet is not available
    """
    if wordnet is None:
        raise ImportError(
            "WordNet is not available. Please download it with: "
            "python -c 'import nltk; nltk.download(\"wordnet\")'"
        )

    vocabulary = set()
    for synset in wordnet.all_synsets():
        for lemma in synset.lemmas():
            # Add lemma name in lowercase
            vocabulary.add(lemma.name().lower().replace("_", " "))

    return vocabulary


def get_wordnet_neologisms(
    docs: List[ProcessedDocument],
    use_lemmas: bool = True,
    min_author_freq: int = 3,
    exclude_proper_nouns: bool = True,
    exclude_stopwords: bool = True,
) -> Set[str]:
    """
    Identify potential neologisms: terms not in WordNet dictionary.

    WordNet is a comprehensive English lexicon covering ~117K word-sense pairs.
    Terms absent from WordNet are strong candidates for:
    - Author-invented philosophical terminology
    - Technical neologisms
    - Specialized domain vocabulary

    Note: WordNet focuses on content words (nouns, verbs, adjectives, adverbs)
    and may not include common function words. Use exclude_stopwords=True to
    filter these out.

    Args:
        docs: List of preprocessed documents
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_author_freq: Minimum occurrences in author corpus (default: 3)
        exclude_proper_nouns: Filter out proper nouns (NNP, NNPS tags) (default: True)
        exclude_stopwords: Filter out common function words (default: True)

    Returns:
        Set of potential neologism terms

    Raises:
        ImportError: If WordNet is not available
    """
    # Load WordNet vocabulary
    wordnet_vocab = _load_wordnet_vocabulary()

    # Get author corpus frequencies
    author_freqs = corpus_frequencies(docs, use_lemmas=use_lemmas)

    # Common stopwords/function words (often not in WordNet)
    stopwords_set = set()
    if exclude_stopwords:
        try:
            from nltk.corpus import stopwords

            stopwords_set = set(stopwords.words("english"))
        except Exception:
            # Minimal fallback if stopwords not available
            stopwords_set = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "if",
                "of",
                "at",
                "by",
                "for",
                "with",
                "to",
                "from",
                "in",
                "on",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
            }

    # Identify terms not in WordNet
    neologisms = set()
    for term, count in author_freqs.items():
        if count < min_author_freq:
            continue

        # Filter stopwords
        if exclude_stopwords and term.lower() in stopwords_set:
            continue

        # Check if term is in WordNet
        if term.lower() not in wordnet_vocab:
            # Optionally filter out proper nouns
            if exclude_proper_nouns:
                # Check if term appears as proper noun in any document
                is_proper = False
                for doc in docs:
                    if any(
                        word == term and pos.startswith("NNP")
                        for word, pos in doc.pos_tags
                    ):
                        is_proper = True
                        break

                if not is_proper:
                    neologisms.add(term)
            else:
                neologisms.add(term)

    return neologisms


def get_capitalized_technical_terms(
    docs: List[ProcessedDocument],
    min_author_freq: int = 3,
    exclude_sentence_initial: bool = True,
    exclude_all_caps: bool = True,
    exclude_proper_nouns: bool = False,
) -> Set[str]:
    """
    Identify capitalized terms that may indicate reified philosophical abstractions.

    Philosophers often capitalize abstract concepts to signal technical usage:
    - Kant: "Theory", "Practice", "Judgment"
    - Hegel: "Logic", "Nature", "Spirit"

    Note: POS taggers often misclassify capitalized philosophical terms as proper
    nouns. Setting exclude_proper_nouns=True will filter both person names AND
    capitalized concepts. Use with caution.

    Args:
        docs: List of preprocessed documents
        min_author_freq: Minimum occurrences in author corpus (default: 3)
        exclude_sentence_initial: Exclude words at sentence start (default: True)
        exclude_all_caps: Exclude words in all capitals like "NATO" (default: True)
        exclude_proper_nouns: Filter terms tagged as NNP/NNPS (default: False)
            WARNING: May also filter capitalized philosophical concepts!

    Returns:
        Set of capitalized technical terms (mid-sentence capitalizations)
    """
    capitalized_terms = Counter()

    # Build set of proper nouns if filtering is enabled
    proper_nouns = set()
    if exclude_proper_nouns:
        for doc in docs:
            for token, pos in doc.pos_tags:
                if pos.startswith("NNP"):
                    proper_nouns.add(token)

    for doc in docs:
        # Work with tokens directly for accurate capitalization info
        for i, token in enumerate(doc.tokens):
            # Skip non-alphabetic tokens
            if not token or not token[0].isalpha():
                continue

            # Check if capitalized
            if token[0].isupper():
                # Exclude all-caps words (likely acronyms)
                if exclude_all_caps and token.isupper() and len(token) > 1:
                    continue

                # Filter proper nouns if requested
                if exclude_proper_nouns and token in proper_nouns:
                    continue

                # Determine if this is sentence-initial
                is_sentence_initial = False
                if i == 0:
                    is_sentence_initial = True
                elif i > 0:
                    # Check if previous token is sentence-ending punctuation
                    prev_token = doc.tokens[i - 1]
                    if prev_token in {".", "!", "?", ";"}:
                        is_sentence_initial = True

                # Skip sentence-initial if flag is set
                if exclude_sentence_initial and is_sentence_initial:
                    continue

                # Count this capitalized term
                capitalized_terms[token] += 1

    # Filter by minimum frequency
    return {
        term for term, count in capitalized_terms.items() if count >= min_author_freq
    }


def get_potential_neologisms(
    docs: List[ProcessedDocument],
    dictionary: Optional[Set[str]] = None,
    use_lemmas: bool = True,
    min_author_freq: int = 3,
    exclude_proper_nouns: bool = True,
) -> Set[str]:
    """
    Identify potential neologisms using custom or WordNet dictionary.

    This is a general-purpose neologism detector that can use either:
    1. WordNet as baseline English dictionary (default)
    2. Custom dictionary (e.g., domain-specific lexicon)

    Args:
        docs: List of preprocessed documents
        dictionary: Optional custom dictionary set. If None, uses WordNet (default: None)
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_author_freq: Minimum occurrences in author corpus (default: 3)
        exclude_proper_nouns: Filter out proper nouns (default: True)

    Returns:
        Set of potential neologism terms

    Raises:
        ImportError: If WordNet is not available and no dictionary provided
    """
    if dictionary is None:
        # Use WordNet as default dictionary
        return get_wordnet_neologisms(
            docs,
            use_lemmas=use_lemmas,
            min_author_freq=min_author_freq,
            exclude_proper_nouns=exclude_proper_nouns,
        )

    # Use custom dictionary
    author_freqs = corpus_frequencies(docs, use_lemmas=use_lemmas)

    neologisms = set()
    for term, count in author_freqs.items():
        if count < min_author_freq:
            continue

        # Check if term is in dictionary
        if term.lower() not in dictionary:
            # Optionally filter out proper nouns
            if exclude_proper_nouns:
                is_proper = False
                for doc in docs:
                    if any(
                        word == term and pos.startswith("NNP")
                        for word, pos in doc.pos_tags
                    ):
                        is_proper = True
                        break

                if not is_proper:
                    neologisms.add(term)
            else:
                neologisms.add(term)

    return neologisms


def get_all_neologism_signals(
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    use_lemmas: bool = True,
    min_author_freq: int = 3,
) -> Dict[str, Set[str]]:
    """
    Get neologism candidates from all detection methods.

    Combines three complementary signals:
    1. Absent from reference corpus (Brown)
    2. Absent from WordNet dictionary
    3. Capitalized mid-sentence (reified abstractions)

    Args:
        docs: List of preprocessed documents
        reference_corpus: Reference corpus frequency distribution
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_author_freq: Minimum occurrences in author corpus (default: 3)

    Returns:
        Dictionary with three sets:
            - "reference": Terms not in reference corpus
            - "wordnet": Terms not in WordNet
            - "capitalized": Mid-sentence capitalized terms
            - "all_neologisms": Union of all three sets
            - "high_confidence": Terms found by multiple methods
    """
    # Get candidates from each method
    from_reference = get_neologism_candidates(
        docs, reference_corpus, use_lemmas=use_lemmas, min_author_freq=min_author_freq
    )

    try:
        from_wordnet = get_wordnet_neologisms(
            docs, use_lemmas=use_lemmas, min_author_freq=min_author_freq
        )
    except ImportError:
        from_wordnet = set()

    capitalized = get_capitalized_technical_terms(docs, min_author_freq=min_author_freq)

    # Combine signals
    all_neologisms = from_reference | from_wordnet | capitalized

    # High confidence: terms found by at least 2 methods
    high_confidence = set()
    for term in all_neologisms:
        count = sum(
            [
                term in from_reference,
                term in from_wordnet,
                term in capitalized,
            ]
        )
        if count >= 2:
            high_confidence.add(term)

    return {
        "reference": from_reference,
        "wordnet": from_wordnet,
        "capitalized": capitalized,
        "all_neologisms": all_neologisms,
        "high_confidence": high_confidence,
    }


# Definitional context extraction patterns
DEFINITIONAL_PATTERNS = [
    # "X is Y" - copular definitions
    (r"\b(\w+(?:-\w+)*)\s+(?:is|are|was|were)\s+", "copular"),
    # "by X I/we mean" - explicit definition
    (r"\bby\s+(\w+(?:-\w+)*)\s+(?:I|we|one)\s+mean", "explicit_mean"),
    # "what I/we call X" - metalinguistic
    (r"\bwhat\s+(?:I|we)\s+call\s+(\w+(?:-\w+)*)", "metalinguistic"),
    # "the concept/notion/idea of X" - conceptual framing
    (
        r"\b(?:the\s+)?(?:concept|notion|idea|category|principle)\s+of\s+(\w+(?:-\w+)*)",
        "conceptual",
    ),
    # "X, which is" - appositive definition
    (r"\b(\w+(?:-\w+)*),?\s+which\s+(?:is|are|means)", "appositive"),
    # "I/we define X as" - explicit definition
    (r"\b(?:I|we)\s+define\s+(\w+(?:-\w+)*)\s+as", "explicit_define"),
    # "X refers to" - referential definition
    (r"\b(\w+(?:-\w+)*)\s+refers?\s+to", "referential"),
    # "X can be understood as" - interpretive definition
    (r"\b(\w+(?:-\w+)*)\s+can\s+be\s+understood\s+as", "interpretive"),
]


def get_definitional_contexts(
    docs: List[ProcessedDocument],
    patterns: Optional[List[Tuple[str, str]]] = None,
    case_sensitive: bool = False,
) -> List[Tuple[str, str, str, str]]:
    """
    Extract sentences where terms are explicitly defined.

    Philosophers often introduce technical terminology with explicit definitions:
    - "Dasein is being-in-the-world"
    - "By abstraction I mean the objectification of social relations"
    - "What I call différance is neither a word nor a concept"
    - "The concept of Being refers to..."

    Args:
        docs: List of preprocessed documents
        patterns: Optional custom list of (regex_pattern, pattern_type) tuples
            If None, uses default DEFINITIONAL_PATTERNS
        case_sensitive: Whether pattern matching is case-sensitive (default: False)

    Returns:
        List of (term, sentence, pattern_type, doc_id) tuples
    """
    if patterns is None:
        patterns = DEFINITIONAL_PATTERNS

    definitional_contexts = []
    flags = 0 if case_sensitive else re.IGNORECASE

    for doc_idx, doc in enumerate(docs):
        doc_id = doc.metadata.get("source_path", f"doc_{doc_idx}")

        for sentence in doc.sentences:
            # Try each pattern
            for pattern, pattern_type in patterns:
                matches = re.finditer(pattern, sentence, flags)
                for match in matches:
                    # Extract the term (first capture group)
                    term = match.group(1)
                    # Store context
                    definitional_contexts.append((term, sentence, pattern_type, doc_id))

    return definitional_contexts


def score_by_definitional_context(
    docs: List[ProcessedDocument],
    terms: Optional[Set[str]] = None,
    patterns: Optional[List[Tuple[str, str]]] = None,
    case_sensitive: bool = False,
) -> Dict[str, int]:
    """
    Score terms by how often they appear in definitional contexts.

    Higher scores indicate terms that receive explicit authorial attention and
    definition, suggesting they are conceptually important.

    Args:
        docs: List of preprocessed documents
        terms: Optional set of terms to score. If None, scores all terms found
        patterns: Optional custom definitional patterns
        case_sensitive: Whether matching is case-sensitive (default: False)

    Returns:
        Dictionary mapping terms to definitional context counts
    """
    contexts = get_definitional_contexts(docs, patterns, case_sensitive)

    # Count definitional occurrences per term
    scores = Counter()
    for term, sentence, pattern_type, doc_id in contexts:
        # Normalize term if not case-sensitive
        normalized_term = term if case_sensitive else term.lower()

        # Only score if in term set (if provided)
        if terms is None or normalized_term in terms:
            scores[normalized_term] += 1

    return dict(scores)


def get_definitional_sentences(
    term: str,
    docs: List[ProcessedDocument],
    patterns: Optional[List[Tuple[str, str]]] = None,
    case_sensitive: bool = False,
) -> List[Tuple[str, str]]:
    """
    Get all definitional sentences for a specific term.

    Useful for understanding how an author defines and uses a technical term.

    Args:
        term: The term to search for
        docs: List of preprocessed documents
        patterns: Optional custom definitional patterns
        case_sensitive: Whether matching is case-sensitive (default: False)

    Returns:
        List of (sentence, pattern_type) tuples containing the term's definitions
    """
    contexts = get_definitional_contexts(docs, patterns, case_sensitive)

    # Filter for this specific term
    term_normalized = term if case_sensitive else term.lower()
    definitional_sentences = []

    for found_term, sentence, pattern_type, doc_id in contexts:
        found_term_normalized = found_term if case_sensitive else found_term.lower()
        if found_term_normalized == term_normalized:
            definitional_sentences.append((sentence, pattern_type))

    return definitional_sentences


def get_highly_defined_terms(
    docs: List[ProcessedDocument],
    min_definitions: int = 2,
    terms: Optional[Set[str]] = None,
    patterns: Optional[List[Tuple[str, str]]] = None,
) -> Set[str]:
    """
    Get terms that appear in multiple definitional contexts.

    Terms defined multiple times are likely to be central technical concepts
    that the author considers important enough to explain repeatedly.

    Args:
        docs: List of preprocessed documents
        min_definitions: Minimum number of definitional contexts (default: 2)
        terms: Optional set of candidate terms to consider
        patterns: Optional custom definitional patterns

    Returns:
        Set of terms appearing in >= min_definitions definitional contexts
    """
    scores = score_by_definitional_context(docs, terms, patterns)

    highly_defined = {
        term for term, count in scores.items() if count >= min_definitions
    }

    return highly_defined


def analyze_definitional_patterns(
    docs: List[ProcessedDocument],
) -> Dict[str, int]:
    """
    Analyze which types of definitional patterns appear most frequently.

    Useful for understanding the author's definitional style.

    Args:
        docs: List of preprocessed documents

    Returns:
        Dictionary mapping pattern types to occurrence counts
    """
    contexts = get_definitional_contexts(docs)

    pattern_counts = Counter()
    for term, sentence, pattern_type, doc_id in contexts:
        pattern_counts[pattern_type] += 1

    return dict(pattern_counts)


def get_terms_with_definitions(
    docs: List[ProcessedDocument],
    candidate_terms: Set[str],
    patterns: Optional[List[Tuple[str, str]]] = None,
) -> Dict[str, List[str]]:
    """
    For each candidate term, retrieve all its definitional sentences.

    Useful for building a glossary or understanding term usage.

    Args:
        docs: List of preprocessed documents
        candidate_terms: Set of terms to look up
        patterns: Optional custom definitional patterns

    Returns:
        Dictionary mapping terms to lists of definitional sentences
    """
    results = {}

    for term in candidate_terms:
        sentences = get_definitional_sentences(term, docs, patterns)
        if sentences:
            # Extract just the sentences (ignore pattern types)
            results[term] = [sent for sent, pattern_type in sentences]

    return results


# POS filtering for philosophical term candidates
CONTENT_WORD_POS_TAGS = {
    "NN",
    "NNS",
    "NNP",
    "NNPS",  # Nouns
    "VB",
    "VBD",
    "VBG",
    "VBN",
    "VBP",
    "VBZ",  # Verbs
    "JJ",
    "JJR",
    "JJS",  # Adjectives
    "RB",
    "RBR",
    "RBS",  # Adverbs
}

FUNCTION_WORD_POS_TAGS = {
    "DT",  # Determiner (the, a, an)
    "IN",  # Preposition/subordinating conjunction (in, of, for)
    "CC",  # Coordinating conjunction (and, or, but)
    "TO",  # to
    "PRP",
    "PRP$",  # Personal/possessive pronouns (I, he, his)
    "WDT",
    "WP",
    "WP$",
    "WRB",  # Wh- words (which, who, where)
    "EX",  # Existential there
    "MD",  # Modal (can, will, should)
    "PDT",  # Predeterminer (all, both)
    "POS",  # Possessive ending ('s)
    "RP",  # Particle (up, off)
    "SYM",  # Symbol
    "UH",  # Interjection (oh, uh)
}


def filter_by_pos_tags(
    docs: List[ProcessedDocument],
    include_tags: Optional[Set[str]] = None,
    exclude_tags: Optional[Set[str]] = None,
    use_lemmas: bool = True,
    min_freq: int = 1,
) -> Set[str]:
    """
    Filter corpus vocabulary by POS tags.

    Args:
        docs: List of preprocessed documents
        include_tags: POS tags to include (default: CONTENT_WORD_POS_TAGS)
        exclude_tags: POS tags to exclude (default: FUNCTION_WORD_POS_TAGS)
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_freq: Minimum frequency threshold (default: 1)

    Returns:
        Set of terms matching POS criteria
    """
    if include_tags is None:
        include_tags = CONTENT_WORD_POS_TAGS

    if exclude_tags is None:
        exclude_tags = FUNCTION_WORD_POS_TAGS

    # Collect terms with their POS tags
    term_pos_map = {}
    for doc in docs:
        for i, (token, pos) in enumerate(doc.pos_tags):
            term = doc.lemmas[i] if use_lemmas and i < len(doc.lemmas) else token

            # Track which POS tags this term appears with
            if term not in term_pos_map:
                term_pos_map[term] = set()
            term_pos_map[term].add(pos)

    # Filter by POS criteria
    filtered_terms = set()
    for term, pos_tags in term_pos_map.items():
        # Check if any of the term's POS tags match include criteria
        if include_tags and not pos_tags.intersection(include_tags):
            continue

        # Check if any of the term's POS tags should be excluded
        if exclude_tags and pos_tags.intersection(exclude_tags):
            continue

        filtered_terms.add(term)

    # Apply frequency filter if needed
    if min_freq > 1:
        term_freqs = corpus_frequencies(docs, use_lemmas=use_lemmas)
        filtered_terms = {
            term for term in filtered_terms if term_freqs.get(term, 0) >= min_freq
        }

    return filtered_terms


def get_philosophical_term_candidates(
    docs: List[ProcessedDocument],
    focus: str = "nouns",
    use_lemmas: bool = True,
    min_freq: int = 3,
    exclude_stopwords: bool = True,
) -> Set[str]:
    """
    Get candidate philosophical terms filtered by POS and frequency.

    Focuses on content words that are likely to be technical philosophical terms.

    Args:
        docs: List of preprocessed documents
        focus: Which word types to focus on (default: "nouns")
            - "nouns": Nouns and noun phrases
            - "verbs": Verbs
            - "adjectives": Adjectives
            - "all_content": All content words (nouns, verbs, adjectives, adverbs)
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_freq: Minimum frequency threshold (default: 3)
        exclude_stopwords: Filter out common stopwords (default: True)

    Returns:
        Set of candidate philosophical terms
    """
    # Define POS tag sets based on focus
    if focus == "nouns":
        include_tags = {"NN", "NNS", "NNP", "NNPS"}
    elif focus == "verbs":
        include_tags = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    elif focus == "adjectives":
        include_tags = {"JJ", "JJR", "JJS"}
    elif focus == "all_content":
        include_tags = CONTENT_WORD_POS_TAGS
    else:
        raise ValueError(
            f"Invalid focus: {focus}. Must be 'nouns', 'verbs', 'adjectives', or 'all_content'"
        )

    # Filter by POS
    candidates = filter_by_pos_tags(
        docs,
        include_tags=include_tags,
        exclude_tags=FUNCTION_WORD_POS_TAGS,
        use_lemmas=use_lemmas,
        min_freq=min_freq,
    )

    # Optionally filter stopwords
    if exclude_stopwords:
        try:
            from nltk.corpus import stopwords

            stop_set = set(stopwords.words("english"))
            candidates = {term for term in candidates if term.lower() not in stop_set}
        except Exception:
            # Fallback to minimal stopwords
            minimal_stops = {
                "the",
                "a",
                "an",
                "and",
                "or",
                "but",
                "if",
                "then",
                "than",
                "of",
                "at",
                "by",
                "for",
                "with",
                "about",
                "to",
                "from",
                "in",
                "on",
                "is",
                "are",
                "was",
                "were",
                "be",
                "been",
                "being",
                "have",
                "has",
                "had",
                "do",
                "does",
                "did",
                "will",
                "would",
                "could",
                "should",
                "may",
                "might",
                "must",
                "can",
                "shall",
            }
            candidates = {
                term for term in candidates if term.lower() not in minimal_stops
            }

    return candidates


def get_compound_terms(
    docs: List[ProcessedDocument],
    min_freq: int = 2,
    max_length: int = 4,
) -> Set[str]:
    """
    Extract compound terms (multi-word phrases) from corpus.

    Philosophical texts often use compound terms like "being-in-the-world",
    "body without organs", "intentional stance", etc.

    This function identifies hyphenated compounds and noun phrases.

    Args:
        docs: List of preprocessed documents
        min_freq: Minimum frequency for compound term (default: 2)
        max_length: Maximum number of words in compound (default: 4)

    Returns:
        Set of compound terms
    """
    compounds = Counter()

    for doc in docs:
        # Extract hyphenated compounds
        for token in doc.tokens:
            if "-" in token and len(token) > 3:
                # Skip tokens that are just punctuation
                if not any(c.isalpha() for c in token):
                    continue
                compounds[token] += 1

        # Extract noun phrases (consecutive nouns/adjectives + noun)
        for i in range(len(doc.pos_tags)):
            for length in range(2, min(max_length + 1, len(doc.pos_tags) - i + 1)):
                # Get sequence of tokens
                sequence_tokens = [doc.tokens[i + j] for j in range(length)]
                sequence_pos = [doc.pos_tags[i + j][1] for j in range(length)]

                # Check if it's a noun phrase (adjectives/nouns ending in noun)
                is_noun_phrase = all(
                    pos.startswith(("NN", "JJ")) for pos in sequence_pos
                ) and sequence_pos[-1].startswith("NN")

                if is_noun_phrase:
                    compound = " ".join(sequence_tokens)
                    compounds[compound] += 1

    # Filter by frequency
    return {term for term, count in compounds.items() if count >= min_freq}


def get_filtered_candidates(
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    use_lemmas: bool = True,
    min_freq: int = 3,
    include_pos_focus: str = "all_content",
    include_compounds: bool = True,
) -> Dict[str, Set[str]]:
    """
    Get comprehensive set of philosophical term candidates with multiple filters.

    Combines POS filtering, frequency filtering, and optional compound extraction.

    Args:
        docs: List of preprocessed documents
        reference_corpus: Reference corpus for rarity detection
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_freq: Minimum frequency threshold (default: 3)
        include_pos_focus: POS focus type (default: "all_content")
        include_compounds: Whether to include compound terms (default: True)

    Returns:
        Dictionary with candidate sets:
            - "single_words": Single-word candidates filtered by POS
            - "compounds": Multi-word/hyphenated compounds (if enabled)
            - "all_candidates": Union of all candidates
    """
    # Get single-word candidates
    single_words = get_philosophical_term_candidates(
        docs,
        focus=include_pos_focus,
        use_lemmas=use_lemmas,
        min_freq=min_freq,
        exclude_stopwords=True,
    )

    # Get compound terms
    compounds = set()
    if include_compounds:
        compounds = get_compound_terms(docs, min_freq=min_freq)

    return {
        "single_words": single_words,
        "compounds": compounds,
        "all_candidates": single_words | compounds,
    }


# Phase 3.6: Hybrid Philosophical Term Scorer

def _is_valid_term(term: str) -> bool:
    """
    Check if a term is valid (not punctuation, minimum length, etc.).

    Args:
        term: Term to validate

    Returns:
        True if term is valid, False otherwise
    """
    # Empty or too short
    if not term or len(term) < 2:
        return False

    # All punctuation
    if all(c in string.punctuation for c in term):
        return False

    # Common abbreviations that shouldn't be terms
    if term.lower() in {'i.e', 'e.g', 'etc', 'vs', 'cf'}:
        return False

    # Must contain at least one letter
    if not any(c.isalpha() for c in term):
        return False

    return True


class PhilosophicalTermScorer:
    """
    Hybrid scorer combining multiple detection methods for philosophical terms.

    Combines five complementary signals:
    1. Relative frequency ratio (corpus-comparative analysis)
    2. TF-IDF score (term frequency × inverse document frequency)
    3. Neologism detection (boolean boost for terms not in reference/WordNet)
    4. Definitional contexts (explicit author definitions)
    5. Capitalization (mid-sentence capitals indicating abstraction)

    Scores are weighted combinations normalized to 0-1 range for each component.
    The total score is the sum of weighted components.

    Example usage:
        >>> scorer = PhilosophicalTermScorer(docs, brown_corpus)
        >>> score = scorer.score_term("dasein")
        >>> print(f"Total: {score['total']:.2f}, Ratio: {score['ratio']:.2f}")
        >>> top_terms = scorer.score_all(min_score=1.0, top_n=20)
    """

    def __init__(
        self,
        docs: List[ProcessedDocument],
        reference_corpus: Counter,
        use_lemmas: bool = True,
        min_author_freq: int = 3,
        weights: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize scorer with corpus and weighting parameters.

        Args:
            docs: List of preprocessed documents
            reference_corpus: Reference corpus frequency distribution
            use_lemmas: Whether to use lemmatized forms (default: True)
            min_author_freq: Minimum occurrences in author corpus (default: 3)
            weights: Optional custom weights for scoring components:
                - "ratio": Relative frequency ratio weight (default: 1.0)
                - "tfidf": TF-IDF weight (default: 1.0)
                - "neologism": Neologism boost weight (default: 0.5)
                - "definitional": Definitional context weight (default: 0.3)
                - "capitalized": Capitalization weight (default: 0.2)
        """
        self.docs = docs
        self.reference_corpus = reference_corpus
        self.use_lemmas = use_lemmas
        self.min_author_freq = min_author_freq

        # Default weights
        default_weights = {
            "ratio": 1.0,
            "tfidf": 1.0,
            "neologism": 0.5,
            "definitional": 0.3,
            "capitalized": 0.2,
        }
        self.weights = weights if weights is not None else default_weights

        # Precompute signals
        self._compute_signals()

    def _compute_signals(self):
        """Precompute all detection signals for efficiency."""
        # Signal 1: Relative frequency ratios
        self.ratios = compare_to_reference(
            self.docs,
            self.reference_corpus,
            use_lemmas=self.use_lemmas,
            min_author_freq=self.min_author_freq,
        )

        # Signal 2: TF-IDF scores
        self.tfidf_scores = tfidf_vs_reference(
            self.docs,
            self.reference_corpus,
            use_lemmas=self.use_lemmas,
            min_author_freq=self.min_author_freq,
        )

        # Signal 3: Neologisms (multiple sources)
        self.neologisms = get_all_neologism_signals(
            self.docs,
            self.reference_corpus,
            use_lemmas=self.use_lemmas,
            min_author_freq=self.min_author_freq,
        )

        # Signal 4: Definitional contexts
        self.definitional_scores = score_by_definitional_context(self.docs, terms=None)

        # Signal 5: Capitalized terms
        self.capitalized_terms = get_capitalized_technical_terms(
            self.docs, min_author_freq=self.min_author_freq
        )

        # Compute normalization factors
        self._compute_normalization()

    def _compute_normalization(self):
        """Compute normalization factors for each signal."""
        # Max ratio (for normalization)
        self.max_ratio = max(self.ratios.values()) if self.ratios else 1.0

        # Max TF-IDF (for normalization)
        self.max_tfidf = max(self.tfidf_scores.values()) if self.tfidf_scores else 1.0

        # Max definitional count
        self.max_definitional = (
            max(self.definitional_scores.values()) if self.definitional_scores else 1.0
        )

    def score_term(self, term: str, normalize: bool = True) -> Dict[str, float]:
        """
        Compute hybrid score for a single term with component breakdown.

        Args:
            term: The term to score
            normalize: Whether to normalize to 0-1 range (default: True)

        Returns:
            Dictionary with:
                - "total": Total weighted score
                - "ratio": Normalized ratio component
                - "tfidf": Normalized TF-IDF component
                - "neologism": Neologism boolean indicator (0 or 1)
                - "definitional": Normalized definitional count
                - "capitalized": Capitalization boolean indicator (0 or 1)
                - "raw_total": Raw score before normalization
        """
        # Component 1: Ratio score
        ratio = self.ratios.get(term, 0)
        ratio_norm = (
            ratio / self.max_ratio if normalize and self.max_ratio > 0 else ratio
        )
        ratio_score = ratio_norm * self.weights["ratio"]

        # Component 2: TF-IDF score
        tfidf = self.tfidf_scores.get(term, 0)
        tfidf_norm = (
            tfidf / self.max_tfidf if normalize and self.max_tfidf > 0 else tfidf
        )
        tfidf_score = tfidf_norm * self.weights["tfidf"]

        # Component 3: Neologism boost (binary signal)
        is_neologism = term in self.neologisms["all_neologisms"]
        neologism_score = (1.0 if is_neologism else 0.0) * self.weights["neologism"]

        # Component 4: Definitional context score
        def_count = self.definitional_scores.get(term, 0)
        def_norm = (
            def_count / self.max_definitional
            if normalize and self.max_definitional > 0
            else def_count
        )
        def_score = def_norm * self.weights["definitional"]

        # Component 5: Capitalization boost (binary signal)
        is_capitalized = term in self.capitalized_terms
        cap_score = (1.0 if is_capitalized else 0.0) * self.weights["capitalized"]

        # Total score
        total = ratio_score + tfidf_score + neologism_score + def_score + cap_score

        return {
            "total": total,
            "ratio": ratio_norm,
            "tfidf": tfidf_norm,
            "neologism": 1.0 if is_neologism else 0.0,
            "definitional": def_norm,
            "capitalized": 1.0 if is_capitalized else 0.0,
            "raw_total": ratio
            + tfidf
            + (1.0 if is_neologism else 0.0)
            + def_count
            + (1.0 if is_capitalized else 0.0),
        }

    def score_all(
        self,
        min_score: float = 0.0,
        top_n: Optional[int] = None,
    ) -> List[Tuple[str, float, Dict[str, float]]]:
        """
        Score all terms in corpus meeting minimum frequency threshold.

        Args:
            min_score: Minimum total score to include term (default: 0.0)
            top_n: Optional limit to top N terms (default: None = all terms)

        Returns:
            List of (term, total_score, components) tuples sorted by score descending
            where components is the full score breakdown dict from score_term()
        """
        # Get all terms from author corpus
        author_freqs = corpus_frequencies(self.docs, use_lemmas=self.use_lemmas)

        # Score each term
        scored_terms = []
        for term in author_freqs:
            # Skip invalid terms (punctuation, too short, etc.)
            if not _is_valid_term(term):
                continue

            if author_freqs[term] >= self.min_author_freq:
                score_breakdown = self.score_term(term, normalize=True)
                total_score = score_breakdown["total"]

                if total_score >= min_score:
                    scored_terms.append((term, total_score, score_breakdown))

        # Sort by total score descending
        scored_terms.sort(key=lambda x: x[1], reverse=True)

        # Limit to top N if specified
        if top_n is not None:
            scored_terms = scored_terms[:top_n]

        return scored_terms

    def get_high_confidence_terms(
        self,
        min_signals: int = 3,
        min_score: float = 1.0,
    ) -> Set[str]:
        """
        Get terms with high confidence based on multiple signal agreement.

        A signal is considered "active" if it contributes meaningfully to the score:
        - Ratio signal: normalized > 0.1
        - TF-IDF signal: normalized > 0.1
        - Neologism signal: present (binary)
        - Definitional signal: count > 0
        - Capitalized signal: present (binary)

        Args:
            min_signals: Minimum number of signals that must fire (default: 3)
            min_score: Minimum total score (default: 1.0)

        Returns:
            Set of high-confidence philosophical terms
        """
        high_conf = set()

        for term, total_score, components in self.score_all(min_score=min_score):
            # Count how many signals fired
            signals_active = sum(
                [
                    components["ratio"] > 0.1,  # Ratio signal
                    components["tfidf"] > 0.1,  # TF-IDF signal
                    components["neologism"] > 0,  # Neologism signal
                    components["definitional"] > 0,  # Definitional signal
                    components["capitalized"] > 0,  # Capitalization signal
                ]
            )

            if signals_active >= min_signals:
                high_conf.add(term)

        return high_conf


def score_philosophical_terms(
    docs: List[ProcessedDocument],
    reference_corpus: Counter,
    use_lemmas: bool = True,
    min_author_freq: int = 3,
    top_n: int = 50,
) -> List[Tuple[str, float]]:
    """
    Convenience function to score philosophical terms with default settings.

    Args:
        docs: List of preprocessed documents
        reference_corpus: Reference corpus frequency distribution
        use_lemmas: Whether to use lemmatized forms (default: True)
        min_author_freq: Minimum occurrences in author corpus (default: 3)
        top_n: Number of top terms to return (default: 50)

    Returns:
        List of (term, total_score) tuples sorted by score descending
    """
    scorer = PhilosophicalTermScorer(
        docs, reference_corpus, use_lemmas=use_lemmas, min_author_freq=min_author_freq
    )

    results = scorer.score_all(min_score=0.0, top_n=top_n)

    # Return simplified (term, score) tuples
    return [(term, score) for term, score, components in results]
