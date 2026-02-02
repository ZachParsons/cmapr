"""
Extract significant terms from sentences containing a search term.

Provides functionality to find sentences containing a term and extract
significant verbs, nouns, and other content words from those sentences
based on philosophical term scoring.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
from collections import defaultdict

from ..corpus.models import ProcessedDocument
from ..preprocessing.tokenize import tokenize_words
from ..preprocessing.tagging import tag_tokens
from ..preprocessing.lemmatize import lemmatize_tagged
from ..analysis.rarity import PhilosophicalTermScorer
from .find import find_sentences, SentenceMatch


# Extended stopwords: common function words + mundane verbs + generic terms
# These are lemmatized forms only (no inflections)
STOPWORDS = {
    # Articles, determiners, pronouns
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "i",
    "you",
    "he",
    "she",
    "it",
    "we",
    "they",
    "them",
    "my",
    "your",
    "his",
    "her",
    "its",
    "our",
    "their",
    "myself",
    "yourself",
    "himself",
    "herself",
    "itself",
    "ourselves",
    "themselves",
    "who",
    "whom",
    "which",
    "what",
    "whose",
    # Common verbs (lemmatized forms only)
    "be",
    "have",
    "do",
    "will",
    "would",
    "shall",
    "should",
    "may",
    "might",
    "must",
    "can",
    "could",
    "go",
    "get",
    "make",
    "take",
    "come",
    "see",
    "know",
    "think",
    "give",
    "find",
    "tell",
    "become",
    "leave",
    "feel",
    "put",
    "bring",
    "begin",
    "keep",
    "hold",
    "write",
    "stand",
    "seem",
    "turn",
    "show",
    "try",
    "call",
    "ask",
    "need",
    "let",
    "mean",
    "say",
    "use",
    "want",
    "work",
    "look",
    "help",
    "endow",
    # Prepositions and conjunctions
    "of",
    "in",
    "to",
    "for",
    "with",
    "on",
    "at",
    "from",
    "by",
    "about",
    "as",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "between",
    "under",
    "since",
    "without",
    "within",
    "along",
    "among",
    "and",
    "or",
    "but",
    "if",
    "then",
    "than",
    "so",
    "because",
    "while",
    "where",
    "when",
    "why",
    "how",
    # Common adverbs
    "not",
    "no",
    "yes",
    "very",
    "too",
    "also",
    "well",
    "only",
    "just",
    "now",
    "then",
    "here",
    "there",
    "up",
    "down",
    "out",
    "over",
    "again",
    "even",
    "still",
    "back",
    "more",
    "most",
    "much",
    "any",
    "some",
    "all",
    "both",
    "each",
    "few",
    "many",
    "other",
    "such",
    "own",
    # Generic/mundane terms (lemmatized)
    "thing",
    "something",
    "anything",
    "everything",
    "nothing",
    "someone",
    "anyone",
    "everyone",
    "no one",
    "way",
    "time",
    "one",
    "two",
    "first",
    "second",
    "last",
    "part",
    "place",
    "case",
    "fact",
    "point",
    "number",
}


# POS tag sets for filtering
NOUN_POS_TAGS = {"NN", "NNS", "NNP", "NNPS"}
VERB_POS_TAGS = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
ADJ_POS_TAGS = {"JJ", "JJR", "JJS"}
ADV_POS_TAGS = {"RB", "RBR", "RBS"}

POS_TAG_GROUPS = {
    "nouns": NOUN_POS_TAGS,
    "verbs": VERB_POS_TAGS,
    "adjectives": ADJ_POS_TAGS,
    "adverbs": ADV_POS_TAGS,
}


@dataclass
class SignificantTermsResult:
    """
    Result of extracting significant terms from a sentence.

    Attributes:
        sentence_match: The original sentence match
        significant_terms: List of (term, score, score_components) tuples
        all_extracted: Set of all terms extracted before filtering
    """

    sentence_match: SentenceMatch
    significant_terms: List[Tuple[str, float, Dict[str, float]]]
    all_extracted: Set[str]

    def __str__(self) -> str:
        """String representation showing sentence and top terms."""
        terms_str = ", ".join([term for term, _, _ in self.significant_terms])
        return f"[{self.sentence_match.doc_id}:{self.sentence_match.sent_index}] {terms_str}"


def extract_significant_terms(
    search_term: str,
    docs: List[ProcessedDocument],
    threshold: float = 0.1,
    pos_types: List[str] = None,
    top_n: int = None,
    match_lemma: bool = False,
    reference_corpus=None,
    scorer_weights: Dict[str, float] = None,
    scoring_mode: str = "corpus_frequency",
) -> List[SignificantTermsResult]:
    """
    Search for a term and extract significant verbs/nouns from matching sentences.

    Args:
        search_term: Term to search for in sentences
        docs: List of preprocessed documents
        threshold: Minimum significance score to include (default: 0.1)
        pos_types: List of POS types to extract (e.g., ["nouns", "verbs"]).
                  If None, extracts both nouns and verbs.
        top_n: Maximum number of terms to return per sentence (default: None for all)
        match_lemma: Whether to match lemmatized forms of search term
        reference_corpus: Reference corpus for scoring (if None, loads Brown corpus)
        scorer_weights: Custom weights for PhilosophicalTermScorer (if None, uses defaults)
        scoring_mode: Scoring method to use:
                     - "corpus_frequency": Frequency-based scoring within corpus (default, best for main content words)
                     - "hybrid": Full PhilosophicalTermScorer (best for rare/distinctive terms)

    Returns:
        List of SignificantTermsResult objects, one per matching sentence

    Example:
        >>> results = extract_significant_terms(
        ...     "abstraction",
        ...     docs,
        ...     threshold=0.1,
        ...     pos_types=["nouns", "verbs"]
        ... )
        >>> for result in results:
        ...     print(f"Sentence {result.sentence_match.sent_index}:")
        ...     for term, score, components in result.significant_terms:
        ...         print(f"  {term}: {score:.2f}")
    """
    # Default to nouns and verbs if not specified
    if pos_types is None:
        pos_types = ["nouns", "verbs"]

    # Build set of POS tags to filter for
    pos_tags_to_extract = set()
    for pos_type in pos_types:
        if pos_type in POS_TAG_GROUPS:
            pos_tags_to_extract.update(POS_TAG_GROUPS[pos_type])
        else:
            raise ValueError(
                f"Invalid POS type: {pos_type}. Must be one of: {list(POS_TAG_GROUPS.keys())}"
            )

    # Find sentences containing the search term
    matches = find_sentences(search_term, docs, match_lemma=match_lemma)

    if not matches:
        return []

    # Get search term lemma for filtering
    search_term_lemma = None
    if match_lemma:
        tokens = tokenize_words(search_term)
        if tokens:
            tagged = tag_tokens(tokens)
            lemmas_list = lemmatize_tagged(tagged)
            search_term_lemma = (
                lemmas_list[0].lower() if lemmas_list else search_term.lower()
            )
    else:
        search_term_lemma = search_term.lower()

    # Initialize scoring based on mode
    if scoring_mode == "hybrid":
        # Load reference corpus if not provided
        if reference_corpus is None:
            from ..analysis.reference import load_reference_corpus

            reference_corpus = load_reference_corpus()

        scorer = PhilosophicalTermScorer(docs, reference_corpus, weights=scorer_weights)

        def score_func(term):
            return scorer.score_term(term)

    elif scoring_mode == "corpus_frequency":
        # Build frequency map for corpus-based scoring
        from collections import Counter

        # Get all terms from the corpus with POS filtering, excluding stopwords
        all_terms = []
        for doc in docs:
            for sent in doc.sentences:
                tokens = tokenize_words(sent)
                if tokens:
                    tagged = tag_tokens(tokens)
                    lemmas_list = lemmatize_tagged(tagged)
                    for (token, pos), lemma in zip(tagged, lemmas_list):
                        lemma_lower = lemma.lower()
                        if pos in pos_tags_to_extract and lemma_lower not in STOPWORDS:
                            all_terms.append(lemma_lower)

        term_freqs = Counter(all_terms)
        max_freq = max(term_freqs.values()) if term_freqs else 1

        # Scoring function: normalized frequency (0-10 scale for interpretability)
        def score_func(term):
            freq = term_freqs.get(term, 0)
            # Normalize to 0-10 scale, with log scaling for better distribution
            import math

            if freq == 0:
                score = 0.0
            else:
                # Log scale: common words get scores 1-10, rare words get lower scores
                score = 10.0 * math.log(1 + freq) / math.log(1 + max_freq)

            return {
                "total": score,
                "frequency": freq,
                "normalized": score / 10.0,
            }
    else:
        raise ValueError(
            f"Invalid scoring_mode: {scoring_mode}. Must be 'hybrid' or 'corpus_frequency'"
        )

    results = []

    for match in matches:
        # Tokenize and tag the sentence
        tokens = tokenize_words(match.sentence)
        if not tokens:
            continue

        tagged = tag_tokens(tokens)
        lemmas = lemmatize_tagged(tagged)

        # Extract terms with desired POS tags, excluding search term, stopwords, and punctuation
        extracted_terms = set()
        import string

        # Include both ASCII and Unicode punctuation
        punctuation = set(string.punctuation) | {""", """, '"', '"', "–", "—", "…", "·"}

        for (token, pos), lemma in zip(tagged, lemmas):
            if pos in pos_tags_to_extract:
                lemma_lower = lemma.lower()
                # Exclude:
                # - The search term itself
                # - Stopwords (common function words, mundane verbs, generic terms)
                # - Punctuation
                # - Single-character symbols
                if (
                    lemma_lower != search_term_lemma
                    and lemma_lower not in STOPWORDS
                    and lemma_lower not in punctuation
                    and not (len(lemma_lower) == 1 and not lemma_lower.isalnum())
                ):
                    extracted_terms.add(lemma_lower)

        if not extracted_terms:
            continue

        # Score each extracted term
        term_scores = []
        for term in extracted_terms:
            score_dict = score_func(term)
            total_score = score_dict["total"]

            # Filter by threshold
            if total_score >= threshold:
                term_scores.append((term, total_score, score_dict))

        # Sort by score (descending)
        term_scores.sort(key=lambda x: x[1], reverse=True)

        # Limit to top N if specified
        if top_n is not None:
            term_scores = term_scores[:top_n]

        # Create result object
        result = SignificantTermsResult(
            sentence_match=match,
            significant_terms=term_scores,
            all_extracted=extracted_terms,
        )
        results.append(result)

    return results


def format_results_by_sentence(
    results: List[SignificantTermsResult], show_scores: bool = False
) -> str:
    """
    Format extraction results grouped by sentence.

    Args:
        results: List of SignificantTermsResult objects
        show_scores: Whether to show score values (default: False)

    Returns:
        Formatted string output

    Example output:
        Sentence 1: contradiction, bourgeoisie, individual, commodity production
        Sentence 2: existence, inhumanity
        Sentence 3: objective crisis, capitalism, proletariat, class consciousness
    """
    lines = []

    for i, result in enumerate(results, 1):
        if show_scores:
            terms_str = ", ".join(
                [f"{term} ({score:.2f})" for term, score, _ in result.significant_terms]
            )
        else:
            terms_str = ", ".join([term for term, _, _ in result.significant_terms])

        lines.append(f"Sentence {i}: {terms_str}")

    return "\n".join(lines)


def format_results_detailed(results: List[SignificantTermsResult]) -> str:
    """
    Format extraction results with full details including sentences and scores.

    Args:
        results: List of SignificantTermsResult objects

    Returns:
        Formatted string output with full details
    """
    lines = []

    for i, result in enumerate(results, 1):
        match = result.sentence_match
        lines.append(
            f"\n[{i}] {match.doc_id or 'document'} (sentence {match.sent_index}):"
        )
        lines.append(f"    {match.sentence}")
        lines.append(f"\n    Significant terms ({len(result.significant_terms)}):")

        for term, score, components in result.significant_terms:
            # Show main score and relevant components based on what's available
            if "ratio" in components and "tfidf" in components:
                # Hybrid scoring mode
                comp_str = (
                    f"ratio={components['ratio']:.2f}, tfidf={components['tfidf']:.2f}"
                )
            elif "frequency" in components:
                # Corpus frequency mode
                comp_str = f"freq={components['frequency']}"
            else:
                # Generic mode - show all available components
                comp_str = ", ".join(
                    f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                    for k, v in components.items()
                    if k != "total"
                )

            lines.append(f"      • {term}: {score:.2f} ({comp_str})")

        lines.append("")  # Blank line between entries

    return "\n".join(lines)


def aggregate_across_sentences(
    results: List[SignificantTermsResult], top_n: int = None
) -> List[Tuple[str, float, int]]:
    """
    Aggregate significant terms across all sentences.

    Returns terms with their average score and occurrence count,
    sorted by average score.

    Args:
        results: List of SignificantTermsResult objects
        top_n: Maximum number of terms to return (default: None for all)

    Returns:
        List of (term, avg_score, count) tuples sorted by avg_score descending
    """
    term_scores = defaultdict(list)

    for result in results:
        for term, score, _ in result.significant_terms:
            term_scores[term].append(score)

    # Calculate averages and counts
    aggregated = []
    for term, scores in term_scores.items():
        avg_score = sum(scores) / len(scores)
        count = len(scores)
        aggregated.append((term, avg_score, count))

    # Sort by average score
    aggregated.sort(key=lambda x: x[1], reverse=True)

    # Limit to top N if specified
    if top_n is not None:
        aggregated = aggregated[:top_n]

    return aggregated
