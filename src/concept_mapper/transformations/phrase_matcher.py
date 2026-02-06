"""
Multi-word phrase matching using lemma n-grams.

Finds occurrences of multi-word phrases in preprocessed documents
by matching lemmatized token sequences.
"""

from dataclasses import dataclass
from typing import List, Tuple
from ..corpus.models import ProcessedDocument


@dataclass
class PhraseMatch:
    """A matched phrase span in a document."""

    start_idx: int  # Starting token index (inclusive)
    end_idx: int  # Ending token index (exclusive)
    tokens: List[str]  # Original tokens in span
    lemmas: List[str]  # Lemmatized tokens in span
    pos_tags: List[str]  # POS tags for span (without words)
    head_idx: int  # Index of head word within span (relative to start)

    @property
    def head_pos(self) -> str:
        """Get POS tag of head word."""
        return self.pos_tags[self.head_idx]

    @property
    def head_token(self) -> str:
        """Get original token of head word."""
        return self.tokens[self.head_idx]

    @property
    def head_lemma(self) -> str:
        """Get lemma of head word."""
        return self.lemmas[self.head_idx]


class PhraseMatcher:
    """
    Match multi-word phrases using lemma-based n-gram matching.

    Uses sliding window over document lemmas to find exact matches
    of phrase patterns, handling inflected variants automatically.
    """

    def find_phrase_matches(
        self,
        phrase_lemmas: List[str],
        doc: ProcessedDocument,
        case_sensitive: bool = False,
    ) -> List[PhraseMatch]:
        """
        Find all occurrences of a phrase in document.

        Args:
            phrase_lemmas: Lemmatized phrase to search for (e.g., ["body", "without", "organ"])
            doc: Preprocessed document to search in
            case_sensitive: Whether to match case-sensitively (default: False)

        Returns:
            List of PhraseMatch objects representing found occurrences

        Example:
            >>> matcher = PhraseMatcher()
            >>> phrase = ["body", "without", "organ"]
            >>> matches = matcher.find_phrase_matches(phrase, doc)
            >>> # Matches both "body without organs" and "bodies without organs"
        """
        matches = []
        n = len(phrase_lemmas)

        # Normalize phrase lemmas for comparison
        if not case_sensitive:
            phrase_lemmas = [lemma.lower() for lemma in phrase_lemmas]

        # Sliding window over document lemmas
        for i in range(len(doc.lemmas) - n + 1):
            # Get window of lemmas
            window = doc.lemmas[i : i + n]

            # Normalize window for comparison
            if not case_sensitive:
                window = [lemma.lower() for lemma in window]

            # Check for match
            if window == phrase_lemmas:
                # Extract POS tags (without words, just tags)
                pos_tags_span = [tag for _, tag in doc.pos_tags[i : i + n]]

                # Find head word in phrase
                head_idx = self._find_head_word(doc.pos_tags[i : i + n])

                # Create match object
                match = PhraseMatch(
                    start_idx=i,
                    end_idx=i + n,
                    tokens=doc.tokens[i : i + n],
                    lemmas=doc.lemmas[i : i + n],  # Use original case
                    pos_tags=pos_tags_span,
                    head_idx=head_idx,
                )
                matches.append(match)

        return matches

    def _find_head_word(self, pos_tags: List[Tuple[str, str]]) -> int:
        """
        Identify head word in phrase using linguistic heuristics.

        For English noun phrases, the head is typically the rightmost noun.
        For verb phrases, it's typically the rightmost verb.

        Args:
            pos_tags: List of (word, tag) tuples for the phrase

        Returns:
            Index of head word within phrase (0-based, relative to phrase start)

        Linguistic Principle:
            English is head-final in noun phrases: "the big red car" → head is "car"
            Also applies to philosophical terms: "body without organs" → head is "organs"
        """
        # Scan right-to-left for noun or verb (content word)
        for i in range(len(pos_tags) - 1, -1, -1):
            _, tag = pos_tags[i]
            if tag.startswith(("NN", "VB")):
                return i

        # Fallback: last word is head
        return len(pos_tags) - 1

    def find_phrase_positions(
        self, phrase_lemmas: List[str], doc: ProcessedDocument
    ) -> List[Tuple[int, int]]:
        """
        Find position spans of phrase matches (simpler version).

        Args:
            phrase_lemmas: Lemmatized phrase to search for
            doc: Document to search in

        Returns:
            List of (start_idx, end_idx) tuples

        Example:
            >>> matcher = PhraseMatcher()
            >>> positions = matcher.find_phrase_positions(["body", "without", "organ"], doc)
            >>> [(5, 8), (12, 15)]  # Found at tokens 5-8 and 12-15
        """
        matches = self.find_phrase_matches(phrase_lemmas, doc)
        return [(m.start_idx, m.end_idx) for m in matches]
