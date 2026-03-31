"""
Node inclusion filter for concept graph construction.

Applies the criteria from docs/specs/graph.md § Nodes > Inclusion criteria:

  1. Content POS — noun, verb, adjective, adverb (checked when pos is supplied)
  2. Length ≥ 4 characters
  3. Not an abbreviation — not all-caps with length ≤ 4
  4. Not a stopword
  5. Minimum corpus frequency (default 3)
  6. Not a fragment — not a prefix of a longer corpus word, unless the term
     itself is in WordNet (so 'sign' is kept even though 'signal' exists)
  7. No invalid characters — only letters and hyphens (rejects '/man/', '[14]')

Applies to both seed nodes (from rarities list) and extracted nodes
(objects/complements of grammatical propositions).  Both roles use the
same criteria — Decision 1, docs/specs/graph.md.
"""

from collections import Counter
from typing import Optional

from concept_mapper.search.extract import (
    ADJ_POS_TAGS,
    ADV_POS_TAGS,
    NOUN_POS_TAGS,
    STOPWORDS,
    VERB_POS_TAGS,
)

CONTENT_POS_TAGS = NOUN_POS_TAGS | VERB_POS_TAGS | ADJ_POS_TAGS | ADV_POS_TAGS

_VALID_CHARS_IMPORT = None  # lazy sentinel


def _valid_chars(term: str) -> bool:
    """Only letters and hyphens allowed (handles /man/, [14] etc.)."""
    return all(c.isalpha() or c == "-" for c in term)


class NodeFilter:
    """
    Decide whether a term qualifies as a graph node.

    Parameters
    ----------
    corpus_vocab : set[str]
        All word forms present in the corpus (used for the fragment check).
    term_freqs : Counter
        Corpus frequency of each (lowercased) term.
    min_freq : int
        Minimum frequency threshold (default 3).
    stopwords : set[str] | None
        Stopword set; defaults to the shared STOPWORDS from search.extract.
    """

    def __init__(
        self,
        corpus_vocab: set[str],
        term_freqs: Counter,
        min_freq: int = 3,
        stopwords: Optional[set] = None,
    ):
        self._term_freqs = term_freqs
        self._min_freq = min_freq
        self._stopwords = stopwords if stopwords is not None else STOPWORDS
        self._corpus_lower = {w.lower() for w in corpus_vocab}
        self._wn_words: Optional[set] = None  # loaded lazily

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_valid(self, term: str, pos: Optional[str] = None) -> tuple:
        """
        Return (True, '') if the term passes all criteria, or
        (False, reason) describing the first failing check.

        Parameters
        ----------
        term : str
            The term to evaluate (may be mixed-case; checks use lowercased form).
        pos : str | None
            NLTK POS tag (e.g. 'NN', 'VB').  When provided the POS criterion
            is enforced; when None the check is skipped (caller guarantees it).
        """
        t = term.lower().strip()
        is_multiword = " " in t

        # Multi-word terms (noun phrases from spaCy) skip single-token checks:
        # length, character validity, and fragment detection don't apply.
        if not is_multiword:
            if pos is not None and pos not in CONTENT_POS_TAGS:
                return False, f"wrong POS ({pos})"

            if len(t) < 4:
                return False, f"too short ({len(t)} chars)"

            if not _valid_chars(term):
                return False, "invalid characters"

            if term.isupper() and len(term) <= 4:
                return False, "abbreviation (all-caps ≤ 4)"

            if self._is_fragment(t):
                return False, "fragment (prefix of longer corpus word)"
        else:
            if pos is not None and pos not in CONTENT_POS_TAGS:
                return False, f"wrong POS ({pos})"

        if t in self._stopwords:
            return False, "stopword"

        freq = self._term_freqs.get(t, 0)
        if freq < self._min_freq:
            return False, f"low frequency ({freq} < {self._min_freq})"

        return True, ""

    def filter(self, terms: list) -> list:
        """Return the subset of terms that pass all criteria."""
        return [t for t in terms if self.is_valid(t)[0]]

    def rejected(self, terms: list) -> dict:
        """
        Return {term: reason} for every term that fails.
        Useful for debugging / inspecting filter decisions.
        """
        out = {}
        for t in terms:
            valid, reason = self.is_valid(t)
            if not valid:
                out[t] = reason
        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _wordnet_words(self) -> set:
        if self._wn_words is None:
            from nltk.corpus import wordnet as wn

            self._wn_words = set(wn.words())
        return self._wn_words

    def _is_fragment(self, term: str) -> bool:
        """
        True if the term looks like a fragment of another corpus word.

        A term is a fragment when it is NOT in WordNet AND either:
          (a) at least one longer corpus word starts with it by ≥ 2 chars
              (e.g. 'structu' → 'structure' [+2], but NOT 'sign-function' →
              'sign-functions' [+1, just a plural]).
          (b) it is ≤ 5 chars AND at least one longer corpus word ends with it
              (e.g. 'tion' → 'proposition', 'ence' → 'evidence').
              Capped at 5 chars to avoid catching multi-char foreign words like
              'aliquid' via OCR-joined tokens ('aliudaliquid').

        Hyphenated terms (sign-function, co-text) skip the check entirely —
        they are compound terms, not fragments.

        The WordNet guard preserves real words: 'sign' passes even though
        'signal' starts with it; 'form' passes even though 'information'
        ends with it.
        """
        if "-" in term:
            return False
        if term in self._wordnet_words:
            return False
        prefix_fragment = any(
            w.startswith(term) and len(w) > len(term) for w in self._corpus_lower
        )
        suffix_fragment = len(term) <= 5 and any(
            w.endswith(term) and len(w) > len(term) for w in self._corpus_lower
        )
        return prefix_fragment or suffix_fragment
