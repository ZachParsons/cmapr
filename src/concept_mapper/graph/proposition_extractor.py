"""
Typed proposition extractor for concept graph construction.

For each pair of terms (A, B), scans sentences containing both and attempts
to classify the relation using pattern matching in priority order:

  1. definition  — explicit authorial definition markers
  2. kind-of     — copular + kind marker ('is a type of')
  3. production  — A produces/generates/implies B
  4. dependence  — A presupposes/requires/depends on B
  5. component   — A, B, C together form X  (composition pattern)

Falls back to cooccurrence (handled by caller) when no pattern matches.

Edge types 'property' and 'opposition' are deferred (see spec).

v1 scanning scope: only sentences where BOTH terms appear.
v2: also scan individual-term sentences and cross-match extracted relations.

Spec ref: docs/specs/graph.md § Edges, § Decisions 4, 9, 10, 12
"""

import re
from dataclasses import dataclass, field
from typing import Optional

from concept_mapper.corpus.models import ProcessedDocument


# ---------------------------------------------------------------------------
# Proposition dataclass
# ---------------------------------------------------------------------------


@dataclass
class Proposition:
    """
    A typed, directed (or undirected) relation between two terms.

    Attributes
    ----------
    source  : subject term (grammatical subject for directed edges)
    target  : object/complement term
    label   : human-readable relation verb/phrase ('produces', 'is a kind of')
    type    : semantic category ('definition','kind-of','production',
              'dependence','component','cooccurrence')
    evidence: the supporting sentence
    directed: True for asymmetric relations; False for component / cooccurrence
    weight  : number of distinct supporting sentences (incremented on merge)
    """

    source: str
    target: str
    label: str
    type: str
    evidence: str
    directed: bool = True
    weight: int = 1


# ---------------------------------------------------------------------------
# Pattern tables
# ---------------------------------------------------------------------------

# Definition markers: {A} is replaced with re.escape(term).
# Each entry: (pattern_template, canonical_label)
_DEFINITION_PATTERNS = [
    # "by 'sign' I mean ..."
    (
        r"by\s+(?:the\s+(?:term|word|expression)\s+)?"
        r"[\u2018\u201c'\"]?{A}[\u2019\u201d'\"]?\s+(?:I\s+|we\s+)?mean",
        "means",
    ),
    # "sign is defined as ..."
    (r"\b{A}\b\s+(?:is|are|was)\s+defined\s+as", "is defined as"),
    # "define sign as ..."
    (r"define\s+{A}\s+as", "is defined as"),
    # "sign denotes ..."
    (r"\b{A}\b\s+denotes\b", "denotes"),
    # "sign stands for ..."
    (r"\b{A}\b\s+stands\s+for\b", "stands for"),
]

_KIND_MARKERS = (
    r"(?:kind|type|sort|species|form|class|subtype|variety|mode|"
    r"instance|example|case)\s+of"
)

_PRODUCTION_VERBS = (
    r"(?:produces?|produced|"
    r"generates?|generated|"
    r"implies?|implied|"
    r"creates?|created|"
    r"causes?|caused|"
    r"yields?|yielded|"
    r"entails?|entailed|"
    r"determines?|determined|"
    r"brings?\s+about|brought\s+about|"
    r"gives?\s+rise\s+to|"
    r"results?\s+in|resulted\s+in)"
)

_DEPENDENCE_PHRASES = (
    r"(?:presupposes?|presupposed|"
    r"depends?\s+on|depended\s+on|"
    r"requires?|required|"
    r"needs?|needed|"
    r"is\s+based\s+on|was\s+based\s+on|"
    r"rests?\s+on|rested\s+on|"
    r"is\s+conditioned\s+by|was\s+conditioned\s+by|"
    r"is\s+grounded\s+in|was\s+grounded\s+in)"
)

_COMPOSITION_VERBS = (
    r"(?:forms?|constitutes?|composes?|makes?\s+up|comprises?|consists?\s+of)"
)


# ---------------------------------------------------------------------------
# Extractor class
# ---------------------------------------------------------------------------


class PropositionExtractor:
    """
    Extract typed propositions between pairs of terms from a document set.

    Parameters
    ----------
    docs : list[ProcessedDocument]
        Pre-processed documents to scan.

    Usage
    -----
    extractor = PropositionExtractor(docs)

    # Single pair
    props = extractor.extract("sign", "code")

    # All pairs from a term list
    props = extractor.extract_all_pairs(["sign", "code", "interpretant"])

    # Composition pattern (Phase 4)
    props = extractor.extract_composition(["sign", "interpretant", "referent"])
    """

    def __init__(self, docs: list):
        self.docs = docs
        self._sentences: list = [
            sent for doc in docs for sent in doc.sentences
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, term_a: str, term_b: str) -> list:
        """
        Return all typed propositions between term_a and term_b.

        Same-type duplicates are merged (weight incremented).
        Different-type propositions between the same pair are all kept
        (multigraph, Decision 9).
        """
        seen: dict = {}

        for sentence in self._sentences_with_both(term_a, term_b):
            for prop in self._extract_from_sentence(sentence, term_a, term_b):
                key = (prop.source.lower(), prop.target.lower(), prop.type)
                if key in seen:
                    seen[key].weight += 1
                else:
                    seen[key] = prop

        return list(seen.values())

    def extract_all_pairs(self, terms: list) -> list:
        """
        Extract propositions for all pairs in the term list.

        Only pairs that co-occur in at least one sentence are processed.
        Returns a flat list of Proposition objects.
        """
        results = []
        for i, term_a in enumerate(terms):
            for term_b in terms[i + 1 :]:
                results.extend(self.extract(term_a, term_b))
        return results

    def extract_composition(self, terms: list) -> list:
        """
        Find composition/constitution patterns across the full term list.

        Pattern: 'A, B, and C form/constitute/compose X'

        Returns:
        - component edges between all co-constituent pairs (undirected)
        - production edges from each constituent → composed entity

        Spec: docs/specs/graph.md § Decision 12
        """
        terms_set = {t.lower() for t in terms}
        seen: dict = {}

        for sentence in self._sentences:
            for prop in self._extract_composition_from_sentence(
                sentence, terms_set
            ):
                key = (prop.source.lower(), prop.target.lower(), prop.type)
                if key in seen:
                    seen[key].weight += 1
                else:
                    seen[key] = prop

        return list(seen.values())

    # ------------------------------------------------------------------
    # Sentence scanning
    # ------------------------------------------------------------------

    def _sentences_with_both(self, term_a: str, term_b: str):
        """Yield sentences containing both terms (v1: substring match for recall)."""
        a = term_a.lower()
        b = term_b.lower()
        for sent in self._sentences:
            sl = sent.lower()
            if a in sl and b in sl:
                yield sent

    # ------------------------------------------------------------------
    # Per-sentence extraction
    # ------------------------------------------------------------------

    def _extract_from_sentence(
        self, sentence: str, term_a: str, term_b: str
    ) -> list:
        """Try all v1 extractors in priority order on one sentence."""
        results = []
        for extractor in (
            self._try_definition,
            self._try_kind_of,
            self._try_production,
            self._try_dependence,
        ):
            p = extractor(sentence, term_a, term_b)
            if p is not None:
                results.append(p)
        return results

    # ------------------------------------------------------------------
    # Typed extractors
    # ------------------------------------------------------------------

    def _try_definition(
        self, sentence: str, term_a: str, term_b: str
    ) -> Optional[Proposition]:
        """
        Detect explicit definition: 'by X I mean Y', 'X is defined as Y'.

        The defined term (A) must match a definition marker pattern; the
        complement term (B) must appear *after* the marker in the sentence.
        Copular disambiguation rule 1.
        """
        for defined, complement in [(term_a, term_b), (term_b, term_a)]:
            ta = re.escape(defined)
            tb = re.escape(complement)

            for pattern_tmpl, label in _DEFINITION_PATTERNS:
                marker_re = pattern_tmpl.replace("{A}", ta)
                m = re.search(marker_re, sentence, re.IGNORECASE)
                if not m:
                    continue
                after = sentence[m.end() :]
                if re.search(r"\b" + tb + r"\w*\b", after, re.IGNORECASE):
                    return Proposition(
                        source=defined,
                        target=complement,
                        label=label,
                        type="definition",
                        evidence=sentence,
                        directed=True,
                    )
        return None

    def _try_kind_of(
        self, sentence: str, term_a: str, term_b: str
    ) -> Optional[Proposition]:
        """
        Detect kind-of: 'A is a type/kind/species of B'.
        Copular disambiguation rule 2.
        """
        for subtype, supertype in [(term_a, term_b), (term_b, term_a)]:
            ta = re.escape(subtype)
            tb = re.escape(supertype)
            pattern = (
                rf"\b{ta}\b.{{0,30}}"
                rf"\bis\s+(?:a|an|the)?\s*{_KIND_MARKERS}.{{0,50}}"
                rf"\b{tb}\b"
            )
            if re.search(pattern, sentence, re.IGNORECASE):
                return Proposition(
                    source=subtype,
                    target=supertype,
                    label="is a kind of",
                    type="kind-of",
                    evidence=sentence,
                    directed=True,
                )
        return None

    def _try_production(
        self, sentence: str, term_a: str, term_b: str
    ) -> Optional[Proposition]:
        """
        Detect production: 'A produces/generates/implies B'.
        """
        for source, target in [(term_a, term_b), (term_b, term_a)]:
            ta = re.escape(source)
            tb = re.escape(target)
            pattern = (
                rf"\b{ta}\b.{{0,80}}"
                rf"\b({_PRODUCTION_VERBS})\b.{{0,80}}"
                rf"\b{tb}\w*\b"
            )
            m = re.search(pattern, sentence, re.IGNORECASE)
            if m:
                label = m.group(1).lower()
                return Proposition(
                    source=source,
                    target=target,
                    label=label,
                    type="production",
                    evidence=sentence,
                    directed=True,
                )
        return None

    def _try_dependence(
        self, sentence: str, term_a: str, term_b: str
    ) -> Optional[Proposition]:
        """
        Detect dependence: 'A presupposes/requires/depends on B'.
        """
        for source, target in [(term_a, term_b), (term_b, term_a)]:
            ta = re.escape(source)
            tb = re.escape(target)
            pattern = (
                rf"\b{ta}\b.{{0,80}}"
                rf"\b({_DEPENDENCE_PHRASES})\b.{{0,80}}"
                rf"\b{tb}\w*\b"
            )
            m = re.search(pattern, sentence, re.IGNORECASE)
            if m:
                label = m.group(1).lower()
                return Proposition(
                    source=source,
                    target=target,
                    label=label,
                    type="dependence",
                    evidence=sentence,
                    directed=True,
                )
        return None

    # ------------------------------------------------------------------
    # Composition pattern (Phase 4 / Decision 12)
    # ------------------------------------------------------------------

    def _extract_composition_from_sentence(
        self, sentence: str, terms_set: set
    ) -> list:
        """
        Detect 'A, B, C form/constitute/compose X' pattern.

        Returns component edges between co-constituents and production
        edges from each constituent to the composed entity.
        """
        verb_match = re.search(
            rf"\b{_COMPOSITION_VERBS}\b", sentence, re.IGNORECASE
        )
        if not verb_match:
            return []

        verb_label = verb_match.group(0).lower()
        verb_start = verb_match.start()

        before = sentence[:verb_start]
        after = sentence[verb_start:]

        constituents = [
            t for t in terms_set
            if re.search(r"\b" + re.escape(t) + r"\b", before, re.IGNORECASE)
        ]
        composed_terms = [
            t for t in terms_set
            if re.search(r"\b" + re.escape(t) + r"\b", after, re.IGNORECASE)
            and t not in set(constituents)
        ]

        if len(constituents) < 2:
            return []

        results = []

        # Undirected component edges between all co-constituent pairs
        for i, ca in enumerate(constituents):
            for cb in constituents[i + 1 :]:
                results.append(
                    Proposition(
                        source=ca,
                        target=cb,
                        label="co-constitutes",
                        type="component",
                        evidence=sentence,
                        directed=False,
                    )
                )

        # Directed production edges: constituent → composed entity
        for constituent in constituents:
            for composed in composed_terms:
                results.append(
                    Proposition(
                        source=constituent,
                        target=composed,
                        label=verb_label,
                        type="production",
                        evidence=sentence,
                        directed=True,
                    )
                )

        return results
