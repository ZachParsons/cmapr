"""
Typed proposition extractor for concept graph construction.

For each pair of terms (A, B), scans sentences containing both and attempts
to classify the relation using pattern matching in priority order:

  1. definition  — explicit authorial definition markers
  2. kind-of     — copular + kind marker ('is a type of')
  3. production  — A produces/generates/implies/expresses/denotes B
  4. dependence  — A presupposes/requires/derives from/governed by B
  5. opposition  — A vs B, A as opposed to B, A contrasts with B
  6. property    — A is [a/an] B (plain copular, no kind marker)
  7. relation    — A <verb> B (any clear verb; label = actual verb from text)
  8. component   — A, B, C together form X  (composition pattern)

Falls back to cooccurrence (handled by caller) when no pattern matches.

v1 scanning scope: only sentences where BOTH terms appear.
v2: also scan individual-term sentences and cross-match extracted relations.

Spec ref: docs/specs/graph.md § Edges, § Decisions 4, 9, 10, 12
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


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
    evidence: List[str] = field(default_factory=list)
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
    # "sign means ..."
    (r"\b{A}\b\s+means\b", "means"),
    # "sign is also known as ..." / "is also called ..."
    (
        r"\b{A}\b\s+(?:is|are|was|were)\s+also\s+(?:known\s+as|called)\b",
        "is also known as",
    ),
    # "sign can/may/might be defined/understood/conceived/construed as ..."
    (
        r"\b{A}\b\s+(?:can|may|might|could)\s+be\s+"
        r"(?:defined|understood|conceived|construed|characterized|described|"
        r"interpreted|taken|regarded|treated|viewed|seen)\s+as",
        "can be defined as",
    ),
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
    r"results?\s+in|resulted\s+in|"
    r"expresses?|expressed|"
    r"denotes?|denoted|"
    r"signifies?|signified|"
    r"represents?|represented|"
    r"refers?\s+to|referred\s+to|"
    r"characterizes?|characterized|"
    r"functions?\s+as|functioned\s+as|"
    r"describes?|described|"
    r"introduces?|introduced|"
    r"encodes?|encoded|"
    r"marks?|marked|"
    r"activates?|activated|"
    r"triggers?|triggered|"
    r"defines?|defined)"
)

_DEPENDENCE_PHRASES = (
    r"(?:presupposes?|presupposed|"
    r"depends?\s+on|depended\s+on|"
    r"requires?|required|"
    r"needs?|needed|"
    r"is\s+based\s+on|was\s+based\s+on|"
    r"rests?\s+on|rested\s+on|"
    r"is\s+conditioned\s+by|was\s+conditioned\s+by|"
    r"is\s+grounded\s+in|was\s+grounded\s+in|"
    r"derives?\s+from|derived\s+from|"
    r"governs?|governed\s+by|"
    r"controls?\s+(?:the\s+)?|controlled\s+by|"
    r"follows?\s+from|followed\s+from|"
    r"arises?\s+from|arose\s+from|"
    r"emerges?\s+from|emerged\s+from|"
    r"is\s+influenced\s+by|was\s+influenced\s+by|"
    r"is\s+shaped\s+by|was\s+shaped\s+by|"
    r"is\s+structured\s+by|was\s+structured\s+by|"
    r"is\s+determined\s+by|was\s+determined\s+by|"
    r"is\s+constrained\s+by|was\s+constrained\s+by|"
    r"applies?\s+to|applied\s+to)"
)

# Broad verb list for the relation fallback — captures any clearly verbal
# connection between two terms that didn't match a more specific type above.
# The actual verb is extracted and used as the edge label.
_RELATION_VERBS = (
    r"(?:relates?\s+to|related\s+to|"
    r"connects?|connected\s+(?:to|with)|"
    r"links?|linked\s+(?:to|with)|"
    r"corresponds?\s+to|corresponded\s+to|"
    r"involves?|involved|"
    r"includes?|included|"
    r"contains?|contained|"
    r"comprises?|comprised|"
    r"associates?\s+with|associated\s+with|"
    r"maps?\s+(?:onto?|to)|mapped\s+(?:onto?|to)|"
    r"points?\s+to|pointed\s+to|"
    r"operates?\s+(?:on|through|in|via)|"
    r"interacts?\s+with|interacted\s+with|"
    r"combines?\s+with|combined\s+with|"
    r"coexists?\s+with|co-exists?\s+with|"
    r"parallels?|paralleled|"
    r"instantiates?|instantiated|"
    r"realizes?|realized|"
    r"implements?|implemented|"
    r"applies?\s+to|applied\s+to|"
    r"modifies?|modified|"
    r"restricts?|restricted)"
)

# Templates use __A__ and __B__ as placeholders (replaced at match time).
# Quantifiers use single braces since these are plain strings, not f-strings.
_OPPOSITION_PATTERNS = [
    # "X vs Y" / "X versus Y" (allow articles/words between marker and term)
    r"\b__A__\b.{0,20}\bvs\.?\b.{0,15}\b__B__\b",
    r"\b__A__\b.{0,20}\bversus\b.{0,20}\b__B__\b",
    # "X as opposed to Y"
    r"\b__A__\b.{0,40}\bas\s+opposed\s+to\b.{0,40}\b__B__\b",
    # "X rather than Y"
    r"\b__A__\b.{0,40}\brather\s+than\b.{0,40}\b__B__\b",
    # "X contrasts with Y" / "X is contrasted with Y"
    r"\b__A__\b.{0,40}\bcontrasts?\s+with\b.{0,40}\b__B__\b",
    r"\b__A__\b.{0,40}\bcontrasted\s+with\b.{0,40}\b__B__\b",
    # "X is the opposite of Y"
    r"\b__A__\b.{0,40}\bopposite\s+of\b.{0,40}\b__B__\b",
    # "unlike X, Y"
    r"\bunlike\b.{0,30}\b__A__\b.{0,60}\b__B__\b",
    # "X, not Y" — tight window to avoid false positives on plain negation
    r"\b__A__\b,?\s+not\b.{0,20}\b__B__\b",
]

_COMPOSITION_VERBS = (
    r"(?:forms?|constitutes?|composes?|makes?\s+up|comprises?|consists?\s+of)"
)

# Hearst-style hypernymy patterns. Each entry uses __SUPER__ for the supertype
# slot and __SUB__ for the subtype. The relation produced is `kind-of` with
# direction subtype → supertype, matching `_try_kind_of`.
_HEARST_PATTERNS = [
    # "supertype such as subtype"
    (r"\b__SUPER__\w*\b.{0,40}\bsuch\s+as\b.{0,40}\b__SUB__\w*\b", "is a kind of"),
    # "such supertype as subtype"
    (r"\bsuch\s+\b__SUPER__\w*\b.{0,40}\bas\b.{0,40}\b__SUB__\w*\b", "is a kind of"),
    # "supertype including subtype"
    (r"\b__SUPER__\w*\b.{0,40}\bincluding\b.{0,40}\b__SUB__\w*\b", "is a kind of"),
    # "supertype, especially/particularly/notably subtype"
    (
        r"\b__SUPER__\w*\b,\s+(?:especially|particularly|notably)\b.{0,40}"
        r"\b__SUB__\w*\b",
        "is a kind of",
    ),
    # "subtype and other supertypes" / "subtype or other supertypes"
    # Allow up to 3 modifier tokens between "other" and the supertype.
    (
        r"\b__SUB__\w*\b\s+(?:and|or)\s+other\s+(?:\w+\s+){0,3}\b__SUPER__\w*\b",
        "is a kind of",
    ),
]


# ---------------------------------------------------------------------------
# Evidence scoring
# ---------------------------------------------------------------------------


def _score_sentence(
    sentence: str,
    term_a: str,
    term_b: str,
    sent_idx: int,
    n_sentences: int,
) -> float:
    """
    Score a candidate evidence sentence; higher = better to surface in tooltip.

    Heuristics (additive):
      +10  contains an explicit definition marker
      +5   both terms appear within 15 words of each other
      -len/400  penalty for long sentences (prefer concise ones)
      -idx/n    penalty for late sentences (prefer intro/early context)
    """
    score = 0.0

    # 1. Definition marker
    if re.search(
        r"\b(?:defined?\s+as|by\b.{0,20}\bI\s+mean|denotes?|stands?\s+for)\b",
        sentence,
        re.IGNORECASE,
    ):
        score += 10.0

    # 2. Proximity: both terms within 15 words
    words = sentence.split()
    ta_l, tb_l = term_a.lower(), term_b.lower()
    ta_idx = [i for i, w in enumerate(words) if ta_l in w.lower()]
    tb_idx = [i for i, w in enumerate(words) if tb_l in w.lower()]
    if ta_idx and tb_idx:
        min_gap = min(abs(ia - ib) for ia in ta_idx for ib in tb_idx)
        if min_gap <= 15:
            score += 5.0

    # 3. Sentence length penalty (prefer under ~200 chars)
    score -= len(sentence) / 400.0

    # 4. Position penalty (prefer early sentences)
    if n_sentences > 1:
        score -= sent_idx / n_sentences

    return score


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

    # Verbs that carry no semantic content — skip these in POS extraction.
    _LIGHT_VERBS = frozenset(
        {
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "make",
            "made",
            "take",
            "took",
            "get",
            "got",
            "put",
            "use",
            "used",
            "show",
            "showed",
            "find",
            "found",
            "give",
            "gave",
            "see",
            "saw",
            "know",
            "knew",
            "think",
            "thought",
            "say",
            "said",
            "tell",
            "told",
            "come",
            "came",
            "go",
            "went",
            "look",
            "looked",
            "work",
            "worked",
            "seem",
            "seemed",
            "appear",
            "appeared",
            "become",
            "became",
            "call",
            "called",
            "let",
            "keep",
            "kept",
            "start",
            "started",
            "try",
            "tried",
            "turn",
            "turned",
            "set",
            "allow",
            "allowed",
            "help",
            "helped",
            "move",
            "moved",
            "begin",
            "began",
            "began",
            "note",
            "noted",
            "mean",
            "meant",
        }
    )
    # Copular verbs — handled by definition / kind-of / property
    _COPULAR = frozenset(
        {
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "being",
        }
    )

    def __init__(self, docs: list):
        self.docs = docs
        self._sentences: list = [sent for doc in docs for sent in doc.sentences]
        # Cache POS-tagged tokens per sentence to avoid re-tagging
        self._pos_cache: Dict[str, Tuple[List[str], List[Tuple[str, str]]]] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self, term_a: str, term_b: str) -> list:
        """
        Return all typed propositions between term_a and term_b.

        Same-type duplicates are merged (weight incremented) and their
        evidence sentences accumulated, ranked by quality, and trimmed to
        the top 3.  Different-type propositions for the same pair are all
        kept (multigraph, Decision 9).
        """
        seen: dict = {}
        # scored_evidence[key] = list of (score, sentence)
        scored_evidence: Dict[tuple, list] = {}
        a_l, b_l = term_a.lower(), term_b.lower()
        n_sents = len(self._sentences)

        for sent_idx, sentence in enumerate(self._sentences):
            sl = sentence.lower()
            if a_l not in sl or b_l not in sl:
                continue
            for prop in self._extract_from_sentence(sentence, term_a, term_b):
                key = (prop.source.lower(), prop.target.lower(), prop.type)
                if key in seen:
                    seen[key].weight += 1
                else:
                    seen[key] = prop
                    scored_evidence[key] = []
                score = _score_sentence(
                    sentence, prop.source, prop.target, sent_idx, n_sents
                )
                scored_evidence[key].append((score, sentence))

        # Rank evidence and attach top-3 to each proposition
        for key, prop in seen.items():
            ranked = sorted(scored_evidence.get(key, []), key=lambda x: -x[0])
            prop.evidence = [s for _, s in ranked[:3]]

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
            for prop in self._extract_composition_from_sentence(sentence, terms_set):
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

    def _extract_from_sentence(self, sentence: str, term_a: str, term_b: str) -> list:
        """Try all extractors in priority order on one sentence."""
        results = []
        for extractor in (
            self._try_definition,
            self._try_kind_of,
            self._try_hearst,
            self._try_production,
            self._try_dependence,
            self._try_opposition,
            self._try_property,
            self._try_relation,
            self._try_pos_verb,  # v2: POS-based fallback, catches any verb
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
                        evidence=[sentence],
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
                    evidence=[sentence],
                    directed=True,
                )
        return None

    def _try_hearst(
        self, sentence: str, term_a: str, term_b: str
    ) -> Optional[Proposition]:
        """
        Detect Hearst-style hypernymy patterns: 'X such as Y', 'X including Y',
        'Y and other X', etc. Produces a `kind-of` proposition oriented
        subtype → supertype, matching `_try_kind_of`.
        """
        for subtype, supertype in [(term_a, term_b), (term_b, term_a)]:
            sub = re.escape(subtype)
            sup = re.escape(supertype)
            for pattern_tmpl, label in _HEARST_PATTERNS:
                pattern = pattern_tmpl.replace("__SUB__", sub).replace("__SUPER__", sup)
                if re.search(pattern, sentence, re.IGNORECASE):
                    return Proposition(
                        source=subtype,
                        target=supertype,
                        label=label,
                        type="kind-of",
                        evidence=[sentence],
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
                    evidence=[sentence],
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
                    evidence=[sentence],
                    directed=True,
                )
        return None

    def _try_opposition(
        self, sentence: str, term_a: str, term_b: str
    ) -> Optional[Proposition]:
        """
        Detect explicit contrast: 'A vs B', 'A as opposed to B', etc.

        Opposition is symmetric — directed=False, label='opposes'.
        """
        ta = re.escape(term_a)
        tb = re.escape(term_b)
        for pattern_tmpl in _OPPOSITION_PATTERNS:
            # Try A…B order
            for a, b in [(ta, tb), (tb, ta)]:
                pattern = pattern_tmpl.replace("__A__", a).replace("__B__", b)
                if re.search(pattern, sentence, re.IGNORECASE):
                    # Always store in canonical (term_a, term_b) order
                    return Proposition(
                        source=term_a,
                        target=term_b,
                        label="opposes",
                        type="opposition",
                        evidence=[sentence],
                        directed=False,
                    )
        return None

    def _try_property(
        self, sentence: str, term_a: str, term_b: str
    ) -> Optional[Proposition]:
        """
        Detect plain copular property: 'A is [a/an] B'.

        Fires only when definition and kind-of patterns did not match
        (they run earlier in the chain). Covers sentences like
        'a rhizome is an open chart' or 'metaphor is a rhetorical figure'.

        Copular disambiguation rule 3 (spec): plain NP complement → property.
        """
        for subject, complement in [(term_a, term_b), (term_b, term_a)]:
            ta = re.escape(subject)
            tb = re.escape(complement)
            # Must NOT contain a kind marker (those are caught by _try_kind_of)
            pattern = (
                rf"\b{ta}\w*\b"
                rf".{{0,40}}"
                rf"\b(?:is|are|was|were)\b"
                rf".{{0,40}}"  # allow articles, adjectives, modifiers
                rf"(?!(?:{_KIND_MARKERS}))"  # negative lookahead for kind markers
                rf"\b{tb}\w*\b"
            )
            if re.search(pattern, sentence, re.IGNORECASE):
                return Proposition(
                    source=subject,
                    target=complement,
                    label="is",
                    type="property",
                    evidence=[sentence],
                    directed=True,
                )
        return None

    def _try_relation(
        self, sentence: str, term_a: str, term_b: str
    ) -> Optional[Proposition]:
        """
        Catch-all SVO extractor: 'A <verb> B' with any verb from a broad list.

        Fires only after all more-specific extractors have run. Captures the
        actual verb from the text as the edge label so the proposition is
        readable ('relates to', 'involves', 'maps onto', etc.).
        """
        for source, target in [(term_a, term_b), (term_b, term_a)]:
            ta = re.escape(source)
            tb = re.escape(target)
            pattern = (
                rf"\b{ta}\w*\b.{{0,80}}"
                rf"\b({_RELATION_VERBS})\b.{{0,80}}"
                rf"\b{tb}\w*\b"
            )
            m = re.search(pattern, sentence, re.IGNORECASE)
            if m:
                label = m.group(1).lower().strip()
                return Proposition(
                    source=source,
                    target=target,
                    label=label,
                    type="relation",
                    evidence=[sentence],
                    directed=True,
                )
        return None

    def _pos_tag(self, sentence: str) -> Tuple[List[str], List[Tuple[str, str]]]:
        """Return (tokens, pos_tags) for sentence, cached."""
        if sentence not in self._pos_cache:
            import nltk

            tokens = nltk.word_tokenize(sentence)
            self._pos_cache[sentence] = (tokens, nltk.pos_tag(tokens))
        return self._pos_cache[sentence]

    def _try_pos_verb(
        self, sentence: str, term_a: str, term_b: str
    ) -> Optional[Proposition]:
        """
        V2: POS-based fallback — find any content verb between the two terms.

        Fires after all pattern-list extractors.  Extracts the text between
        the two terms, POS-tags it, and uses the first non-copular,
        non-light verb found as the edge label (type = 'relation').

        Passive voice ('A is VBN by B') is detected and direction reversed
        so the actual agent is the source.

        Max gap between terms: 120 characters (avoids spurious long-range
        matches in run-on sentences).
        """
        MAX_GAP_TOKENS = 15  # skip pairs separated by too many tokens

        # Use the full sentence for accurate POS context
        tokens, tagged = self._pos_tag(sentence)
        tokens_lower = [w.lower() for w, _ in tagged]

        for source, target in [(term_a, term_b), (term_b, term_a)]:
            src_word = source.lower()
            tgt_word = target.lower()

            # Find first token that starts with the source term
            src_idx = next(
                (
                    i
                    for i, w in enumerate(tokens_lower)
                    if w.startswith(src_word[:4]) and src_word in w
                ),
                None,
            )
            if src_idx is None:
                src_idx = next(
                    (i for i, w in enumerate(tokens_lower) if w == src_word), None
                )
            if src_idx is None:
                continue

            # Find first token that starts with the target term, after src_idx
            tgt_idx = next(
                (
                    i
                    for i, w in enumerate(tokens_lower)
                    if i > src_idx and w.startswith(tgt_word[:4]) and tgt_word in w
                ),
                None,
            )
            if tgt_idx is None:
                tgt_idx = next(
                    (
                        i
                        for i, w in enumerate(tokens_lower)
                        if i > src_idx and w == tgt_word
                    ),
                    None,
                )
            if tgt_idx is None:
                continue

            gap_tagged = tagged[src_idx + 1 : tgt_idx]
            if len(gap_tagged) > MAX_GAP_TOKENS:
                continue

            content_verbs = [
                (w.lower(), pos)
                for w, pos in gap_tagged
                if pos.startswith("VB")
                and w.lower() not in self._COPULAR
                and w.lower() not in self._LIGHT_VERBS
            ]
            if not content_verbs:
                continue

            verb, vpos = content_verbs[0]

            # Passive detection: copular token precedes VBN in the gap
            if vpos == "VBN":
                verb_gap_idx = next(
                    i for i, (w, _) in enumerate(gap_tagged) if w.lower() == verb
                )
                is_passive = any(
                    w.lower() in self._COPULAR for w, _ in gap_tagged[:verb_gap_idx]
                )
            else:
                is_passive = False

            act_src, act_tgt = (target, source) if is_passive else (source, target)
            return Proposition(
                source=act_src,
                target=act_tgt,
                label=verb,
                type="relation",
                evidence=[sentence],
                directed=True,
            )

        return None

    # ------------------------------------------------------------------
    # Composition pattern (Phase 4 / Decision 12)
    # ------------------------------------------------------------------

    def _extract_composition_from_sentence(self, sentence: str, terms_set: set) -> list:
        """
        Detect 'A, B, C form/constitute/compose X' pattern.

        Returns component edges between co-constituents and production
        edges from each constituent to the composed entity.
        """
        verb_match = re.search(rf"\b{_COMPOSITION_VERBS}\b", sentence, re.IGNORECASE)
        if not verb_match:
            return []

        verb_label = verb_match.group(0).lower()
        verb_start = verb_match.start()

        before = sentence[:verb_start]
        after = sentence[verb_start:]

        constituents = [
            t
            for t in terms_set
            if re.search(r"\b" + re.escape(t) + r"\b", before, re.IGNORECASE)
        ]
        composed_terms = [
            t
            for t in terms_set
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
                        evidence=[sentence],
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
                        evidence=[sentence],
                        directed=True,
                    )
                )

        return results
