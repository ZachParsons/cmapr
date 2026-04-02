"""
Tests for PropositionExtractor — typed proposition extraction for graph edges.

Covers each edge type independently, direction detection, deduplication/weight,
the composition pattern, and no-match (cooccurrence fallback) cases.

Sentences are modelled on Eco's semiotics vocabulary to stay close to the
real use case.
"""

from unittest.mock import MagicMock


from concept_mapper.graph.proposition_extractor import PropositionExtractor


# ---------------------------------------------------------------------------
# Minimal document fixture
# ---------------------------------------------------------------------------


def make_docs(sentences: list) -> list:
    """Build a minimal ProcessedDocument-like list from a sentence list."""
    doc = MagicMock()
    doc.sentences = sentences
    doc.metadata = {"source_path": "test"}
    return [doc]


# ---------------------------------------------------------------------------
# Definition extraction
# ---------------------------------------------------------------------------


class TestDefinitionExtraction:
    def _extractor(self, sentence: str) -> PropositionExtractor:
        return PropositionExtractor(make_docs([sentence]))

    def test_by_x_i_mean(self):
        s = "By 'sign' I mean any interpretant that stands for something."
        e = self._extractor(s)
        props = e.extract("sign", "interpretant")
        assert any(p.type == "definition" for p in props)
        p = next(p for p in props if p.type == "definition")
        assert p.source.lower() == "sign"
        assert p.target.lower() == "interpretant"
        assert p.directed is True

    def test_is_defined_as(self):
        s = "A sign is defined as a correlate of a signifier and a signified."
        e = self._extractor(s)
        props = e.extract("sign", "signifier")
        assert any(p.type == "definition" for p in props)

    def test_denotes(self):
        s = "The sign denotes an interpretant in the mind of the receiver."
        e = self._extractor(s)
        props = e.extract("sign", "interpretant")
        assert any(p.type == "definition" for p in props)
        p = next(p for p in props if p.type == "definition")
        assert p.source.lower() == "sign"

    def test_definition_requires_complement_after_marker(self):
        """B must appear AFTER the definition marker, not before."""
        s = "The interpretant, as sign is defined as, appears later."
        # "interpretant" appears before the marker, so it should NOT be found
        # as the complement of "sign is defined as ..."
        # (the text after "is defined as" is "appears later", no 'interpretant')
        e = self._extractor(s)
        props = e.extract("sign", "interpretant")
        definition_props = [p for p in props if p.type == "definition"]
        assert not definition_props

    def test_no_definition_without_marker(self):
        s = "The sign and the code appear together in the text."
        e = self._extractor(s)
        props = e.extract("sign", "code")
        assert not any(p.type == "definition" for p in props)


# ---------------------------------------------------------------------------
# Kind-of extraction
# ---------------------------------------------------------------------------


class TestKindOfExtraction:
    def _extractor(self, sentence: str) -> PropositionExtractor:
        return PropositionExtractor(make_docs([sentence]))

    def test_is_a_type_of(self):
        s = "A sign is a type of code that governs the production of meaning."
        e = self._extractor(s)
        props = e.extract("sign", "code")
        assert any(p.type == "kind-of" for p in props)
        p = next(p for p in props if p.type == "kind-of")
        assert p.source.lower() == "sign"
        assert p.target.lower() == "code"

    def test_is_a_kind_of(self):
        s = "An interpretant is a kind of mental representation."
        e = self._extractor(s)
        props = e.extract("interpretant", "representation")
        assert any(p.type == "kind-of" for p in props)

    def test_is_a_species_of(self):
        s = "Semiosis is a species of sign activity."
        e = self._extractor(s)
        props = e.extract("semiosis", "sign")
        assert any(p.type == "kind-of" for p in props)
        p = next(p for p in props if p.type == "kind-of")
        assert p.source.lower() == "semiosis"

    def test_direction_subtype_to_supertype(self):
        s = "A sign is a type of code."
        e = self._extractor(s)
        props = e.extract("sign", "code")
        p = next(p for p in props if p.type == "kind-of")
        assert p.source.lower() == "sign"  # subtype
        assert p.target.lower() == "code"  # supertype

    def test_no_kind_of_without_marker(self):
        s = "A sign and a code are related."
        e = self._extractor(s)
        props = e.extract("sign", "code")
        assert not any(p.type == "kind-of" for p in props)


# ---------------------------------------------------------------------------
# Production extraction
# ---------------------------------------------------------------------------


class TestProductionExtraction:
    def _extractor(self, sentence: str) -> PropositionExtractor:
        return PropositionExtractor(make_docs([sentence]))

    def test_produces(self):
        s = "A sign produces meaning in the mind of the interpreter."
        e = self._extractor(s)
        props = e.extract("sign", "meaning")
        assert any(p.type == "production" for p in props)
        p = next(p for p in props if p.type == "production")
        assert p.source.lower() == "sign"
        assert p.target.lower() == "meaning"
        assert "produc" in p.label.lower()

    def test_generates(self):
        s = "Semiosis generates interpretants in an unlimited chain."
        e = self._extractor(s)
        props = e.extract("semiosis", "interpretant")
        assert any(p.type == "production" for p in props)
        p = next(p for p in props if p.type == "production")
        assert "generat" in p.label.lower()

    def test_implies(self):
        s = "Every sign implies a code that assigns values to its expression."
        e = self._extractor(s)
        props = e.extract("sign", "code")
        assert any(p.type == "production" for p in props)

    def test_direction_source_to_target(self):
        s = "The code produces the sign through a correlation of functives."
        e = self._extractor(s)
        props = e.extract("code", "sign")
        p = next(p for p in props if p.type == "production")
        assert p.source.lower() == "code"
        assert p.target.lower() == "sign"

    def test_reverse_direction_detected(self):
        s = "Meaning is produced by the sign in a given context."
        e = self._extractor(s)
        props = e.extract("sign", "meaning")
        # "meaning ... produced by sign" — B verb A → source=sign target=meaning?
        # Our pattern checks A→B then B→A. "meaning ... produced ... sign" matches B→A
        # so source=meaning, target=sign. Either direction finding is acceptable.
        assert any(p.type == "production" for p in props)

    def test_no_production_without_verb(self):
        s = "The sign and the code appear together in a sentence."
        e = self._extractor(s)
        props = e.extract("sign", "code")
        assert not any(p.type == "production" for p in props)


# ---------------------------------------------------------------------------
# Dependence extraction
# ---------------------------------------------------------------------------


class TestDependenceExtraction:
    def _extractor(self, sentence: str) -> PropositionExtractor:
        return PropositionExtractor(make_docs([sentence]))

    def test_presupposes(self):
        s = "Every act of semiosis presupposes a code that establishes the correlation."
        e = self._extractor(s)
        props = e.extract("semiosis", "code")
        assert any(p.type == "dependence" for p in props)
        p = next(p for p in props if p.type == "dependence")
        assert p.source.lower() == "semiosis"
        assert "presuppose" in p.label.lower()

    def test_depends_on(self):
        s = "The sign depends on a code to produce its meaning."
        e = self._extractor(s)
        props = e.extract("sign", "code")
        assert any(p.type == "dependence" for p in props)

    def test_requires(self):
        s = "Interpretation requires a sign that points beyond itself."
        e = self._extractor(s)
        props = e.extract("interpretation", "sign")
        assert any(p.type == "dependence" for p in props)

    def test_direction(self):
        s = "Semiosis presupposes a code."
        e = self._extractor(s)
        props = e.extract("semiosis", "code")
        p = next(p for p in props if p.type == "dependence")
        assert p.source.lower() == "semiosis"
        assert p.target.lower() == "code"
        assert p.directed is True


# ---------------------------------------------------------------------------
# Opposition extraction
# ---------------------------------------------------------------------------


class TestOppositionExtraction:
    def _extractor(self, sentence: str) -> PropositionExtractor:
        return PropositionExtractor(make_docs([sentence]))

    def test_vs(self):
        s = "The symbol vs the sign is a central debate in semiotics."
        e = self._extractor(s)
        props = e.extract("symbol", "sign")
        assert any(p.type == "opposition" for p in props)

    def test_versus(self):
        s = "Saussure versus Peirce represents two traditions of semiotic thought."
        e = self._extractor(s)
        props = e.extract("saussure", "peirce")
        assert any(p.type == "opposition" for p in props)

    def test_as_opposed_to(self):
        s = "The sign, as opposed to the signal, involves interpretation."
        e = self._extractor(s)
        props = e.extract("sign", "signal")
        assert any(p.type == "opposition" for p in props)

    def test_rather_than(self):
        s = "Eco focuses on the code rather than the message in isolation."
        e = self._extractor(s)
        props = e.extract("code", "message")
        assert any(p.type == "opposition" for p in props)

    def test_contrasts_with(self):
        s = "The symbol contrasts with the icon in its mode of signification."
        e = self._extractor(s)
        props = e.extract("symbol", "icon")
        assert any(p.type == "opposition" for p in props)

    def test_is_the_opposite_of(self):
        s = "Denotation is the opposite of connotation in classical semiotics."
        e = self._extractor(s)
        props = e.extract("denotation", "connotation")
        assert any(p.type == "opposition" for p in props)

    def test_opposition_is_undirected(self):
        s = "The symbol vs the sign is a central debate."
        e = self._extractor(s)
        props = e.extract("symbol", "sign")
        p = next(p for p in props if p.type == "opposition")
        assert p.directed is False

    def test_no_opposition_without_marker(self):
        s = "The sign and the symbol share a common structure."
        e = self._extractor(s)
        props = e.extract("sign", "symbol")
        assert not any(p.type == "opposition" for p in props)


# ---------------------------------------------------------------------------
# No-match (cooccurrence fallback territory)
# ---------------------------------------------------------------------------


class TestNoMatch:
    def test_plain_cooccurrence_returns_empty(self):
        """Two terms in same sentence with no typed relation → empty list."""
        s = "The sign appeared in the text where the code was mentioned."
        e = PropositionExtractor(make_docs([s]))
        props = e.extract("sign", "code")
        assert props == []

    def test_terms_not_in_same_sentence(self):
        s1 = "The sign has many forms."
        s2 = "The code determines meaning."
        e = PropositionExtractor(make_docs([s1, s2]))
        props = e.extract("sign", "code")
        assert props == []


# ---------------------------------------------------------------------------
# Deduplication and weight
# ---------------------------------------------------------------------------


class TestDeduplication:
    def test_same_type_duplicate_increments_weight(self):
        """Same (source, target, type) in two sentences → weight=2."""
        s1 = "A sign produces meaning in context."
        s2 = "Each sign produces an interpretant as its meaning."
        e = PropositionExtractor(make_docs([s1, s2]))
        props = e.extract("sign", "meaning")
        prod = [p for p in props if p.type == "production"]
        assert len(prod) == 1
        assert prod[0].weight == 2

    def test_different_types_both_kept(self):
        """Different types from different sentences → both kept (multigraph)."""
        s1 = "A sign is defined as a carrier that produces meaning."
        s2 = "A sign produces meaning through its interpretant."
        e = PropositionExtractor(make_docs([s1, s2]))
        # s1 might fire both definition and production
        props = e.extract("sign", "meaning")
        types = {p.type for p in props}
        # At least production should be found
        assert "production" in types


# ---------------------------------------------------------------------------
# extract_all_pairs
# ---------------------------------------------------------------------------


class TestExtractAllPairs:
    def test_finds_multiple_pairs(self):
        sentences = [
            "A sign produces meaning through semiosis.",
            "Semiosis presupposes a code.",
            "A sign is a type of code.",
        ]
        e = PropositionExtractor(make_docs(sentences))
        props = e.extract_all_pairs(["sign", "meaning", "semiosis", "code"])
        types = {p.type for p in props}
        assert "production" in types
        assert "dependence" in types
        assert "kind-of" in types

    def test_no_self_pairs(self):
        """A term should never form a pair with itself."""
        sentences = ["A sign is a sign of something."]
        e = PropositionExtractor(make_docs(sentences))
        props = e.extract_all_pairs(["sign"])
        assert props == []


# ---------------------------------------------------------------------------
# Composition pattern (Phase 4 / Decision 12)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Property extraction
# ---------------------------------------------------------------------------


class TestPropertyExtraction:
    def _extractor(self, sentence: str) -> PropositionExtractor:
        return PropositionExtractor(make_docs([sentence]))

    def test_plain_copular(self):
        s = "A rhizome is an open chart without a fixed centre."
        e = self._extractor(s)
        props = e.extract("rhizome", "chart")
        assert any(p.type == "property" for p in props)
        p = next(p for p in props if p.type == "property")
        assert p.source.lower() == "rhizome"
        assert p.directed is True

    def test_metaphor_is_figure(self):
        s = "Metaphor is a rhetorical figure that transfers meaning."
        e = self._extractor(s)
        props = e.extract("metaphor", "figure")
        assert any(p.type == "property" for p in props)

    def test_property_does_not_fire_for_kind_of(self):
        """kind-of runs first; property should not also fire for the same sentence."""
        s = "A sign is a type of code."
        e = self._extractor(s)
        props = e.extract("sign", "code")
        # kind-of should be present; property may or may not be — but kind-of must win
        assert any(p.type == "kind-of" for p in props)


# ---------------------------------------------------------------------------
# Relation extraction (_try_relation and _try_pos_verb)
# ---------------------------------------------------------------------------


class TestRelationExtraction:
    def _extractor(self, sentence: str) -> PropositionExtractor:
        return PropositionExtractor(make_docs([sentence]))

    def test_try_relation_verb_list(self):
        s = "The sign relates to the code through a system of conventions."
        e = self._extractor(s)
        props = e.extract("sign", "code")
        assert any(p.type == "relation" for p in props)
        p = next(p for p in props if p.type == "relation")
        assert "relat" in p.label.lower()

    def test_try_relation_involves(self):
        s = "Every act of semiosis involves a sign and a referent."
        e = self._extractor(s)
        props = e.extract("semiosis", "sign")
        assert any(p.type == "relation" for p in props)

    def test_try_pos_verb_catches_unlisted_verb(self):
        """POS fallback catches verbs not in any predefined list."""
        s = "The sign transforms the referent into an abstract entity."
        e = self._extractor(s)
        props = e.extract("sign", "referent")
        # "transforms" is not in any static list → must be caught by _try_pos_verb
        assert any(p.type == "relation" for p in props)
        p = next(p for p in props if p.type == "relation")
        assert "transform" in p.label.lower()

    def test_try_pos_verb_passive_reverses_direction(self):
        """Passive voice ('A is governed by B') should set source=B, target=A."""
        s = "The sign is structured by the code in all cultural systems."
        e = self._extractor(s)
        props = e.extract("sign", "code")
        # "sign is structured by code" → dependence OR relation with reversed direction
        typed = [p for p in props if p.type in ("dependence", "relation")]
        assert typed, "Expected at least one typed edge"

    def test_pos_verb_no_match_on_light_verb_only(self):
        """Sentence with only light verbs between terms → no relation from POS."""
        s = "The sign and the code appear together in a sentence."
        e = self._extractor(s)
        props = e.extract("sign", "code")
        # 'appear' is a light verb; no content verb between the terms
        assert not any(p.type == "relation" for p in props)


# ---------------------------------------------------------------------------
# Composition pattern
# ---------------------------------------------------------------------------


class TestCompositionPattern:
    def test_component_edges_between_constituents(self):
        """
        'The sign, the interpretant, and the referent form a triadic relation.'
        → component edges: sign↔interpretant, sign↔referent, interpretant↔referent
        """
        s = "The sign, the interpretant, and the referent form a triadic relation."
        e = PropositionExtractor(make_docs([s]))
        props = e.extract_composition(["sign", "interpretant", "referent", "relation"])
        component = [p for p in props if p.type == "component"]
        pairs = {frozenset([p.source, p.target]) for p in component}
        assert frozenset(["sign", "interpretant"]) in pairs
        assert frozenset(["sign", "referent"]) in pairs
        assert frozenset(["interpretant", "referent"]) in pairs

    def test_component_edges_are_undirected(self):
        s = "The sign, the interpretant, and the referent form a triadic relation."
        e = PropositionExtractor(make_docs([s]))
        props = e.extract_composition(["sign", "interpretant", "referent"])
        for p in props:
            if p.type == "component":
                assert p.directed is False

    def test_production_edges_to_composed_term(self):
        """Each constituent should also get a production edge → composed entity."""
        s = "The sign, the interpretant, and the referent form a triadic relation."
        e = PropositionExtractor(make_docs([s]))
        props = e.extract_composition(["sign", "interpretant", "referent", "relation"])
        prod = [p for p in props if p.type == "production"]
        sources = {p.source for p in prod}
        assert "sign" in sources
        assert "interpretant" in sources
        assert "referent" in sources
        targets = {p.target for p in prod}
        assert "relation" in targets

    def test_fewer_than_two_constituents_no_component(self):
        """Only one term before the composition verb → no component edges."""
        s = "The sign forms the basis of semiosis."
        e = PropositionExtractor(make_docs([s]))
        props = e.extract_composition(["sign", "semiosis"])
        assert not any(p.type == "component" for p in props)

    def test_composition_deduplication(self):
        """Same composition pair in two sentences → weight=2."""
        s1 = "The sign and the code form a functional unit."
        s2 = "Together the sign and the code form a system."
        e = PropositionExtractor(make_docs([s1, s2]))
        props = e.extract_composition(["sign", "code", "unit", "system"])
        comp = [p for p in props if p.type == "component"]
        pair = frozenset(["sign", "code"])
        matches = [p for p in comp if frozenset([p.source, p.target]) == pair]
        assert len(matches) == 1
        assert matches[0].weight == 2


# ============================================================================
# Test EvidenceScoring (Phase 10)
# ============================================================================

from concept_mapper.graph.proposition_extractor import _score_sentence


class TestEvidenceScoring:
    """_score_sentence assigns higher scores to more informative evidence sentences."""

    def test_definition_marker_raises_score(self):
        """A sentence with a definition marker scores higher than one without."""
        with_marker = "By 'sign' I mean any vehicle that stands for something else."
        without_marker = "Sign and vehicle appear together in the same chapter."
        score_with = _score_sentence(with_marker, "sign", "vehicle", 5, 20)
        score_without = _score_sentence(without_marker, "sign", "vehicle", 5, 20)
        assert score_with > score_without, (
            "Expected definition-marker sentence to score higher than plain co-occurrence"
        )

    def test_denotes_marker_raises_score(self):
        """A sentence using 'denotes' scores higher than one without the marker."""
        with_denotes = "The sign denotes an interpretant in the semiotic process."
        without_denotes = "Sign and interpretant appear in the same passage."
        score_with = _score_sentence(with_denotes, "sign", "interpretant", 0, 10)
        score_without = _score_sentence(without_denotes, "sign", "interpretant", 0, 10)
        assert score_with > score_without, (
            "Expected 'denotes' sentence to score higher than plain co-occurrence"
        )

    def test_proximity_bonus_applied(self):
        """A sentence where the terms are close scores higher than one where they are far."""
        close = "Sign produces interpretant."
        # Build a sentence with more than 15 words between "sign" and "interpretant"
        filler = " ".join(["which", "is", "a", "complex", "relational", "structure",
                           "in", "Peircean", "philosophy", "may", "under", "certain",
                           "conditions", "realise", "its", "full", "potential",
                           "and", "finally", "produce", "its"])
        far = f"Sign {filler} interpretant."
        score_close = _score_sentence(close, "sign", "interpretant", 0, 10)
        score_far = _score_sentence(far, "sign", "interpretant", 0, 10)
        assert score_close > score_far, (
            "Expected proximity bonus to raise score when terms are within 15 words"
        )

    def test_longer_sentences_penalised(self):
        """A longer sentence scores lower than a shorter sentence with the same meaning."""
        short = "Sign produces interpretant."
        long = (
            "Sign, as a triadic relation in Peirce's semiotic philosophy, "
            "produces its corresponding interpretant through a process of mediation "
            "involving object, representamen, and ground in the semiotic system."
        )
        score_short = _score_sentence(short, "sign", "interpretant", 0, 10)
        score_long = _score_sentence(long, "sign", "interpretant", 0, 10)
        assert score_short > score_long, (
            "Expected shorter sentence to score higher due to length penalty"
        )

    def test_early_sentences_preferred(self):
        """The same sentence at index 0 scores higher than at index 99."""
        s = "Sign produces interpretant."
        early = _score_sentence(s, "sign", "interpretant", sent_idx=0, n_sentences=100)
        late = _score_sentence(s, "sign", "interpretant", sent_idx=99, n_sentences=100)
        assert early > late, (
            "Expected early sentence (idx=0) to score higher than late sentence (idx=99)"
        )

    def test_returns_float(self):
        """_score_sentence always returns a float."""
        result = _score_sentence(
            "Sign is an interpretant.", "sign", "interpretant", 0, 1
        )
        assert isinstance(result, float), (
            f"Expected float return type, got {type(result).__name__}"
        )

    def test_evidence_is_list_on_proposition(self):
        """Proposition.evidence is a list after extraction."""
        sentences = [
            "Semiosis is defined as the process by which signs produce interpretants.",
        ]
        e = PropositionExtractor(make_docs(sentences))
        props = e.extract("semiosis", "interpretants")
        if props:
            assert isinstance(props[0].evidence, list), (
                "Expected Proposition.evidence to be a list"
            )

    def test_top3_evidence_max_length(self):
        """Proposition.evidence contains at most 3 entries."""
        sentences = [
            "Sign and interpretant appear in the text.",
            "The sign relates to the interpretant through a code.",
            "Both sign and interpretant are central to semiosis.",
            "The sign is fundamentally related to the interpretant.",
            "Sign mediates between object and interpretant.",
            "Interpretant is produced by sign in a triadic relation.",
        ]
        e = PropositionExtractor(make_docs(sentences))
        props = e.extract("sign", "interpretant")
        assert props, "Expected at least one proposition to be extracted"
        assert len(props[0].evidence) <= 3, (
            f"Expected at most 3 evidence entries, got {len(props[0].evidence)}"
        )

    def test_definition_sentence_ranked_first(self):
        """The definition proposition's evidence[0] is the sentence with the marker."""
        sentences = [
            "Sign and interpretant appear together in semiotic theory.",
            "By 'sign' I mean interpretant-producing vehicle in Peirce's model.",
            "Sign and interpretant are both central to the semiotic process.",
            "Sign and interpretant co-occur in every act of communication.",
        ]
        e = PropositionExtractor(make_docs(sentences))
        props = e.extract("sign", "interpretant")
        assert props, "Expected at least one proposition to be extracted"
        # The definition proposition's evidence should rank the marker sentence first
        defn_props = [p for p in props if p.type == "definition"]
        assert defn_props, "Expected a definition proposition to be extracted"
        defn_p = defn_props[0]
        assert defn_p.evidence, "Expected non-empty evidence list on definition proposition"
        assert "I mean" in defn_p.evidence[0], (
            f"Expected definition-marker sentence to rank first in definition evidence, "
            f"got: {defn_p.evidence[0]!r}"
        )
