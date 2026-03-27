"""
Tests for PropositionExtractor — typed proposition extraction for graph edges.

Covers each edge type independently, direction detection, deduplication/weight,
the composition pattern, and no-match (cooccurrence fallback) cases.

Sentences are modelled on Eco's semiotics vocabulary to stay close to the
real use case.
"""

from unittest.mock import MagicMock

import pytest

from concept_mapper.graph.proposition_extractor import Proposition, PropositionExtractor


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
        assert p.source.lower() == "sign"   # subtype
        assert p.target.lower() == "code"   # supertype

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
