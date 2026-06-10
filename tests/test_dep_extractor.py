"""
Tests for sentence-centric dependency-parse proposition extraction
(graph/dep_extractor.py). Skipped when the spacy extra / en_core_web_sm
model is not installed (mirrors the noun-chunk tests' gating).

Each test plants a syntactic construction the regex chain mishandles or
misses entirely and asserts the parse-based extractor reads it correctly.
"""

import pytest


def _spacy_available() -> bool:
    try:
        import spacy  # noqa: PLC0415

        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _spacy_available(),
    reason="spaCy en_core_web_sm not installed",
)


def _extract(sentences, terms, **kwargs):
    from concept_mapper.graph.dep_extractor import DependencyExtractor

    return DependencyExtractor(sentences, terms, **kwargs).extract_all()


def _by_type(props):
    return {(p.source, p.target, p.type) for p in props}


class TestVerbConstructions:
    def test_active_svo(self):
        props = _extract(
            ["Semiosis produces the interpretant."], ["semiosis", "interpretant"]
        )
        assert ("semiosis", "interpretant", "production") in _by_type(props)

    def test_passive_with_agent_recovers_direction(self):
        # Regex chain cannot invert passives; the parse can.
        props = _extract(
            ["The interpretant is produced by semiosis."],
            ["semiosis", "interpretant"],
        )
        assert ("semiosis", "interpretant", "production") in _by_type(props)

    def test_passive_defined_as(self):
        props = _extract(
            ["Semiosis is defined as an inferential process."],
            ["semiosis", "process"],
        )
        assert any(p.source == "semiosis" and p.type == "definition" for p in props)

    def test_prepositional_object_dependence(self):
        props = _extract(["Semiosis depends on the sign."], ["semiosis", "sign"])
        assert ("semiosis", "sign", "dependence") in _by_type(props)

    def test_coordinated_objects_expand(self):
        props = _extract(
            ["Semiosis produces signs and interpretants."],
            ["semiosis", "sign", "interpretant"],
        )
        triples = _by_type(props)
        assert ("semiosis", "sign", "production") in triples
        assert ("semiosis", "interpretant", "production") in triples

    def test_unmapped_verb_yields_generic_relation_for_known_terms(self):
        props = _extract(["Semiosis transforms the sign."], ["semiosis", "sign"])
        assert ("semiosis", "sign", "relation") in _by_type(props)


class TestCopularAndAppositive:
    def test_copular_kind_of(self):
        props = _extract(["Semiosis is a kind of process."], ["semiosis", "process"])
        assert ("semiosis", "process", "kind-of") in _by_type(props)

    def test_plain_copular_noun_is_kind_of(self):
        props = _extract(["A sign is a correlate."], ["sign", "correlate"])
        assert ("sign", "correlate", "kind-of") in _by_type(props)

    def test_appositive_kind_of(self):
        props = _extract(
            ["Semiosis, a species of inference, never halts."],
            ["semiosis", "inference"],
        )
        assert ("semiosis", "inference", "kind-of") in _by_type(props)


class TestSentenceCentricRecall:
    def test_single_seed_sentence_yields_new_node_proposition(self):
        # Only one seed term in the sentence: the regex pairwise loop would
        # skip it entirely; the parse still extracts against the other noun.
        props = _extract(["Semiosis produces the interpretant."], ["semiosis"])
        assert ("semiosis", "interpretant", "production") in _by_type(props)

    def test_new_nodes_suppressed_when_disabled(self):
        props = _extract(
            ["Semiosis produces the interpretant."],
            ["semiosis"],
            include_new_nodes=False,
        )
        assert props == []

    def test_generic_relations_never_mint_new_nodes(self):
        props = _extract(["Semiosis transforms the landscape."], ["semiosis"])
        assert _by_type(props) == set()

    def test_evidence_carries_the_sentence(self):
        sentence = "Semiosis depends on the sign."
        props = _extract([sentence], ["semiosis", "sign"])
        prop = next(p for p in props if p.type == "dependence")
        assert prop.evidence == [sentence]

    def test_repeated_relation_accumulates_weight(self):
        props = _extract(
            ["Semiosis produces the sign.", "Semiosis produces a sign."],
            ["semiosis", "sign"],
        )
        prop = next(p for p in props if p.type == "production")
        assert prop.weight == 2
        assert len(prop.evidence) == 2


class TestBuilderIntegration:
    def test_dependency_engine_via_builder(self):
        from concept_mapper.corpus.models import Document
        from concept_mapper.preprocessing.pipeline import preprocess
        from concept_mapper.graph.builders import build_proposition_graph

        text = (
            "The interpretant is produced by semiosis. Semiosis is a kind of process."
        )
        doc = preprocess(Document(text=text, metadata={}))
        g = build_proposition_graph(
            [doc],
            ["semiosis", "interpretant", "process"],
            pmi_threshold=0.0,
            engine="dependency",
        )
        edge = g.get_edge("semiosis", "interpretant")
        assert edge is not None and edge["relation_type"] == "production"
        edge2 = g.get_edge("semiosis", "process")
        assert edge2 is not None and edge2["relation_type"] == "kind-of"
