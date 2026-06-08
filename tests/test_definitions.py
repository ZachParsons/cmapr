"""
Tests for composite extractive definition derivation (analysis/definitions.py).

These cover the offline path (no sentence-embeddings dependency): every node is
guaranteed a text-derived definition, ranked from its concordance, with the
coverage fallback chain (best sentence → edge gloss → first occurrence).
See docs/plans/node-definitions.md.
"""

from concept_mapper.corpus.models import ProcessedDocument
from concept_mapper.graph.model import ConceptGraph
from concept_mapper.analysis.definitions import (
    derive_definitions,
    _pattern_score,
    _edge_gloss,
)


def _doc(sentences):
    return ProcessedDocument(
        raw_text=" ".join(sentences),
        sentences=sentences,
        tokens=[],
        lemmas=[],
        pos_tags=[],
        metadata={"source_path": "d.txt"},
        sentence_locations=[],
    )


class TestPatternScore:
    def test_explicit_definition_scores_high(self):
        s = "Abduction is defined as a form of inference."
        assert _pattern_score(s, "abduction") >= 0.9

    def test_passing_mention_scores_zero(self):
        s = "We discussed abduction at the seminar yesterday."
        assert _pattern_score(s, "abduction") == 0.0

    def test_pattern_must_match_the_term(self):
        # A definition of a *different* term doesn't credit "abduction".
        s = "Deduction is defined as a form of inference."
        assert _pattern_score(s, "abduction") == 0.0


class TestDeriveDefinitions:
    def test_picks_most_definitional_sentence(self):
        docs = [
            _doc(
                [
                    "We discussed the sign at the seminar.",
                    "A sign is defined as a correlate of expression and content.",
                    "The book about the sign is on the shelf.",
                ]
            )
        ]
        g = ConceptGraph(directed=True)
        g.add_node("sign", label="sign")
        n = derive_definitions(g, docs)
        assert n == 1
        assert g.get_node("sign")["definition"].startswith("A sign is defined as")

    def test_every_node_is_defined(self):
        docs = [
            _doc(
                [
                    "A sign is defined as a correlate.",
                    "Semiosis unfolds through interpretation of the sign.",
                ]
            )
        ]
        g = ConceptGraph(directed=True)
        g.add_node("sign", label="sign")
        g.add_node("semiosis", label="semiosis")
        derive_definitions(g, docs)
        assert all(g.get_node(t).get("definition") for t in g.nodes())

    def test_edge_gloss_fallback_when_no_occurrence(self):
        # 'phoneme' never appears in a sentence, but has a kind-of edge.
        docs = [_doc(["A sign is defined as a correlate."])]
        g = ConceptGraph(directed=True)
        g.add_node("phoneme", label="phoneme")
        g.add_node("unit", label="unit")
        g.add_edge("phoneme", "unit", type="kind-of")
        derive_definitions(g, docs)
        assert g.get_node("phoneme")["definition"] == "a kind of unit"

    def test_skips_nodes_with_existing_definition(self):
        docs = [_doc(["A sign is defined as a correlate."])]
        g = ConceptGraph(directed=True)
        g.add_node("sign", label="sign", definition="curated definition")
        derive_definitions(g, docs)
        assert g.get_node("sign")["definition"] == "curated definition"

    def test_attaches_location_source(self):
        from concept_mapper.corpus.models import SentenceLocation

        doc = ProcessedDocument(
            raw_text="x",
            sentences=["A sign is defined as a correlate of expression."],
            tokens=[],
            lemmas=[],
            pos_tags=[],
            metadata={},
            sentence_locations=[
                SentenceLocation(sent_index=0, chapter="1", chapter_title="Signs")
            ],
        )
        g = ConceptGraph(directed=True)
        g.add_node("sign", label="sign")
        derive_definitions(g, [doc])
        assert g.get_node("sign")["definition_source"] == "Ch. 1 — Signs"


class TestEdgeGloss:
    def test_priority_prefers_kind_of_over_relation(self):
        g = ConceptGraph(directed=True)
        g.add_node("a")
        g.add_node("b")
        g.add_node("c")
        g.add_edge("a", "b", type="relation")
        g.add_edge("a", "c", type="kind-of")
        assert _edge_gloss(g, "a") == "a kind of c"

    def test_none_when_no_edges(self):
        g = ConceptGraph(directed=True)
        g.add_node("a")
        assert _edge_gloss(g, "a") is None
