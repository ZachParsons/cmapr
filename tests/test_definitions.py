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
    compose_definition,
    compose_definition_parts,
    compose_definitions,
    derive_definitions,
    _COMPOSE_PART_ORDER,
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


class TestComposeDefinitions:
    def _graph(self):
        g = ConceptGraph(directed=True)
        for n in ("semiosis", "process", "sign", "stasis"):
            g.add_node(n, label=n)
        g.add_edge("semiosis", "process", relation_type="kind-of", weight=2)
        g.add_edge("semiosis", "stasis", relation_type="opposition", weight=1)
        g.add_edge("sign", "semiosis", relation_type="production", weight=1)
        return g

    def test_composes_clauses_in_type_priority_order(self):
        g = self._graph()
        n = compose_definitions(g)
        assert n == 4  # every node touches at least one typed edge
        assert (
            g.get_node("semiosis")["composed_definition"]
            == "Semiosis — a kind of process; produced by sign; opposed to stasis."
        )

    def test_incoming_kind_of_lists_kinds(self):
        g = self._graph()
        compose_definitions(g)
        assert (
            g.get_node("process")["composed_definition"]
            == "Process — whose kinds include semiosis."
        )

    def test_cooccurrence_only_node_gets_association_fallback(self):
        # Coverage guarantee: a node with only cooccurrence edges still gets a
        # composed definition (association phrasing, not a typed assertion).
        g = ConceptGraph(directed=True)
        g.add_node("a", label="a")
        g.add_node("b", label="b")
        g.add_edge("a", "b", relation_type="cooccurrence", weight=3)
        assert compose_definitions(g) == 2
        assert g.get_node("a")["composed_definition"] == "A — co-occurs with b."
        # All scaffold parts are honestly absent.
        assert all(v is None for v in g.get_node("a")["composed_parts"].values())

    def test_parts_scaffold_has_every_relation_type(self):
        g = self._graph()
        parts = compose_definition_parts(g, "semiosis")
        assert tuple(parts.keys()) == _COMPOSE_PART_ORDER
        assert parts["kind-of"] == "a kind of process"
        assert parts["production"] == "produced by sign"
        assert parts["opposition"] == "opposed to stasis"
        assert parts["definition"] is None
        assert parts["dependence"] is None

    def test_compose_definitions_attaches_parts(self):
        g = self._graph()
        compose_definitions(g)
        assert g.get_node("semiosis")["composed_parts"]["kind-of"] == (
            "a kind of process"
        )

    def test_weight_orders_terms_and_caps_per_clause(self):
        g = ConceptGraph(directed=True)
        g.add_node("x", label="x")
        for name, w in (("k1", 1), ("k2", 4), ("k3", 2), ("k4", 3)):
            g.add_node(name, label=name)
            g.add_edge("x", name, relation_type="kind-of", weight=w)
        assert compose_definition(g, "x") == "X — a kind of k2, k4 and k3."

    def test_multi_type_merged_edge_yields_both_clauses(self):
        # `cmapr merge` additive schema: one edge carrying relation_types.
        g = ConceptGraph(directed=True)
        g.add_node("a", label="a")
        g.add_node("b", label="b")
        g.add_edge("a", "b", relation_types=["kind-of", "production"], weight=1)
        assert compose_definition(g, "a") == "A — a kind of b; producing b."

    def test_round_trip_type_key_supported(self):
        # Graphs reloaded from D3 JSON carry `type` instead of `relation_type`.
        g = ConceptGraph(directed=True)
        g.add_node("a", label="a")
        g.add_node("b", label="b")
        g.add_edge("a", "b", type="dependence")
        assert compose_definition(g, "a") == "A — dependent on b."

    def test_recompute_drops_stale_value(self):
        g = self._graph()
        compose_definitions(g)
        g.graph.remove_edge("semiosis", "process")
        g.graph.remove_edge("semiosis", "stasis")
        g.graph.remove_edge("sign", "semiosis")
        compose_definitions(g)
        assert "composed_definition" not in g.get_node("semiosis")
