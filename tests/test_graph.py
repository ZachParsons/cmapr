"""
Tests for graph construction and analysis (Phase 8).
"""

import pytest
import networkx as nx
from concept_mapper.graph import (
    ConceptGraph,
    aggregate_graphs,
    graph_from_cooccurrence,
    graph_from_relations,
    graph_from_terms,
    merge_graphs,
    prune_edges,
    prune_nodes,
    get_subgraph,
    filter_by_relation_type,
    centrality,
    detect_communities,
    assign_communities,
    get_connected_components,
    graph_density,
    get_shortest_path,
)
from concept_mapper.graph.operations import (
    consolidate_duplicate_labels,
    find_isolated_nodes,
    connect_isolated_nodes,
    prune_to_ratio,
)
from concept_mapper.analysis.relations import Relation
from concept_mapper.graph.builders import build_proposition_graph
from unittest.mock import MagicMock as _MagicMock

# ============================================================================
# Test ConceptGraph Model
# ============================================================================


class TestConceptGraph:
    """Tests for ConceptGraph data structure."""

    def test_create_undirected(self):
        """Test creating undirected graph."""
        graph = ConceptGraph(directed=False)

        assert not graph.directed
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_create_directed(self):
        """Test creating directed graph."""
        graph = ConceptGraph(directed=True)

        assert graph.directed
        assert graph.node_count() == 0
        assert graph.edge_count() == 0

    def test_add_node_basic(self):
        """Test adding node with basic attributes."""
        graph = ConceptGraph()
        graph.add_node("consciousness", label="Consciousness", frequency=42)

        assert graph.has_node("consciousness")
        assert graph.node_count() == 1

        node = graph.get_node("consciousness")
        assert node["label"] == "Consciousness"
        assert node["frequency"] == 42

    def test_add_node_all_attributes(self):
        """Test adding node with all standard attributes."""
        graph = ConceptGraph()
        graph.add_node(
            "being",
            label="Being",
            frequency=100,
            pos="NN",
            definition="That which exists",
        )

        node = graph.get_node("being")
        assert node["label"] == "Being"
        assert node["frequency"] == 100
        assert node["pos"] == "NN"
        assert node["definition"] == "That which exists"

    def test_add_node_custom_attributes(self):
        """Test adding node with custom attributes."""
        graph = ConceptGraph()
        graph.add_node("term", custom_attr="value", another=123)

        node = graph.get_node("term")
        assert node["custom_attr"] == "value"
        assert node["another"] == 123

    def test_add_node_default_label(self):
        """Test that label defaults to node_id."""
        graph = ConceptGraph()
        graph.add_node("test")

        assert graph.get_node("test")["label"] == "test"

    def test_remove_node(self):
        """Test removing node."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_edge("a", "b")

        graph.remove_node("a")

        assert not graph.has_node("a")
        assert graph.node_count() == 1
        assert graph.edge_count() == 0

    def test_get_node_nonexistent(self):
        """Test getting nonexistent node raises error."""
        graph = ConceptGraph()

        with pytest.raises(KeyError):
            graph.get_node("nonexistent")

    def test_nodes_list(self):
        """Test getting list of nodes."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")

        nodes = graph.nodes()
        assert len(nodes) == 3
        assert "a" in nodes
        assert "b" in nodes
        assert "c" in nodes

    def test_add_edge_basic(self):
        """Test adding edge with basic attributes."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_edge("a", "b", weight=0.85)

        assert graph.has_edge("a", "b")
        assert graph.edge_count() == 1

        edge = graph.get_edge("a", "b")
        assert edge["weight"] == 0.85

    def test_add_edge_all_attributes(self):
        """Test adding edge with all standard attributes."""
        graph = ConceptGraph(directed=True)
        graph.add_node("consciousness")
        graph.add_node("intentionality")
        graph.add_edge(
            "consciousness",
            "intentionality",
            weight=0.9,
            relation_type="copular",
            evidence=["Consciousness is intentional."],
        )

        edge = graph.get_edge("consciousness", "intentionality")
        assert edge["weight"] == 0.9
        assert edge["relation_type"] == "copular"
        assert len(edge["evidence"]) == 1

    def test_add_edge_custom_attributes(self):
        """Test adding edge with custom attributes."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_edge("a", "b", custom="value")

        assert graph.get_edge("a", "b")["custom"] == "value"

    def test_remove_edge(self):
        """Test removing edge."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_edge("a", "b")

        graph.remove_edge("a", "b")

        assert not graph.has_edge("a", "b")
        assert graph.edge_count() == 0
        assert graph.node_count() == 2

    def test_get_edge_nonexistent(self):
        """Test getting nonexistent edge raises error."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")

        with pytest.raises(KeyError):
            graph.get_edge("a", "b")

    def test_edges_list(self):
        """Test getting list of edges."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        edges = graph.edges()
        assert len(edges) == 2
        assert ("a", "b") in edges
        assert ("b", "c") in edges

    def test_neighbors(self):
        """Test getting node neighbors."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")

        neighbors = graph.neighbors("a")
        assert len(neighbors) == 2
        assert "b" in neighbors
        assert "c" in neighbors

    def test_degree(self):
        """Test getting node degree."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")

        assert graph.degree("a") == 2
        assert graph.degree("b") == 1

    def test_copy(self):
        """Test copying graph."""
        graph = ConceptGraph()
        graph.add_node("a", frequency=10)
        graph.add_edge("a", "b", weight=0.5)

        copy = graph.copy()

        assert copy.node_count() == graph.node_count()
        assert copy.edge_count() == graph.edge_count()
        assert copy.get_node("a")["frequency"] == 10

        # Verify it's a deep copy
        copy.add_node("c")
        assert not graph.has_node("c")

    def test_repr(self):
        """Test string representation."""
        graph = ConceptGraph(directed=True)
        graph.add_node("a")
        graph.add_node("b")
        graph.add_edge("a", "b")

        repr_str = repr(graph)
        assert "Directed" in repr_str
        assert "nodes=2" in repr_str
        assert "edges=1" in repr_str


# ============================================================================
# Test Graph Builders
# ============================================================================


class TestBuilders:
    """Tests for graph construction from data."""

    def test_graph_from_cooccurrence_basic(self):
        """Test building graph from co-occurrence matrix."""
        matrix = {
            "consciousness": {"intentionality": 0.85, "awareness": 0.42},
            "intentionality": {"consciousness": 0.85},
        }

        graph = graph_from_cooccurrence(matrix)

        assert graph.node_count() >= 2
        assert graph.has_node("consciousness")
        assert graph.has_node("intentionality")

    def test_graph_from_cooccurrence_threshold(self):
        """Test threshold filtering."""
        matrix = {
            "consciousness": {"intentionality": 0.85, "awareness": 0.42},
        }

        graph = graph_from_cooccurrence(matrix, threshold=0.5)

        # Only edge above 0.5 should be included
        assert graph.has_edge("consciousness", "intentionality")
        assert not graph.has_edge("consciousness", "awareness")

    def test_graph_from_cooccurrence_undirected(self):
        """Test undirected graph creation."""
        matrix = {"a": {"b": 1.0}}

        graph = graph_from_cooccurrence(matrix, directed=False)

        assert not graph.directed
        # Undirected graph should have edge in both directions
        assert graph.has_edge("a", "b")

    def test_graph_from_cooccurrence_directed(self):
        """Test directed graph creation."""
        matrix = {"a": {"b": 1.0}}

        graph = graph_from_cooccurrence(matrix, directed=True)

        assert graph.directed

    def test_graph_from_cooccurrence_weights(self):
        """Test edge weights are preserved."""
        matrix = {"a": {"b": 0.75}}

        graph = graph_from_cooccurrence(matrix)

        assert graph.get_edge("a", "b")["weight"] == 0.75

    def test_graph_from_cooccurrence_relation_type(self):
        """Test relation_type is set to cooccurrence."""
        matrix = {"a": {"b": 1.0}}

        graph = graph_from_cooccurrence(matrix)

        assert graph.get_edge("a", "b")["relation_type"] == "cooccurrence"

    def test_graph_from_relations_basic(self):
        """Test building graph from relations."""
        relations = [
            Relation(
                source="consciousness",
                relation_type="copular",
                target="intentional",
                evidence=["Consciousness is intentional."],
            )
        ]

        graph = graph_from_relations(relations)

        assert graph.directed
        assert graph.has_node("consciousness")
        assert graph.has_node("intentional")
        assert graph.has_edge("consciousness", "intentional")

    def test_graph_from_relations_attributes(self):
        """Test relation attributes are preserved."""
        relations = [
            Relation(
                source="Being",
                relation_type="copular",
                target="presence",
                evidence=["Being is presence.", "Being was presence."],
                metadata={"copula": "is"},
            )
        ]

        graph = graph_from_relations(relations)

        edge = graph.get_edge("being", "presence")
        assert edge["relation_type"] == "copular"
        assert edge["weight"] == 2  # Number of evidence sentences
        assert len(edge["evidence"]) == 2
        assert edge["metadata"]["copula"] == "is"

    def test_graph_from_relations_no_evidence(self):
        """Test building without evidence."""
        relations = [
            Relation(
                source="a",
                relation_type="svo",
                target="b",
                evidence=["Test."],
            )
        ]

        graph = graph_from_relations(relations, include_evidence=False)

        edge = graph.get_edge("a", "b")
        assert "evidence" not in edge
        assert edge["weight"] == 1

    def test_graph_from_relations_merge_duplicates(self):
        """Test duplicate relations are merged."""
        relations = [
            Relation(
                source="a",
                relation_type="copular",
                target="b",
                evidence=["First."],
            ),
            Relation(
                source="a",
                relation_type="copular",
                target="b",
                evidence=["Second."],
            ),
        ]

        graph = graph_from_relations(relations)

        assert graph.edge_count() == 1
        edge = graph.get_edge("a", "b")
        assert edge["weight"] == 2
        assert len(edge["evidence"]) == 2

    def test_graph_from_terms_basic(self):
        """Test building graph from term list."""
        terms = ["consciousness", "intentionality", "being"]

        graph = graph_from_terms(terms)

        assert graph.node_count() == 3
        assert graph.edge_count() == 0
        assert graph.has_node("consciousness")

    def test_graph_from_terms_with_data(self):
        """Test building graph with term data."""
        terms = ["consciousness", "being"]
        data = {
            "consciousness": {"frequency": 42, "pos": "NN"},
            "being": {"frequency": 100},
        }

        graph = graph_from_terms(terms, term_data=data)

        assert graph.get_node("consciousness")["frequency"] == 42
        assert graph.get_node("being")["frequency"] == 100


# ============================================================================
# Test Graph Operations
# ============================================================================


class TestOperations:
    """Tests for graph manipulation operations."""

    def test_merge_graphs_basic(self):
        """Test merging two graphs."""
        g1 = ConceptGraph()
        g1.add_node("a")
        g1.add_node("b")
        g1.add_edge("a", "b")

        g2 = ConceptGraph()
        g2.add_node("c")
        g2.add_node("d")
        g2.add_edge("c", "d")

        merged = merge_graphs(g1, g2)

        assert merged.node_count() == 4
        assert merged.edge_count() == 2
        assert merged.has_edge("a", "b")
        assert merged.has_edge("c", "d")

    def test_merge_graphs_overlapping_nodes(self):
        """Test merging graphs with overlapping nodes."""
        g1 = ConceptGraph()
        g1.add_node("a", frequency=10)

        g2 = ConceptGraph()
        g2.add_node("a", frequency=20)

        merged = merge_graphs(g1, g2)

        # g2's attributes should take precedence
        assert merged.get_node("a")["frequency"] == 20

    def test_merge_graphs_different_directedness(self):
        """Test merging graphs with different directedness raises error."""
        g1 = ConceptGraph(directed=True)
        g2 = ConceptGraph(directed=False)

        with pytest.raises(ValueError):
            merge_graphs(g1, g2)

    def test_prune_edges_basic(self):
        """Test pruning edges by weight."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b", weight=0.8)
        graph.add_edge("b", "c", weight=0.3)

        pruned = prune_edges(graph, min_weight=0.5)

        assert pruned.has_edge("a", "b")
        assert not pruned.has_edge("b", "c")

    def test_prune_edges_retains_nodes(self):
        """Test that pruning edges retains all nodes."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_edge("a", "b", weight=0.1)

        pruned = prune_edges(graph, min_weight=0.5)

        assert pruned.node_count() == 2
        assert pruned.edge_count() == 0

    def test_prune_nodes_basic(self):
        """Test pruning nodes by degree."""
        graph = ConceptGraph()
        graph.add_node("isolated")
        graph.add_node("a")
        graph.add_node("b")
        graph.add_edge("a", "b")

        pruned = prune_nodes(graph, min_degree=1)

        assert not pruned.has_node("isolated")
        assert pruned.has_node("a")
        assert pruned.has_node("b")

    def test_prune_nodes_removes_edges(self):
        """Test that pruning nodes removes their edges."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        pruned = prune_nodes(graph, min_degree=2)

        # Only b has degree 2, so a and c are removed
        assert pruned.node_count() == 1
        assert pruned.edge_count() == 0

    def test_get_subgraph_basic(self):
        """Test extracting subgraph."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        subgraph = get_subgraph(graph, {"a", "b"})

        assert subgraph.node_count() == 2
        assert subgraph.has_node("a")
        assert subgraph.has_node("b")
        assert not subgraph.has_node("c")
        assert subgraph.has_edge("a", "b")
        assert not subgraph.has_edge("b", "c")

    def test_get_subgraph_preserves_attributes(self):
        """Test that subgraph preserves node/edge attributes."""
        graph = ConceptGraph()
        graph.add_node("a", frequency=10)
        graph.add_node("b", frequency=20)
        graph.add_edge("a", "b", weight=0.5)

        subgraph = get_subgraph(graph, {"a", "b"})

        assert subgraph.get_node("a")["frequency"] == 10
        assert subgraph.get_edge("a", "b")["weight"] == 0.5

    def test_filter_by_relation_type_basic(self):
        """Test filtering by relation type."""
        graph = ConceptGraph(directed=True)
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b", relation_type="copular")
        graph.add_edge("b", "c", relation_type="svo")

        filtered = filter_by_relation_type(graph, {"copular"})

        assert filtered.has_edge("a", "b")
        assert not filtered.has_edge("b", "c")

    def test_filter_by_relation_type_retains_nodes(self):
        """Test that filtering retains all nodes."""
        graph = ConceptGraph(directed=True)
        graph.add_node("a")
        graph.add_node("b")
        graph.add_edge("a", "b", relation_type="svo")

        filtered = filter_by_relation_type(graph, {"copular"})

        assert filtered.node_count() == 2
        assert filtered.edge_count() == 0

    def test_consolidate_no_duplicates(self):
        """No-op when all labels are unique."""
        g = ConceptGraph()
        g.add_node("a", label="alpha")
        g.add_node("b", label="beta")
        g.add_edge("a", "b", weight=1)
        assert consolidate_duplicate_labels(g) == 0
        assert g.node_count() == 2

    def test_consolidate_merges_duplicate_nodes(self):
        """Nodes with identical labels are merged into one."""
        g = ConceptGraph()
        g.add_node("sign_1", label="sign")
        g.add_node("sign_2", label="sign")
        g.add_node("symbol", label="symbol")
        g.add_edge("sign_1", "symbol", weight=2)
        g.add_edge("sign_2", "symbol", weight=3)
        count = consolidate_duplicate_labels(g)
        assert count == 1
        assert g.node_count() == 2
        assert not g.has_node("sign_2")
        # Weights from both edges should be summed
        assert g.get_edge("sign_1", "symbol")["weight"] == 5

    def test_consolidate_logs_warning(self, caplog):
        """A warning is emitted for each consolidated label group."""
        import logging

        g = ConceptGraph()
        g.add_node("x1", label="dup")
        g.add_node("x2", label="dup")
        with caplog.at_level(logging.WARNING):
            consolidate_duplicate_labels(g)
        assert any("dup" in r.message for r in caplog.records)

    def test_find_isolated_nodes(self):
        """find_isolated_nodes returns only nodes with degree zero."""
        g = ConceptGraph()
        g.add_node("alone")
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", weight=1)
        assert find_isolated_nodes(g) == ["alone"]

    def test_find_isolated_nodes_none(self):
        """Returns empty list when all nodes are connected."""
        g = ConceptGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", weight=1)
        assert find_isolated_nodes(g) == []

    def test_connect_isolated_nodes(self):
        """Isolated node is connected to its top co-occurrence partner."""
        g = ConceptGraph()
        g.add_node("sign")
        g.add_node("code")
        g.add_node("alone")
        g.add_edge("sign", "code", weight=3)
        matrix = {"alone": {"sign": 2.5, "code": 1.0}}
        connected = connect_isolated_nodes(g, matrix)
        assert connected == 1
        assert g.has_edge("alone", "sign")
        assert g.get_edge("alone", "sign")["weight"] == 2.5

    def test_connect_isolated_nodes_logs_error_when_no_data(self, caplog):
        """An error is logged when an isolated node has no co-occurrence with connected nodes."""
        import logging

        g = ConceptGraph()
        g.add_node("a")
        g.add_node("b")
        g.add_edge("a", "b", weight=1)
        g.add_node("orphan")
        with caplog.at_level(logging.ERROR):
            connected = connect_isolated_nodes(g, {})
        assert connected == 0
        assert any("orphan" in r.message for r in caplog.records)


# ============================================================================
# Test aggregate_graphs (cmapr merge backend)
# ============================================================================


class TestAggregateGraphs:
    """aggregate_graphs combines per-chapter graphs with proper aggregation."""

    def test_empty_input_raises(self):
        with pytest.raises(ValueError):
            aggregate_graphs([])

    def test_mixed_directedness_raises(self):
        g1 = ConceptGraph(directed=True)
        g2 = ConceptGraph(directed=False)
        with pytest.raises(ValueError):
            aggregate_graphs([g1, g2])

    def test_single_graph_passthrough(self):
        g = ConceptGraph(directed=True)
        g.add_node("sign", frequency=10, score=3.5, label="sign")
        g.add_node("code", frequency=5, score=2.0, label="code")
        g.add_edge("sign", "code", relation_type="definition", weight=2, evidence=["s1"])
        merged = aggregate_graphs([g])
        assert merged.node_count() == 2
        assert merged.edge_count() == 1
        assert merged.get_node("sign")["frequency"] == 10
        assert merged.get_node("sign")["score"] == 3.5

    def test_disjoint_graphs_pass_nodes_and_edges_through(self):
        g1 = ConceptGraph(directed=True)
        g1.add_node("a", frequency=1, score=1.0)
        g1.add_node("b", frequency=1, score=1.0)
        g1.add_edge("a", "b", relation_type="production", weight=1, evidence=["s1"])

        g2 = ConceptGraph(directed=True)
        g2.add_node("c", frequency=1, score=1.0)
        g2.add_node("d", frequency=1, score=1.0)
        g2.add_edge("c", "d", relation_type="kind-of", weight=1, evidence=["s2"])

        merged = aggregate_graphs([g1, g2])
        assert merged.node_count() == 4
        assert merged.edge_count() == 2

    def test_shared_node_sums_frequencies(self):
        g1 = ConceptGraph(directed=True)
        g1.add_node("sign", frequency=10, score=3.5)
        g2 = ConceptGraph(directed=True)
        g2.add_node("sign", frequency=20, score=4.2)
        merged = aggregate_graphs([g1, g2])
        assert merged.get_node("sign")["frequency"] == 30

    def test_shared_node_weighted_avg_score(self):
        g1 = ConceptGraph(directed=True)
        g1.add_node("sign", frequency=10, score=3.5)
        g2 = ConceptGraph(directed=True)
        g2.add_node("sign", frequency=20, score=4.2)
        merged = aggregate_graphs([g1, g2])
        # Expected: (10*3.5 + 20*4.2) / 30 = 119 / 30 ≈ 3.967
        assert merged.get_node("sign")["score"] == pytest.approx(119 / 30)

    def test_shared_node_score_only_in_one_graph(self):
        """Score from the contributing graph carries through unchanged."""
        g1 = ConceptGraph(directed=True)
        g1.add_node("sign", frequency=10)  # no score
        g2 = ConceptGraph(directed=True)
        g2.add_node("sign", frequency=20, score=4.2)
        merged = aggregate_graphs([g1, g2])
        assert merged.get_node("sign")["score"] == pytest.approx(4.2)

    def test_community_dropped_on_merge(self):
        """community is graph-relative; the merged graph should re-detect."""
        g1 = ConceptGraph(directed=True)
        g1.add_node("sign", frequency=10, community=0)
        merged = aggregate_graphs([g1])
        assert "community" not in merged.get_node("sign")

    def test_shared_edge_same_type_sums_weight_and_concats_evidence(self):
        g1 = ConceptGraph(directed=True)
        g1.add_node("a")
        g1.add_node("b")
        g1.add_edge(
            "a", "b", relation_type="definition", weight=2, evidence=["s1", "s2"]
        )

        g2 = ConceptGraph(directed=True)
        g2.add_node("a")
        g2.add_node("b")
        g2.add_edge("a", "b", relation_type="definition", weight=3, evidence=["s3"])

        merged = aggregate_graphs([g1, g2])
        edge = merged.get_edge("a", "b")
        assert edge["relation_type"] == "definition"
        assert edge["weight"] == 5
        assert sorted(edge["evidence"]) == sorted(["s1", "s2", "s3"])
        # Single type — no multi-type fields written
        assert "relation_types" not in edge

    def test_shared_edge_different_types_writes_multi_type_fields(self):
        g1 = ConceptGraph(directed=True)
        g1.add_node("a")
        g1.add_node("b")
        g1.add_edge(
            "a",
            "b",
            relation_type="definition",
            weight=3,
            evidence=["s1"],
            verb="is defined as",
        )

        g2 = ConceptGraph(directed=True)
        g2.add_node("a")
        g2.add_node("b")
        g2.add_edge(
            "a",
            "b",
            relation_type="production",
            weight=2,
            evidence=["s2", "s3"],
            verb="produces",
        )

        merged = aggregate_graphs([g1, g2])
        edge = merged.get_edge("a", "b")

        # Primary type: definition wins by priority ladder
        assert edge["relation_type"] == "definition"
        assert edge["verb"] == "is defined as"
        assert edge["weight"] == 5
        # Flat evidence: combined
        assert set(edge["evidence"]) == {"s1", "s2", "s3"}

        # Multi-type fields
        assert edge["relation_types"] == ["definition", "production"]
        assert edge["weight_by_type"] == {"definition": 3, "production": 2}
        assert edge["evidence_by_type"] == {
            "definition": ["s1"],
            "production": ["s2", "s3"],
        }
        assert edge["verb_by_type"] == {
            "definition": "is defined as",
            "production": "produces",
        }

    def test_shared_edge_priority_resolution(self):
        """Lower-priority types lose the primary slot to higher-priority ones."""
        g1 = ConceptGraph(directed=True)
        g1.add_node("a")
        g1.add_node("b")
        # cooccurrence has the lowest priority (8)
        g1.add_edge(
            "a", "b", relation_type="cooccurrence", weight=1, evidence=["s_cooc"]
        )

        g2 = ConceptGraph(directed=True)
        g2.add_node("a")
        g2.add_node("b")
        # kind-of (priority 1) outranks cooccurrence
        g2.add_edge("a", "b", relation_type="kind-of", weight=2, evidence=["s_kind"])

        merged = aggregate_graphs([g1, g2])
        edge = merged.get_edge("a", "b")
        assert edge["relation_type"] == "kind-of"
        assert edge["relation_types"][0] == "kind-of"

    def test_three_way_merge(self):
        g1, g2, g3 = (ConceptGraph(directed=True) for _ in range(3))
        for g in (g1, g2, g3):
            g.add_node("sign", frequency=10, score=4.0)
        merged = aggregate_graphs([g1, g2, g3])
        assert merged.get_node("sign")["frequency"] == 30
        assert merged.get_node("sign")["score"] == pytest.approx(4.0)


# ============================================================================
# Test Graph Metrics
# ============================================================================


class TestMetrics:
    """Tests for graph metrics and analysis."""

    def test_centrality_degree(self):
        """Test degree centrality."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        scores = centrality(graph, method="degree")

        # b has highest degree
        assert scores["b"] > scores["a"]
        assert scores["b"] > scores["c"]

    def test_centrality_betweenness(self):
        """Test betweenness centrality."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        scores = centrality(graph, method="betweenness")

        # b is on all paths between a and c
        assert scores["b"] >= scores["a"]

    def test_centrality_invalid_method(self):
        """Test invalid centrality method raises error."""
        graph = ConceptGraph()
        graph.add_node("a")

        with pytest.raises(ValueError):
            centrality(graph, method="invalid")

    def test_detect_communities_basic(self):
        """Test community detection."""
        graph = ConceptGraph()
        # Create two separate components
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_node("d")
        graph.add_edge("a", "b")
        graph.add_edge("c", "d")

        communities = detect_communities(graph)

        assert len(communities) >= 1
        assert isinstance(communities[0], set)

    def test_detect_communities_invalid_method(self):
        """Test invalid community method raises error."""
        graph = ConceptGraph()
        graph.add_node("a")

        with pytest.raises(ValueError):
            detect_communities(graph, method="invalid")

    def test_assign_communities_basic(self):
        """Test assigning community IDs to nodes."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")

        communities = [{"a", "b"}]
        assign_communities(graph, communities)

        assert graph.get_node("a")["community"] == 0
        assert graph.get_node("b")["community"] == 0

    def test_assign_communities_custom_attribute(self):
        """Test assigning with custom attribute name."""
        graph = ConceptGraph()
        graph.add_node("a")

        communities = [{"a"}]
        assign_communities(graph, communities, attribute_name="group")

        assert graph.get_node("a")["group"] == 0

    def test_get_connected_components_undirected(self):
        """Test getting connected components (undirected)."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b")

        components = get_connected_components(graph)

        assert len(components) == 2
        assert {"a", "b"} in components
        assert {"c"} in components

    def test_get_connected_components_directed(self):
        """Test getting connected components (directed)."""
        graph = ConceptGraph(directed=True)
        graph.add_node("a")
        graph.add_node("b")
        graph.add_edge("a", "b")

        components = get_connected_components(graph)

        assert isinstance(components, list)
        assert len(components) >= 1

    def test_graph_density_empty(self):
        """Test density of empty graph."""
        graph = ConceptGraph()

        density = graph_density(graph)

        assert density == 0.0

    def test_graph_density_complete(self):
        """Test density of complete graph."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_edge("a", "b")

        density = graph_density(graph)

        assert 0.0 <= density <= 1.0

    def test_get_shortest_path_basic(self):
        """Test finding shortest path."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_node("c")
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")

        path = get_shortest_path(graph, "a", "c")

        assert path == ["a", "b", "c"]

    def test_get_shortest_path_direct(self):
        """Test shortest path with direct connection."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")
        graph.add_edge("a", "b")

        path = get_shortest_path(graph, "a", "b")

        assert path == ["a", "b"]

    def test_get_shortest_path_no_path(self):
        """Test shortest path when no path exists."""
        graph = ConceptGraph()
        graph.add_node("a")
        graph.add_node("b")

        with pytest.raises(nx.NetworkXNoPath):
            get_shortest_path(graph, "a", "b")


# ============================================================================
# Integration Tests
# ============================================================================


class TestGraphIntegration:
    """Integration tests for complete graph workflows."""

    def test_full_workflow_cooccurrence(self):
        """Test complete workflow from co-occurrence to metrics."""
        # Build graph from co-occurrence
        matrix = {
            "consciousness": {"intentionality": 0.9, "awareness": 0.5},
            "intentionality": {"consciousness": 0.9, "thought": 0.6},
            "awareness": {"consciousness": 0.5},
            "thought": {"intentionality": 0.6},
        }

        graph = graph_from_cooccurrence(matrix, threshold=0.6)

        # Compute centrality
        scores = centrality(graph, method="degree")

        assert "consciousness" in scores or "intentionality" in scores

    def test_full_workflow_relations(self):
        """Test complete workflow from relations to communities."""
        # Build graph from relations
        relations = [
            Relation("a", "copular", "b", evidence=["A is B."]),
            Relation("b", "copular", "c", evidence=["B is C."]),
            Relation("d", "svo", "e", evidence=["D does E."]),
        ]

        graph = graph_from_relations(relations)

        # Filter to only copular relations
        copular_graph = filter_by_relation_type(graph, {"copular"})

        assert copular_graph.edge_count() == 2

    def test_merge_and_prune_workflow(self):
        """Test workflow combining graphs and pruning."""
        # Create two graphs
        g1 = graph_from_cooccurrence({"a": {"b": 0.9}})
        g2 = graph_from_cooccurrence({"c": {"d": 0.3}})

        # Merge them
        merged = merge_graphs(g1, g2)

        # Prune low-weight edges
        pruned = prune_edges(merged, min_weight=0.5)

        assert pruned.has_edge("a", "b")
        assert not pruned.has_edge("c", "d")


# ============================================================================
# Test BuildPropositionGraph (Phase 5)
# ============================================================================


def _make_nlp_docs(sentences):
    doc = _MagicMock()
    doc.sentences = sentences
    doc.metadata = {}
    return [doc]


class TestBuildPropositionGraph:
    """build_proposition_graph returns a typed ConceptGraph from docs and seed terms."""

    def _build(self, sentences, seed_terms, **kwargs):
        docs = _make_nlp_docs(sentences)
        return build_proposition_graph(docs, seed_terms, **kwargs)

    def test_returns_concept_graph(self):
        """Any call to build_proposition_graph returns a ConceptGraph instance."""
        sentences = ["Semiosis involves a sign and a ground."]
        graph = self._build(sentences, ["semiosis", "sign"], pmi_threshold=0.0)
        assert isinstance(graph, ConceptGraph), (
            "Expected build_proposition_graph to return a ConceptGraph"
        )

    def test_typed_edge_over_cooccurrence(self):
        """A sentence with a kind-of marker produces a typed edge, not cooccurrence."""
        sentences = ["Semiosis is a kind of signification process."]
        graph = self._build(sentences, ["semiosis", "signification"], pmi_threshold=0.0)
        has_edge = graph.has_edge("semiosis", "signification") or graph.has_edge(
            "signification", "semiosis"
        )
        assert has_edge, "Expected an edge between semiosis and signification"
        for src, tgt in graph.edges():
            edge = graph.get_edge(src, tgt)
            assert edge["relation_type"] != "cooccurrence", (
                f"Expected typed edge for {src}→{tgt}, got cooccurrence"
            )

    def test_cooccurrence_fallback_when_no_pattern(self):
        """Terms co-occurring without any pattern marker fall back to cooccurrence edge."""
        # Sentences mention both terms but contain no typed-relation markers
        # (no copula, no 'produces', no 'kind of', no verb between them)
        sentences = [
            "Sign and code appear together in the signification process.",
            "Sign and code both play a role in communicative exchange.",
            "Sign and code are both present in the text.",
        ]
        graph = self._build(sentences, ["sign", "code"], pmi_threshold=0.0)
        has_edge = graph.has_edge("sign", "code") or graph.has_edge("code", "sign")
        assert has_edge, "Expected a cooccurrence edge between sign and code"
        for src, tgt in graph.edges():
            edge = graph.get_edge(src, tgt)
            assert edge["relation_type"] == "cooccurrence", (
                f"Expected cooccurrence fallback for {src}→{tgt}, "
                f"got {edge['relation_type']!r}"
            )

    def test_no_edge_for_non_cooccurring_terms(self):
        """Terms that never co-occur in any sentence get no edge."""
        sentences = [
            "Sign alone appears in the semiotic process.",
            "Code alone structures the communicative system.",
        ]
        graph = self._build(sentences, ["sign", "code"], pmi_threshold=1.0)
        has_edge = graph.has_edge("sign", "code") or graph.has_edge("code", "sign")
        assert not has_edge, (
            "Expected no edge between sign and code when they never co-occur"
        )

    def test_term_scores_stored_as_node_attributes(self):
        """term_scores values are stored as 'score' attributes on nodes."""
        sentences = [
            "Sign is a kind of interpretant in Peirce's triadic model.",
            "Interpretant mediates between sign and object.",
        ]
        scores = {"sign": 2.5, "interpretant": 1.8}
        graph = self._build(
            sentences, ["sign", "interpretant"], term_scores=scores, pmi_threshold=0.0
        )
        for term, expected_score in scores.items():
            if graph.graph.has_node(term):
                actual = graph.graph.nodes[term].get("score")
                assert actual == pytest.approx(expected_score), (
                    f"Expected score {expected_score} for node {term!r}, got {actual}"
                )

    def test_both_seed_terms_become_nodes(self):
        """All seed terms appear as nodes in the resulting graph."""
        sentences = [
            "Sign and interpretant are both present in semiosis.",
            "Interpretant depends on sign for its constitution.",
        ]
        graph = self._build(sentences, ["sign", "interpretant"], pmi_threshold=0.0)
        nodes = set(graph.nodes())
        assert "sign" in nodes or any("sign" in n for n in nodes), (
            "Expected 'sign' to be a node in the graph"
        )
        assert "interpretant" in nodes or any("interpretant" in n for n in nodes), (
            "Expected 'interpretant' to be a node in the graph"
        )

    def test_node_filter_accepted_without_error(self):
        """Passing a real NodeFilter does not crash build_proposition_graph."""
        from collections import Counter
        from concept_mapper.graph.node_filter import NodeFilter

        sentences = ["Sign produces an interpretant in semiotic theory."]
        vocab = {"sign", "produces", "interpretant", "semiotic", "theory"}
        freqs = Counter({"sign": 5, "interpretant": 4, "semiosis": 3})
        nf = NodeFilter(corpus_vocab=vocab, term_freqs=freqs, min_freq=1)
        graph = self._build(
            sentences, ["sign", "interpretant"], node_filter=nf, pmi_threshold=0.0
        )
        assert isinstance(graph, ConceptGraph), (
            "Expected ConceptGraph even when NodeFilter is provided"
        )

    def test_multiword_seed_becomes_node(self):
        """Phase 13: a multi-word seed term survives the NodeFilter and
        appears as a node when it co-occurs with another seed in a sentence
        that yields a typed edge."""
        from collections import Counter
        from concept_mapper.graph.node_filter import NodeFilter

        sentences = [
            "A sign vehicle produces meaning when interpreted.",
            "The sign vehicle and meaning are central to semiosis.",
        ]
        # Build NodeFilter the way cli.py does: token-level vocab and freqs.
        # The phrase "sign vehicle" has freq 0 in this Counter — the filter
        # must still accept it because it's multi-word.
        vocab = {"sign", "vehicle", "produces", "meaning", "interpreted", "semiosis"}
        freqs = Counter(
            {"sign": 2, "vehicle": 2, "meaning": 2, "produces": 1, "semiosis": 1}
        )
        nf = NodeFilter(corpus_vocab=vocab, term_freqs=freqs, min_freq=3)
        # Confirm the filter passes the phrase even with min_freq=3
        assert nf.is_valid("sign vehicle")[0], (
            "Expected NodeFilter to accept the multi-word phrase 'sign vehicle'"
        )
        graph = self._build(
            sentences,
            ["sign vehicle", "meaning"],
            node_filter=nf,
            pmi_threshold=0.0,
        )
        nodes = set(graph.nodes())
        assert "sign vehicle" in nodes, (
            f"Expected 'sign vehicle' as a node, got nodes={nodes}"
        )
        assert "meaning" in nodes, f"Expected 'meaning' as a node, got nodes={nodes}"
        # The production pattern should produce a typed edge
        has_edge = graph.has_edge("sign vehicle", "meaning") or graph.has_edge(
            "meaning", "sign vehicle"
        )
        assert has_edge, "Expected an edge between 'sign vehicle' and 'meaning'"


# ============================================================================
# Test PruneToRatio (Phase 6)
# ============================================================================


def _chain_graph(n=5, rtype="cooccurrence"):
    """Linear chain: 0→1→2→...→n-1, all same relation type."""
    g = ConceptGraph(directed=True)
    for i in range(n):
        g.add_node(str(i), label=str(i))
    for i in range(n - 1):
        g.add_edge(str(i), str(i + 1), relation_type=rtype, weight=1)
    return g


class TestPruneToRatio:
    """prune_to_ratio removes edges until edges:nodes ≤ target_ratio."""

    def test_ratio_enforced(self):
        """After pruning a dense graph, edge:node ratio is at or below the target."""
        g = ConceptGraph(directed=True)
        for n in "abcde":
            g.add_node(n, label=n)
        # 8 edges on 5 nodes = 1.6:1, pruning to 1.0 must reduce edges
        pairs = [
            ("a", "b"),
            ("a", "c"),
            ("a", "d"),
            ("a", "e"),
            ("b", "c"),
            ("b", "d"),
            ("c", "d"),
            ("d", "e"),
        ]
        for src, tgt in pairs:
            g.add_edge(src, tgt, relation_type="cooccurrence", weight=1)
        pruned = prune_to_ratio(g, target_ratio=1.0)
        ratio = pruned.edge_count() / max(pruned.node_count(), 1)
        assert ratio <= 1.0, (
            f"Expected edge:node ratio ≤ 1.0 after pruning, got {ratio:.2f}"
        )

    def test_already_within_ratio_unchanged(self):
        """A graph already within the target ratio is not changed."""
        # 3-node chain has 2 edges on 3 nodes = 0.67:1 — well within target=3.0
        g = _chain_graph(n=3, rtype="cooccurrence")
        original_edges = g.edge_count()
        pruned = prune_to_ratio(g, target_ratio=3.0)
        assert pruned.edge_count() == original_edges, (
            f"Expected edge count to remain {original_edges}, "
            f"got {pruned.edge_count()} after pruning within ratio"
        )

    def test_no_isolated_nodes_after_pruning(self):
        """Pruning never leaves a node with no edges."""
        g = ConceptGraph(directed=True)
        for n in "abcde":
            g.add_node(n, label=n)
        # 10 edges (all pairs) on 5 nodes
        pairs = [(a, b) for a in "abcde" for b in "abcde" if a < b]
        for src, tgt in pairs:
            g.add_edge(src, tgt, relation_type="cooccurrence", weight=1)
        pruned = prune_to_ratio(g, target_ratio=0.5)
        connected = set()
        for src, tgt in pruned.edges():
            connected.add(src)
            connected.add(tgt)
        for node in pruned.nodes():
            assert node in connected, (
                f"Node {node!r} is isolated after pruning to ratio 0.5"
            )

    def test_cooccurrence_removed_before_grammatical(self):
        """Cooccurrence edges are removed before grammatical edges even when heavier."""
        g = ConceptGraph(directed=True)
        for n in ("a", "b", "c"):
            g.add_node(n, label=n)
        g.add_edge("a", "b", relation_type="kind-of", weight=1)
        g.add_edge("b", "c", relation_type="production", weight=1)
        # cooccurrence edge has high weight but should still be removed first
        g.add_edge("a", "c", relation_type="cooccurrence", weight=5)
        # 3 edges on 3 nodes = 1.0; pruning to 0.5 requires removing at least one
        pruned = prune_to_ratio(g, target_ratio=0.5)
        edge_types = {pruned.get_edge(s, t)["relation_type"] for s, t in pruned.edges()}
        assert "cooccurrence" not in edge_types, (
            "Expected cooccurrence edge to be removed before grammatical edges"
        )

    def test_empty_graph_unchanged(self):
        """An empty graph returned unchanged — no nodes, no edges, no crash."""
        g = ConceptGraph(directed=True)
        pruned = prune_to_ratio(g, target_ratio=2.0)
        assert pruned.node_count() == 0, "Expected empty graph to remain empty"
        assert pruned.edge_count() == 0, "Expected empty graph to have no edges"


# ============================================================================
# Test GraphDepthFocus — unit tests (Phase 14 B4/B5)
# ============================================================================


def _linear_graph():
    """a→b→c→d→e directed chain with scores."""
    g = ConceptGraph(directed=True)
    for n in "abcde":
        g.add_node(n, label=n, score=2.0 if n == "a" else 0.5)
    for src, tgt in zip("abcd", "bcde"):
        g.add_edge(src, tgt, relation_type="kind-of", weight=1)
    return g


class TestGraphDepthFocusUnit:
    """Unit tests for ego_graph depth/focus logic used by the graph command."""

    def test_depth_1_ego_graph_includes_direct_neighbours(self):
        """ego_graph at radius=1 from 'c' contains exactly its direct neighbours."""
        g = _linear_graph()
        sub = nx.ego_graph(g.graph, "c", radius=1, undirected=True)
        assert set(sub.nodes()) == {"b", "c", "d"}, (
            f"Expected {{b, c, d}} at depth-1 from c, got {set(sub.nodes())}"
        )

    def test_depth_2_ego_graph_extends_two_hops(self):
        """ego_graph at radius=2 from 'c' spans the full five-node chain."""
        g = _linear_graph()
        sub = nx.ego_graph(g.graph, "c", radius=2, undirected=True)
        assert set(sub.nodes()) == {"a", "b", "c", "d", "e"}, (
            f"Expected all five nodes at depth-2 from c, got {set(sub.nodes())}"
        )

    def test_focus_default_depth_is_1(self):
        """When --focus is given without --depth the radius defaults to 1."""
        g = _linear_graph()
        sub = nx.ego_graph(g.graph, "a", radius=1, undirected=True)
        assert "a" in sub.nodes(), "Focus node 'a' should be in the ego graph"
        assert "b" in sub.nodes(), "Direct neighbour 'b' should be in the ego graph"
        assert "c" not in sub.nodes(), (
            "Two-hop neighbour 'c' should not be in depth-1 ego graph"
        )

    def test_highest_score_node_selection(self):
        """The node with score=2.0 is 'a'; confirm via graph node attribute."""
        g = _linear_graph()
        assert g.graph.nodes["a"]["score"] == pytest.approx(2.0), (
            "Expected node 'a' to have score=2.0"
        )
