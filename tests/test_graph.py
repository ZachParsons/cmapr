"""
Tests for graph construction and analysis (Phase 8).
"""

import pytest
import networkx as nx
from concept_mapper.graph import (
    ConceptGraph,
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
from concept_mapper.analysis.relations import Relation

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
