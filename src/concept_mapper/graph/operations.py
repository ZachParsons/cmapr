"""
Graph manipulation operations.

This module provides functions for modifying and extracting from ConceptGraphs.
"""

import logging
from collections import defaultdict
from typing import Dict, List, Set
from concept_mapper.graph.model import ConceptGraph

logger = logging.getLogger(__name__)


def find_isolated_nodes(graph: ConceptGraph) -> List[str]:
    """Return node IDs that have no edges (degree == 0)."""
    return [n for n in graph.nodes() if graph._graph.degree(n) == 0]


def connect_isolated_nodes(
    graph: ConceptGraph,
    cooccurrence_matrix: Dict[str, Dict[str, float]],
    min_degree: int = 1,
) -> int:
    """
    Connect sparse nodes by finding their strongest co-occurrence partner
    among already-sufficiently-connected nodes in the graph.

    Processes all nodes with degree < min_degree.  Use min_degree=1 to
    connect only truly isolated nodes (degree 0); use min_degree=2 for a
    hybrid approach that also connects leaf nodes (degree 1).

    Logs a warning for each connection made and an error for any node that
    cannot be connected (no co-occurrence data with connected nodes).

    Args:
        graph: ConceptGraph to modify in-place
        cooccurrence_matrix: Nested dict of term -> term -> score
        min_degree: Nodes with degree < min_degree are candidates (default: 1)

    Returns:
        Number of sparse nodes that were successfully connected
    """
    sparse = [n for n in graph.nodes() if graph._graph.degree(n) < min_degree]
    if not sparse:
        return 0

    # Anchor set: nodes already meeting the degree threshold
    anchors = {n for n in graph.nodes() if graph._graph.degree(n) >= min_degree}
    connected = 0

    for node_id in sparse:
        scores = cooccurrence_matrix.get(node_id, {})
        best = max(
            ((partner, score) for partner, score in scores.items()
             if partner in anchors and partner != node_id),
            key=lambda x: x[1],
            default=None,
        )
        if best is None:
            logger.error(
                "Sparse node %r (degree<%d) has no co-occurrence data with any "
                "anchor node — cannot connect. Fix upstream graph building.",
                node_id,
                min_degree,
            )
            continue

        partner, score = best
        logger.warning(
            "Sparse node %r connected to %r via co-occurrence fallback (score=%.3f)",
            node_id,
            partner,
            score,
        )
        graph.add_edge(node_id, partner, weight=score, relation_type="cooccurrence")
        anchors.add(node_id)
        connected += 1

    return connected


def consolidate_duplicate_labels(graph: ConceptGraph) -> int:
    """
    Merge nodes that share the same label into a single canonical node.

    Logs a warning for each set of duplicates so the upstream cause can be
    investigated and fixed.  All edges attached to duplicate nodes are
    re-wired to the canonical node; weights are summed and evidence lists
    are concatenated when edges already exist between the same pair.

    Args:
        graph: ConceptGraph to consolidate (mutated in-place)

    Returns:
        Number of label groups that were consolidated (0 means no duplicates)
    """
    label_to_ids: dict = defaultdict(list)
    for node_id in graph.nodes():
        label = graph.get_node(node_id).get("label", node_id)
        label_to_ids[label].append(node_id)

    consolidations = 0
    for label, node_ids in label_to_ids.items():
        if len(node_ids) <= 1:
            continue

        logger.warning(
            "Consolidating %d nodes with duplicate label %r: %s",
            len(node_ids),
            label,
            node_ids,
        )
        consolidations += 1
        canonical = node_ids[0]
        nx_graph = graph._graph

        for dup in node_ids[1:]:
            edges_to_add = []

            # Outgoing (and undirected) edges from dup
            for u, v, data in list(nx_graph.edges(dup, data=True)):
                other = v if u == dup else u
                if other != canonical:
                    edges_to_add.append((canonical, other, dict(data)))

            # Incoming edges (directed graphs only)
            if graph.directed:
                for u, v, data in list(nx_graph.in_edges(dup, data=True)):
                    if u != canonical:
                        edges_to_add.append((u, canonical, dict(data)))

            for u, v, data in edges_to_add:
                if graph.has_edge(u, v):
                    existing = graph.get_edge(u, v)
                    merged = {**existing, **data}
                    merged["weight"] = existing.get("weight", 1) + data.get("weight", 1)
                    combined_evidence = existing.get("evidence", []) + data.get("evidence", [])
                    if combined_evidence:
                        merged["evidence"] = combined_evidence
                    graph.add_edge(u, v, **merged)
                else:
                    graph.add_edge(u, v, **data)

            graph.remove_node(dup)

    return consolidations


def merge_graphs(g1: ConceptGraph, g2: ConceptGraph) -> ConceptGraph:
    """
    Merge two graphs into a new graph.

    Combines nodes and edges from both graphs. If both graphs have the same
    node/edge, attributes from g2 take precedence.

    Args:
        g1: First graph
        g2: Second graph

    Returns:
        New ConceptGraph containing nodes and edges from both graphs

    Raises:
        ValueError: If graphs have different directedness

    Example:
        >>> g1 = ConceptGraph()
        >>> g1.add_node("consciousness")
        >>> g2 = ConceptGraph()
        >>> g2.add_node("intentionality")
        >>> merged = merge_graphs(g1, g2)
        >>> merged.node_count()
        2
    """
    if g1.directed != g2.directed:
        raise ValueError("Cannot merge directed and undirected graphs")

    result = ConceptGraph(directed=g1.directed)

    # Add nodes from g1
    for node in g1.nodes():
        attrs = g1.get_node(node)
        result.add_node(node, **attrs)

    # Add nodes from g2 (may overwrite g1's attributes)
    for node in g2.nodes():
        attrs = g2.get_node(node)
        result.add_node(node, **attrs)

    # Add edges from g1
    for source, target in g1.edges():
        attrs = g1.get_edge(source, target)
        result.add_edge(source, target, **attrs)

    # Add edges from g2 (may overwrite g1's attributes)
    for source, target in g2.edges():
        attrs = g2.get_edge(source, target)
        result.add_edge(source, target, **attrs)

    return result


def prune_edges(graph: ConceptGraph, min_weight: float) -> ConceptGraph:
    """
    Remove edges with weight below threshold.

    Creates a new graph with only edges meeting the weight threshold.
    Nodes without any edges are retained.

    Args:
        graph: Input graph
        min_weight: Minimum edge weight to retain

    Returns:
        New ConceptGraph with low-weight edges removed

    Example:
        >>> graph = ConceptGraph()
        >>> graph.add_node("a")
        >>> graph.add_node("b")
        >>> graph.add_edge("a", "b", weight=0.5)
        >>> pruned = prune_edges(graph, min_weight=1.0)
        >>> pruned.has_edge("a", "b")
        False
        >>> pruned.node_count()
        2
    """
    result = ConceptGraph(directed=graph.directed)

    # Copy all nodes
    for node in graph.nodes():
        attrs = graph.get_node(node)
        result.add_node(node, **attrs)

    # Copy edges that meet weight threshold
    for source, target in graph.edges():
        attrs = graph.get_edge(source, target)
        weight = attrs.get("weight", 0.0)

        if weight >= min_weight:
            result.add_edge(source, target, **attrs)

    return result


def prune_nodes(graph: ConceptGraph, min_degree: int) -> ConceptGraph:
    """
    Remove nodes with degree below threshold.

    Creates a new graph with only nodes meeting the degree threshold.
    Removes all edges connected to removed nodes.

    Args:
        graph: Input graph
        min_degree: Minimum node degree to retain

    Returns:
        New ConceptGraph with low-degree nodes removed

    Example:
        >>> graph = ConceptGraph()
        >>> graph.add_node("isolated")
        >>> graph.add_node("a")
        >>> graph.add_node("b")
        >>> graph.add_edge("a", "b")
        >>> pruned = prune_nodes(graph, min_degree=1)
        >>> pruned.has_node("isolated")
        False
        >>> pruned.has_node("a")
        True
    """
    result = ConceptGraph(directed=graph.directed)

    # Identify nodes to keep
    nodes_to_keep = {node for node in graph.nodes() if graph.degree(node) >= min_degree}

    # Copy nodes that meet degree threshold
    for node in nodes_to_keep:
        attrs = graph.get_node(node)
        result.add_node(node, **attrs)

    # Copy edges where both endpoints are kept
    for source, target in graph.edges():
        if source in nodes_to_keep and target in nodes_to_keep:
            attrs = graph.get_edge(source, target)
            result.add_edge(source, target, **attrs)

    return result


def get_subgraph(graph: ConceptGraph, terms: Set[str]) -> ConceptGraph:
    """
    Extract a subgraph containing only specified terms.

    Creates a new graph with only the specified nodes and edges between them.

    Args:
        graph: Input graph
        terms: Set of node IDs to include

    Returns:
        New ConceptGraph containing only specified nodes and their connections

    Example:
        >>> graph = ConceptGraph()
        >>> graph.add_node("a")
        >>> graph.add_node("b")
        >>> graph.add_node("c")
        >>> graph.add_edge("a", "b")
        >>> graph.add_edge("b", "c")
        >>> subgraph = get_subgraph(graph, {"a", "b"})
        >>> subgraph.has_node("c")
        False
        >>> subgraph.has_edge("a", "b")
        True
    """
    result = ConceptGraph(directed=graph.directed)

    # Copy specified nodes
    for node in terms:
        if graph.has_node(node):
            attrs = graph.get_node(node)
            result.add_node(node, **attrs)

    # Copy edges between specified nodes
    for source, target in graph.edges():
        if source in terms and target in terms:
            attrs = graph.get_edge(source, target)
            result.add_edge(source, target, **attrs)

    return result


def filter_by_relation_type(
    graph: ConceptGraph,
    relation_types: Set[str],
) -> ConceptGraph:
    """
    Extract edges with specific relation types.

    Creates a new graph containing only edges with the specified relation types.
    All nodes are retained.

    Args:
        graph: Input graph
        relation_types: Set of relation types to include (e.g., {"copular", "svo"})

    Returns:
        New ConceptGraph with only specified relation types

    Example:
        >>> graph = ConceptGraph(directed=True)
        >>> graph.add_node("a")
        >>> graph.add_node("b")
        >>> graph.add_edge("a", "b", relation_type="copular")
        >>> filtered = filter_by_relation_type(graph, {"copular"})
        >>> filtered.has_edge("a", "b")
        True
    """
    result = ConceptGraph(directed=graph.directed)

    # Copy all nodes
    for node in graph.nodes():
        attrs = graph.get_node(node)
        result.add_node(node, **attrs)

    # Copy edges with matching relation types
    for source, target in graph.edges():
        attrs = graph.get_edge(source, target)
        rel_type = attrs.get("relation_type")

        if rel_type in relation_types:
            result.add_edge(source, target, **attrs)

    return result
