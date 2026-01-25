"""
Graph manipulation operations.

This module provides functions for modifying and extracting from ConceptGraphs.
"""

from typing import Set
from concept_mapper.graph.model import ConceptGraph


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
