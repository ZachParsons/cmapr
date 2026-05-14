"""
Edge / node pruning operations on a ConceptGraph.

Three strategies:

* ``prune_edges`` — drop edges below a minimum weight.
* ``prune_nodes`` — drop nodes below a minimum degree.
* ``prune_to_ratio`` — drop edges until the edge:node ratio is at or
  below the target. Cooccurrence-first, weakest-first; never isolates a
  node. Used by ``cmapr graph``, ``cmapr merge``, ``cmapr cluster``.

All three return a new graph; the input is not mutated.
"""

from concept_mapper.graph.model import ConceptGraph


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


def prune_to_ratio(
    graph: ConceptGraph,
    target_ratio: float = 3.0,
) -> ConceptGraph:
    """
    Prune edges until edge:node ratio ≤ target_ratio.

    Pruning order (spec § Implementation priorities #3):
      1. Remove cooccurrence edges (lowest priority), except where they are
         the sole connection for a node.
      2. Remove lowest-weight non-cooccurrence edges, again protecting any
         edge that is the last connection for either endpoint.

    Operates on a copy; the original graph is not mutated.

    Args:
        graph       : ConceptGraph to prune
        target_ratio: Maximum edges-per-node ratio to reach (default 3.0)

    Returns:
        Pruned ConceptGraph (same nodes, fewer edges)
    """
    result = graph.copy()

    def _ratio(g: ConceptGraph) -> float:
        n = g.node_count()
        return g.edge_count() / n if n > 0 else 0.0

    def _sole_connection(g: ConceptGraph, src: str, tgt: str) -> bool:
        """True if removing this edge would isolate either endpoint."""
        return g._graph.degree(src) <= 1 or g._graph.degree(tgt) <= 1

    # --- Pass 1: remove cooccurrence edges (weakest first) ---
    if _ratio(result) > target_ratio:
        cooc_edges = sorted(
            [
                (src, tgt, result.get_edge(src, tgt).get("weight", 1))
                for src, tgt in result.edges()
                if result.get_edge(src, tgt).get("relation_type") == "cooccurrence"
            ],
            key=lambda x: x[2],  # ascending weight → remove weakest first
        )
        for src, tgt, _ in cooc_edges:
            if _ratio(result) <= target_ratio:
                break
            if not _sole_connection(result, src, tgt):
                result.remove_edge(src, tgt)

    # --- Pass 2: remove lowest-weight typed edges if still over ratio ---
    if _ratio(result) > target_ratio:
        typed_edges = sorted(
            [
                (src, tgt, result.get_edge(src, tgt).get("weight", 1))
                for src, tgt in result.edges()
            ],
            key=lambda x: x[2],
        )
        for src, tgt, _ in typed_edges:
            if _ratio(result) <= target_ratio:
                break
            if not _sole_connection(result, src, tgt):
                result.remove_edge(src, tgt)

    return result
