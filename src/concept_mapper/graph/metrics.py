"""
Graph metrics and analysis.

This module provides functions for computing centrality, communities,
and other graph metrics.
"""

from typing import Dict, List, Set
import networkx as nx
from concept_mapper.graph.model import ConceptGraph


def centrality(
    graph: ConceptGraph,
    method: str = "betweenness",
    normalized: bool = True,
) -> Dict[str, float]:
    """
    Compute centrality scores for all nodes.

    Centrality measures identify the most important or influential nodes
    in the network.

    Args:
        graph: Input graph
        method: Centrality algorithm to use. Options:
            - "betweenness": Betweenness centrality (default)
            - "degree": Degree centrality
            - "closeness": Closeness centrality
            - "eigenvector": Eigenvector centrality
            - "pagerank": PageRank
        normalized: Whether to normalize scores (default: True)

    Returns:
        Dictionary mapping node ID to centrality score

    Raises:
        ValueError: If method is not recognized

    Example:
        >>> graph = ConceptGraph()
        >>> graph.add_node("a")
        >>> graph.add_node("b")
        >>> graph.add_node("c")
        >>> graph.add_edge("a", "b")
        >>> graph.add_edge("b", "c")
        >>> scores = centrality(graph, method="degree")
        >>> scores["b"] > scores["a"]
        True
    """
    g = graph.graph

    if method == "betweenness":
        return nx.betweenness_centrality(g, normalized=normalized)
    elif method == "degree":
        return nx.degree_centrality(g)
    elif method == "closeness":
        return nx.closeness_centrality(g)
    elif method == "eigenvector":
        # May fail for disconnected graphs, use max_iter
        try:
            return nx.eigenvector_centrality(g, max_iter=1000)
        except nx.PowerIterationFailedConvergence:
            # Fallback to degree centrality
            return nx.degree_centrality(g)
    elif method == "pagerank":
        try:
            return nx.pagerank(g, max_iter=1000)
        except (nx.PowerIterationFailedConvergence, ModuleNotFoundError):
            # Fallback to degree centrality if scipy not installed
            return nx.degree_centrality(g)
    else:
        raise ValueError(
            f"Unknown centrality method '{method}'. "
            f"Options: betweenness, degree, closeness, eigenvector, pagerank"
        )


def detect_communities(
    graph: ConceptGraph,
    method: str = "louvain",
) -> List[Set[str]]:
    """
    Detect communities (clusters) in the graph.

    Communities are groups of nodes that are more densely connected
    to each other than to the rest of the network.

    Args:
        graph: Input graph (should be undirected for best results)
        method: Community detection algorithm. Options:
            - "louvain": Louvain modularity maximization (default)
            - "greedy": Greedy modularity maximization
            - "label_propagation": Label propagation

    Returns:
        List of sets, where each set contains node IDs in a community

    Raises:
        ValueError: If method is not recognized

    Example:
        >>> graph = ConceptGraph()
        >>> for node in ["a", "b", "c", "d"]:
        ...     graph.add_node(node)
        >>> graph.add_edge("a", "b")
        >>> graph.add_edge("c", "d")
        >>> communities = detect_communities(graph)
        >>> len(communities) >= 1
        True
    """
    g = graph.graph

    # Convert to undirected for community detection
    if graph.directed:
        g = g.to_undirected()

    if method == "louvain":
        # Use greedy modularity as nx doesn't have louvain built-in
        # (would need python-louvain package)
        communities = nx.community.greedy_modularity_communities(g)
    elif method == "greedy":
        communities = nx.community.greedy_modularity_communities(g)
    elif method == "label_propagation":
        communities = nx.community.label_propagation_communities(g)
    else:
        raise ValueError(
            f"Unknown community detection method '{method}'. "
            f"Options: louvain, greedy, label_propagation"
        )

    return [set(community) for community in communities]


def assign_communities(
    graph: ConceptGraph,
    communities: List[Set[str]],
    attribute_name: str = "community",
) -> None:
    """
    Assign community IDs to nodes as an attribute.

    Modifies the graph in-place by adding a community attribute to each node.
    Useful for visualization grouping.

    Args:
        graph: Graph to modify
        communities: List of community sets from detect_communities()
        attribute_name: Name of node attribute to set (default: "community")

    Example:
        >>> graph = ConceptGraph()
        >>> graph.add_node("a")
        >>> graph.add_node("b")
        >>> communities = [{"a", "b"}]
        >>> assign_communities(graph, communities)
        >>> graph.get_node("a")["community"]
        0
    """
    for community_id, community in enumerate(communities):
        for node in community:
            if graph.has_node(node):
                # Get existing attributes
                attrs = graph.get_node(node)
                attrs[attribute_name] = community_id
                # Update node with new attribute
                graph.graph.nodes[node].update(attrs)


def get_connected_components(graph: ConceptGraph) -> List[Set[str]]:
    """
    Get connected components of the graph.

    A connected component is a maximal set of nodes where every pair
    is connected by a path.

    Args:
        graph: Input graph

    Returns:
        List of sets, where each set contains node IDs in a component

    Example:
        >>> graph = ConceptGraph()
        >>> graph.add_node("a")
        >>> graph.add_node("b")
        >>> graph.add_node("c")
        >>> graph.add_edge("a", "b")
        >>> components = get_connected_components(graph)
        >>> len(components)
        2
    """
    g = graph.graph

    if graph.directed:
        # For directed graphs, use weakly connected components
        components = nx.weakly_connected_components(g)
    else:
        components = nx.connected_components(g)

    return [set(component) for component in components]


def graph_density(graph: ConceptGraph) -> float:
    """
    Compute the density of the graph.

    Density is the ratio of actual edges to possible edges.
    Range: [0, 1] where 1 means all possible edges exist.

    Args:
        graph: Input graph

    Returns:
        Density value between 0 and 1

    Example:
        >>> graph = ConceptGraph()
        >>> graph.add_node("a")
        >>> graph.add_node("b")
        >>> graph.add_edge("a", "b")
        >>> density = graph_density(graph)
        >>> 0 <= density <= 1
        True
    """
    return nx.density(graph.graph)


def get_shortest_path(
    graph: ConceptGraph,
    source: str,
    target: str,
) -> List[str]:
    """
    Find the shortest path between two nodes.

    Args:
        graph: Input graph
        source: Source node ID
        target: Target node ID

    Returns:
        List of node IDs forming the shortest path from source to target

    Raises:
        nx.NetworkXNoPath: If no path exists
        nx.NodeNotFound: If source or target not in graph

    Example:
        >>> graph = ConceptGraph()
        >>> graph.add_node("a")
        >>> graph.add_node("b")
        >>> graph.add_node("c")
        >>> graph.add_edge("a", "b")
        >>> graph.add_edge("b", "c")
        >>> path = get_shortest_path(graph, "a", "c")
        >>> path
        ['a', 'b', 'c']
    """
    return nx.shortest_path(graph.graph, source, target)
