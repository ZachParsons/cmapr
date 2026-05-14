"""
Graph manipulation operations — dedup, isolation handling, subgraph extraction.

The pruning / aggregation / clustering families have been split off into
sibling modules:

* :mod:`concept_mapper.graph.pruning` — ``prune_edges``, ``prune_nodes``,
  ``prune_to_ratio``
* :mod:`concept_mapper.graph.aggregation` — ``merge_graphs``,
  ``aggregate_graphs``
* :mod:`concept_mapper.graph.cluster` — ``cluster_by_structure``

What's left here is the tidy/extract operations: detecting and connecting
isolated nodes, consolidating duplicate labels (used at export time),
extracting a subgraph by term set, and filtering by relation type.
"""

import logging
from collections import defaultdict
from typing import Dict, Set

from concept_mapper.graph.model import ConceptGraph

logger = logging.getLogger(__name__)


def find_isolated_nodes(graph: ConceptGraph) -> list:
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
            (
                (partner, score)
                for partner, score in scores.items()
                if partner in anchors and partner != node_id
            ),
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
        attrs = graph.get_node(node_id)
        label = attrs.get("label", node_id)
        # Clustered graphs intentionally repeat labels across chapters
        # (sign__Chapter 1, sign__Chapter 2). Dedup by (label, cluster)
        # so the namespacing survives.
        cluster = attrs.get("chapter") or attrs.get("section")
        key = (label, cluster) if cluster is not None else label
        label_to_ids[key].append(node_id)

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
                    combined_evidence = existing.get("evidence", []) + data.get(
                        "evidence", []
                    )
                    if combined_evidence:
                        merged["evidence"] = combined_evidence
                    graph.add_edge(u, v, **merged)
                else:
                    graph.add_edge(u, v, **data)

            graph.remove_node(dup)

    return consolidations


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
