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


# Edge-type priority for resolving the primary type when multiple types
# describe the same (source, target) pair after merge. Mirrors the ladder
# in graph/builders.py:_TYPE_PRIORITY so merge respects the same precedence
# build_proposition_graph uses.
_TYPE_PRIORITY = {
    "definition": 0,
    "kind-of": 1,
    "production": 2,
    "dependence": 3,
    "component": 4,
    "opposition": 5,
    "property": 6,
    "relation": 7,
    "cooccurrence": 8,
}


def aggregate_graphs(graphs: List[ConceptGraph]) -> ConceptGraph:
    """
    Merge multiple ConceptGraphs with proper attribute aggregation.

    Distinct from ``merge_graphs`` (which is last-write-wins). Used by
    ``cmapr merge`` to combine per-chapter graphs into a unified view.

    Aggregation rules:

    * **Shared nodes**: ``frequency`` is summed; ``score`` is the
      frequency-weighted mean of contributing scores; ``community`` is
      dropped (rerun community detection on the merged graph if needed);
      ``label`` and other attributes use last-write-wins.
    * **Shared edges (same source, target, relation_type)**: ``weight``
      is summed, ``evidence`` lists are concatenated.
    * **Shared edges (same pair, different types)**: a *single* edge is
      written carrying additive multi-type fields:
      ``relation_types`` (list, ordered by _TYPE_PRIORITY),
      ``weight_by_type`` (dict), ``evidence_by_type`` (dict),
      ``verb_by_type`` (dict). The flat ``relation_type`` / ``weight``
      / ``evidence`` / ``verb`` attributes still reflect the
      highest-priority type so single-edge consumers keep working.

    Args:
        graphs: list of ConceptGraphs (≥1). All must share directedness.

    Returns:
        New ConceptGraph with merged nodes and edges.

    Raises:
        ValueError: empty input or mixed directedness.
    """
    if not graphs:
        raise ValueError("aggregate_graphs requires at least one graph")
    if any(g.directed != graphs[0].directed for g in graphs):
        raise ValueError("Cannot aggregate graphs with mixed directedness")

    result = ConceptGraph(directed=graphs[0].directed)

    # ------------------------------------------------------------------
    # Phase 1 — node aggregation
    # ------------------------------------------------------------------
    node_acc: Dict[str, Dict] = {}
    for g in graphs:
        for n in g.nodes():
            attrs = g.get_node(n)
            slot = node_acc.setdefault(
                n,
                {
                    "freq_sum": 0,
                    "score_terms": [],  # list of (weight, score)
                    "label": None,
                    "other": {},
                },
            )
            f = attrs.get("frequency", 0) or 0
            slot["freq_sum"] += f
            if "score" in attrs:
                # Weight by this graph's contribution to the node's frequency,
                # falling back to 1 when freq is absent or zero so score-only
                # graphs still participate in the average.
                w = f if f > 0 else 1
                slot["score_terms"].append((w, attrs["score"]))
            slot["label"] = attrs.get("label", slot["label"] or n)
            for k, v in attrs.items():
                if k in {"frequency", "score", "label", "community"}:
                    continue
                slot["other"][k] = v  # last-write-wins for unspecified attrs

    for n, slot in node_acc.items():
        merged: Dict = {"label": slot["label"] or n}
        if slot["freq_sum"] > 0:
            merged["frequency"] = slot["freq_sum"]
        if slot["score_terms"]:
            total_w = sum(w for w, _ in slot["score_terms"])
            merged["score"] = (
                sum(w * s for w, s in slot["score_terms"]) / total_w
                if total_w > 0
                else sum(s for _, s in slot["score_terms"]) / len(slot["score_terms"])
            )
        merged.update(slot["other"])
        result.add_node(n, **merged)

    # ------------------------------------------------------------------
    # Phase 2 — edge aggregation, grouped by (source, target, type)
    # ------------------------------------------------------------------
    edge_acc: Dict[tuple, Dict[str, Dict]] = {}
    for g in graphs:
        for s, t in g.edges():
            attrs = g.get_edge(s, t)
            rel = attrs.get("relation_type", "cooccurrence")
            pair = edge_acc.setdefault((s, t), {})
            type_slot = pair.setdefault(
                rel, {"weight": 0, "evidence": [], "verbs": []}
            )
            type_slot["weight"] += attrs.get("weight", 1)
            evidence = attrs.get("evidence")
            if isinstance(evidence, list):
                type_slot["evidence"].extend(evidence)
            elif isinstance(evidence, str) and evidence:
                type_slot["evidence"].append(evidence)
            verb = attrs.get("verb")
            if verb:
                type_slot["verbs"].append(verb)

    for (s, t), per_type in edge_acc.items():
        ordered = sorted(
            per_type.items(),
            key=lambda kv: _TYPE_PRIORITY.get(kv[0], 99),
        )
        primary_type, primary_data = ordered[0]
        edge_attrs: Dict = {
            "relation_type": primary_type,
            "weight": sum(d["weight"] for _, d in ordered),
            "evidence": [e for _, d in ordered for e in d["evidence"]],
            "verb": primary_data["verbs"][0] if primary_data["verbs"] else primary_type,
        }
        if len(ordered) > 1:
            edge_attrs["relation_types"] = [tp for tp, _ in ordered]
            edge_attrs["weight_by_type"] = {tp: d["weight"] for tp, d in ordered}
            edge_attrs["evidence_by_type"] = {
                tp: list(d["evidence"]) for tp, d in ordered
            }
            edge_attrs["verb_by_type"] = {
                tp: (d["verbs"][0] if d["verbs"] else tp) for tp, d in ordered
            }
        result.add_edge(s, t, **edge_attrs)

    return result


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
