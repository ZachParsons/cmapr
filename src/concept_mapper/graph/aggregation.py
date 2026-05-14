"""
Graph aggregation — combining multiple ConceptGraphs into one.

Two flavours:

* ``aggregate_graphs`` — attribute-aware: sums frequencies, weighted-avg
  scores, sums edge weights, concatenates evidence, preserves multi-type
  info on shared pairs via additive fields. Powers ``cmapr merge``.

* ``merge_graphs`` — last-write-wins primitive. Useful as a building
  block; not what ``cmapr merge`` calls. Kept for callers that want the
  simpler behaviour.

Complement of ``cluster_by_structure`` (which preserves per-chapter
structure instead of collapsing it).
"""

from typing import Dict, List

from concept_mapper.graph.model import ConceptGraph


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
            type_slot = pair.setdefault(rel, {"weight": 0, "evidence": [], "verbs": []})
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
    Merge two graphs into a new graph (last-write-wins).

    Combines nodes and edges from both graphs. If both graphs have the same
    node/edge, attributes from g2 take precedence. Use ``aggregate_graphs``
    for proper attribute aggregation (sum, weighted-avg, etc.).

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
