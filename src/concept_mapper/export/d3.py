"""
D3.js JSON export for interactive visualizations.

This module exports ConceptGraphs to D3-compatible JSON format for
force-directed network visualizations in web browsers.
"""

import json
from pathlib import Path
from typing import Dict, Any, List
from concept_mapper.graph.model import ConceptGraph
from concept_mapper.graph.metrics import (
    centrality,
    detect_communities,
    assign_communities,
)
from concept_mapper.graph.operations import consolidate_duplicate_labels, find_isolated_nodes
from concept_mapper.validation import validate_concept_graph


def to_d3_dict(
    graph: ConceptGraph,
    include_evidence: bool = False,
    size_by: str = "frequency",
    compute_communities: bool = True,
    max_evidence: int = 3,
) -> Dict[str, Any]:
    """
    Convert graph to D3.js-compatible dictionary.

    Returns a dictionary with nodes and links arrays suitable for D3.js
    force-directed graph visualizations.

    Args:
        graph: ConceptGraph to convert
        include_evidence: Include evidence sentences in edge metadata (default: False)
        size_by: Node size metric ("frequency", "degree", "betweenness") (default: "frequency")
        compute_communities: Detect and assign community groups (default: True)
        max_evidence: Maximum evidence sentences per edge (default: 3)

    Returns:
        Dictionary with "nodes" and "links" keys for D3.js
    """
    # Validate graph is not empty
    validate_concept_graph(graph, require_edges=False)

    import logging
    _log = logging.getLogger(__name__)

    # Work on a copy so the caller's graph is not mutated
    graph = graph.copy()
    consolidate_duplicate_labels(graph)

    for node_id in find_isolated_nodes(graph):
        _log.error(
            "Isolated node %r has no edges and will be excluded from the diagram. "
            "Fix upstream: ensure every term has at least one extracted relation or "
            "co-occurrence, or run connect_isolated_nodes() before export.",
            node_id,
        )
        graph.remove_node(node_id)

    # Compute communities if requested and not already assigned
    if compute_communities:
        has_community = any("community" in graph.get_node(n) for n in graph.nodes())
        if not has_community and graph.node_count() > 0:
            try:
                communities = detect_communities(graph)
                assign_communities(graph, communities)
            except Exception:
                # Skip community detection if it fails
                pass

    # Compute size metric
    sizes = _compute_node_sizes(graph, size_by)

    # Build nodes array
    nodes = []
    for node_id in graph.nodes():
        node_attrs = graph.get_node(node_id)
        node_data = {
            "id": node_id,
            "label": node_attrs.get("label", node_id),
            "size": sizes.get(node_id, 1.0),
        }

        # Add optional attributes
        if "frequency" in node_attrs:
            node_data["frequency"] = node_attrs["frequency"]
        if "pos" in node_attrs:
            node_data["pos"] = node_attrs["pos"]
        if "definition" in node_attrs:
            node_data["definition"] = node_attrs["definition"]
        if "community" in node_attrs:
            node_data["group"] = node_attrs["community"]
        else:
            node_data["group"] = 0

        nodes.append(node_data)

    # Build links array
    links = []
    for source, target in graph.edges():
        edge_attrs = graph.get_edge(source, target)
        link_data = {
            "source": source,
            "target": target,
            "weight": edge_attrs.get("weight", 1.0),
        }

        # Add optional attributes
        if "relation_type" in edge_attrs:
            link_data["label"] = edge_attrs["relation_type"]

        metadata = edge_attrs.get("metadata", {})
        verb = (
            metadata.get("verb")
            or metadata.get("copula")
            or metadata.get("preposition")
        )
        if verb:
            link_data["verb"] = verb

        if include_evidence and "evidence" in edge_attrs:
            evidence = edge_attrs["evidence"]
            # Limit evidence to max_evidence sentences
            link_data["evidence"] = evidence[:max_evidence]

        links.append(link_data)

    # Create D3 JSON structure
    return {
        "nodes": nodes,
        "links": links,
    }


def export_d3_json(
    graph: ConceptGraph,
    path: Path,
    include_evidence: bool = False,
    size_by: str = "frequency",
    compute_communities: bool = True,
    max_evidence: int = 3,
) -> None:
    """
    Export graph to D3.js force-directed layout JSON format.

    Creates a JSON file with nodes and links arrays suitable for D3.js
    force-directed graph visualizations.

    Args:
        graph: ConceptGraph to export
        path: Output file path
        include_evidence: Include evidence sentences in edge metadata (default: False)
        size_by: Node size metric ("frequency", "degree", "betweenness") (default: "frequency")
        compute_communities: Detect and assign community groups (default: True)
        max_evidence: Maximum evidence sentences per edge (default: 3)

    Example:
        >>> from concept_mapper.graph import ConceptGraph
        >>> from pathlib import Path
        >>> graph = ConceptGraph()
        >>> graph.add_node("consciousness", frequency=42)
        >>> graph.add_node("being", frequency=28)
        >>> graph.add_edge("consciousness", "being", weight=0.85)
        >>> export_d3_json(graph, Path("output/graph.json"))

    D3 JSON Schema:
        {
          "nodes": [
            {
              "id": "term",
              "label": "Term",
              "group": 0,
              "size": 10,
              "frequency": 42,
              "pos": "NN",
              ...
            }
          ],
          "links": [
            {
              "source": "term1",
              "target": "term2",
              "weight": 0.85,
              "label": "copular",
              "evidence": ["Example sentence."],
              ...
            }
          ]
        }
    """
    # Get D3 data dict
    d3_data = to_d3_dict(
        graph,
        include_evidence=include_evidence,
        size_by=size_by,
        compute_communities=compute_communities,
        max_evidence=max_evidence,
    )

    # Write to file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(d3_data, f, indent=2, ensure_ascii=False)


def _compute_node_sizes(
    graph: ConceptGraph,
    size_by: str,
) -> Dict[str, float]:
    """
    Compute node sizes based on metric.

    Args:
        graph: ConceptGraph
        size_by: Metric to use ("frequency", "degree", "betweenness")

    Returns:
        Dictionary mapping node ID to size value
    """
    if size_by == "frequency":
        # Use frequency attribute if available
        sizes = {}
        for node_id in graph.nodes():
            node_attrs = graph.get_node(node_id)
            sizes[node_id] = node_attrs.get("frequency", 1.0)
        return sizes

    elif size_by == "degree":
        # Use degree centrality
        scores = centrality(graph, method="degree")
        # Normalize to reasonable range (1-20)
        if scores:
            max_score = max(scores.values())
            min_score = min(scores.values())
            if max_score > min_score:
                return {
                    node: 1 + 19 * (score - min_score) / (max_score - min_score)
                    for node, score in scores.items()
                }
        return scores

    elif size_by == "betweenness":
        # Use betweenness centrality
        scores = centrality(graph, method="betweenness")
        # Normalize to reasonable range (1-20)
        if scores:
            max_score = max(scores.values())
            min_score = min(scores.values())
            if max_score > min_score:
                return {
                    node: 1 + 19 * (score - min_score) / (max_score - min_score)
                    for node, score in scores.items()
                }
        return scores

    else:
        # Default to uniform size
        return {node: 1.0 for node in graph.nodes()}


def load_d3_json(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load D3 JSON file.

    Args:
        path: Path to D3 JSON file

    Returns:
        Dictionary with "nodes" and "links" keys

    Example:
        >>> data = load_d3_json(Path("output/graph.json"))
        >>> print(f"Nodes: {len(data['nodes'])}, Links: {len(data['links'])}")
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
