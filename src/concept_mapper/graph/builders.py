"""
Graph construction from co-occurrence and relations.

This module provides functions to build ConceptGraphs from analysis results.
"""

from typing import Dict, List, Optional
from concept_mapper.graph.model import ConceptGraph
from concept_mapper.analysis.relations import Relation


def graph_from_cooccurrence(
    matrix: Dict[str, Dict[str, float]],
    threshold: float = 0.0,
    directed: bool = False,
) -> ConceptGraph:
    """
    Build a graph from a co-occurrence matrix.

    Creates nodes for each term and edges for co-occurrences above the threshold.
    Edge weights are set to the co-occurrence values.

    Args:
        matrix: Co-occurrence matrix as nested dict (term -> term -> score)
        threshold: Minimum co-occurrence value to create an edge (default: 0.0)
        directed: Whether to create a directed graph (default: False)

    Returns:
        ConceptGraph with nodes for terms and weighted edges for co-occurrences

    Example:
        >>> matrix = {"consciousness": {"intentionality": 0.85, "awareness": 0.42}}
        >>> graph = graph_from_cooccurrence(matrix, threshold=0.5)
        >>> graph.has_edge("consciousness", "intentionality")
        True
        >>> graph.has_edge("consciousness", "awareness")
        False
    """
    graph = ConceptGraph(directed=directed)

    # Collect all terms
    terms = set(matrix.keys())
    for term_cooccurs in matrix.values():
        terms.update(term_cooccurs.keys())

    # Add nodes for all terms
    for term in terms:
        graph.add_node(term, label=term)

    # Add edges for co-occurrences above threshold
    for term1, cooccurs in matrix.items():
        for term2, score in cooccurs.items():
            # Skip if below threshold
            if score < threshold:
                continue

            # Skip self-loops
            if term1 == term2:
                continue

            # For undirected graphs, only add each edge once
            if not directed and graph.has_edge(term2, term1):
                continue

            graph.add_edge(
                term1,
                term2,
                weight=score,
                relation_type="cooccurrence",
            )

    return graph


def graph_from_relations(
    relations: List[Relation],
    include_evidence: bool = True,
    term_filter: Optional[set] = None,
) -> ConceptGraph:
    """
    Build a directed graph from extracted relations.

    Creates nodes for source and target concepts, with directed edges
    labeled by relation type.

    Args:
        relations: List of Relation objects from relation extraction
        include_evidence: Whether to include evidence sentences in edge attributes
        term_filter: If provided, only include edges where both source and target
                     are in this set (lowercased). Prevents non-term SVO endpoints
                     from inflating node count with unconnectable leaf nodes.

    Returns:
        Directed ConceptGraph with labeled edges

    Example:
        >>> from concept_mapper.analysis.relations import Relation
        >>> relations = [
        ...     Relation("consciousness", "copular", "intentional",
        ...              evidence=["Consciousness is intentional."])
        ... ]
        >>> graph = graph_from_relations(relations)
        >>> graph.directed
        True
        >>> graph.get_edge("consciousness", "intentional")["relation_type"]
        'copular'
    """
    graph = ConceptGraph(directed=True)

    # Normalise filter set once
    _filter = {t.lower() for t in term_filter} if term_filter else None

    # Process each relation
    for relation in relations:
        source = relation.source.lower()
        target = relation.target.lower()

        # Skip if either endpoint is outside the allowed term set
        if _filter is not None and (source not in _filter or target not in _filter):
            continue

        # Add nodes if they don't exist
        if not graph.has_node(source):
            graph.add_node(source, label=source)
        if not graph.has_node(target):
            graph.add_node(target, label=target)

        # Prepare edge attributes
        edge_attrs = {
            "relation_type": relation.relation_type,
            "weight": len(relation.evidence),  # Weight by evidence count
        }

        if include_evidence:
            edge_attrs["evidence"] = relation.evidence

        # Add metadata
        if relation.metadata:
            edge_attrs["metadata"] = relation.metadata

        # If edge already exists, merge evidence
        if graph.has_edge(source, target):
            existing = graph.get_edge(source, target)
            # Combine evidence if present
            if include_evidence and "evidence" in existing:
                existing["evidence"].extend(edge_attrs["evidence"])
                edge_attrs["evidence"] = existing["evidence"]
            # Increase weight
            edge_attrs["weight"] = existing.get("weight", 1) + edge_attrs["weight"]

        graph.add_edge(source, target, **edge_attrs)

    return graph


def graph_from_terms(
    terms: List[str],
    term_data: Optional[Dict[str, Dict]] = None,
) -> ConceptGraph:
    """
    Build a graph with nodes for terms (no edges).

    Useful as a starting point before adding edges from analysis.

    Args:
        terms: List of term strings
        term_data: Optional dict mapping term -> attributes
                  (e.g., frequency, pos, definition)

    Returns:
        ConceptGraph with nodes but no edges

    Example:
        >>> terms = ["consciousness", "intentionality"]
        >>> data = {"consciousness": {"frequency": 42, "pos": "NN"}}
        >>> graph = graph_from_terms(terms, term_data=data)
        >>> graph.node_count()
        2
        >>> graph.get_node("consciousness")["frequency"]
        42
    """
    graph = ConceptGraph(directed=False)

    for term in terms:
        attrs = {}
        if term_data and term in term_data:
            attrs = term_data[term]

        graph.add_node(term, label=term, **attrs)

    return graph
