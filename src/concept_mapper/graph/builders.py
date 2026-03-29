"""
Graph construction from co-occurrence and relations.

This module provides functions to build ConceptGraphs from analysis results.
"""

import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional
from concept_mapper.graph.model import ConceptGraph
from concept_mapper.analysis.relations import Relation

if TYPE_CHECKING:
    from concept_mapper.analysis.contextual_relations import ContextualRelation
    from concept_mapper.graph.node_filter import NodeFilter


def build_proposition_graph(
    docs: list,
    seed_terms: list,
    node_filter: Optional["NodeFilter"] = None,
    pmi_threshold: float = 1.0,
    term_scores: Optional[Dict[str, float]] = None,
) -> ConceptGraph:
    """
    Build a typed proposition graph from a document set and seed term list.

    For each pair of seed terms that co-occur in at least one sentence, attempts
    to extract a typed proposition (definition, kind-of, production, dependence,
    component). Falls back to a cooccurrence edge when PMI ≥ pmi_threshold and
    no typed proposition was found.

    Multigraph note: ConceptGraph is backed by a DiGraph (one edge per directed
    pair). When multiple proposition types exist for the same pair the
    highest-priority type wins: definition > kind-of > production > dependence >
    component > cooccurrence.

    Args:
        docs         : list of ProcessedDocument objects
        seed_terms   : terms from the rarities list; all become nodes
        node_filter  : optional NodeFilter; applied to extracted (non-seed) nodes
        pmi_threshold: minimum PMI for a cooccurrence fallback edge (default 1.0)
        term_scores  : optional dict mapping term → rarity score; stored as node attr

    Returns:
        ConceptGraph with typed edges
    """
    from concept_mapper.graph.proposition_extractor import (
        Proposition,
        PropositionExtractor,
    )

    _TYPE_PRIORITY = {
        "definition": 0,
        "kind-of": 1,
        "production": 2,
        "dependence": 3,
        "component": 4,
        "cooccurrence": 5,
    }

    extractor = PropositionExtractor(docs)
    all_sentences = extractor._sentences
    n_total = max(len(all_sentences), 1)

    seed_lower = [t.lower() for t in seed_terms]
    seed_set = set(seed_lower)

    # Per-term sentence counts for PMI
    term_counts: Dict[str, int] = {
        term: sum(1 for s in all_sentences if term in s.lower()) for term in seed_lower
    }

    # Accumulate best proposition per (source, target) pair
    best: Dict[tuple, Proposition] = {}

    def _keep(prop: Proposition) -> None:
        """Store prop if it beats the current best for its (source, target) pair."""
        key = (prop.source.lower(), prop.target.lower())
        rkey = (prop.target.lower(), prop.source.lower())
        current = best.get(key) or best.get(rkey)
        if current is None or _TYPE_PRIORITY.get(prop.type, 9) < _TYPE_PRIORITY.get(
            current.type, 9
        ):
            # Remove reverse key if present so direction is updated
            best.pop(rkey, None)
            best[key] = prop

    # Typed propositions for all seed pairs
    for i, term_a in enumerate(seed_lower):
        for term_b in seed_lower[i + 1 :]:
            for prop in extractor.extract(term_a, term_b):
                _keep(prop)

    # Composition pattern across full term list
    for prop in extractor.extract_composition(seed_lower):
        _keep(prop)

    # Cooccurrence fallback for uncovered pairs
    covered = {frozenset(k) for k in best}
    for i, term_a in enumerate(seed_lower):
        for term_b in seed_lower[i + 1 :]:
            if frozenset([term_a, term_b]) in covered:
                continue
            n_ab = sum(
                1 for s in all_sentences if term_a in s.lower() and term_b in s.lower()
            )
            if n_ab == 0:
                continue
            na, nb = term_counts.get(term_a, 0), term_counts.get(term_b, 0)
            pmi = math.log2(n_ab * n_total / (na * nb)) if na > 0 and nb > 0 else 0.0
            if pmi >= pmi_threshold:
                _keep(
                    Proposition(
                        source=term_a,
                        target=term_b,
                        label="co-occurs with",
                        type="cooccurrence",
                        evidence="",
                        directed=False,
                        weight=n_ab,
                    )
                )

    # Normalise term_scores keys to lowercase for lookup
    scores_lower: Dict[str, float] = (
        {k.lower(): v for k, v in term_scores.items()} if term_scores else {}
    )

    # Build graph
    graph = ConceptGraph(directed=True)

    for (source, target), prop in best.items():
        # NodeFilter: reject extracted nodes that fail inclusion criteria
        if node_filter is not None:
            if any(
                node not in seed_set and not node_filter.is_valid(node)[0]
                for node in (source, target)
            ):
                continue

        for node in (source, target):
            if not graph.has_node(node):
                node_attrs: Dict[str, Any] = {"label": node}
                if node in scores_lower:
                    node_attrs["score"] = scores_lower[node]
                graph.add_node(node, **node_attrs)

        graph.add_edge(
            source,
            target,
            relation_type=prop.type,
            verb=prop.label,
            weight=prop.weight,
            evidence=[prop.evidence] if prop.evidence else [],
        )

    return graph


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


def graph_from_contextual_relations(
    relations: List["ContextualRelation"],
    term_filter: Optional[set] = None,
) -> ConceptGraph:
    """
    Build a directed graph from ContextualRelation objects produced by analyze_context().

    Designed for batch analysis: collects relations from analyzing multiple terms
    and merges duplicate edges (same source/target/type) by taking the max score
    and combining evidence. Cooccurrence edges are normalised to alphabetical
    source/target order so A~B and B~A (from analysing A then B) collapse to one edge.

    Args:
        relations: Combined list of ContextualRelation from one or more analyze_context calls
        term_filter: If provided, only include edges where both endpoints are in
                     this set (lowercased). Keeps the graph focused on known terms.

    Returns:
        Directed ConceptGraph with relation_type and weight on each edge
    """
    graph = ConceptGraph(directed=True)
    _filter = {t.lower() for t in term_filter} if term_filter else None

    for relation in relations:
        source = relation.source.lower()
        target = relation.target.lower()

        if source == target:
            continue

        if _filter is not None and (source not in _filter or target not in _filter):
            continue

        # Normalise cooccurrence direction so A~B and B~A merge
        if relation.relation_type == "cooccurrence" and source > target:
            source, target = target, source

        for node in (source, target):
            if not graph.has_node(node):
                graph.add_node(node, label=node)

        edge_attrs = {
            "relation_type": relation.relation_type,
            "weight": relation.score,
            "evidence": list(relation.evidence),
        }
        if relation.metadata:
            edge_attrs["metadata"] = relation.metadata

        if graph.has_edge(source, target):
            existing = graph.get_edge(source, target)
            edge_attrs["weight"] = max(existing.get("weight", 0), relation.score)
            edge_attrs["evidence"] = list(
                set(existing.get("evidence", []) + edge_attrs["evidence"])
            )

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
