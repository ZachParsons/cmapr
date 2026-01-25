"""
Graph construction and analysis for concept networks.

This module provides functionality for building, manipulating, and analyzing
graphs of philosophical concepts and their relationships.

Main components:
- ConceptGraph: Core graph data structure
- Builders: Construct graphs from co-occurrence and relations
- Operations: Merge, prune, filter graphs
- Metrics: Centrality, communities, paths
"""

from concept_mapper.graph.model import ConceptGraph
from concept_mapper.graph.builders import (
    graph_from_cooccurrence,
    graph_from_relations,
    graph_from_terms,
)
from concept_mapper.graph.operations import (
    merge_graphs,
    prune_edges,
    prune_nodes,
    get_subgraph,
    filter_by_relation_type,
)
from concept_mapper.graph.metrics import (
    centrality,
    detect_communities,
    assign_communities,
    get_connected_components,
    graph_density,
    get_shortest_path,
)

__all__ = [
    # Model
    "ConceptGraph",
    # Builders
    "graph_from_cooccurrence",
    "graph_from_relations",
    "graph_from_terms",
    # Operations
    "merge_graphs",
    "prune_edges",
    "prune_nodes",
    "get_subgraph",
    "filter_by_relation_type",
    # Metrics
    "centrality",
    "detect_communities",
    "assign_communities",
    "get_connected_components",
    "graph_density",
    "get_shortest_path",
]
