"""
Graph data structures for concept networks.

This module provides a wrapper around NetworkX graphs with domain-specific
functionality for representing philosophical concepts and their relationships.
"""

from typing import Any, Optional, Dict, List
import networkx as nx


class ConceptGraph:
    """
    A graph representing concepts and their relationships.

    Wraps NetworkX DiGraph with domain-specific attributes and methods.

    Node attributes:
        - label: Display name for the concept
        - frequency: Number of occurrences in corpus
        - pos: Part-of-speech tag(s)
        - definition: Definition or description
        - (custom attributes can be added)

    Edge attributes:
        - weight: Numeric strength of relationship
        - relation_type: Type of relation (svo, copular, prep, cooccurrence)
        - evidence: Example sentences demonstrating the relation
        - (custom attributes can be added)

    Example:
        >>> graph = ConceptGraph(directed=True)
        >>> graph.add_node("consciousness", label="Consciousness", frequency=42)
        >>> graph.add_node("intentionality", label="Intentionality", frequency=28)
        >>> graph.add_edge("consciousness", "intentionality",
        ...                weight=0.85, relation_type="copular",
        ...                evidence=["Consciousness is intentional."])
    """

    def __init__(self, directed: bool = False):
        """
        Initialize a ConceptGraph.

        Args:
            directed: If True, create a directed graph (for relations).
                     If False, create an undirected graph (for co-occurrence).
        """
        self._graph = nx.DiGraph() if directed else nx.Graph()
        self._directed = directed

    @property
    def directed(self) -> bool:
        """Whether this is a directed graph."""
        return self._directed

    @property
    def graph(self) -> nx.Graph:
        """Access underlying NetworkX graph."""
        return self._graph

    # ========================================================================
    # Node operations
    # ========================================================================

    def add_node(
        self,
        node_id: str,
        label: Optional[str] = None,
        frequency: Optional[int] = None,
        pos: Optional[str] = None,
        definition: Optional[str] = None,
        **attrs: Any,
    ) -> None:
        """
        Add a node to the graph.

        Args:
            node_id: Unique identifier for the node (typically the term)
            label: Display label (defaults to node_id)
            frequency: Number of occurrences in corpus
            pos: Part-of-speech tag
            definition: Definition or description
            **attrs: Additional custom attributes
        """
        attributes = {}
        if label is not None:
            attributes["label"] = label
        else:
            attributes["label"] = node_id

        if frequency is not None:
            attributes["frequency"] = frequency
        if pos is not None:
            attributes["pos"] = pos
        if definition is not None:
            attributes["definition"] = definition

        attributes.update(attrs)
        self._graph.add_node(node_id, **attributes)

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the graph."""
        self._graph.remove_node(node_id)

    def has_node(self, node_id: str) -> bool:
        """Check if a node exists in the graph."""
        return self._graph.has_node(node_id)

    def get_node(self, node_id: str) -> Dict[str, Any]:
        """
        Get node attributes.

        Args:
            node_id: Node identifier

        Returns:
            Dictionary of node attributes

        Raises:
            KeyError: If node doesn't exist
        """
        if not self.has_node(node_id):
            raise KeyError(f"Node '{node_id}' not found in graph")
        return dict(self._graph.nodes[node_id])

    def nodes(self) -> List[str]:
        """Get list of all node IDs."""
        return list(self._graph.nodes())

    def node_count(self) -> int:
        """Get number of nodes in graph."""
        return self._graph.number_of_nodes()

    # ========================================================================
    # Edge operations
    # ========================================================================

    def add_edge(
        self,
        source: str,
        target: str,
        weight: Optional[float] = None,
        relation_type: Optional[str] = None,
        evidence: Optional[List[str]] = None,
        **attrs: Any,
    ) -> None:
        """
        Add an edge to the graph.

        Args:
            source: Source node ID
            target: Target node ID
            weight: Numeric strength of relationship
            relation_type: Type of relation (svo, copular, prep, cooccurrence)
            evidence: Example sentences demonstrating the relation
            **attrs: Additional custom attributes
        """
        attributes = {}
        if weight is not None:
            attributes["weight"] = weight
        if relation_type is not None:
            attributes["relation_type"] = relation_type
        if evidence is not None:
            attributes["evidence"] = evidence

        attributes.update(attrs)
        self._graph.add_edge(source, target, **attributes)

    def remove_edge(self, source: str, target: str) -> None:
        """Remove an edge from the graph."""
        self._graph.remove_edge(source, target)

    def has_edge(self, source: str, target: str) -> bool:
        """Check if an edge exists in the graph."""
        return self._graph.has_edge(source, target)

    def get_edge(self, source: str, target: str) -> Dict[str, Any]:
        """
        Get edge attributes.

        Args:
            source: Source node ID
            target: Target node ID

        Returns:
            Dictionary of edge attributes

        Raises:
            KeyError: If edge doesn't exist
        """
        if not self.has_edge(source, target):
            raise KeyError(f"Edge '{source}' -> '{target}' not found in graph")
        return dict(self._graph.edges[source, target])

    def edges(self) -> List[tuple]:
        """
        Get list of all edges.

        Returns:
            List of (source, target) tuples
        """
        return list(self._graph.edges())

    def edge_count(self) -> int:
        """Get number of edges in graph."""
        return self._graph.number_of_edges()

    # ========================================================================
    # Graph operations
    # ========================================================================

    def neighbors(self, node_id: str) -> List[str]:
        """
        Get neighbors of a node.

        Args:
            node_id: Node identifier

        Returns:
            List of neighbor node IDs
        """
        return list(self._graph.neighbors(node_id))

    def degree(self, node_id: str) -> int:
        """
        Get degree of a node.

        Args:
            node_id: Node identifier

        Returns:
            Number of edges connected to the node
        """
        return self._graph.degree(node_id)

    def copy(self) -> "ConceptGraph":
        """
        Create a deep copy of the graph.

        Returns:
            New ConceptGraph instance with copied data
        """
        new_graph = ConceptGraph(directed=self._directed)
        new_graph._graph = self._graph.copy()
        return new_graph

    def __repr__(self) -> str:
        """String representation of the graph."""
        graph_type = "Directed" if self._directed else "Undirected"
        return (
            f"ConceptGraph({graph_type}, "
            f"nodes={self.node_count()}, edges={self.edge_count()})"
        )

    def __str__(self) -> str:
        """String representation of the graph."""
        return self.__repr__()
