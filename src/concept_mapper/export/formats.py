"""
Alternative export formats for graph visualization tools.

This module provides export functions for various graph formats:
- GraphML (for Gephi, yEd, Cytoscape)
- DOT (for Graphviz)
- CSV (nodes.csv + edges.csv)
"""

import csv
from pathlib import Path
import networkx as nx
from concept_mapper.graph.model import ConceptGraph


def export_graphml(
    graph: ConceptGraph,
    path: Path,
) -> None:
    """
    Export graph to GraphML format for Gephi, yEd, Cytoscape.

    GraphML is an XML-based graph format that preserves all node and edge
    attributes. Compatible with most graph visualization tools.

    Args:
        graph: ConceptGraph to export
        path: Output file path (.graphml)

    Example:
        >>> from concept_mapper.graph import ConceptGraph
        >>> from pathlib import Path
        >>> graph = ConceptGraph()
        >>> graph.add_node("consciousness", frequency=42)
        >>> graph.add_edge("consciousness", "being", weight=0.85)
        >>> export_graphml(graph, Path("output/graph.graphml"))

    Note:
        Evidence lists are converted to comma-separated strings for compatibility.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get underlying NetworkX graph
    g = graph.graph.copy()

    # Convert list attributes to strings for GraphML compatibility
    for node in g.nodes():
        node_attrs = g.nodes[node]
        for key, value in list(node_attrs.items()):
            if isinstance(value, list):
                node_attrs[key] = ", ".join(str(v) for v in value)

    for source, target in g.edges():
        edge_attrs = g.edges[source, target]
        for key, value in list(edge_attrs.items()):
            if isinstance(value, list):
                edge_attrs[key] = " | ".join(str(v) for v in value)

    # Write GraphML
    nx.write_graphml(g, str(path))


def export_dot(
    graph: ConceptGraph,
    path: Path,
    layout: str = "dot",
) -> None:
    """
    Export graph to DOT format for Graphviz.

    DOT is a plain text graph description language used by Graphviz
    for creating publication-quality graph layouts.

    Args:
        graph: ConceptGraph to export
        path: Output file path (.dot)
        layout: Graphviz layout algorithm (dot, neato, fdp, sfdp, circo, twopi)

    Raises:
        ImportError: If pydot is not installed

    Example:
        >>> export_dot(graph, Path("output/graph.dot"))
        # Then render with: dot -Tpng graph.dot -o graph.png

    Layouts:
        - dot: Hierarchical, directed graphs
        - neato: Spring model, undirected graphs
        - fdp: Force-directed placement
        - sfdp: Scalable force-directed (large graphs)
        - circo: Circular layout
        - twopi: Radial layout

    Note:
        Requires pydot package: pip install pydot
    """
    try:
        import pydot  # noqa: F401
    except ImportError:
        raise ImportError(
            "DOT export requires pydot package. Install with: pip install pydot"
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get underlying NetworkX graph
    g = graph.graph

    # Create DOT representation using NetworkX
    dot_data = nx.nx_pydot.to_pydot(g)

    # Set graph attributes
    dot_data.set_layout(layout)
    if graph.directed:
        dot_data.set_rankdir("LR")  # Left to right for directed graphs

    # Write DOT file
    with open(path, "w", encoding="utf-8") as f:
        f.write(dot_data.to_string())


def export_csv(
    graph: ConceptGraph,
    output_dir: Path,
    nodes_filename: str = "nodes.csv",
    edges_filename: str = "edges.csv",
) -> None:
    """
    Export graph to CSV format (nodes.csv + edges.csv).

    Creates two CSV files for easy import into spreadsheets or databases.
    This format is useful for manual inspection and editing.

    Args:
        graph: ConceptGraph to export
        output_dir: Output directory
        nodes_filename: Filename for nodes CSV (default: "nodes.csv")
        edges_filename: Filename for edges CSV (default: "edges.csv")

    Example:
        >>> export_csv(graph, Path("output/"))
        # Creates: output/nodes.csv, output/edges.csv

    Nodes CSV format:
        id,label,frequency,pos,definition,community
        consciousness,Consciousness,42,NN,Mental awareness,0

    Edges CSV format:
        source,target,weight,relation_type,evidence
        consciousness,being,0.85,copular,"Consciousness is being."
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    nodes_path = output_dir / nodes_filename
    edges_path = output_dir / edges_filename

    # Collect all node attributes to determine columns
    node_attrs_set = set()
    for node_id in graph.nodes():
        node_attrs_set.update(graph.get_node(node_id).keys())

    # Always include id and label
    node_columns = ["id", "label"]
    # Add other attributes in consistent order
    other_attrs = sorted(node_attrs_set - {"label"})
    node_columns.extend(other_attrs)

    # Write nodes CSV
    with open(nodes_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=node_columns)
        writer.writeheader()

        for node_id in graph.nodes():
            node_attrs = graph.get_node(node_id)
            row = {"id": node_id}
            for key in node_columns[1:]:  # Skip 'id'
                value = node_attrs.get(key, "")
                # Convert lists to semicolon-separated strings
                if isinstance(value, list):
                    value = "; ".join(str(v) for v in value)
                row[key] = value
            writer.writerow(row)

    # Collect all edge attributes to determine columns
    edge_attrs_set = set()
    for source, target in graph.edges():
        edge_attrs_set.update(graph.get_edge(source, target).keys())

    # Always include source and target
    edge_columns = ["source", "target"]
    # Add other attributes in consistent order
    other_attrs = sorted(edge_attrs_set)
    edge_columns.extend(other_attrs)

    # Write edges CSV
    with open(edges_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=edge_columns)
        writer.writeheader()

        for source, target in graph.edges():
            edge_attrs = graph.get_edge(source, target)
            row = {"source": source, "target": target}
            for key in edge_columns[2:]:  # Skip 'source' and 'target'
                value = edge_attrs.get(key, "")
                # Convert lists to semicolon-separated strings
                if isinstance(value, list):
                    value = "; ".join(str(v) for v in value)
                row[key] = value
            writer.writerow(row)


def export_gexf(
    graph: ConceptGraph,
    path: Path,
) -> None:
    """
    Export graph to GEXF format (Graph Exchange XML Format).

    GEXF is a modern XML format for graphs, optimized for Gephi.
    Supports dynamic graphs and rich metadata.

    Args:
        graph: ConceptGraph to export
        path: Output file path (.gexf)

    Example:
        >>> export_gexf(graph, Path("output/graph.gexf"))
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get underlying NetworkX graph
    g = graph.graph.copy()

    # Convert list attributes to strings for GEXF compatibility
    for node in g.nodes():
        node_attrs = g.nodes[node]
        for key, value in list(node_attrs.items()):
            if isinstance(value, list):
                node_attrs[key] = ", ".join(str(v) for v in value)

    for source, target in g.edges():
        edge_attrs = g.edges[source, target]
        for key, value in list(edge_attrs.items()):
            if isinstance(value, list):
                edge_attrs[key] = " | ".join(str(v) for v in value)

    # Write GEXF
    nx.write_gexf(g, str(path))


def export_json_graph(
    graph: ConceptGraph,
    path: Path,
) -> None:
    """
    Export graph to NetworkX node-link JSON format.

    This is a simple JSON format that can be loaded back into NetworkX.

    Args:
        graph: ConceptGraph to export
        path: Output file path (.json)

    Example:
        >>> export_json_graph(graph, Path("output/graph.json"))
    """
    import json

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Get underlying NetworkX graph
    g = graph.graph

    # Convert to node-link data
    data = nx.node_link_data(g)

    # Write JSON
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
