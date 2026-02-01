"""
Tests for export and visualization (Phase 9).
"""

import pytest
import json
import csv
from pathlib import Path
from concept_mapper.graph import ConceptGraph, graph_from_cooccurrence
from concept_mapper.export import (
    export_d3_json,
    load_d3_json,
    export_graphml,
    export_dot,
    export_csv,
    export_gexf,
    export_json_graph,
    generate_html,
)
from concept_mapper.analysis.relations import Relation
from concept_mapper.graph import graph_from_relations

# Check if pydot is available
try:
    import pydot  # noqa: F401

    HAS_PYDOT = True
except ImportError:
    HAS_PYDOT = False


@pytest.fixture
def sample_graph():
    """Create a sample graph for testing."""
    graph = ConceptGraph(directed=False)
    graph.add_node("consciousness", label="Consciousness", frequency=42, pos="NN")
    graph.add_node("being", label="Being", frequency=28, pos="NN")
    graph.add_node("intentionality", label="Intentionality", frequency=15, pos="NN")
    graph.add_edge("consciousness", "being", weight=0.85, relation_type="cooccurrence")
    graph.add_edge(
        "consciousness", "intentionality", weight=0.92, relation_type="cooccurrence"
    )
    return graph


@pytest.fixture
def sample_directed_graph():
    """Create a sample directed graph with relations."""
    relations = [
        Relation(
            source="consciousness",
            relation_type="copular",
            target="intentional",
            evidence=[
                "Consciousness is intentional.",
                "Consciousness is always intentional.",
            ],
            metadata={"copula": "is"},
        ),
        Relation(
            source="being",
            relation_type="copular",
            target="presence",
            evidence=["Being is presence."],
            metadata={"copula": "is"},
        ),
    ]
    return graph_from_relations(relations, include_evidence=True)


@pytest.fixture
def temp_output_dir(tmp_path):
    """Create temporary output directory."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir


# ============================================================================
# Test D3 JSON Export
# ============================================================================


class TestD3Export:
    """Tests for D3.js JSON export."""

    def test_export_d3_json_basic(self, sample_graph, temp_output_dir):
        """Test basic D3 JSON export."""
        output_path = temp_output_dir / "graph.json"
        export_d3_json(sample_graph, output_path)

        assert output_path.exists()

        # Load and verify structure
        data = load_d3_json(output_path)
        assert "nodes" in data
        assert "links" in data
        assert len(data["nodes"]) == 3
        assert len(data["links"]) == 2

    def test_export_d3_json_node_structure(self, sample_graph, temp_output_dir):
        """Test D3 JSON node structure."""
        output_path = temp_output_dir / "graph.json"
        export_d3_json(sample_graph, output_path)

        data = load_d3_json(output_path)

        # Check first node
        node = data["nodes"][0]
        assert "id" in node
        assert "label" in node
        assert "size" in node
        assert "group" in node

    def test_export_d3_json_link_structure(self, sample_graph, temp_output_dir):
        """Test D3 JSON link structure."""
        output_path = temp_output_dir / "graph.json"
        export_d3_json(sample_graph, output_path)

        data = load_d3_json(output_path)

        # Check first link
        link = data["links"][0]
        assert "source" in link
        assert "target" in link
        assert "weight" in link

    def test_export_d3_json_with_evidence(self, sample_directed_graph, temp_output_dir):
        """Test D3 JSON export with evidence."""
        output_path = temp_output_dir / "graph.json"
        export_d3_json(sample_directed_graph, output_path, include_evidence=True)

        data = load_d3_json(output_path)

        # Check that evidence is included
        link = data["links"][0]
        assert "evidence" in link
        assert isinstance(link["evidence"], list)
        assert len(link["evidence"]) > 0

    def test_export_d3_json_without_evidence(
        self, sample_directed_graph, temp_output_dir
    ):
        """Test D3 JSON export without evidence."""
        output_path = temp_output_dir / "graph.json"
        export_d3_json(sample_directed_graph, output_path, include_evidence=False)

        data = load_d3_json(output_path)

        # Evidence should not be included
        link = data["links"][0]
        assert "evidence" not in link

    def test_export_d3_json_max_evidence(self, sample_directed_graph, temp_output_dir):
        """Test max_evidence parameter."""
        output_path = temp_output_dir / "graph.json"
        export_d3_json(
            sample_directed_graph, output_path, include_evidence=True, max_evidence=1
        )

        data = load_d3_json(output_path)

        # Evidence should be limited to 1
        link = data["links"][0]
        assert len(link["evidence"]) <= 1

    def test_export_d3_json_size_by_frequency(self, sample_graph, temp_output_dir):
        """Test node sizing by frequency."""
        output_path = temp_output_dir / "graph.json"
        export_d3_json(sample_graph, output_path, size_by="frequency")

        data = load_d3_json(output_path)

        # Node with higher frequency should have larger size
        nodes_by_id = {n["id"]: n for n in data["nodes"]}
        assert (
            nodes_by_id["consciousness"]["size"] > nodes_by_id["intentionality"]["size"]
        )

    def test_export_d3_json_size_by_degree(self, sample_graph, temp_output_dir):
        """Test node sizing by degree centrality."""
        output_path = temp_output_dir / "graph.json"
        export_d3_json(sample_graph, output_path, size_by="degree")

        data = load_d3_json(output_path)

        # Consciousness has degree 2, others have degree 1 or 2
        nodes_by_id = {n["id"]: n for n in data["nodes"]}
        assert "size" in nodes_by_id["consciousness"]

    def test_export_d3_json_creates_directory(self, tmp_path):
        """Test that export creates output directory."""
        output_path = tmp_path / "nested" / "dir" / "graph.json"
        graph = ConceptGraph()
        graph.add_node("test")

        export_d3_json(graph, output_path)

        assert output_path.exists()

    def test_load_d3_json(self, sample_graph, temp_output_dir):
        """Test loading D3 JSON."""
        output_path = temp_output_dir / "graph.json"
        export_d3_json(sample_graph, output_path)

        data = load_d3_json(output_path)

        assert isinstance(data, dict)
        assert "nodes" in data
        assert "links" in data


# ============================================================================
# Test Alternative Formats
# ============================================================================


class TestAlternativeFormats:
    """Tests for alternative export formats."""

    def test_export_graphml(self, sample_graph, temp_output_dir):
        """Test GraphML export."""
        output_path = temp_output_dir / "graph.graphml"
        export_graphml(sample_graph, output_path)

        assert output_path.exists()

        # Check it's valid XML
        content = output_path.read_text()
        assert "<?xml version=" in content
        assert "<graphml" in content
        assert "<graph" in content

    def test_export_graphml_preserves_attributes(self, sample_graph, temp_output_dir):
        """Test that GraphML preserves node/edge attributes."""
        output_path = temp_output_dir / "graph.graphml"
        export_graphml(sample_graph, output_path)

        content = output_path.read_text()
        # Check that attributes are present in the file
        assert "frequency" in content or "label" in content

    @pytest.mark.skipif(not HAS_PYDOT, reason="pydot not installed")
    def test_export_dot(self, sample_graph, temp_output_dir):
        """Test DOT export."""
        output_path = temp_output_dir / "graph.dot"
        export_dot(sample_graph, output_path)

        assert output_path.exists()

        content = output_path.read_text()
        assert "graph" in content.lower() or "digraph" in content.lower()

    @pytest.mark.skipif(not HAS_PYDOT, reason="pydot not installed")
    def test_export_dot_directed(self, sample_directed_graph, temp_output_dir):
        """Test DOT export for directed graph."""
        output_path = temp_output_dir / "graph.dot"
        export_dot(sample_directed_graph, output_path)

        assert output_path.exists()

        content = output_path.read_text()
        # Directed graphs should have "digraph"
        assert "digraph" in content.lower() or "graph" in content.lower()

    def test_export_dot_without_pydot(self, sample_graph, temp_output_dir):
        """Test that DOT export raises helpful error without pydot."""
        if HAS_PYDOT:
            pytest.skip("pydot is installed")

        output_path = temp_output_dir / "graph.dot"

        with pytest.raises(ImportError, match="pydot"):
            export_dot(sample_graph, output_path)

    def test_export_csv_basic(self, sample_graph, temp_output_dir):
        """Test CSV export."""
        export_csv(sample_graph, temp_output_dir)

        nodes_path = temp_output_dir / "nodes.csv"
        edges_path = temp_output_dir / "edges.csv"

        assert nodes_path.exists()
        assert edges_path.exists()

    def test_export_csv_nodes_structure(self, sample_graph, temp_output_dir):
        """Test CSV nodes structure."""
        export_csv(sample_graph, temp_output_dir)

        nodes_path = temp_output_dir / "nodes.csv"

        with open(nodes_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 3
        assert "id" in rows[0]
        assert "label" in rows[0]

    def test_export_csv_edges_structure(self, sample_graph, temp_output_dir):
        """Test CSV edges structure."""
        export_csv(sample_graph, temp_output_dir)

        edges_path = temp_output_dir / "edges.csv"

        with open(edges_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert "source" in rows[0]
        assert "target" in rows[0]
        assert "weight" in rows[0]

    def test_export_csv_custom_filenames(self, sample_graph, temp_output_dir):
        """Test CSV export with custom filenames."""
        export_csv(
            sample_graph,
            temp_output_dir,
            nodes_filename="custom_nodes.csv",
            edges_filename="custom_edges.csv",
        )

        assert (temp_output_dir / "custom_nodes.csv").exists()
        assert (temp_output_dir / "custom_edges.csv").exists()

    def test_export_gexf(self, sample_graph, temp_output_dir):
        """Test GEXF export."""
        output_path = temp_output_dir / "graph.gexf"
        export_gexf(sample_graph, output_path)

        assert output_path.exists()

        content = output_path.read_text()
        assert "<?xml version=" in content
        assert "<gexf" in content

    def test_export_json_graph(self, sample_graph, temp_output_dir):
        """Test NetworkX JSON export."""
        output_path = temp_output_dir / "graph.json"
        export_json_graph(sample_graph, output_path)

        assert output_path.exists()

        with open(output_path, "r") as f:
            data = json.load(f)

        # Should have NetworkX node-link format
        assert "nodes" in data or "links" in data or "directed" in data


# ============================================================================
# Test HTML Generation
# ============================================================================


class TestHTMLGeneration:
    """Tests for HTML visualization generation."""

    def test_generate_html_basic(self, sample_graph, temp_output_dir):
        """Test basic HTML generation."""
        html_path = generate_html(sample_graph, temp_output_dir)

        assert html_path.exists()
        assert html_path.name == "index.html"

        # Check that data file was also created
        data_path = temp_output_dir / "graph_data.json"
        assert data_path.exists()

    def test_generate_html_content(self, sample_graph, temp_output_dir):
        """Test HTML content structure."""
        html_path = generate_html(sample_graph, temp_output_dir)

        content = html_path.read_text()

        # Check essential HTML elements
        assert "<!DOCTYPE html>" in content
        assert "<html" in content
        assert "d3js.org" in content.lower() or "d3.js" in content.lower()
        assert "<svg" in content

    def test_generate_html_with_title(self, sample_graph, temp_output_dir):
        """Test HTML generation with custom title."""
        html_path = generate_html(sample_graph, temp_output_dir, title="My Network")

        content = html_path.read_text()
        assert "My Network" in content

    def test_generate_html_with_dimensions(self, sample_graph, temp_output_dir):
        """Test HTML generation with custom dimensions."""
        html_path = generate_html(sample_graph, temp_output_dir, width=800, height=600)

        content = html_path.read_text()
        assert "800" in content
        assert "600" in content

    def test_generate_html_with_evidence(self, sample_directed_graph, temp_output_dir):
        """Test HTML generation with evidence."""
        generate_html(sample_directed_graph, temp_output_dir, include_evidence=True)

        # Check that data file includes evidence
        data_path = temp_output_dir / "graph_data.json"
        data = load_d3_json(data_path)

        if data["links"]:
            # At least one link should have evidence
            assert any("evidence" in link for link in data["links"])

    def test_generate_html_returns_path(self, sample_graph, temp_output_dir):
        """Test that generate_html returns path."""
        html_path = generate_html(sample_graph, temp_output_dir)

        assert isinstance(html_path, Path)
        assert html_path.exists()


# ============================================================================
# Integration Tests
# ============================================================================


class TestExportIntegration:
    """Integration tests for complete export workflows."""

    def test_export_all_formats(self, sample_graph, temp_output_dir):
        """Test exporting to all formats."""
        # D3 JSON
        export_d3_json(sample_graph, temp_output_dir / "d3.json")
        assert (temp_output_dir / "d3.json").exists()

        # GraphML
        export_graphml(sample_graph, temp_output_dir / "graph.graphml")
        assert (temp_output_dir / "graph.graphml").exists()

        # DOT (if pydot available)
        if HAS_PYDOT:
            export_dot(sample_graph, temp_output_dir / "graph.dot")
            assert (temp_output_dir / "graph.dot").exists()

        # CSV
        export_csv(sample_graph, temp_output_dir)
        assert (temp_output_dir / "nodes.csv").exists()
        assert (temp_output_dir / "edges.csv").exists()

        # GEXF
        export_gexf(sample_graph, temp_output_dir / "graph.gexf")
        assert (temp_output_dir / "graph.gexf").exists()

        # HTML
        html_path = generate_html(sample_graph, temp_output_dir / "viz")
        assert html_path.exists()

    def test_export_from_cooccurrence(self, temp_output_dir):
        """Test exporting graph built from co-occurrence."""
        matrix = {
            "consciousness": {"being": 0.85, "intentionality": 0.92},
            "being": {"consciousness": 0.85},
            "intentionality": {"consciousness": 0.92},
        }

        graph = graph_from_cooccurrence(matrix, threshold=0.5)

        # Export to D3
        export_d3_json(graph, temp_output_dir / "cooccur.json")
        data = load_d3_json(temp_output_dir / "cooccur.json")

        assert len(data["nodes"]) == 3
        assert len(data["links"]) > 0

    def test_export_from_relations(self, sample_directed_graph, temp_output_dir):
        """Test exporting graph built from relations."""
        # Export with evidence
        export_d3_json(
            sample_directed_graph,
            temp_output_dir / "relations.json",
            include_evidence=True,
        )

        data = load_d3_json(temp_output_dir / "relations.json")

        # Should have evidence in links
        assert any("evidence" in link for link in data["links"])

    def test_round_trip_d3_json(self, sample_graph, temp_output_dir):
        """Test exporting and loading D3 JSON."""
        output_path = temp_output_dir / "graph.json"

        # Export
        export_d3_json(sample_graph, output_path)

        # Load
        data = load_d3_json(output_path)

        # Verify data integrity
        assert len(data["nodes"]) == sample_graph.node_count()
        assert len(data["links"]) == sample_graph.edge_count()

    def test_empty_graph_exports(self, temp_output_dir):
        """Test exporting empty graph raises validation error."""
        from concept_mapper.validation import EmptyOutputError
        import pytest

        graph = ConceptGraph()

        # Should raise EmptyOutputError
        with pytest.raises(EmptyOutputError, match="Cannot save empty graph"):
            export_d3_json(graph, temp_output_dir / "empty.json")

        with pytest.raises(EmptyOutputError, match="Cannot save empty graph"):
            export_graphml(graph, temp_output_dir / "empty.graphml")

        with pytest.raises(EmptyOutputError, match="Cannot save empty graph"):
            export_csv(graph, temp_output_dir / "empty")
