"""
Tests for CLI interface (Phase 10).
"""

import pytest
import json
from click.testing import CliRunner
from concept_mapper.cli import cli


@pytest.fixture
def runner():
    """Create Click test runner."""
    return CliRunner()


@pytest.fixture
def sample_text_file(tmp_path):
    """Create a sample text file with philosophical terminology."""
    text_file = tmp_path / "sample.txt"
    # Use a more substantial text that will produce detectable rare terms
    text_file.write_text(
        "Geist is a fundamental concept in Hegel's dialectical philosophy. "
        "Geist refers to the self-developing rationality that animates history and thought. "
        "Aufhebung describes the dialectical movement of sublation and preservation. "
        "Through Aufhebung, contradictions are both negated and preserved at a higher level. "
        "Selbstbewusstsein characterizes the fundamental structure of self-consciousness. "
        "Selbstbewusstsein cannot be understood apart from recognition by another consciousness. "
        "The dialectical process grounds the entire structure of thought and reality. "
        "The dialectic is not mere succession but the negation of negation. "
        "Anerkennung distinguishes mutual recognition from mere acknowledgment. "
        "Understanding Anerkennung is crucial for Hegel's social philosophy. "
        "Sittlichkeit involves the actualization of ethical life through social institutions. "
        "Sittlichkeit contrasts with abstract morality and immediate desire. "
        "The Phenomenology investigates the structures of consciousness and spirit. "
        "The phenomenological method involves the dialectical unfolding of shapes of consciousness. "
        "Mediation plays a crucial role in the interpretation of concrete universals. "
        "The dialectical circle describes how the end returns to the beginning enriched by development."
    )
    return text_file


@pytest.fixture
def sample_corpus_json(tmp_path):
    """Create a sample preprocessed corpus JSON file."""
    from concept_mapper.corpus.loader import load_file
    from concept_mapper.preprocessing.pipeline import preprocess

    # Create a text file with philosophical terms that should be detected
    text_file = tmp_path / "source.txt"
    text_file.write_text(
        "Geist is a fundamental concept in Hegel's dialectical philosophy. "
        "Geist refers to the self-developing rationality that animates history. "
        "Aufhebung describes the dialectical movement of sublation. "
        "Through Aufhebung, contradictions are both negated and preserved. "
        "Selbstbewusstsein characterizes the fundamental structure of self-consciousness. "
        "Selbstbewusstsein cannot be understood apart from recognition. "
        "The dialectical process grounds the entire structure of thought. "
        "The dialectic is not mere succession but negation of negation. "
        "Anerkennung distinguishes mutual recognition from acknowledgment. "
        "Understanding Anerkennung is crucial for social philosophy. "
        "Sittlichkeit involves the actualization of ethical life. "
        "Sittlichkeit contrasts with abstract morality and immediate desire."
    )

    # Process it
    doc = load_file(text_file)
    processed = preprocess(doc)

    # Save as JSON
    corpus_file = tmp_path / "corpus.json"
    serialized = [
        {
            "raw_text": processed.raw_text,
            "sentences": processed.sentences,
            "tokens": processed.tokens,
            "lemmas": processed.lemmas,
            "pos_tags": processed.pos_tags,
            "metadata": processed.metadata,
        }
    ]

    with open(corpus_file, "w") as f:
        json.dump(serialized, f)

    return corpus_file


@pytest.fixture
def sample_terms_json(tmp_path):
    """Create a sample terms JSON file."""
    from concept_mapper.terms.models import TermList
    from concept_mapper.terms.manager import TermManager

    terms = TermList.from_dict(
        {
            "terms": [
                {"term": "Geist", "pos": "NN"},
                {"term": "Aufhebung", "pos": "NN"},
                {"term": "Selbstbewusstsein", "pos": "NN"},
                {"term": "Anerkennung", "pos": "NN"},
                {"term": "Sittlichkeit", "pos": "NN"},
            ]
        }
    )

    terms_file = tmp_path / "terms.json"
    manager = TermManager(terms)
    manager.export_to_json(terms_file)

    return terms_file


# ============================================================================
# Test CLI Framework
# ============================================================================


class TestCLIFramework:
    """Tests for basic CLI framework."""

    def test_cli_help(self, runner):
        """Test main CLI help."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "Concept Mapper" in result.output
        assert "ingest" in result.output

    def test_cli_version_flags(self, runner):
        """Test CLI with verbose and output-dir flags."""
        result = runner.invoke(
            cli, ["--verbose", "--output-dir", "/tmp/test", "--help"]
        )

        assert result.exit_code == 0


# ============================================================================
# Test Ingest Command
# ============================================================================


class TestIngestCommand:
    """Tests for ingest command."""

    def test_ingest_file(self, runner, sample_text_file, tmp_path):
        """Test ingesting a single file."""
        output_file = tmp_path / "corpus.json"

        result = runner.invoke(
            cli, ["ingest", str(sample_text_file), "--output", str(output_file)]
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)

        assert isinstance(data, list)
        assert len(data) == 1
        assert "raw_text" in data[0]
        assert "sentences" in data[0]
        assert "tokens" in data[0]

    def test_ingest_directory_without_recursive(self, runner, tmp_path):
        """Test that directory requires --recursive flag."""
        result = runner.invoke(cli, ["ingest", str(tmp_path)])

        assert result.exit_code == 1
        assert "recursive" in result.output.lower()

    def test_ingest_verbose(self, runner, sample_text_file, tmp_path):
        """Test ingest with verbose output."""
        output_file = tmp_path / "corpus.json"

        result = runner.invoke(
            cli,
            [
                "--verbose",
                "ingest",
                str(sample_text_file),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert "Loading" in result.output
        assert "Processing" in result.output


# ============================================================================
# Test Rarities Command
# ============================================================================


class TestRaritiesCommand:
    """Tests for rarities command."""

    def test_rarities_basic(self, runner, sample_corpus_json, tmp_path):
        """Test basic rarities detection."""
        output_file = tmp_path / "terms.json"

        result = runner.invoke(
            cli,
            [
                "rarities",
                str(sample_corpus_json),
                "--threshold",
                "0.0",
                "--top-n",
                "10",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_rarities_methods(self, runner, sample_corpus_json, tmp_path):
        """Test different detection methods."""
        for method in ["ratio", "tfidf", "hybrid"]:
            output_file = tmp_path / f"terms_{method}.json"

            result = runner.invoke(
                cli,
                [
                    "rarities",
                    str(sample_corpus_json),
                    "--method",
                    method,
                    "--threshold",
                    "0.0",
                    "--output",
                    str(output_file),
                ],
            )

            assert result.exit_code == 0

    def test_rarities_displays_results(self, runner, sample_corpus_json):
        """Test that rarities displays results."""
        result = runner.invoke(
            cli,
            ["rarities", str(sample_corpus_json), "--threshold", "0.0", "--top-n", "5"],
        )

        assert result.exit_code == 0
        assert "rare terms" in result.output.lower()


# ============================================================================
# Test Search Command
# ============================================================================


class TestSearchCommand:
    """Tests for search command."""

    def test_search_basic(self, runner, sample_corpus_json):
        """Test basic search."""
        result = runner.invoke(cli, ["search", str(sample_corpus_json), "Geist"])

        assert result.exit_code == 0
        assert "occurrence" in result.output.lower()

    def test_search_with_context(self, runner, sample_corpus_json):
        """Test search with context."""
        result = runner.invoke(
            cli, ["search", str(sample_corpus_json), "being", "--context", "1"]
        )

        assert result.exit_code == 0

    def test_search_no_matches(self, runner, sample_corpus_json):
        """Test search with no matches."""
        result = runner.invoke(cli, ["search", str(sample_corpus_json), "nonexistent"])

        assert result.exit_code == 0
        assert "No matches" in result.output

    def test_search_with_output(self, runner, sample_corpus_json, tmp_path):
        """Test search with output file."""
        output_file = tmp_path / "results.txt"

        result = runner.invoke(
            cli,
            [
                "search",
                str(sample_corpus_json),
                "Geist",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_search_with_diagram(self, runner, sample_corpus_json):
        """Test search with sentence diagram generation."""
        result = runner.invoke(
            cli, ["search", str(sample_corpus_json), "dialectic", "--diagram"]
        )

        assert result.exit_code == 0
        assert "occurrence(s)" in result.output
        assert "Diagram:" in result.output
        # Check for dependency parse output
        assert "root" in result.output or "SENTENCE DIAGRAM" in result.output

    def test_search_with_diagram_format(self, runner, sample_corpus_json):
        """Test search with different diagram formats."""
        result = runner.invoke(
            cli,
            [
                "search",
                str(sample_corpus_json),
                "dialectic",
                "--diagram",
                "--diagram-format",
                "tree",
            ],
        )

        assert result.exit_code == 0
        assert "Diagram:" in result.output

    def test_search_with_diagram_output(self, runner, sample_corpus_json, tmp_path):
        """Test search with diagram output to file."""
        output_file = tmp_path / "diagrams.txt"
        result = runner.invoke(
            cli,
            [
                "search",
                str(sample_corpus_json),
                "dialectic",
                "--diagram",
                "-o",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "Diagram:" in content
        assert "dialectic" in content.lower()


# ============================================================================
# Test Concordance Command
# ============================================================================


class TestConcordanceCommand:
    """Tests for concordance command."""

    def test_concordance_basic(self, runner, sample_corpus_json):
        """Test basic concordance."""
        result = runner.invoke(cli, ["concordance", str(sample_corpus_json), "Geist"])

        assert result.exit_code == 0
        assert "KWIC" in result.output

    def test_concordance_width(self, runner, sample_corpus_json):
        """Test concordance with custom width."""
        result = runner.invoke(
            cli, ["concordance", str(sample_corpus_json), "being", "--width", "30"]
        )

        assert result.exit_code == 0

    def test_concordance_no_matches(self, runner, sample_corpus_json):
        """Test concordance with no matches."""
        result = runner.invoke(
            cli, ["concordance", str(sample_corpus_json), "nonexistent"]
        )

        assert result.exit_code == 0
        assert "No matches" in result.output


# ============================================================================
# Test Graph Command
# ============================================================================


class TestGraphCommand:
    """Tests for graph command."""

    def test_graph_cooccurrence(
        self, runner, sample_corpus_json, sample_terms_json, tmp_path
    ):
        """Test graph building with co-occurrence."""
        output_file = tmp_path / "graph.json"

        result = runner.invoke(
            cli,
            [
                "graph",
                str(sample_corpus_json),
                "--terms",
                str(sample_terms_json),
                "--method",
                "cooccurrence",
                "--threshold",
                "0.0",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify JSON structure
        with open(output_file) as f:
            data = json.load(f)

        assert "nodes" in data
        assert "links" in data

    def test_graph_relations(
        self, runner, sample_corpus_json, sample_terms_json, tmp_path
    ):
        """Test graph building with relations."""
        output_file = tmp_path / "graph.json"

        result = runner.invoke(
            cli,
            [
                "graph",
                str(sample_corpus_json),
                "--terms",
                str(sample_terms_json),
                "--method",
                "relations",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_graph_requires_terms(self, runner, sample_corpus_json):
        """Test that graph command requires --terms."""
        result = runner.invoke(cli, ["graph", str(sample_corpus_json)])

        assert result.exit_code != 0


# ============================================================================
# Test Export Command
# ============================================================================


class TestExportCommand:
    """Tests for export command."""

    @pytest.fixture
    def sample_graph_json(self, tmp_path):
        """Create a sample graph JSON file."""
        graph_data = {
            "nodes": [
                {
                    "id": "consciousness",
                    "label": "Consciousness",
                    "size": 10,
                    "group": 0,
                },
                {"id": "being", "label": "Being", "size": 8, "group": 0},
            ],
            "links": [{"source": "consciousness", "target": "being", "weight": 0.85}],
        }

        graph_file = tmp_path / "graph.json"
        with open(graph_file, "w") as f:
            json.dump(graph_data, f)

        return graph_file

    def test_export_html(self, runner, sample_graph_json, tmp_path):
        """Test exporting to HTML."""
        output_dir = tmp_path / "viz"

        result = runner.invoke(
            cli,
            [
                "export",
                str(sample_graph_json),
                "--format",
                "html",
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "index.html").exists()

    def test_export_csv(self, runner, sample_graph_json, tmp_path):
        """Test exporting to CSV."""
        output_dir = tmp_path / "csv"

        result = runner.invoke(
            cli,
            [
                "export",
                str(sample_graph_json),
                "--format",
                "csv",
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        assert (output_dir / "nodes.csv").exists()
        assert (output_dir / "edges.csv").exists()

    def test_export_graphml(self, runner, sample_graph_json, tmp_path):
        """Test exporting to GraphML."""
        output_file = tmp_path / "graph.graphml"

        result = runner.invoke(
            cli,
            [
                "export",
                str(sample_graph_json),
                "--format",
                "graphml",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_export_with_title(self, runner, sample_graph_json, tmp_path):
        """Test export with custom title."""
        output_dir = tmp_path / "viz"

        result = runner.invoke(
            cli,
            [
                "export",
                str(sample_graph_json),
                "--format",
                "html",
                "--title",
                "My Network",
                "--output",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0
        # Verify title is in HTML
        html_file = output_dir / "index.html"
        content = html_file.read_text()
        assert "My Network" in content


# ============================================================================
# Integration Tests
# ============================================================================


class TestCLIIntegration:
    """Integration tests for complete CLI workflows."""

    def test_full_workflow(self, runner, sample_text_file, tmp_path):
        """Test complete workflow from ingest to export."""
        corpus_file = tmp_path / "corpus.json"
        terms_file = tmp_path / "terms.json"
        graph_file = tmp_path / "graph.json"
        viz_dir = tmp_path / "viz"

        # 1. Ingest
        result = runner.invoke(
            cli, ["ingest", str(sample_text_file), "--output", str(corpus_file)]
        )
        assert result.exit_code == 0
        assert corpus_file.exists()

        # 2. Detect rarities
        result = runner.invoke(
            cli,
            [
                "rarities",
                str(corpus_file),
                "--threshold",
                "0.0",
                "--top-n",
                "10",
                "--output",
                str(terms_file),
            ],
        )
        assert result.exit_code == 0
        assert terms_file.exists()

        # 3. Build graph
        result = runner.invoke(
            cli,
            [
                "graph",
                str(corpus_file),
                "--terms",
                str(terms_file),
                "--method",
                "cooccurrence",
                "--output",
                str(graph_file),
            ],
        )
        assert result.exit_code == 0
        assert graph_file.exists()

        # 4. Export to HTML
        result = runner.invoke(
            cli,
            ["export", str(graph_file), "--format", "html", "--output", str(viz_dir)],
        )
        assert result.exit_code == 0
        assert (viz_dir / "index.html").exists()


class TestSourceDerivedFilenames:
    """Test source-derived filename functionality."""

    def test_ingest_source_derived_naming(self, runner, sample_text_file, tmp_path):
        """Test that ingest uses source-derived output filename."""
        output_dir = tmp_path / "output"

        result = runner.invoke(
            cli,
            [
                "--output-dir",
                str(output_dir),
                "ingest",
                str(sample_text_file),
            ],
        )

        assert result.exit_code == 0
        # Should create sample.json (not corpus.json)
        expected_corpus = output_dir / "corpus" / "sample.json"
        assert expected_corpus.exists()
        assert not (output_dir / "corpus" / "corpus.json").exists()

    def test_rarities_source_derived_naming(self, runner, sample_corpus_json, tmp_path):
        """Test that rarities uses source-derived output filename."""
        output_dir = tmp_path / "output"
        corpus_file = sample_corpus_json

        result = runner.invoke(
            cli,
            [
                "--output-dir",
                str(output_dir),
                "rarities",
                str(corpus_file),
                "--threshold",
                "0.0",
                "--top-n",
                "5",
            ],
        )

        assert result.exit_code == 0
        # Should create corpus.json (not terms.json) because corpus was named corpus.json
        expected_terms = output_dir / "terms" / "corpus.json"
        assert expected_terms.exists()
        assert not (output_dir / "terms" / "terms.json").exists()

    def test_graph_source_derived_naming(self, runner, sample_corpus_json, tmp_path):
        """Test that graph uses source-derived output filename."""
        output_dir = tmp_path / "output"
        corpus_file = sample_corpus_json

        # First create terms file
        terms_file = tmp_path / "source_terms.json"
        runner.invoke(
            cli,
            [
                "rarities",
                str(corpus_file),
                "--threshold",
                "0.0",
                "--top-n",
                "5",
                "--output",
                str(terms_file),
            ],
        )

        result = runner.invoke(
            cli,
            [
                "--output-dir",
                str(output_dir),
                "graph",
                str(corpus_file),
                "--terms",
                str(terms_file),
            ],
        )

        assert result.exit_code == 0
        # Should create corpus.json (not graph.json)
        expected_graph = output_dir / "graphs" / "corpus.json"
        assert expected_graph.exists()
        assert not (output_dir / "graphs" / "graph.json").exists()

    def test_graph_method_suffix(self, runner, sample_corpus_json, tmp_path):
        """Test that graph adds method suffix for non-default methods."""
        output_dir = tmp_path / "output"
        corpus_file = sample_corpus_json

        # First create terms file
        terms_file = tmp_path / "source_terms.json"
        runner.invoke(
            cli,
            [
                "rarities",
                str(corpus_file),
                "--threshold",
                "0.0",
                "--top-n",
                "5",
                "--output",
                str(terms_file),
            ],
        )

        result = runner.invoke(
            cli,
            [
                "--output-dir",
                str(output_dir),
                "graph",
                str(corpus_file),
                "--terms",
                str(terms_file),
                "--method",
                "relations",
            ],
        )

        assert result.exit_code == 0
        # Should create corpus_relations.json
        expected_graph = output_dir / "graphs" / "corpus_relations.json"
        assert expected_graph.exists()

    def test_export_source_derived_naming(self, runner, tmp_path):
        """Test that export uses source-derived output paths."""
        from concept_mapper.graph import ConceptGraph
        from concept_mapper.export import export_d3_json

        output_dir = tmp_path / "output"

        # Create a simple graph file with source-derived name
        graph_file = tmp_path / "sample1_analytic.json"
        graph = ConceptGraph()
        graph.add_node("term1")
        graph.add_node("term2")
        graph.add_edge("term1", "term2", weight=0.5)
        export_d3_json(graph, graph_file)

        result = runner.invoke(
            cli,
            [
                "--output-dir",
                str(output_dir),
                "export",
                str(graph_file),
                "--format",
                "html",
            ],
        )

        assert result.exit_code == 0
        # Should create sample1_analytic/ directory (not visualization/)
        expected_viz = output_dir / "exports" / "sample1_analytic" / "index.html"
        assert expected_viz.exists()
        assert not (output_dir / "exports" / "visualization" / "index.html").exists()

    def test_multiple_texts_no_overwrite(self, runner, tmp_path):
        """Test that processing multiple texts doesn't cause overwrites."""
        output_dir = tmp_path / "output"

        # Create two different text files
        text1 = tmp_path / "sample1.txt"
        text1.write_text(
            "Geist is a concept. Aufhebung is dialectical. "
            "Geist and Aufhebung are fundamental to Hegel."
        )

        text2 = tmp_path / "sample2.txt"
        text2.write_text(
            "Praxis is action. Abstraction is a process. "
            "Praxis and abstraction are central to Philosopher."
        )

        # Ingest first text
        result1 = runner.invoke(
            cli,
            ["--output-dir", str(output_dir), "ingest", str(text1)],
        )
        assert result1.exit_code == 0

        corpus1 = output_dir / "corpus" / "sample1.json"
        assert corpus1.exists()
        corpus1_content = corpus1.read_text()

        # Ingest second text
        result2 = runner.invoke(
            cli,
            ["--output-dir", str(output_dir), "ingest", str(text2)],
        )
        assert result2.exit_code == 0

        corpus2 = output_dir / "corpus" / "sample2.json"
        assert corpus2.exists()

        # Verify first corpus wasn't overwritten
        assert corpus1.exists()
        assert corpus1.read_text() == corpus1_content
        assert corpus1.read_text() != corpus2.read_text()

    def test_explicit_output_override_still_works(
        self, runner, sample_text_file, tmp_path
    ):
        """Test that explicit -o flag still works and overrides default naming."""
        custom_output = tmp_path / "my_custom_corpus.json"

        result = runner.invoke(
            cli,
            ["ingest", str(sample_text_file), "--output", str(custom_output)],
        )

        assert result.exit_code == 0
        assert custom_output.exists()

    def test_full_workflow_with_source_derived_names(
        self, runner, sample_text_file, tmp_path
    ):
        """Test complete workflow using source-derived filenames."""
        output_dir = tmp_path / "output"

        # 1. Ingest
        result = runner.invoke(
            cli,
            ["--output-dir", str(output_dir), "ingest", str(sample_text_file)],
        )
        assert result.exit_code == 0

        corpus_file = output_dir / "corpus" / "sample.json"
        assert corpus_file.exists()

        # 2. Rarities
        result = runner.invoke(
            cli,
            [
                "--output-dir",
                str(output_dir),
                "rarities",
                str(corpus_file),
                "--threshold",
                "0.0",
                "--top-n",
                "5",
            ],
        )
        assert result.exit_code == 0

        terms_file = output_dir / "terms" / "sample.json"
        assert terms_file.exists()

        # 3. Graph
        result = runner.invoke(
            cli,
            [
                "--output-dir",
                str(output_dir),
                "graph",
                str(corpus_file),
                "--terms",
                str(terms_file),
            ],
        )
        assert result.exit_code == 0

        graph_file = output_dir / "graphs" / "sample.json"
        assert graph_file.exists()

        # 4. Export
        result = runner.invoke(
            cli,
            [
                "--output-dir",
                str(output_dir),
                "export",
                str(graph_file),
                "--format",
                "html",
            ],
        )
        assert result.exit_code == 0

        viz_file = output_dir / "exports" / "sample" / "index.html"
        assert viz_file.exists()
