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
    """Create a sample text file."""
    text_file = tmp_path / "sample.txt"
    text_file.write_text(
        "Consciousness is intentional. Being is presence. "
        "The concept of abstraction transforms social relations into things."
    )
    return text_file


@pytest.fixture
def sample_corpus_json(tmp_path):
    """Create a sample preprocessed corpus JSON file."""
    from concept_mapper.corpus.loader import load_file
    from concept_mapper.preprocessing.pipeline import preprocess

    # Create a simple text file
    text_file = tmp_path / "source.txt"
    text_file.write_text(
        "Consciousness is intentional. Being is presence in time. "
        "Abstraction transforms processes into things."
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

    terms = TermList(
        [
            {"term": "consciousness", "pos": "NN"},
            {"term": "being", "pos": "NN"},
            {"term": "abstraction", "pos": "NN"},
        ]
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
        result = runner.invoke(
            cli, ["search", str(sample_corpus_json), "consciousness"]
        )

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
                "consciousness",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()


# ============================================================================
# Test Concordance Command
# ============================================================================


class TestConcordanceCommand:
    """Tests for concordance command."""

    def test_concordance_basic(self, runner, sample_corpus_json):
        """Test basic concordance."""
        result = runner.invoke(
            cli, ["concordance", str(sample_corpus_json), "consciousness"]
        )

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
