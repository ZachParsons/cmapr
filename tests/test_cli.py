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
        graph_file = tmp_path / "eco_spl.json"
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
        # Should create eco_spl/ directory (not visualization/)
        expected_viz = output_dir / "exports" / "eco_spl" / "index.html"
        assert expected_viz.exists()
        assert not (output_dir / "exports" / "visualization" / "index.html").exists()

    def test_multiple_texts_no_overwrite(self, runner, tmp_path):
        """Test that processing multiple texts doesn't cause overwrites."""
        output_dir = tmp_path / "output"

        # Create two different text files
        text1 = tmp_path / "eco_spl.txt"
        text1.write_text(
            "Geist is a concept. Aufhebung is dialectical. "
            "Geist and Aufhebung are fundamental to Hegel."
        )

        text2 = tmp_path / "eco_spl_alt.txt"
        text2.write_text(
            "Intentionality is directedness. Consciousness is awareness. "
            "Intentionality and consciousness are central to Brentano."
        )

        # Ingest first text
        result1 = runner.invoke(
            cli,
            ["--output-dir", str(output_dir), "ingest", str(text1)],
        )
        assert result1.exit_code == 0

        corpus1 = output_dir / "corpus" / "eco_spl.json"
        assert corpus1.exists()
        corpus1_content = corpus1.read_text()

        # Ingest second text
        result2 = runner.invoke(
            cli,
            ["--output-dir", str(output_dir), "ingest", str(text2)],
        )
        assert result2.exit_code == 0

        corpus2 = output_dir / "corpus" / "eco_spl_alt.json"
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


class TestWindowHelpers:
    """Unit tests for window parsing and slot helpers."""

    def test_parse_window_sentence_zero(self):
        from concept_mapper.cli import _parse_window

        assert _parse_window("s0") == ("s", 0)

    def test_parse_window_sentence_one(self):
        from concept_mapper.cli import _parse_window

        assert _parse_window("s1") == ("s", 1)

    def test_parse_window_paragraph(self):
        from concept_mapper.cli import _parse_window

        assert _parse_window("p0") == ("p", 0)
        assert _parse_window("p2") == ("p", 2)

    def test_parse_window_uppercase_entity(self):
        from concept_mapper.cli import _parse_window

        assert _parse_window("S1") == ("s", 1)
        assert _parse_window("P1") == ("p", 1)

    def test_parse_window_invalid_entity(self):
        import click
        from concept_mapper.cli import _parse_window

        with pytest.raises(click.BadParameter):
            _parse_window("x1")

    def test_parse_window_non_integer_radius(self):
        import click
        from concept_mapper.cli import _parse_window

        with pytest.raises(click.BadParameter):
            _parse_window("sabc")

    def test_parse_window_too_short(self):
        import click
        from concept_mapper.cli import _parse_window

        with pytest.raises(click.BadParameter):
            _parse_window("s")

    def test_parse_window_negative_radius(self):
        import click
        from concept_mapper.cli import _parse_window

        with pytest.raises(click.BadParameter):
            _parse_window("s-1")

    def test_offset_label_zero(self):
        from concept_mapper.cli import _offset_label

        assert _offset_label(0, 1) == "current:"

    def test_offset_label_prev_radius_one(self):
        from concept_mapper.cli import _offset_label

        assert _offset_label(-1, 1) == "previous:"

    def test_offset_label_next_radius_one(self):
        from concept_mapper.cli import _offset_label

        assert _offset_label(1, 1) == "next:"

    def test_offset_label_large_radius(self):
        from concept_mapper.cli import _offset_label

        assert _offset_label(-2, 2) == "prev 2:"
        assert _offset_label(2, 2) == "next 2:"
        assert _offset_label(-1, 2) == "previous:"
        assert _offset_label(1, 2) == "next:"

    def test_compute_window_slots_sentence_radius_zero(self):
        from concept_mapper.corpus.models import ProcessedDocument
        from concept_mapper.search.find import SentenceMatch
        from concept_mapper.cli import _compute_window_slots

        doc = ProcessedDocument(
            raw_text="A. B. C.",
            sentences=["A.", "B.", "C."],
            tokens=[],
            lemmas=[],
            pos_tags=[],
            metadata={},
        )
        match = SentenceMatch(
            sentence="B.", doc_id="d", sent_index=1, term_positions=[0], term="b"
        )
        slots = _compute_window_slots(match, doc, "s", 0)
        assert len(slots) == 1
        assert slots[0] == (0, ["B."])

    def test_compute_window_slots_sentence_radius_one(self):
        from concept_mapper.corpus.models import ProcessedDocument
        from concept_mapper.search.find import SentenceMatch
        from concept_mapper.cli import _compute_window_slots

        doc = ProcessedDocument(
            raw_text="A. B. C.",
            sentences=["A.", "B.", "C."],
            tokens=[],
            lemmas=[],
            pos_tags=[],
            metadata={},
        )
        match = SentenceMatch(
            sentence="B.", doc_id="d", sent_index=1, term_positions=[0], term="b"
        )
        slots = _compute_window_slots(match, doc, "s", 1)
        assert len(slots) == 3
        offsets = [o for o, _ in slots]
        assert offsets == [-1, 0, 1]
        assert slots[0][1] == ["A."]
        assert slots[1][1] == ["B."]
        assert slots[2][1] == ["C."]

    def test_compute_window_slots_boundary_no_prev(self):
        from concept_mapper.corpus.models import ProcessedDocument
        from concept_mapper.search.find import SentenceMatch
        from concept_mapper.cli import _compute_window_slots

        doc = ProcessedDocument(
            raw_text="A. B.",
            sentences=["A.", "B."],
            tokens=[],
            lemmas=[],
            pos_tags=[],
            metadata={},
        )
        # Match is at index 0, so prev slot should be empty
        match = SentenceMatch(
            sentence="A.", doc_id="d", sent_index=0, term_positions=[0], term="a"
        )
        slots = _compute_window_slots(match, doc, "s", 1)
        prev_slot = next(s for o, s in slots if o == -1)
        assert prev_slot == []

    def test_compute_window_slots_paragraph_mode(self):
        from concept_mapper.corpus.models import ProcessedDocument
        from concept_mapper.search.find import SentenceMatch
        from concept_mapper.cli import _compute_window_slots

        # Two paragraphs: sentences 0-1 in para 0, sentences 2-3 in para 1
        doc = ProcessedDocument(
            raw_text="A. B. C. D.",
            sentences=["A.", "B.", "C.", "D."],
            tokens=[],
            lemmas=[],
            pos_tags=[],
            metadata={},
            paragraph_indices=[0, 0, 1, 1],
        )
        # Match is in paragraph 1 (sent_index=2)
        match = SentenceMatch(
            sentence="C.", doc_id="d", sent_index=2, term_positions=[0], term="c"
        )
        slots = _compute_window_slots(match, doc, "p", 0)
        assert len(slots) == 1
        _, sentences = slots[0]
        assert set(sentences) == {"C.", "D."}


class TestAnalyzeWindowCommand:
    """Tests for analyze --window option."""

    def test_window_s0_runs(self, runner, sample_corpus_json):
        """analyze -w s0 should succeed and print window analysis."""
        result = runner.invoke(
            cli, ["analyze", str(sample_corpus_json), "dialectic", "-w", "s0"]
        )
        assert result.exit_code == 0
        assert "Window analysis" in result.output
        assert "current:" in result.output

    def test_window_s1_shows_slots(self, runner, sample_corpus_json):
        """analyze -w s1 should show previous/current/next slots."""
        result = runner.invoke(
            cli, ["analyze", str(sample_corpus_json), "dialectic", "-w", "s1"]
        )
        assert result.exit_code == 0
        assert "previous:" in result.output
        assert "current:" in result.output
        assert "next:" in result.output

    def test_window_long_form(self, runner, sample_corpus_json):
        """--window should be equivalent to -w."""
        result = runner.invoke(
            cli,
            ["analyze", str(sample_corpus_json), "dialectic", "--window", "s0"],
        )
        assert result.exit_code == 0
        assert "Window analysis" in result.output

    def test_window_no_matches(self, runner, sample_corpus_json):
        """analyze -w s0 with unknown term should report no occurrences."""
        result = runner.invoke(
            cli,
            ["analyze", str(sample_corpus_json), "xyznonexistentterm", "-w", "s0"],
        )
        assert result.exit_code == 0
        assert "No occurrences" in result.output

    def test_window_invalid_format(self, runner, sample_corpus_json):
        """analyze with a malformed -w value should exit with an error."""
        result = runner.invoke(
            cli, ["analyze", str(sample_corpus_json), "dialectic", "-w", "x99"]
        )
        assert result.exit_code != 0

    def test_window_shows_path(self, runner, sample_corpus_json):
        """Output should include a 'path:' line for each occurrence."""
        result = runner.invoke(
            cli, ["analyze", str(sample_corpus_json), "dialectic", "-w", "s0"]
        )
        assert result.exit_code == 0
        assert "path:" in result.output

    def test_window_shows_found_sentence(self, runner, sample_corpus_json):
        """Output should include the found sentence and a dependency tree for each occurrence."""
        result = runner.invoke(
            cli, ["analyze", str(sample_corpus_json), "dialectic", "-w", "s0"]
        )
        assert result.exit_code == 0
        assert "sentence:" in result.output
        assert "(root)" in result.output

    def test_window_top_n(self, runner, sample_corpus_json):
        """--top-n should limit terms per slot."""
        result = runner.invoke(
            cli,
            [
                "analyze",
                str(sample_corpus_json),
                "dialectic",
                "-w",
                "s0",
                "--top-n",
                "2",
            ],
        )
        assert result.exit_code == 0

    def test_window_p0(self, runner, sample_corpus_json):
        """analyze -w p0 (paragraph mode) should succeed."""
        result = runner.invoke(
            cli, ["analyze", str(sample_corpus_json), "dialectic", "-w", "p0"]
        )
        assert result.exit_code == 0
        assert "paragraph" in result.output


class TestSectionFilters:
    """Unit tests for _location_passes_filters and integration tests for --start-from-section / --exclude-sections."""

    # ------------------------------------------------------------------ #
    # _location_passes_filters unit tests                                 #
    # ------------------------------------------------------------------ #

    def test_none_location_passes(self):
        """A None location always passes (no data to filter on)."""
        from concept_mapper.cli import _location_passes_filters

        assert _location_passes_filters(None, start_section="1") is True

    def test_no_filters_always_passes(self):
        """With no filters, any location passes."""
        from concept_mapper.cli import _location_passes_filters
        from concept_mapper.corpus.models import SentenceLocation

        loc = SentenceLocation(sent_index=0, chapter="0")
        assert _location_passes_filters(loc) is True

    def test_start_section_excludes_low_chapter(self):
        """A chapter below start_section is excluded."""
        from concept_mapper.cli import _location_passes_filters
        from concept_mapper.corpus.models import SentenceLocation

        loc = SentenceLocation(sent_index=0, chapter="0")
        assert _location_passes_filters(loc, start_section="1") is False

    def test_start_section_allows_equal_chapter(self):
        """A chapter equal to start_section passes."""
        from concept_mapper.cli import _location_passes_filters
        from concept_mapper.corpus.models import SentenceLocation

        loc = SentenceLocation(sent_index=0, chapter="1")
        assert _location_passes_filters(loc, start_section="1") is True

    def test_start_section_allows_higher_chapter(self):
        """A chapter above start_section passes."""
        from concept_mapper.cli import _location_passes_filters
        from concept_mapper.corpus.models import SentenceLocation

        loc = SentenceLocation(sent_index=0, chapter="3")
        assert _location_passes_filters(loc, start_section="1") is True

    def test_start_section_none_chapter_excluded(self):
        """A location with no chapter label is treated as front-matter and excluded."""
        from concept_mapper.cli import _location_passes_filters
        from concept_mapper.corpus.models import SentenceLocation

        loc = SentenceLocation(sent_index=0, chapter=None)
        assert _location_passes_filters(loc, start_section="1") is False

    def test_start_section_decimal_comparison(self):
        """Decimal chapter numbers are compared as floats."""
        from concept_mapper.cli import _location_passes_filters
        from concept_mapper.corpus.models import SentenceLocation

        loc_low = SentenceLocation(sent_index=0, chapter="0.5")
        loc_high = SentenceLocation(sent_index=1, chapter="1.5")
        assert _location_passes_filters(loc_low, start_section="1") is False
        assert _location_passes_filters(loc_high, start_section="1") is True

    def test_exclude_pattern_matches_chapter_title(self):
        """exclude_pattern matches against chapter_title."""
        from concept_mapper.cli import _location_passes_filters
        from concept_mapper.corpus.models import SentenceLocation

        loc = SentenceLocation(sent_index=0, chapter="10", chapter_title="Index")
        assert _location_passes_filters(loc, exclude_pattern="index") is False

    def test_exclude_pattern_matches_section_title(self):
        """exclude_pattern matches against section_title."""
        from concept_mapper.cli import _location_passes_filters
        from concept_mapper.corpus.models import SentenceLocation

        loc = SentenceLocation(sent_index=0, chapter="10", section_title="Bibliography")
        assert _location_passes_filters(loc, exclude_pattern="bibliography") is False

    def test_exclude_pattern_case_insensitive(self):
        """exclude_pattern matching is case-insensitive."""
        from concept_mapper.cli import _location_passes_filters
        from concept_mapper.corpus.models import SentenceLocation

        loc = SentenceLocation(sent_index=0, chapter="10", chapter_title="APPENDIX A")
        assert _location_passes_filters(loc, exclude_pattern="appendix") is False

    def test_exclude_pattern_no_match_passes(self):
        """A location whose titles don't match exclude_pattern passes."""
        from concept_mapper.cli import _location_passes_filters
        from concept_mapper.corpus.models import SentenceLocation

        loc = SentenceLocation(sent_index=0, chapter="2", chapter_title="Semantics")
        assert _location_passes_filters(loc, exclude_pattern="bibliography") is True

    def test_both_filters_applied(self):
        """Both start_section and exclude_pattern are applied together."""
        from concept_mapper.cli import _location_passes_filters
        from concept_mapper.corpus.models import SentenceLocation

        # Fails start_section
        loc_front = SentenceLocation(sent_index=0, chapter="0", chapter_title="TOC")
        assert (
            _location_passes_filters(
                loc_front, start_section="1", exclude_pattern="bibliography"
            )
            is False
        )
        # Passes start_section but fails exclude_pattern
        loc_back = SentenceLocation(
            sent_index=99, chapter="12", chapter_title="Bibliography"
        )
        assert (
            _location_passes_filters(
                loc_back, start_section="1", exclude_pattern="bibliography"
            )
            is False
        )
        # Passes both
        loc_main = SentenceLocation(
            sent_index=50, chapter="3", chapter_title="Semantics"
        )
        assert (
            _location_passes_filters(
                loc_main, start_section="1", exclude_pattern="bibliography"
            )
            is True
        )

    # ------------------------------------------------------------------ #
    # _filter_sentence_matches / _filter_relations helpers                #
    # ------------------------------------------------------------------ #

    def test_filter_sentence_matches_no_filters(self):
        """_filter_sentence_matches returns all matches when no filters set."""
        from concept_mapper.cli import _filter_sentence_matches
        from concept_mapper.corpus.models import SentenceLocation
        from concept_mapper.search.find import SentenceMatch

        loc = SentenceLocation(sent_index=0, chapter="0")
        match = SentenceMatch(
            sentence="hello world",
            doc_id="d1",
            sent_index=0,
            term_positions=[0],
            term="hello",
            location=loc,
        )
        assert _filter_sentence_matches([match]) == [match]

    def test_filter_sentence_matches_excludes(self):
        """_filter_sentence_matches removes matches below start_section."""
        from concept_mapper.cli import _filter_sentence_matches
        from concept_mapper.corpus.models import SentenceLocation
        from concept_mapper.search.find import SentenceMatch

        loc_low = SentenceLocation(sent_index=0, chapter="0")
        loc_high = SentenceLocation(sent_index=1, chapter="2")
        m_low = SentenceMatch(
            sentence="front",
            doc_id="d1",
            sent_index=0,
            term_positions=[0],
            term="x",
            location=loc_low,
        )
        m_high = SentenceMatch(
            sentence="body",
            doc_id="d1",
            sent_index=1,
            term_positions=[0],
            term="x",
            location=loc_high,
        )
        result = _filter_sentence_matches([m_low, m_high], start_section="1")
        assert result == [m_high]

    # ------------------------------------------------------------------ #
    # CLI integration: --start-from-section and --exclude-sections        #
    # ------------------------------------------------------------------ #

    def test_analyze_start_from_section_accepts_option(
        self, runner, sample_corpus_json
    ):
        """analyze --start-from-section should not error out."""
        result = runner.invoke(
            cli,
            [
                "analyze",
                str(sample_corpus_json),
                "dialectic",
                "--start-from-section",
                "1",
            ],
        )
        assert result.exit_code == 0

    def test_analyze_exclude_sections_accepts_option(self, runner, sample_corpus_json):
        """analyze --exclude-sections should not error out."""
        result = runner.invoke(
            cli,
            [
                "analyze",
                str(sample_corpus_json),
                "dialectic",
                "--exclude-sections",
                "bibliography",
            ],
        )
        assert result.exit_code == 0

    def test_analyze_exclude_sections_removes_matched_content(self, runner, tmp_path):
        """Terms in excluded sections should not appear in output."""
        import json as _json
        from concept_mapper.corpus.loader import load_file
        from concept_mapper.preprocessing.pipeline import preprocess
        from concept_mapper.corpus.models import SentenceLocation

        # Build corpus with two sentences in different chapters
        text_file = tmp_path / "two_chapters.txt"
        text_file.write_text(
            "Dialectical method is essential. The index covers all entries."
        )
        doc = load_file(text_file)
        processed = preprocess(doc)
        # Manually assign locations: first sentence chapter 1, second chapter "Index"
        processed.sentence_locations = [
            SentenceLocation(sent_index=0, chapter="1", chapter_title="Main"),
            SentenceLocation(sent_index=1, chapter="2", chapter_title="Index"),
        ]
        corpus_file = tmp_path / "corpus.json"
        with open(corpus_file, "w") as f:
            _json.dump([processed.to_dict()], f)

        result = runner.invoke(
            cli,
            [
                "search",
                str(corpus_file),
                "index",
                "--exclude-sections",
                "^Index$",
            ],
        )
        assert result.exit_code == 0
        assert "No matches found" in result.output

    def test_search_start_from_section_accepts_option(self, runner, sample_corpus_json):
        """search --start-from-section should not error out."""
        result = runner.invoke(
            cli,
            [
                "search",
                str(sample_corpus_json),
                "dialectic",
                "--start-from-section",
                "1",
            ],
        )
        assert result.exit_code == 0

    def test_search_exclude_sections_accepts_option(self, runner, sample_corpus_json):
        """search --exclude-sections should not error out."""
        result = runner.invoke(
            cli,
            [
                "search",
                str(sample_corpus_json),
                "dialectic",
                "--exclude-sections",
                "bibliography",
            ],
        )
        assert result.exit_code == 0

    def test_window_with_start_from_section(self, runner, sample_corpus_json):
        """analyze --window with --start-from-section should not error out."""
        result = runner.invoke(
            cli,
            [
                "analyze",
                str(sample_corpus_json),
                "dialectic",
                "-w",
                "s0",
                "--start-from-section",
                "1",
            ],
        )
        assert result.exit_code == 0


class TestReplaceCommand:
    """Tests for replace command."""

    def test_replace_single_word(self, runner, sample_corpus_json):
        """Test replacing single word with synonym."""
        result = runner.invoke(
            cli, ["replace", str(sample_corpus_json), "dialectical", "dynamic"]
        )
        assert result.exit_code == 0
        assert "dynamic" in result.output
        assert "dialectical" not in result.output

    def test_replace_with_preview(self, runner, sample_corpus_json):
        """Test replace with preview flag."""
        result = runner.invoke(
            cli,
            ["replace", str(sample_corpus_json), "dialectical", "dynamic", "--preview"],
        )
        assert result.exit_code == 0
        assert "Preview of changes:" in result.output
        assert "Total length:" in result.output

    def test_replace_with_output_file(self, runner, sample_corpus_json, tmp_path):
        """Test replace with output file."""
        output_file = tmp_path / "replaced.txt"
        result = runner.invoke(
            cli,
            [
                "replace",
                str(sample_corpus_json),
                "dialectical",
                "dynamic",
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()
        content = output_file.read_text()
        assert "dynamic" in content
        assert "dialectical" not in content

    def test_replace_preserves_inflection(self, runner, tmp_path):
        """Test that replacement preserves grammatical inflections."""
        # Create corpus with various inflections
        from concept_mapper.corpus.loader import load_file
        from concept_mapper.preprocessing.pipeline import preprocess

        text_file = tmp_path / "inflection_test.txt"
        text_file.write_text("The cat runs. The cats ran quickly.")

        doc = load_file(text_file)
        processed = preprocess(doc)

        corpus_file = tmp_path / "corpus.json"
        with open(corpus_file, "w") as f:
            json.dump([processed.to_dict()], f)

        result = runner.invoke(cli, ["replace", str(corpus_file), "run", "sprint"])
        assert result.exit_code == 0
        # Should preserve tense: "runs" → "sprints", "ran" → "sprinted"
        assert "sprints" in result.output
        assert "sprinted" in result.output

    def test_replace_phrase_to_single(self, runner, tmp_path):
        """Test replacing multi-word phrase with single word."""
        from concept_mapper.corpus.loader import load_file
        from concept_mapper.preprocessing.pipeline import preprocess

        text_file = tmp_path / "phrase_test.txt"
        text_file.write_text("The body without organs is a concept.")

        doc = load_file(text_file)
        processed = preprocess(doc)

        corpus_file = tmp_path / "corpus.json"
        with open(corpus_file, "w") as f:
            json.dump([processed.to_dict()], f)

        result = runner.invoke(
            cli, ["replace", str(corpus_file), "body,without,organ", "medium"]
        )
        assert result.exit_code == 0
        assert "medium" in result.output
        assert "body without" not in result.output

    def test_replace_phrase_to_phrase(self, runner, tmp_path):
        """Test replacing phrase with another phrase."""
        from concept_mapper.corpus.loader import load_file
        from concept_mapper.preprocessing.pipeline import preprocess

        text_file = tmp_path / "phrase_test.txt"
        # Use clearer context so POS tagger identifies "organs" as noun
        text_file.write_text("The body without organs is important.")

        doc = load_file(text_file)
        processed = preprocess(doc)

        corpus_file = tmp_path / "corpus.json"
        with open(corpus_file, "w") as f:
            json.dump([processed.to_dict()], f)

        result = runner.invoke(
            cli,
            [
                "replace",
                str(corpus_file),
                "body,without,organ",
                "blank,resistant,field",
            ],
        )
        assert result.exit_code == 0
        assert "blank resistant" in result.output
        assert "field" in result.output

    def test_replace_no_matches(self, runner, sample_corpus_json):
        """Test replace when term not found in corpus."""
        result = runner.invoke(
            cli, ["replace", str(sample_corpus_json), "nonexistent", "replacement"]
        )
        assert result.exit_code == 0
        # Should return original text unchanged
        assert "dialectical" in result.output or "Geist" in result.output

    def test_replace_empty_corpus(self, runner, tmp_path):
        """Test replace on empty corpus."""
        corpus_file = tmp_path / "empty.json"
        corpus_file.write_text("[]")

        result = runner.invoke(
            cli, ["replace", str(corpus_file), "word", "replacement"]
        )
        assert result.exit_code == 1
        assert "Empty corpus" in result.output
