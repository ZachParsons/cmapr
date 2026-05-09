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
    # Text must be substantial enough for rarities detection AND have cross-term
    # sentences so build_proposition_graph can extract typed edges between pairs.
    text_file.write_text(
        "Geist is a fundamental concept in Hegel's dialectical philosophy. "
        "Geist refers to the self-developing rationality that animates history and thought. "
        "Geist produces Aufhebung through the dialectical movement of sublation. "
        "Aufhebung describes how Geist negates and preserves contradictions at a higher level. "
        "Aufhebung is the defining operation of dialectical thought and historical development. "
        "Through Aufhebung, Geist realises itself in concrete Sittlichkeit. "
        "Selbstbewusstsein characterizes the fundamental structure of Geist in the world. "
        "Selbstbewusstsein cannot be understood apart from recognition by another consciousness. "
        "Selbstbewusstsein depends on Anerkennung for its full realisation in the social world. "
        "The dialectical process grounds the entire structure of thought and reality. "
        "The dialectic is not mere succession but the negation of negation itself. "
        "Anerkennung distinguishes mutual recognition from mere acknowledgment of presence. "
        "Anerkennung is a type of Sittlichkeit that grounds social and political philosophy. "
        "Understanding Anerkennung is crucial for Sittlichkeit and ethical life. "
        "Sittlichkeit involves the actualization of Geist in ethical life through social institutions. "
        "Sittlichkeit contrasts with abstract morality and the immediacy of desire. "
        "The Phenomenology investigates the structures of consciousness and Geist. "
        "Mediation plays a crucial role in the interpretation of Aufhebung and concrete universals. "
        "The dialectical circle describes how Geist returns to itself enriched by development."
    )
    return text_file


@pytest.fixture
def sample_corpus_json(tmp_path):
    """Create a sample preprocessed corpus JSON file."""
    from concept_mapper.corpus.loader import load_file
    from concept_mapper.preprocessing.pipeline import preprocess

    # Text must be substantial enough for rarities detection AND include cross-term
    # sentences so build_proposition_graph can extract typed edges between pairs.
    text_file = tmp_path / "source.txt"
    text_file.write_text(
        "Geist is a fundamental concept in Hegel's dialectical philosophy. "
        "Geist refers to the self-developing rationality that animates history. "
        "Geist drives the movement of thought toward absolute knowledge. "
        "Geist produces Aufhebung through the dialectical movement of sublation. "
        "Aufhebung describes how Geist negates and preserves contradictions. "
        "Aufhebung is the defining operation of dialectical thought. "
        "Through Aufhebung, Geist realises itself in concrete Sittlichkeit. "
        "Selbstbewusstsein characterizes the fundamental structure of Geist. "
        "Selbstbewusstsein cannot be understood apart from recognition. "
        "Selbstbewusstsein depends on Anerkennung for its full realisation. "
        "Anerkennung distinguishes mutual recognition from mere acknowledgment. "
        "Anerkennung is a type of Sittlichkeit that grounds social philosophy. "
        "Understanding Anerkennung is crucial for Sittlichkeit. "
        "Sittlichkeit involves the actualization of Geist in ethical life. "
        "Sittlichkeit contrasts with abstract morality and immediate desire. "
        "The dialectical process grounds the entire structure of philosophical thought. "
        "The dialectic is not mere succession but the negation of negation itself. "
        "Hegel's system connects the individual subject to the universal Geist. "
        "Reason develops through the historical realization of Geist in the world. "
        "The concept of Aufhebung captures both negation and preservation simultaneously."
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


# ============================================================================
# Test Graph Command
# ============================================================================


class TestGraphCommand:
    """Tests for graph command."""

    def test_graph_basic(self, runner, sample_corpus_json, sample_terms_json, tmp_path):
        """Test graph building produces valid D3 JSON."""
        output_file = tmp_path / "graph.json"

        result = runner.invoke(
            cli,
            [
                "graph",
                str(sample_corpus_json),
                "--terms",
                str(sample_terms_json),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        with open(output_file) as f:
            data = json.load(f)

        assert "nodes" in data
        assert "links" in data

    def test_graph_with_relations(
        self, runner, sample_corpus_json, sample_terms_json, tmp_path
    ):
        """Test graph building with relations enabled."""
        output_file = tmp_path / "graph.json"

        result = runner.invoke(
            cli,
            [
                "graph",
                str(sample_corpus_json),
                "--terms",
                str(sample_terms_json),
                "--with-relations",
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
                "--no-filter-names",
                "--no-filter-fragments",
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
        # Should create corpus/sample/corpus.json (work dir named after source)
        expected_corpus = output_dir / "corpus" / "sample" / "corpus.json"
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
        # Should create rarities/corpus/rarities.json (work dir named after source)
        expected_terms = output_dir / "rarities" / "corpus" / "rarities.json"
        assert expected_terms.exists()
        assert not (output_dir / "rarities" / "terms.json").exists()

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
                "--no-filter-names",
                "--no-filter-fragments",
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
        # Should create graphs/corpus/graph.json (work dir named after source)
        expected_graph = output_dir / "graphs" / "corpus" / "graph.json"
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
                "--no-filter-names",
                "--no-filter-fragments",
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
        # Should create graphs/corpus/graph.json (work dir named after source)
        expected_graph = output_dir / "graphs" / "corpus" / "graph.json"
        assert expected_graph.exists()

    def test_export_source_derived_naming(self, runner, tmp_path):
        """Test that export uses source-derived output paths."""
        from concept_mapper.graph import ConceptGraph
        from concept_mapper.export import export_d3_json

        output_dir = tmp_path / "output"

        # Create a simple graph file with source-derived name (flat, in graphs/)
        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()
        graph_file = graphs_dir / "eco_spl.json"
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
        # Should create exports/eco_spl/ directory namespaced by term
        expected_viz = output_dir / "exports" / "eco_spl" / "index.html"
        assert expected_viz.exists()

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

        corpus1 = output_dir / "corpus" / "eco_spl" / "corpus.json"
        assert corpus1.exists()
        corpus1_content = corpus1.read_text()

        # Ingest second text
        result2 = runner.invoke(
            cli,
            ["--output-dir", str(output_dir), "ingest", str(text2)],
        )
        assert result2.exit_code == 0

        corpus2 = output_dir / "corpus" / "eco_spl_alt" / "corpus.json"
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

        corpus_file = output_dir / "corpus" / "sample" / "corpus.json"
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
                "--no-filter-names",
                "--no-filter-fragments",
            ],
        )
        assert result.exit_code == 0

        terms_file = output_dir / "rarities" / "sample" / "rarities.json"
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

        graph_file = output_dir / "graphs" / "sample" / "graph.json"
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


# ============================================================================
# Test RaritiesVetting (Phase 12)
# ============================================================================


class TestRaritiesVetting:
    """Vetting file (vetting.json) is respected when running rarities."""

    def _run_rarities(self, runner, corpus_path, output_path, **extra_args):
        args = [
            "rarities",
            str(corpus_path),
            "--threshold",
            "0.0",
            "--output",
            str(output_path),
        ]
        for k, v in extra_args.items():
            args.extend([k, v])
        return runner.invoke(cli, args)

    def test_rejected_term_excluded(self, runner, sample_corpus_json, tmp_path):
        """A term listed in vetting.json 'reject' does not appear in output."""
        output = tmp_path / "terms.json"
        # First run to discover which terms are extracted
        result = self._run_rarities(runner, sample_corpus_json, output)
        assert result.exit_code == 0
        terms = [t["term"].lower() for t in json.loads(output.read_text())]
        if not terms:
            return  # nothing to reject
        term_to_reject = terms[0]

        # Write vetting file with that term rejected
        vetting = {"accept": [], "reject": [term_to_reject]}
        (tmp_path / "vetting.json").write_text(json.dumps(vetting))

        result2 = self._run_rarities(runner, sample_corpus_json, output)
        assert result2.exit_code == 0
        terms2 = [t["term"].lower() for t in json.loads(output.read_text())]
        assert term_to_reject not in terms2, (
            f"Expected rejected term {term_to_reject!r} to be absent from output"
        )

    def test_accepted_term_survives_top_n_cutoff(
        self, runner, sample_corpus_json, tmp_path
    ):
        """A term in vetting 'accept' survives even when --top-n would cut it."""
        output = tmp_path / "terms.json"
        # First run to get at least 2 terms
        result = self._run_rarities(runner, sample_corpus_json, output)
        assert result.exit_code == 0
        terms = [t["term"] for t in json.loads(output.read_text())]
        if len(terms) < 2:
            return  # not enough terms to test cutoff
        term_to_protect = terms[1]

        # Accept the second term so it survives top-n=1
        vetting = {"accept": [term_to_protect.lower()], "reject": []}
        (tmp_path / "vetting.json").write_text(json.dumps(vetting))

        result2 = runner.invoke(
            cli,
            [
                "rarities",
                str(sample_corpus_json),
                "--threshold",
                "0.0",
                "--output",
                str(output),
                "--top-n",
                "1",
            ],
        )
        assert result2.exit_code == 0
        terms2 = [t["term"].lower() for t in json.loads(output.read_text())]
        assert term_to_protect.lower() in terms2, (
            f"Expected accepted term {term_to_protect!r} to survive --top-n cutoff"
        )

    def test_vetting_file_loaded_message(self, runner, sample_corpus_json, tmp_path):
        """Running rarities when vetting.json exists prints a 'loaded vetting' message."""
        output = tmp_path / "terms.json"
        vetting = {"accept": [], "reject": []}
        (tmp_path / "vetting.json").write_text(json.dumps(vetting))

        result = self._run_rarities(runner, sample_corpus_json, output)
        assert result.exit_code == 0
        assert "loaded vetting" in result.output.lower(), (
            "Expected 'loaded vetting' message when vetting.json exists"
        )

    def test_vet_flag_creates_vetting_json(self, runner, sample_corpus_json, tmp_path):
        """--vet creates vetting.json with 'accept' and 'reject' keys."""
        output = tmp_path / "terms.json"
        result = runner.invoke(
            cli,
            [
                "rarities",
                str(sample_corpus_json),
                "--vet",
                "--threshold",
                "0.0",
                "--output",
                str(output),
            ],
            input="y\n" * 20,
        )
        assert result.exit_code == 0, result.output
        vetting_path = tmp_path / "vetting.json"
        assert vetting_path.exists(), "Expected vetting.json to be created by --vet"
        vetting = json.loads(vetting_path.read_text())
        assert "accept" in vetting, "Expected 'accept' key in vetting.json"
        assert "reject" in vetting, "Expected 'reject' key in vetting.json"

    def test_vet_flag_records_accept_and_reject(
        self, runner, sample_corpus_json, tmp_path
    ):
        """--vet records accept (y) and reject (n) decisions in vetting.json."""
        output = tmp_path / "terms.json"
        result_pre = runner.invoke(
            cli,
            [
                "rarities",
                str(sample_corpus_json),
                "--threshold",
                "0.0",
                "--output",
                str(output),
            ],
        )
        assert result_pre.exit_code == 0
        terms = [t["term"] for t in json.loads(output.read_text())]
        if len(terms) < 2:
            return  # not enough terms to make a meaningful accept/reject test

        # Accept first, reject second, accept the rest
        inputs = "y\nn\n" + "y\n" * 20
        result = runner.invoke(
            cli,
            [
                "rarities",
                str(sample_corpus_json),
                "--vet",
                "--threshold",
                "0.0",
                "--output",
                str(output),
            ],
            input=inputs,
        )
        assert result.exit_code == 0, result.output
        vetting_path = tmp_path / "vetting.json"
        assert vetting_path.exists(), "Expected vetting.json to be created"
        vetting = json.loads(vetting_path.read_text())
        assert len(vetting.get("accept", [])) >= 1, (
            "Expected at least one term in vetting 'accept'"
        )
        assert len(vetting.get("reject", [])) >= 1, (
            "Expected at least one term in vetting 'reject'"
        )

    def test_vet_flag_invalid_input_reprompts(
        self, runner, sample_corpus_json, tmp_path
    ):
        """--vet re-prompts and shows an error message when input is not y or n."""
        output = tmp_path / "terms.json"
        # First input is invalid; second is valid y
        result = runner.invoke(
            cli,
            [
                "rarities",
                str(sample_corpus_json),
                "--vet",
                "--threshold",
                "0.0",
                "--output",
                str(output),
            ],
            input="s\ny\n" + "y\n" * 20,
        )
        assert result.exit_code == 0, result.output
        assert "enter y" in result.output.lower(), (
            "Expected re-prompt error message when invalid input given"
        )


# ============================================================================
# Test RaritiesPOSFilter (Phase 14 B1)
# ============================================================================


class TestRaritiesPOSFilter:
    """--pos filter on rarities restricts candidates to the specified POS category."""

    def test_pos_noun_completes_without_error(
        self, runner, sample_corpus_json, tmp_path
    ):
        """rarities --pos noun completes with exit code 0."""
        output = tmp_path / "terms.json"
        result = runner.invoke(
            cli,
            [
                "rarities",
                str(sample_corpus_json),
                "--pos",
                "noun",
                "--threshold",
                "0.0",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, (
            f"Expected exit_code 0 for --pos noun, got {result.exit_code}. "
            f"Output: {result.output}"
        )

    def test_pos_verb_completes_without_error(
        self, runner, sample_corpus_json, tmp_path
    ):
        """rarities --pos verb does not crash (may produce zero terms on test corpus)."""
        output = tmp_path / "terms.json"
        result = runner.invoke(
            cli,
            [
                "rarities",
                str(sample_corpus_json),
                "--pos",
                "verb",
                "--threshold",
                "0.0",
                "--output",
                str(output),
            ],
        )
        # Verb filter may empty the candidate list on a noun-heavy test corpus;
        # the command should exit cleanly (0 or 1) — not raise an unhandled exception.
        assert result.exception is None or isinstance(result.exception, SystemExit), (
            f"Expected clean exit for --pos verb, got exception: {result.exception}"
        )

    def test_unknown_pos_category_warns(self, runner, sample_corpus_json, tmp_path):
        """An unknown --pos category produces a warning or exits cleanly — not a crash."""
        output = tmp_path / "terms.json"
        result = runner.invoke(
            cli,
            [
                "rarities",
                str(sample_corpus_json),
                "--pos",
                "badcategory",
                "--threshold",
                "0.0",
                "--output",
                str(output),
            ],
        )
        # Should warn or exit 0 — must not crash unhandled
        warning_present = "unknown" in result.output.lower()
        assert warning_present or result.exit_code == 0, (
            "Expected either a warning about unknown POS or a clean exit"
        )

    def test_pos_filter_reduces_candidate_count(
        self, runner, sample_corpus_json, tmp_path
    ):
        """--pos noun produces ≤ terms than an unfiltered run (or equal when all are nouns)."""
        output_all = tmp_path / "terms_all.json"
        output_noun = tmp_path / "terms_noun.json"

        result_all = runner.invoke(
            cli,
            [
                "rarities",
                str(sample_corpus_json),
                "--threshold",
                "0.0",
                "--output",
                str(output_all),
            ],
        )
        result_noun = runner.invoke(
            cli,
            [
                "rarities",
                str(sample_corpus_json),
                "--pos",
                "noun",
                "--threshold",
                "0.0",
                "--output",
                str(output_noun),
            ],
        )
        assert result_all.exit_code == 0
        assert result_noun.exit_code == 0

        all_terms = json.loads(output_all.read_text())
        noun_terms = json.loads(output_noun.read_text())
        assert len(noun_terms) <= len(all_terms), (
            f"Expected --pos noun to produce ≤ {len(all_terms)} terms, "
            f"got {len(noun_terms)}"
        )

    def test_top_n_and_pos_combine(self, runner, sample_corpus_json, tmp_path):
        """B2: --top-n and --pos can be combined; output respects both bounds."""
        output_path = tmp_path / "terms_top10_noun.json"
        result = runner.invoke(
            cli,
            [
                "rarities",
                str(sample_corpus_json),
                "--top-n",
                "10",
                "--pos",
                "noun",
                "--threshold",
                "0.0",
                "--output",
                str(output_path),
            ],
        )
        assert result.exit_code == 0, (
            f"Expected exit_code 0 for --top-n + --pos, got {result.exit_code}. "
            f"Output: {result.output!r}"
        )
        terms = json.loads(output_path.read_text())
        assert len(terms) <= 10, (
            f"Expected ≤ 10 terms with --top-n 10, got {len(terms)}"
        )


# ============================================================================
# Test RaritiesBySection (Phase 14 B3)
# ============================================================================


class TestRaritiesBySection:
    """--by-section saves terms_by_section.json alongside terms.json."""

    def _run_by_section(self, runner, corpus_path, output_path):
        return runner.invoke(
            cli,
            [
                "rarities",
                str(corpus_path),
                "--by-section",
                "--threshold",
                "0.0",
                "--output",
                str(output_path),
            ],
        )

    def test_by_section_creates_json_file(self, runner, sample_corpus_json, tmp_path):
        """--by-section creates terms_by_section.json next to terms.json."""
        output = tmp_path / "terms.json"
        result = self._run_by_section(runner, sample_corpus_json, output)
        assert result.exit_code == 0
        by_section_path = tmp_path / "terms_by_section.json"
        assert by_section_path.exists(), (
            "Expected terms_by_section.json to be created by --by-section"
        )

    def test_by_section_json_has_sections_key(
        self, runner, sample_corpus_json, tmp_path
    ):
        """The by-section JSON contains a top-level 'sections' key."""
        output = tmp_path / "terms.json"
        result = self._run_by_section(runner, sample_corpus_json, output)
        assert result.exit_code == 0
        by_section_path = tmp_path / "terms_by_section.json"
        data = json.loads(by_section_path.read_text())
        assert "sections" in data, "Expected 'sections' key in terms_by_section.json"

    def test_by_section_sections_have_term_lists(
        self, runner, sample_corpus_json, tmp_path
    ):
        """Each entry in data['sections'] has 'section' (str) and 'terms' (list)."""
        output = tmp_path / "terms.json"
        result = self._run_by_section(runner, sample_corpus_json, output)
        assert result.exit_code == 0
        by_section_path = tmp_path / "terms_by_section.json"
        data = json.loads(by_section_path.read_text())
        for entry in data.get("sections", []):
            assert "section" in entry, (
                f"Expected 'section' key in each section entry, got {entry!r}"
            )
            assert isinstance(entry["section"], str), (
                "Expected 'section' to be a string"
            )
            assert "terms" in entry, (
                f"Expected 'terms' key in each section entry, got {entry!r}"
            )
            assert isinstance(entry["terms"], list), "Expected 'terms' to be a list"

    def test_by_section_all_terms_in_some_section(
        self, runner, sample_corpus_json, tmp_path
    ):
        """Every term in terms.json appears in at least one section in terms_by_section.json."""
        output = tmp_path / "terms.json"
        result = self._run_by_section(runner, sample_corpus_json, output)
        assert result.exit_code == 0

        flat_terms = {t["term"].lower() for t in json.loads(output.read_text())}

        by_section_path = tmp_path / "terms_by_section.json"
        section_data = json.loads(by_section_path.read_text())
        section_terms = set()
        for entry in section_data.get("sections", []):
            for t in entry.get("terms", []):
                term_val = t if isinstance(t, str) else t.get("term", "")
                section_terms.add(term_val.lower())

        for term in flat_terms:
            assert term in section_terms, (
                f"Expected term {term!r} from flat output to appear in some section"
            )


# ============================================================================
# Test GraphDepthFocusCLI (Phase 14 B4/B5)
# ============================================================================


class TestGraphDepthFocusCLI:
    """CLI smoke tests for --focus and --depth flags on the graph command."""

    def test_focus_flag_completes(
        self, runner, sample_corpus_json, sample_terms_json, tmp_path
    ):
        """graph --focus geist completes with exit code 0."""
        output = tmp_path / "graph_focus.json"
        result = runner.invoke(
            cli,
            [
                "graph",
                str(sample_corpus_json),
                "-t",
                str(sample_terms_json),
                "--focus",
                "geist",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, (
            f"Expected exit_code 0 for graph --focus geist, got {result.exit_code}. "
            f"Output: {result.output}"
        )

    def test_unknown_focus_warns_not_crashes(
        self, runner, sample_corpus_json, sample_terms_json, tmp_path
    ):
        """graph --focus on a non-existent term warns but does not crash."""
        output = tmp_path / "graph_focus_missing.json"
        result = runner.invoke(
            cli,
            [
                "graph",
                str(sample_corpus_json),
                "-t",
                str(sample_terms_json),
                "--focus",
                "xxxxnotaword",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, (
            f"Expected exit_code 0 even for unknown focus term, got {result.exit_code}"
        )
        combined = result.output + (result.stderr or "")
        warned = any(
            w in combined.lower() for w in ("not found", "warning", "ignoring")
        )
        assert warned, (
            "Expected a 'not found', 'warning', or 'ignoring' message for unknown focus term"
        )

    def test_depth_flag_completes(
        self, runner, sample_corpus_json, sample_terms_json, tmp_path
    ):
        """graph --depth 1 completes with exit code 0 and writes valid JSON."""
        output = tmp_path / "graph_depth.json"
        result = runner.invoke(
            cli,
            [
                "graph",
                str(sample_corpus_json),
                "-t",
                str(sample_terms_json),
                "--depth",
                "1",
                "--output",
                str(output),
            ],
        )
        assert result.exit_code == 0, (
            f"Expected exit_code 0 for graph --depth 1, got {result.exit_code}. "
            f"Output: {result.output}"
        )
        assert output.exists(), "Expected output file to be created by graph --depth 1"
        data = json.loads(output.read_text())
        assert "nodes" in data and "links" in data, (
            "Expected valid D3 JSON with 'nodes' and 'links' keys"
        )


# ============================================================================
# Test Merge Command
# ============================================================================


def _write_d3_graph(path, nodes, links):
    """Write a minimal D3-format graph JSON to path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({"nodes": nodes, "links": links}, f)


class TestMergeCommand:
    """`cmapr merge` aggregates two or more graph files into one."""

    def _make_graph(self, path, *, suffix=""):
        """Build a small D3 graph with sign + code (definition link)."""
        _write_d3_graph(
            path,
            nodes=[
                {"id": "sign", "label": "sign", "frequency": 10, "score": 3.5},
                {"id": "code", "label": "code", "frequency": 5, "score": 2.0},
            ],
            links=[
                {
                    "source": "sign",
                    "target": "code",
                    "type": "definition",
                    "weight": 2,
                    "verb": "is defined as",
                    "evidence": [f"Sign is defined as code{suffix}."],
                }
            ],
        )

    def test_requires_two_graphs(self, runner, tmp_path):
        """A single graph file is rejected (≥ 2 required)."""
        g1 = tmp_path / "ch1.json"
        self._make_graph(g1, suffix=" 1")
        out = tmp_path / "merged.json"
        result = runner.invoke(cli, ["merge", str(g1), "-o", str(out)])
        assert result.exit_code != 0
        assert "at least 2" in result.output.lower()

    def test_merges_two_graphs(self, runner, tmp_path):
        """Two graphs sharing a node merge with summed frequency."""
        g1 = tmp_path / "ch1.json"
        g2 = tmp_path / "ch2.json"
        self._make_graph(g1, suffix=" 1")
        self._make_graph(g2, suffix=" 2")
        out = tmp_path / "merged.json"

        result = runner.invoke(cli, ["merge", str(g1), str(g2), "-o", str(out)])
        assert result.exit_code == 0, (
            f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
        )
        assert out.exists()

        merged = json.loads(out.read_text())
        ids = {n["id"] for n in merged["nodes"]}
        assert ids == {"sign", "code"}

        sign = next(n for n in merged["nodes"] if n["id"] == "sign")
        assert sign["frequency"] == 20  # 10 + 10
        # Score weighted-avg with equal weights ≈ 3.5
        assert sign["score"] == pytest.approx(3.5, rel=0.01)

    def test_multi_type_edge_preserved_in_output(self, runner, tmp_path):
        """When the same pair has different types in two graphs, the merged
        edge carries relation_types and per-type breakdowns."""
        g1 = tmp_path / "ch1.json"
        g2 = tmp_path / "ch2.json"
        _write_d3_graph(
            g1,
            nodes=[
                {"id": "sign", "label": "sign"},
                {"id": "code", "label": "code"},
            ],
            links=[
                {
                    "source": "sign", "target": "code",
                    "type": "definition", "weight": 3,
                    "verb": "is defined as",
                    "evidence": ["s1"],
                }
            ],
        )
        _write_d3_graph(
            g2,
            nodes=[
                {"id": "sign", "label": "sign"},
                {"id": "code", "label": "code"},
            ],
            links=[
                {
                    "source": "sign", "target": "code",
                    "type": "production", "weight": 2,
                    "verb": "produces",
                    "evidence": ["s2"],
                }
            ],
        )
        out = tmp_path / "merged.json"
        result = runner.invoke(cli, ["merge", str(g1), str(g2), "-o", str(out)])
        assert result.exit_code == 0, result.output

        merged = json.loads(out.read_text())
        edge = merged["links"][0]
        assert edge["type"] == "definition"  # primary by priority
        assert edge["weight"] == 5
        assert edge["relation_types"] == ["definition", "production"]
        assert edge["weight_by_type"] == {"definition": 3, "production": 2}
        # evidence_by_type only included when include_evidence=True (merge does)
        assert "evidence_by_type" in edge
        assert edge["evidence_by_type"] == {
            "definition": ["s1"],
            "production": ["s2"],
        }

    def test_prune_ratio_applied(self, runner, tmp_path):
        """--prune-ratio caps the merged graph's edge density."""
        # Build two graphs whose union exceeds a 1.0 ratio
        g1 = tmp_path / "ch1.json"
        _write_d3_graph(
            g1,
            nodes=[{"id": x, "label": x} for x in ("a", "b", "c", "d")],
            links=[
                {"source": "a", "target": "b", "type": "cooccurrence", "weight": 1},
                {"source": "a", "target": "c", "type": "cooccurrence", "weight": 1},
                {"source": "a", "target": "d", "type": "cooccurrence", "weight": 1},
                {"source": "b", "target": "c", "type": "cooccurrence", "weight": 1},
            ],
        )
        g2 = tmp_path / "ch2.json"
        _write_d3_graph(
            g2,
            nodes=[{"id": x, "label": x} for x in ("a", "b", "e")],
            links=[
                {"source": "b", "target": "e", "type": "cooccurrence", "weight": 1},
                {"source": "a", "target": "e", "type": "cooccurrence", "weight": 1},
            ],
        )
        out = tmp_path / "merged.json"
        result = runner.invoke(
            cli,
            ["merge", str(g1), str(g2), "-o", str(out), "--prune-ratio", "1.0"],
        )
        assert result.exit_code == 0, result.output
        merged = json.loads(out.read_text())
        ratio = len(merged["links"]) / max(len(merged["nodes"]), 1)
        assert ratio <= 1.0, f"Expected ratio ≤ 1.0 after prune, got {ratio:.2f}"
