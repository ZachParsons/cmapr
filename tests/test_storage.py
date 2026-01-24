"""
Tests for storage backend and utilities.
"""

import json
from pathlib import Path

import pytest

from src.concept_mapper.storage import (
    JSONBackend,
    ensure_output_structure,
    get_cache_path,
    get_output_path,
    validate_file_path,
)


class TestJSONBackend:
    """Test JSON storage backend."""

    @pytest.fixture
    def backend(self):
        """Create JSON backend instance."""
        return JSONBackend()

    @pytest.fixture
    def temp_dir(self, tmp_path):
        """Create temporary directory for tests."""
        return tmp_path / "test_storage"

    def test_save_and_load_analysis(self, backend, temp_dir):
        """Test saving and loading analysis results."""
        data = {
            "frequencies": {"term1": 10, "term2": 5},
            "total_words": 100,
        }

        output_path = temp_dir / "analysis.json"
        backend.save_analysis(data, output_path)

        assert output_path.exists()

        loaded = backend.load_analysis(output_path)
        assert loaded == data

    def test_save_creates_parent_directories(self, backend, temp_dir):
        """Test that save operations create parent directories."""
        nested_path = temp_dir / "nested" / "deep" / "data.json"

        data = {"test": "value"}
        backend.save_analysis(data, nested_path)

        assert nested_path.exists()
        assert nested_path.parent.exists()

    def test_save_corpus(self, backend, temp_dir):
        """Test saving corpus data."""
        corpus_data = {
            "documents": [
                {"text": "Sample text", "metadata": {"title": "Doc 1"}},
                {"text": "Another text", "metadata": {"title": "Doc 2"}},
            ]
        }

        output_path = temp_dir / "corpus.json"
        backend.save_corpus(corpus_data, output_path)

        loaded = backend.load_corpus(output_path)
        assert loaded == corpus_data

    def test_save_term_list(self, backend, temp_dir):
        """Test saving term list."""
        term_list = {
            "terms": [
                {"term": "daseinology", "frequency": 6},
                {"term": "temporalization", "frequency": 5},
            ]
        }

        output_path = temp_dir / "terms.json"
        backend.save_term_list(term_list, output_path)

        loaded = backend.load_term_list(output_path)
        assert loaded == term_list

    def test_save_graph(self, backend, temp_dir):
        """Test saving graph data."""
        graph_data = {
            "nodes": [{"id": "term1"}, {"id": "term2"}],
            "links": [{"source": "term1", "target": "term2", "weight": 0.5}],
        }

        output_path = temp_dir / "graph.json"
        backend.save_graph(graph_data, output_path)

        loaded = backend.load_graph(output_path)
        assert loaded == graph_data

    def test_pretty_printing(self, temp_dir):
        """Test that JSON is formatted with indentation."""
        backend = JSONBackend(indent=2)
        data = {"key": "value", "nested": {"inner": "data"}}

        output_path = temp_dir / "formatted.json"
        backend.save_analysis(data, output_path)

        # Check that file contains newlines (pretty-printed)
        content = output_path.read_text()
        assert "\n" in content
        assert "  " in content  # Check for indentation


class TestFilesystemUtils:
    """Test filesystem utility functions."""

    def test_ensure_output_structure(self, tmp_path):
        """Test creating standard output directory structure."""
        base_dir = tmp_path / "output"
        dirs = ensure_output_structure(base_dir)

        assert dirs["base"].exists()
        assert dirs["corpus"].exists()
        assert dirs["analysis"].exists()
        assert dirs["graphs"].exists()
        assert dirs["d3"].exists()
        assert dirs["cache"].exists()

        # Verify structure
        assert dirs["d3"] == base_dir / "graphs" / "d3"

    def test_validate_file_path_existing(self, tmp_path):
        """Test validating an existing file path."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")

        validated = validate_file_path(test_file, must_exist=True)
        assert validated.exists()

    def test_validate_file_path_nonexistent_required(self, tmp_path):
        """Test that validation fails for missing required file."""
        test_file = tmp_path / "missing.txt"

        with pytest.raises(FileNotFoundError):
            validate_file_path(test_file, must_exist=True)

    def test_validate_file_path_directory_fails(self, tmp_path):
        """Test that validation fails when directory provided instead of file."""
        test_dir = tmp_path / "testdir"
        test_dir.mkdir()

        with pytest.raises(ValueError, match="Expected file, got directory"):
            validate_file_path(test_dir)

    def test_get_output_path(self, tmp_path):
        """Test getting output path for a file."""
        path = get_output_path(
            "test.json", subdir="analysis", base_dir=tmp_path, ensure_dir=True
        )

        assert path == tmp_path / "analysis" / "test.json"
        assert path.parent.exists()

    def test_get_cache_path(self, tmp_path):
        """Test getting cache file path."""
        path = get_cache_path("brown_freqs.json", base_dir=tmp_path, ensure_dir=True)

        assert path == tmp_path / "cache" / "brown_freqs.json"
        assert path.parent.exists()
