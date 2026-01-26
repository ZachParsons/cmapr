"""
JSON-based storage backend.

Simple, human-readable storage using JSON serialization.
Default backend for development and small-to-medium corpora.
"""

import json
from pathlib import Path
from typing import Any, Dict

from .backend import StorageBackend
from concept_mapper.validation import (
    validate_corpus,
    validate_term_list,
    validate_graph,
)


class JSONBackend(StorageBackend):
    """
    JSON-based storage backend.

    Stores all data as JSON files with optional pretty-printing
    for human readability.
    """

    def __init__(self, indent: int = 2, ensure_ascii: bool = False):
        """
        Initialize JSON backend.

        Args:
            indent: Number of spaces for indentation (None for compact)
            ensure_ascii: If False, allow Unicode characters in output
        """
        self.indent = indent
        self.ensure_ascii = ensure_ascii

    def save_corpus(self, corpus_data: Any, path: Path) -> None:
        """Save corpus data as JSON."""
        # Validate corpus is not empty
        validate_corpus(corpus_data)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(
                corpus_data, f, indent=self.indent, ensure_ascii=self.ensure_ascii
            )

    def load_corpus(self, path: Path) -> Any:
        """Load corpus data from JSON."""
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_term_list(self, term_list: Any, path: Path) -> None:
        """Save term list as JSON."""
        # Validate term list is not empty
        validate_term_list(term_list)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(term_list, f, indent=self.indent, ensure_ascii=self.ensure_ascii)

    def load_term_list(self, path: Path) -> Any:
        """Load term list from JSON."""
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_graph(self, graph: Any, path: Path) -> None:
        """Save graph as JSON."""
        # Validate graph is not empty
        validate_graph(graph, require_edges=False)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(graph, f, indent=self.indent, ensure_ascii=self.ensure_ascii)

    def load_graph(self, path: Path) -> Any:
        """Load graph from JSON."""
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)

    def save_analysis(self, data: Dict[str, Any], path: Path) -> None:
        """Save analysis results as JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=self.indent, ensure_ascii=self.ensure_ascii)

    def load_analysis(self, path: Path) -> Dict[str, Any]:
        """Load analysis results from JSON."""
        with Path(path).open("r", encoding="utf-8") as f:
            return json.load(f)
