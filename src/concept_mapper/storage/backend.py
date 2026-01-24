"""
Abstract base class for storage backends.

Defines the interface for persisting and loading various data types
used throughout the concept mapping pipeline.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.

    Implementations should handle serialization/deserialization of:
    - ProcessedDocument objects (corpus data)
    - TermList objects (curated terms)
    - Graph objects (concept maps)
    - Analysis results (frequencies, co-occurrence matrices)
    """

    @abstractmethod
    def save_corpus(self, corpus_data: Any, path: Path) -> None:
        """
        Save preprocessed corpus data.

        Args:
            corpus_data: Corpus or list of ProcessedDocument objects
            path: Output file path
        """
        pass

    @abstractmethod
    def load_corpus(self, path: Path) -> Any:
        """
        Load preprocessed corpus data.

        Args:
            path: Input file path

        Returns:
            Corpus or list of ProcessedDocument objects
        """
        pass

    @abstractmethod
    def save_term_list(self, term_list: Any, path: Path) -> None:
        """
        Save curated term list.

        Args:
            term_list: TermList object
            path: Output file path
        """
        pass

    @abstractmethod
    def load_term_list(self, path: Path) -> Any:
        """
        Load curated term list.

        Args:
            path: Input file path

        Returns:
            TermList object
        """
        pass

    @abstractmethod
    def save_graph(self, graph: Any, path: Path) -> None:
        """
        Save concept graph.

        Args:
            graph: ConceptGraph or networkx graph object
            path: Output file path
        """
        pass

    @abstractmethod
    def load_graph(self, path: Path) -> Any:
        """
        Load concept graph.

        Args:
            path: Input file path

        Returns:
            ConceptGraph or networkx graph object
        """
        pass

    @abstractmethod
    def save_analysis(self, data: Dict[str, Any], path: Path) -> None:
        """
        Save analysis results (frequencies, co-occurrence, etc.).

        Args:
            data: Dictionary of analysis results
            path: Output file path
        """
        pass

    @abstractmethod
    def load_analysis(self, path: Path) -> Dict[str, Any]:
        """
        Load analysis results.

        Args:
            path: Input file path

        Returns:
            Dictionary of analysis results
        """
        pass
