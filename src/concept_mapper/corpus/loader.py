"""File loading utilities for text corpus."""

from pathlib import Path


def load_text(file_path: str) -> str:
    """Load text content from a file."""
    return Path(file_path).read_text(encoding="utf-8")
