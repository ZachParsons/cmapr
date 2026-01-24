"""
File loading utilities for text corpus.

Provides functions to load individual files or entire directories
into Document and Corpus objects.
"""

from pathlib import Path
from typing import Dict, Optional, Union

from .models import Corpus, Document


def load_text(file_path: Union[str, Path]) -> str:
    """
    Load text content from a file with encoding fallback.

    Attempts UTF-8 first, falls back to Latin-1 if UTF-8 fails.
    This handles most text files including those with special characters.

    Args:
        file_path: Path to text file

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback to Latin-1 for files with special encoding
        return path.read_text(encoding="latin-1")


def load_file(file_path: Union[str, Path], metadata: Optional[Dict] = None) -> Document:
    """
    Load a single file into a Document object.

    Args:
        file_path: Path to text file
        metadata: Optional metadata dict (if None, extracts from filename)

    Returns:
        Document object with text and metadata

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)
    text = load_text(path)

    # Create metadata if not provided
    if metadata is None:
        metadata = {
            "source_path": str(path.resolve()),
            "filename": path.name,
            "title": path.stem,  # Filename without extension
        }
    else:
        # Ensure source_path is set
        if "source_path" not in metadata:
            metadata["source_path"] = str(path.resolve())

    return Document(text=text, metadata=metadata)


def load_directory(
    directory_path: Union[str, Path], pattern: str = "*.txt", recursive: bool = False
) -> Corpus:
    """
    Load all matching files from a directory into a Corpus.

    Args:
        directory_path: Path to directory
        pattern: Glob pattern for matching files (default: "*.txt")
        recursive: If True, search subdirectories recursively

    Returns:
        Corpus containing all loaded documents

    Raises:
        NotADirectoryError: If path is not a directory
        FileNotFoundError: If directory doesn't exist
    """
    dir_path = Path(directory_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    # Use rglob for recursive, glob for non-recursive
    glob_method = dir_path.rglob if recursive else dir_path.glob
    file_paths = sorted(glob_method(pattern))

    corpus = Corpus()
    for file_path in file_paths:
        if file_path.is_file():
            doc = load_file(file_path)
            corpus.add_document(doc)

    return corpus
