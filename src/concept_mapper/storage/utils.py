"""
Filesystem utilities for storage operations.

Provides helper functions for managing output directories
and validating file paths.
"""

from pathlib import Path
from typing import Optional


def ensure_output_structure(base_dir: Path = Path("output")) -> dict[str, Path]:
    """
    Create standard output directory structure.

    Creates the following directories:
    - output/corpus/
    - output/analysis/
    - output/graphs/
    - output/graphs/d3/
    - output/cache/

    Args:
        base_dir: Base output directory (default: "output")

    Returns:
        Dictionary mapping directory names to Path objects
    """
    base_dir = Path(base_dir)

    directories = {
        "base": base_dir,
        "corpus": base_dir / "corpus",
        "analysis": base_dir / "analysis",
        "graphs": base_dir / "graphs",
        "d3": base_dir / "graphs" / "d3",
        "cache": base_dir / "cache",
    }

    for dir_path in directories.values():
        dir_path.mkdir(parents=True, exist_ok=True)

    return directories


def validate_file_path(path: Path, must_exist: bool = False) -> Path:
    """
    Validate and normalize a file path.

    Args:
        path: File path to validate
        must_exist: If True, raise error if file doesn't exist

    Returns:
        Normalized Path object

    Raises:
        FileNotFoundError: If must_exist=True and file doesn't exist
        ValueError: If path is a directory when file expected
    """
    path = Path(path).resolve()

    if must_exist and not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.exists() and path.is_dir():
        raise ValueError(f"Expected file, got directory: {path}")

    return path


def get_output_path(
    filename: str,
    subdir: str = "analysis",
    base_dir: Path = Path("output"),
    ensure_dir: bool = True,
) -> Path:
    """
    Get standardized output path for a file.

    Args:
        filename: Output filename
        subdir: Subdirectory within output (e.g., 'analysis', 'graphs')
        base_dir: Base output directory
        ensure_dir: If True, create directory if it doesn't exist

    Returns:
        Full path to output file
    """
    output_dir = Path(base_dir) / subdir

    if ensure_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir / filename


def get_cache_path(
    cache_name: str, base_dir: Path = Path("output"), ensure_dir: bool = True
) -> Path:
    """
    Get path for cached data.

    Args:
        cache_name: Name of cache file (e.g., 'brown_corpus_freqs.json')
        base_dir: Base output directory
        ensure_dir: If True, create cache directory if it doesn't exist

    Returns:
        Full path to cache file
    """
    return get_output_path(cache_name, subdir="cache", base_dir=base_dir, ensure_dir=ensure_dir)
