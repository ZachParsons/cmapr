"""
Filesystem utilities for storage operations.

Provides helper functions for managing output directories
and validating file paths.
"""

from pathlib import Path


def ensure_output_structure(base_dir: Path = Path("output")) -> dict[str, Path]:
    """
    Create standard output directory structure.

    Creates the following directories:
    - output/corpus/
    - output/terms/
    - output/graphs/
    - output/exports/
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
        "terms": base_dir / "terms",
        "graphs": base_dir / "graphs",
        "exports": base_dir / "exports",
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
    subdir: str = "terms",
    base_dir: Path = Path("output"),
    ensure_dir: bool = True,
) -> Path:
    """
    Get standardized output path for a file.

    Args:
        filename: Output filename
        subdir: Subdirectory within output (e.g., 'corpus', 'terms', 'graphs', 'exports')
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
    return get_output_path(
        cache_name, subdir="cache", base_dir=base_dir, ensure_dir=ensure_dir
    )


def derive_identifier(file_path: Path) -> str:
    """
    Extract identifier from filename stem.

    The identifier is derived from the filename by:
    1. Taking the file stem (filename without extension)
    2. Replacing spaces with underscores
    3. Removing parentheses and other problematic characters
    4. Keeping only alphanumeric characters, hyphens, and underscores
    5. Truncating to 100 characters if needed

    Args:
        file_path: Path to file

    Returns:
        Sanitized identifier string

    Examples:
        >>> derive_identifier(Path("sample1_analytic_pragmatism.txt"))
        'sample1_analytic_pragmatism'
        >>> derive_identifier(Path("My Text (2024).txt"))
        'My_Text_2024'
        >>> derive_identifier(Path("file-with-dashes.txt"))
        'file-with-dashes'
    """
    stem = file_path.stem

    # Handle empty stem
    if not stem:
        return "untitled"

    # Sanitize: replace spaces with underscores, remove parentheses
    identifier = stem.replace(" ", "_").replace("(", "").replace(")", "")

    # Remove other problematic characters, keep alphanumeric + hyphens + underscores
    identifier = "".join(c for c in identifier if c.isalnum() or c in "-_")

    # Handle case where all characters were removed
    if not identifier:
        return "untitled"

    # Truncate if too long (max 100 chars)
    return identifier[:100] if len(identifier) > 100 else identifier


def infer_output_path(
    input_path: Path,
    output_dir: Path,
    subdir: str,
    suffix: str = "",
    extension: str = ".json",
) -> Path:
    """
    Infer output path from input path.

    Derives an identifier from the input filename and constructs
    an output path in the standard directory structure.

    Args:
        input_path: Input file path
        output_dir: Base output directory
        subdir: Subdirectory within output (e.g., 'corpus', 'terms', 'graphs', 'exports')
        suffix: Optional suffix to append to identifier (e.g., '_cooccurrence')
        extension: File extension (default: '.json')

    Returns:
        Path object: output_dir/subdir/identifier{suffix}{extension}

    Examples:
        >>> infer_output_path(Path("sample1.txt"), Path("output"), "corpus")
        Path('output/corpus/sample1.json')
        >>> infer_output_path(Path("data.txt"), Path("out"), "graphs", "_cooccur", ".json")
        Path('out/graphs/data_cooccur.json')
    """
    identifier = derive_identifier(input_path)
    output_subdir = output_dir / subdir
    output_subdir.mkdir(parents=True, exist_ok=True)
    return output_subdir / f"{identifier}{suffix}{extension}"
