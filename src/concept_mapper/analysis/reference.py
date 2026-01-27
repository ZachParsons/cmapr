"""
Reference corpus utilities.

Provides functions for loading and caching reference corpora
(e.g., Brown corpus) for comparative frequency analysis.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Optional

from nltk.corpus import brown

from ..storage.utils import get_cache_path


def load_reference_corpus(
    name: str = "brown", cache: bool = True, cache_dir: Optional[Path] = None
) -> Counter:
    """
    Load reference corpus word frequencies.

    Checks for bundled reference data first (data/reference/), then user cache
    (output/cache/). If not found, computes from NLTK corpus and caches.

    Args:
        name: Reference corpus name ("brown" supported currently)
        cache: If True, cache frequencies to disk after first computation
        cache_dir: Directory for cache files (default: output/cache/)

    Returns:
        Counter mapping words to their frequencies in reference corpus

    Example:
        >>> ref_freq = load_reference_corpus("brown")
        >>> ref_freq["the"] > ref_freq["philosophy"]
        True
    """
    if name != "brown":
        raise ValueError(
            f"Unsupported reference corpus: {name}. Only 'brown' is currently supported."
        )

    # Check bundled reference data first
    bundled_path = (
        Path(__file__).parent.parent.parent.parent
        / "data"
        / "reference"
        / f"{name}_corpus_freqs.json"
    )
    if bundled_path.exists():
        with bundled_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            return Counter(data)

    # Check user cache
    if cache:
        cache_path = get_cache_path(
            f"{name}_corpus_freqs.json",
            base_dir=cache_dir or Path("output"),
            ensure_dir=True,
        )

        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
                return Counter(data)

    # Compute frequencies from Brown corpus
    print(f"Computing {name} corpus frequencies (this may take a moment)...")
    words = [word.lower() for word in brown.words()]
    freq = Counter(words)

    # Cache for future use
    if cache:
        print(f"Caching frequencies to {cache_path}")
        with cache_path.open("w", encoding="utf-8") as f:
            json.dump(dict(freq), f)

    return freq


def get_reference_vocabulary(name: str = "brown") -> set[str]:
    """
    Get all unique words in reference corpus.

    Args:
        name: Reference corpus name

    Returns:
        Set of all words in corpus

    Example:
        >>> vocab = get_reference_vocabulary("brown")
        >>> "the" in vocab
        True
        >>> len(vocab) > 10000
        True
    """
    freq = load_reference_corpus(name)
    return set(freq.keys())


def get_reference_size(name: str = "brown") -> int:
    """
    Get total word count in reference corpus.

    Args:
        name: Reference corpus name

    Returns:
        Total number of words

    Example:
        >>> size = get_reference_size("brown")
        >>> size > 1000000
        True
    """
    freq = load_reference_corpus(name)
    return sum(freq.values())
