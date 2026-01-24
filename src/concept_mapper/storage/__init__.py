"""
Storage layer for concept mapper.

Provides abstract storage backend interface and implementations
for persisting intermediate and final outputs.
"""

from .backend import StorageBackend
from .json_backend import JSONBackend
from .utils import (
    ensure_output_structure,
    get_cache_path,
    get_output_path,
    validate_file_path,
)

__all__ = [
    "StorageBackend",
    "JSONBackend",
    "ensure_output_structure",
    "get_cache_path",
    "get_output_path",
    "validate_file_path",
]
