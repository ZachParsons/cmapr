"""
Corpus management module.

Provides data structures and utilities for loading and managing
text corpora.
"""

from .loader import load_directory, load_file, load_text
from .models import Corpus, Document, ProcessedDocument

__all__ = [
    "Corpus",
    "Document",
    "ProcessedDocument",
    "load_text",
    "load_file",
    "load_directory",
]
