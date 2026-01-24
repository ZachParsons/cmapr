"""
Term list management for human-in-the-loop curation.

Provides data structures and operations for managing curated philosophical
term lists - the key concepts selected from statistical detection for further
analysis and visualization.
"""

from .models import TermEntry, TermList
from .manager import TermManager
from .suggester import suggest_terms_from_analysis

__all__ = [
    "TermEntry",
    "TermList",
    "TermManager",
    "suggest_terms_from_analysis",
]
