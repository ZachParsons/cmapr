"""
Search and concordance tools for finding and viewing terms in context.

Provides functionality for:
- Finding sentences containing terms
- KWIC (Key Word In Context) concordance display
- Context windows (sentences before/after)
- Term dispersion analysis (where terms appear)
"""

from .find import SentenceMatch, find_sentences
from .concordance import KWICLine, concordance, format_kwic_lines
from .context import ContextWindow, get_context
from .dispersion import (
    dispersion,
    get_dispersion_summary,
    compare_dispersion,
    dispersion_plot_data,
    get_concentrated_regions,
)

__all__ = [
    "SentenceMatch",
    "find_sentences",
    "KWICLine",
    "concordance",
    "format_kwic_lines",
    "ContextWindow",
    "get_context",
    "dispersion",
    "get_dispersion_summary",
    "compare_dispersion",
    "dispersion_plot_data",
    "get_concentrated_regions",
]
