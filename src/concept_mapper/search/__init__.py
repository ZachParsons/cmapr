"""
Search and concordance tools for finding and viewing terms in context.

Provides functionality for:
- Finding sentences containing terms
- KWIC (Key Word In Context) concordance display
- Context windows (sentences before/after)
- Term dispersion analysis (where terms appear)
- Extracting significant terms from matching sentences
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
from .extract import (
    SignificantTermsResult,
    extract_significant_terms,
    format_results_by_sentence,
    format_results_detailed,
    aggregate_across_sentences,
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
    "SignificantTermsResult",
    "extract_significant_terms",
    "format_results_by_sentence",
    "format_results_detailed",
    "aggregate_across_sentences",
]
