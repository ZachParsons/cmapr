"""
Text transformation utilities for concept mapping.

Provides synonym replacement with inflection preservation, maintaining
grammatical correctness when replacing terms in philosophical texts.
"""

from .inflection import InflectionGenerator
from .phrase_matcher import PhraseMatcher, PhraseMatch
from .replacement import SynonymReplacer, ReplacementSpec
from .text_reconstruction import TextReconstructor

__all__ = [
    'InflectionGenerator',
    'PhraseMatcher',
    'PhraseMatch',
    'SynonymReplacer',
    'ReplacementSpec',
    'TextReconstructor',
]
