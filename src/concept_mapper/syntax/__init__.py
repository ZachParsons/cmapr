"""
Syntactic analysis module.

Provides dependency parsing and sentence diagramming capabilities.
"""

from .diagram import parse_sentence, diagram_sentence, print_dependency_tree

__all__ = ["parse_sentence", "diagram_sentence", "print_dependency_tree"]
