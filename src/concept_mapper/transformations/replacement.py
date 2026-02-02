"""
Core synonym replacement with inflection preservation.

Replaces terms with synonyms while maintaining grammatical correctness
through POS-aware inflection generation.
"""

from dataclasses import dataclass
from typing import List, Union, Tuple
from ..corpus.models import ProcessedDocument
from .inflection import InflectionGenerator
from .phrase_matcher import PhraseMatcher, PhraseMatch
from .text_reconstruction import TextReconstructor


@dataclass
class ReplacementSpec:
    """Specification for a synonym replacement operation."""

    source_lemma: Union[str, List[str]]  # Single word or phrase lemmas
    target_lemma: Union[str, List[str]]  # Replacement (single or phrase)

    @property
    def is_phrase(self) -> bool:
        """Check if source is multi-word phrase."""
        return isinstance(self.source_lemma, list)

    @property
    def target_is_phrase(self) -> bool:
        """Check if target is multi-word phrase."""
        return isinstance(self.target_lemma, list)


class SynonymReplacer:
    """
    Replace synonyms while preserving grammatical inflections.

    Handles both single-word and multi-word replacements, maintaining
    tense, number, degree, and other grammatical features.
    """

    def __init__(self):
        self.inflector = InflectionGenerator()
        self.matcher = PhraseMatcher()
        self.reconstructor = TextReconstructor()

    def replace_in_document(
        self,
        spec: ReplacementSpec,
        doc: ProcessedDocument,
        case_sensitive: bool = False
    ) -> str:
        """
        Replace all occurrences of source with target in document.

        Args:
            spec: Replacement specification
            doc: Preprocessed document
            case_sensitive: Whether matching should be case-sensitive

        Returns:
            Modified text with replacements applied

        Examples:
            >>> replacer = SynonymReplacer()
            >>> spec = ReplacementSpec("run", "sprint")
            >>> text = replacer.replace_in_document(spec, doc)
            >>> # "running" → "sprinting", "ran" → "sprinted"
        """
        if spec.is_phrase:
            return self._replace_phrase(spec, doc, case_sensitive)
        else:
            return self._replace_single(spec, doc, case_sensitive)

    def _replace_single(
        self,
        spec: ReplacementSpec,
        doc: ProcessedDocument,
        case_sensitive: bool
    ) -> str:
        """
        Replace single-word term with inflection preservation.

        Args:
            spec: Replacement spec (both source and target are single words)
            doc: Document to process
            case_sensitive: Case-sensitive matching

        Returns:
            Modified text
        """
        replacements = []
        source_lemma = spec.source_lemma if case_sensitive else spec.source_lemma.lower()

        # Find matching tokens by lemma
        for i, lemma in enumerate(doc.lemmas):
            compare_lemma = lemma if case_sensitive else lemma.lower()

            if compare_lemma == source_lemma:
                # Get POS tag for this token
                _, pos_tag = doc.pos_tags[i]

                # Generate inflected form of target
                inflected = self.inflector.inflect(spec.target_lemma, pos_tag)

                # Preserve capitalization pattern of original token
                original_token = doc.tokens[i]
                inflected = self._match_capitalization(inflected, original_token)

                # Record replacement
                replacements.append((i, inflected))

        # Rebuild text with replacements
        return self.reconstructor.rebuild(doc.tokens, replacements)

    def _replace_phrase(
        self,
        spec: ReplacementSpec,
        doc: ProcessedDocument,
        case_sensitive: bool
    ) -> str:
        """
        Replace multi-word phrase.

        Handles three cases:
        1. Multi → single word (e.g., "bodies without organs" → "mediums")
        2. Multi → multi word (e.g., "body without organs" → "blank resistant field")

        Args:
            spec: Replacement spec with phrase as source
            doc: Document to process
            case_sensitive: Case-sensitive matching

        Returns:
            Modified text
        """
        # Find phrase matches
        matches = self.matcher.find_phrase_matches(
            spec.source_lemma,
            doc,
            case_sensitive
        )

        # Generate replacements for each match
        phrase_replacements = []

        for match in matches:
            # Generate replacement text
            if not spec.target_is_phrase:
                # Multi → single word: inflect based on phrase head
                replacement = self._inflect_for_phrase_head(
                    spec.target_lemma,
                    match
                )
            else:
                # Multi → multi word: inflect head word of target phrase
                replacement = self._inflect_phrase(
                    spec.target_lemma,
                    match
                )

            # Preserve sentence-initial capitalization
            if match.tokens[0][0].isupper():
                replacement = self._capitalize_first_word(replacement)

            # Record phrase replacement (start, end, replacement_text)
            phrase_replacements.append((match.start_idx, match.end_idx, replacement))

        # Rebuild text with phrase replacements
        return self.reconstructor.rebuild_phrases(doc.tokens, phrase_replacements)

    def _inflect_for_phrase_head(
        self,
        target_lemma: str,
        match: PhraseMatch
    ) -> str:
        """
        Inflect single target word based on phrase head.

        Example:
            Phrase: "bodies without organs" (head="bodies", POS=NNS)
            Target: "medium"
            Result: "mediums" (inflected to plural)

        Args:
            target_lemma: Single word to inflect
            match: Phrase match containing head word info

        Returns:
            Inflected form of target
        """
        # Get head word's POS tag
        head_pos = match.head_pos

        # Inflect target to match head's POS
        inflected = self.inflector.inflect(target_lemma, head_pos)

        return inflected

    def _inflect_phrase(
        self,
        target_lemmas: List[str],
        match: PhraseMatch
    ) -> str:
        """
        Inflect multi-word target phrase based on source phrase head.

        Example:
            Source: "bodies without organs" (head="bodies", POS=NNS)
            Target: ["blank", "resistant", "field"]
            Result: "blank resistant fields" (head "field" inflected to plural)

        Args:
            target_lemmas: List of lemmas for target phrase
            match: Source phrase match with head info

        Returns:
            Inflected target phrase as string
        """
        # Identify head word in target phrase (assume last content word)
        # For simplicity, use last word as head
        head_idx = len(target_lemmas) - 1

        # Inflect head word
        head_pos = match.head_pos
        inflected_head = self.inflector.inflect(target_lemmas[head_idx], head_pos)

        # Build phrase with inflected head
        phrase_tokens = target_lemmas[:head_idx] + [inflected_head]

        # Join with spaces
        return ' '.join(phrase_tokens)

    def _match_capitalization(self, target: str, original: str) -> str:
        """
        Match capitalization pattern of original token.

        Handles:
        - All caps: "RUNNING" → "SPRINTING"
        - Title case: "Running" → "Sprinting"
        - Lower case: "running" → "sprinting"

        Args:
            target: New word to capitalize
            original: Original word with capitalization pattern

        Returns:
            Target with matched capitalization
        """
        if not original or not target:
            return target

        if original.isupper():
            # All caps
            return target.upper()
        elif original[0].isupper():
            # Title case (first letter capitalized)
            return target.capitalize()
        else:
            # Lower case
            return target.lower()

    def _capitalize_first_word(self, text: str) -> str:
        """
        Capitalize first letter of text (for sentence-initial preservation).

        Args:
            text: Text to capitalize

        Returns:
            Text with first letter capitalized
        """
        if not text:
            return text
        return text[0].upper() + text[1:] if len(text) > 1 else text.upper()
