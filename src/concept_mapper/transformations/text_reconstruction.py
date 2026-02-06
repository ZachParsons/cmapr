"""
Reconstruct text after replacements, preserving formatting.

Handles spacing, punctuation attachment, and contractions to produce
natural-looking output text.
"""

from typing import List, Tuple


class TextReconstructor:
    """
    Rebuild text from tokens with replacements while preserving spacing/punctuation.
    """

    # Punctuation that attaches to previous token (no space before)
    ATTACH_LEFT = {".", ",", "!", "?", ";", ":", ")", "]", "}", "'"}

    # Punctuation that attaches to next token (no space after)
    ATTACH_RIGHT = {"(", "[", "{"}

    # Contraction parts that attach to previous token
    CONTRACTIONS = {"'s", "'t", "'re", "'ve", "'d", "'ll", "n't", "'m"}

    def rebuild(
        self, original_tokens: List[str], replacements: List[Tuple[int, str]]
    ) -> str:
        """
        Rebuild text from tokens with replacements.

        Args:
            original_tokens: Original token list from ProcessedDocument
            replacements: List of (index, replacement_text) tuples

        Returns:
            Reconstructed text with proper spacing and formatting

        Example:
            >>> reconstructor = TextReconstructor()
            >>> tokens = ['The', 'cat', 'ran', 'quickly', '.']
            >>> replacements = [(2, 'sprinted'), (3, 'swiftly')]
            >>> reconstructor.rebuild(tokens, replacements)
            'The cat sprinted swiftly.'
        """
        # Create replacement map for O(1) lookup
        replacement_map = dict(replacements)

        # Build new token list with replacements applied
        new_tokens = []
        for i, token in enumerate(original_tokens):
            if i in replacement_map:
                new_tokens.append(replacement_map[i])
            else:
                new_tokens.append(token)

        # Join with smart spacing
        return self._join_with_spacing(new_tokens)

    def _join_with_spacing(self, tokens: List[str]) -> str:
        """
        Join tokens with appropriate spacing rules.

        Handles:
        - Normal spacing between words
        - Punctuation attachment (no space before .,!? etc.)
        - Opening punctuation (no space after ([{ etc.)
        - Contractions (attach to previous word)

        Args:
            tokens: List of tokens to join

        Returns:
            Joined string with proper spacing
        """
        if not tokens:
            return ""

        result = []

        for i, token in enumerate(tokens):
            if i == 0:
                # First token: no space before
                result.append(token)
            elif self._is_attach_left(token):
                # Punctuation/contraction: attach to previous token
                result.append(token)
            elif i > 0 and self._is_attach_right(tokens[i - 1]):
                # Previous token was opening punctuation: no space
                result.append(token)
            else:
                # Normal case: add space before token
                result.append(" ")
                result.append(token)

        return "".join(result)

    def _is_attach_left(self, token: str) -> bool:
        """
        Check if token should attach to previous token (no space before).

        Args:
            token: Token to check

        Returns:
            True if token attaches left
        """
        # Check if token is punctuation or contraction
        return token in self.ATTACH_LEFT or token in self.CONTRACTIONS

    def _is_attach_right(self, token: str) -> bool:
        """
        Check if token should attach to next token (no space after).

        Args:
            token: Token to check

        Returns:
            True if token attaches right
        """
        return token in self.ATTACH_RIGHT

    def rebuild_phrases(
        self,
        original_tokens: List[str],
        phrase_replacements: List[Tuple[int, int, str]],
    ) -> str:
        """
        Rebuild text with multi-word phrase replacements.

        Args:
            original_tokens: Original token list
            phrase_replacements: List of (start_idx, end_idx, replacement_text) tuples

        Returns:
            Reconstructed text with phrase replacements applied

        Example:
            >>> reconstructor = TextReconstructor()
            >>> tokens = ['The', 'body', 'without', 'organs', 'resists', '.']
            >>> replacements = [(1, 4, 'medium')]
            >>> reconstructor.rebuild_phrases(tokens, replacements)
            'The medium resists.'
        """
        # Sort replacements by start index (reverse order for safe deletion)
        sorted_replacements = sorted(
            phrase_replacements, key=lambda x: x[0], reverse=True
        )

        # Create new token list
        new_tokens = original_tokens.copy()

        # Apply phrase replacements (working backwards to maintain indices)
        for start_idx, end_idx, replacement in sorted_replacements:
            # Replace span with single token
            new_tokens[start_idx:end_idx] = [replacement]

        # Join with smart spacing
        return self._join_with_spacing(new_tokens)
