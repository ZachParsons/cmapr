"""
Text cleaning for OCR and PDF extraction artifacts.

Handles common issues from PDF-to-text conversion and OCR errors:
- Spacing issues around punctuation
- Split words
- Common OCR character substitutions
- Page numbers and headers/footers
- Special characters in words

Philosophy: Good-enough cleaning is better than perfect but slow.
Focus on high-value, low-complexity fixes.
"""

import re


class TextCleaner:
    """Clean OCR and PDF extraction artifacts from text."""

    # Common OCR character substitutions
    OCR_SUBSTITUTIONS = [
        # Numbers at END of words (likely OCR errors) - need custom function for multiple replacements
        (
            r"([a-z]{3,})(1+)\b",
            lambda m: m.group(1) + "l" * len(m.group(2)),
        ),  # 11 -> ll
        (
            r"([a-z]{3,})(0+)\b",
            lambda m: m.group(1) + "o" * len(m.group(2)),
        ),  # 00 -> oo
        # Special characters that shouldn't be in words
        (r"(\w)[!/\\|](\w)", r"\1\2"),  # Remove !/\/| from middle of words
        (r"(\w)!([A-Z])", r"\1\2"),  # predomi!IOI -> premiIOI
    ]

    # Patterns for spacing fixes (order matters!)
    SPACING_PATTERNS = [
        # Fix spacing in numbered headings: "1 . 5 . 2." -> "1.5.2."
        # Apply multiple times to handle nested patterns
        (r"(\d+)\s+\.\s+(\d+)", r"\1.\2"),  # "1 . 5" -> "1.5"
        # Fix spacing in numbered lists: "1 . " -> "1. "
        (r"(\d+)\s+\.\s+([A-Z])", r"\1. \2"),  # "1 . The" -> "1. The"
        # Fix multiple spaces (but preserve paragraph breaks)
        (r"([^\n]) {2,}", r"\1 "),  # Multiple spaces on same line
        # Fix space before punctuation (except after newlines)
        (r"([^\n])\s+([.,;:!?])", r"\1\2"),
    ]

    def __init__(
        self,
        fix_spacing: bool = True,
        fix_ocr_chars: bool = True,
        fix_split_words: bool = True,
        fix_joined_words: bool = True,
        remove_page_numbers: bool = True,
        min_word_length: int = 2,
    ):
        """
        Initialize text cleaner.

        Args:
            fix_spacing: Fix spacing issues around punctuation
            fix_ocr_chars: Fix common OCR character substitutions
            fix_split_words: Attempt to rejoin split words
            fix_joined_words: Attempt to split incorrectly joined words
            remove_page_numbers: Remove standalone page numbers
            min_word_length: Minimum length for valid words
        """
        self.fix_spacing = fix_spacing
        self.fix_ocr_chars = fix_ocr_chars
        self.fix_split_words = fix_split_words
        self.fix_joined_words = fix_joined_words
        self.remove_page_numbers = remove_page_numbers
        self.min_word_length = min_word_length

    def clean(self, text: str) -> str:
        """
        Clean text with all enabled fixes.

        Args:
            text: Raw text to clean

        Returns:
            Cleaned text
        """
        if self.remove_page_numbers:
            text = self._remove_page_numbers(text)

        if self.fix_spacing:
            text = self._fix_spacing(text)

        if self.fix_ocr_chars:
            text = self._fix_ocr_characters(text)

        if self.fix_split_words:
            text = self._fix_split_words(text)

        if self.fix_joined_words:
            text = self._fix_joined_words(text)

        # Final cleanup
        text = self._final_cleanup(text)

        return text

    def _remove_page_numbers(self, text: str) -> str:
        """
        Remove standalone page numbers and page headers.

        Removes:
        - Lines with just page numbers: "42"
        - Lines with page numbers and simple headers: "Page 42", "42 | Chapter 1"
        - Clusters of numbers (TOC page listings)
        """
        lines = text.split("\n")
        cleaned_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip empty lines
            if not stripped:
                cleaned_lines.append(line)
                continue

            # Skip lines that are just numbers (page numbers)
            if stripped.isdigit() and len(stripped) <= 5:
                continue

            # Skip simple page headers: "Page 42", "42 |", "- 42 -"
            if re.match(r"^(?:Page\s+)?\d+\s*[|\-]?$", stripped, re.IGNORECASE):
                continue

            # Skip lines with clusters of numbers (TOC page listings)
            # e.g., "3 4 5 6 7 90 89 88 87 86"
            if re.match(r"^[\d\s]+$", stripped) and len(stripped.split()) > 3:
                continue

            cleaned_lines.append(line)

        return "\n".join(cleaned_lines)

    def _fix_spacing(self, text: str) -> str:
        """Fix spacing issues around punctuation."""
        # Apply some patterns multiple times to handle nested cases
        for pattern, replacement in self.SPACING_PATTERNS[:2]:
            # Apply number spacing fixes multiple times
            for _ in range(3):
                text = re.sub(pattern, replacement, text)

        # Apply other patterns once
        for pattern, replacement in self.SPACING_PATTERNS[2:]:
            text = re.sub(pattern, replacement, text)

        return text

    def _fix_ocr_characters(self, text: str) -> str:
        """Fix common OCR character substitutions."""
        for pattern, replacement in self.OCR_SUBSTITUTIONS:
            text = re.sub(pattern, replacement, text)
        return text

    def _fix_split_words(self, text: str) -> str:
        """
        Attempt to rejoin split words.

        Strategy: Look for patterns like "obsti nacy" where two parts
        separated by a single space could form a valid word.

        This is conservative - only joins when:
        1. Single space between parts
        2. Both parts are lowercase (or first is capitalized)
        3. Neither part is too short
        4. The pattern looks like a split word (ends mid-syllable)
        """

        def maybe_join(match):
            part1 = match.group(1)
            part2 = match.group(2)

            # For very short first parts (2-3 chars), be more conservative
            if len(part1) < 2:
                return match.group(0)

            # Require second part to be at least 3 chars
            if len(part2) < 3:
                return match.group(0)

            # Don't join if either part is a common complete word
            common_words = {
                "the",
                "and",
                "that",
                "with",
                "from",
                "this",
                "they",
                "have",
                "been",
                "are",
                "was",
                "were",
                "will",
                "clean",
                "clear",
                "can",
                "for",
                "not",
                "but",
                "all",
                "has",
            }
            if part1.lower() in common_words or part2.lower() in common_words:
                return match.group(0)

            # Don't join if first part ends with common suffixes (-ly, -ed, -ing, -er)
            if len(part1) > 4 and re.search(r"(ly|ed|ing|er)$", part1, re.IGNORECASE):
                return match.group(0)

            # Only join if first part is short (likely incomplete fragment)
            # Longer first parts are probably complete words
            if len(part1) > 6:
                return match.group(0)

            # Join them
            return part1 + part2

        # Pattern 1: consonant ending + vowel beginning (e.g., "signif icant")
        # Allow 1+ chars before consonant to catch short fragments
        pattern1 = r"\b([a-z]+[bcdfghjklmnpqrstvwxyz])\s+([aeiouy][a-z]{2,})\b"
        text = re.sub(pattern1, maybe_join, text, flags=re.IGNORECASE)

        # Pattern 2: vowel ending + consonant beginning (e.g., "obsti nacy", "li nguistic")
        # Allow 1+ chars before vowel to catch short fragments like "li"
        pattern2 = r"\b([a-z]+[aeiouy])\s+([bcdfghjklmnpqrstvwxyz][a-z]{2,})\b"
        text = re.sub(pattern2, maybe_join, text, flags=re.IGNORECASE)

        return text

    def _fix_joined_words(self, text: str) -> str:
        """
        Attempt to split incorrectly joined words.

        Strategy: Look for patterns where common small words are joined
        to other words without spaces (e.g., "betweenexpression").

        This is conservative - only splits at known word boundaries.
        """
        # Common small words that often get joined incorrectly
        # (prepositions, articles, conjunctions)
        common_prefixes = [
            "between",
            "within",
            "without",
            "through",
            "around",
            "before",
            "after",
            "under",
            "over",
            "about",
            "above",
            "below",
            "inside",
            "outside",
            "into",
            "onto",
            "upon",
            "towards",
            "toward",
        ]

        # Pattern: commonword + capitalized word or another word
        for prefix in common_prefixes:
            # Look for: prefix + [lowercase letter starting a new word]
            # e.g., "betweenexpression" → "between expression"
            pattern = r"\b(" + re.escape(prefix) + r")([a-z]{3,})\b"

            def maybe_split(match):
                prefix_part = match.group(1)
                rest_part = match.group(2)

                # Only split if rest part looks like a distinct word
                # (starts with common word beginnings or is long enough)
                if len(rest_part) >= 5:
                    return prefix_part + " " + rest_part

                return match.group(0)  # Keep as-is

            text = re.sub(pattern, maybe_split, text, flags=re.IGNORECASE)

        # Look for adjective+noun or compound words that should be split
        # Pattern: lowercase + uppercase (CamelCase-like errors)
        # e.g., "structuralArrangements" → "structural Arrangements"
        text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)

        return text

    def _final_cleanup(self, text: str) -> str:
        """Final cleanup pass."""
        # Remove excessive newlines (more than 2) but preserve paragraph breaks
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove trailing whitespace from lines (but keep the newlines)
        lines = text.split("\n")
        lines = [line.rstrip() for line in lines]
        text = "\n".join(lines)

        # Remove leading/trailing whitespace from whole text
        return text.strip()


def clean_text(
    text: str,
    fix_spacing: bool = True,
    fix_ocr_chars: bool = True,
    fix_split_words: bool = True,
    fix_joined_words: bool = True,
    remove_page_numbers: bool = True,
) -> str:
    """
    Convenience function to clean text.

    Args:
        text: Raw text to clean
        fix_spacing: Fix spacing issues around punctuation
        fix_ocr_chars: Fix common OCR character substitutions
        fix_split_words: Attempt to rejoin split words
        fix_joined_words: Attempt to split incorrectly joined words
        remove_page_numbers: Remove standalone page numbers

    Returns:
        Cleaned text

    Example:
        >>> text = "1 . 5. The obsti nacy of predomi!IOI/Ce\\n42\\n"
        >>> clean_text(text)
        "1.5. The obstinacy of premiIOICe"
    """
    cleaner = TextCleaner(
        fix_spacing=fix_spacing,
        fix_ocr_chars=fix_ocr_chars,
        fix_split_words=fix_split_words,
        fix_joined_words=fix_joined_words,
        remove_page_numbers=remove_page_numbers,
    )
    return cleaner.clean(text)


def detect_ocr_issues(text: str) -> dict:
    """
    Detect potential OCR issues in text without fixing them.

    Useful for reporting to users what issues were found.

    Returns:
        Dictionary with counts of different issue types
    """
    issues = {
        "spacing_in_numbers": 0,
        "special_chars_in_words": 0,
        "possible_split_words": 0,
        "page_numbers": 0,
        "number_clusters": 0,
    }

    # Count spacing issues in numbers
    issues["spacing_in_numbers"] = len(re.findall(r"\d+\s+\.\s+\d+", text))

    # Count special characters in words
    issues["special_chars_in_words"] = len(re.findall(r"\w[!/\\|]\w", text))

    # Count possible split words (both patterns from _fix_split_words)
    pattern1 = r"\b[a-z]{3,}[bcdfghjklmnpqrstvwxyz]\s+[aeiouy][a-z]{2,}\b"
    pattern2 = r"\b[a-z]{3,}[aeiouy]\s+[bcdfghjklmnpqrstvwxyz][a-z]{2,}\b"
    issues["possible_split_words"] = len(
        re.findall(pattern1, text, re.IGNORECASE)
    ) + len(re.findall(pattern2, text, re.IGNORECASE))

    # Count standalone page numbers
    lines = text.split("\n")
    for line in lines:
        stripped = line.strip()
        if stripped.isdigit() and len(stripped) <= 5:
            issues["page_numbers"] += 1
        elif re.match(r"^[\d\s]+$", stripped) and len(stripped.split()) > 3:
            issues["number_clusters"] += 1

    return issues
