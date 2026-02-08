"""
Tests for text cleaning functionality.
"""

from concept_mapper.preprocessing.cleaning import (
    TextCleaner,
    clean_text,
    detect_ocr_issues,
)


class TestTextCleaner:
    """Tests for TextCleaner class."""

    def test_fix_spacing_in_numbers(self):
        """Test fixing spacing in numbered headings."""
        cleaner = TextCleaner()
        text = "1 . 5. The Introduction"
        result = cleaner.clean(text)
        assert "1.5." in result
        assert "1 . 5." not in result

    def test_fix_spacing_multiple_patterns(self):
        """Test various spacing patterns."""
        cleaner = TextCleaner()
        text = "1 . 5 . 2. Section  heading with  multiple   spaces ."
        result = cleaner.clean(text)
        assert "1.5.2." in result
        assert "  " not in result  # Multiple spaces reduced
        assert " ." not in result  # Space before period removed

    def test_fix_ocr_characters(self):
        """Test fixing common OCR character substitutions."""
        cleaner = TextCleaner()

        # Test 1 -> l at END of words
        text = "The interpretatio11 was clear"
        result = cleaner.clean(text)
        assert "interpretatioll" in result  # 11 -> ll at end

        # Test special characters in words
        text = "predomi!IOI/Ce of the signifier"
        result = cleaner.clean(text)
        # Should remove special chars from middle of words
        assert result.count("!") < text.count("!") or result.count("/") < text.count(
            "/"
        )

    def test_remove_page_numbers(self):
        """Test removing standalone page numbers."""
        cleaner = TextCleaner()

        # Standalone page number
        text = "Some content here.\n42\nMore content."
        result = cleaner.clean(text)
        assert "\n42\n" not in result

        # Page header
        text = "Content\nPage 25\nMore content"
        result = cleaner.clean(text)
        assert "Page 25" not in result

        # Number clusters (TOC page listings)
        text = "Table of Contents\n3 4 5 6 7 90 89 88\nChapter 1"
        result = cleaner.clean(text)
        assert "3 4 5 6 7" not in result

    def test_fix_split_words(self):
        """Test rejoining split words."""
        cleaner = TextCleaner()

        # Common split pattern: consonant + vowel
        text = "The obsti nacy of signs"
        result = cleaner.clean(text)
        assert "obstinacy" in result

        # Should not join actual separate words
        text = "The cat and dog"
        result = cleaner.clean(text)
        assert "cat and dog" in result  # Not "catand"

    def test_clean_with_all_fixes(self):
        """Test cleaning with multiple issues."""
        cleaner = TextCleaner()
        text = """
1 . 5. The obsti nacy of predomi!IOI/Ce

Content here.
42
More content with interpretatio11.
        """
        result = cleaner.clean(text)

        # Check all fixes applied
        assert "1.5." in result
        assert "obstinacy" in result
        assert "\n42\n" not in result

    def test_selective_cleaning(self):
        """Test disabling specific cleaning steps."""
        # Only fix spacing, not OCR chars
        cleaner = TextCleaner(fix_spacing=True, fix_ocr_chars=False)
        text = "1 . 5. predomi!IOI/Ce"
        result = cleaner.clean(text)
        assert "1.5." in result
        # OCR chars should still be present
        assert "!" in result

    def test_convenience_function(self):
        """Test clean_text convenience function."""
        text = "1 . 5. The obsti nacy\n42\n"
        result = clean_text(text)

        assert "1.5." in result
        assert "obstinacy" in result
        assert "\n42\n" not in result

    def test_empty_text(self):
        """Test cleaning empty text."""
        cleaner = TextCleaner()
        assert cleaner.clean("") == ""
        assert cleaner.clean("   ") == ""

    def test_text_without_issues(self):
        """Test that clean text is mostly preserved."""
        cleaner = TextCleaner()
        text = "This is perfectly clean text. No issues here."
        result = cleaner.clean(text)
        # Main content should be preserved (might have minor whitespace changes)
        assert "perfectly clean text" in result
        assert "No issues here" in result

    def test_preserve_intentional_formatting(self):
        """Test that intentional formatting is preserved."""
        cleaner = TextCleaner()
        text = "First paragraph.\n\nSecond paragraph."
        result = cleaner.clean(text)
        # Should preserve paragraph breaks
        assert "\n\n" in result


class TestDetectOCRIssues:
    """Tests for OCR issue detection."""

    def test_detect_spacing_issues(self):
        """Test detection of spacing issues in numbers."""
        text = "1 . 5. and 2 . 3. are sections"
        issues = detect_ocr_issues(text)
        assert issues["spacing_in_numbers"] == 2

    def test_detect_special_chars(self):
        """Test detection of special characters in words."""
        text = "predomi!IOI/Ce and other!words"
        issues = detect_ocr_issues(text)
        assert issues["special_chars_in_words"] >= 2

    def test_detect_split_words(self):
        """Test detection of possible split words."""
        text = "The obsti nacy of concep tual analysis"
        issues = detect_ocr_issues(text)
        assert issues["possible_split_words"] >= 2

    def test_detect_page_numbers(self):
        """Test detection of page numbers."""
        text = "Content\n42\nMore content\nPage 25\nEven more"
        issues = detect_ocr_issues(text)
        assert issues["page_numbers"] >= 1

    def test_detect_number_clusters(self):
        """Test detection of number clusters."""
        text = "TOC\n3 4 5 6 7 8 9 10\nChapter 1"
        issues = detect_ocr_issues(text)
        assert issues["number_clusters"] >= 1

    def test_clean_text_no_issues(self):
        """Test detection on clean text returns minimal issues."""
        text = "This is clean text without any OCR problems."
        issues = detect_ocr_issues(text)
        # Should have very few or no issues (heuristics aren't perfect)
        total_issues = sum(issues.values())
        assert total_issues < 3  # Allow some false positives from heuristics


class TestEdgeCases:
    """Test edge cases and corner cases."""

    def test_numbers_in_context(self):
        """Test that legitimate numbers are not corrupted."""
        cleaner = TextCleaner()
        text = "The year 2024 has 365 days."
        result = cleaner.clean(text)
        assert "2024" in result
        assert "365" in result

    def test_abbreviations(self):
        """Test that abbreviations are handled correctly."""
        cleaner = TextCleaner()
        text = "Dr. Smith and Prof. Jones wrote in Sec. 1.5."
        result = cleaner.clean(text)
        assert "Dr." in result
        assert "Prof." in result
        assert "Sec." in result or "1.5." in result

    def test_urls_and_emails(self):
        """Test that URLs and emails are not broken."""
        cleaner = TextCleaner()
        text = "Visit http://example.com or email test@example.com"
        result = cleaner.clean(text)
        # URLs/emails might be affected by special char cleaning, but shouldn't break completely
        assert "example.com" in result

    def test_unicode_characters(self):
        """Test handling of unicode characters."""
        cleaner = TextCleaner()
        text = "The café has naïve décor"
        result = cleaner.clean(text)
        assert "café" in result
        assert "naïve" in result
        assert "décor" in result
