"""
Tests for paragraph segmentation.
"""

from src.concept_mapper.preprocessing.segment import (
    segment_paragraphs,
    get_paragraph_indices,
    get_paragraph_spans,
)


class TestSegmentParagraphs:
    """Tests for segment_paragraphs function."""

    def test_double_newline_separation(self):
        """Test basic paragraph separation with blank lines."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        paragraphs = segment_paragraphs(text)

        assert len(paragraphs) == 3
        assert paragraphs[0] == "First paragraph."
        assert paragraphs[1] == "Second paragraph."
        assert paragraphs[2] == "Third paragraph."

    def test_multiple_blank_lines(self):
        """Test handling of multiple consecutive blank lines."""
        text = "Para 1.\n\n\n\nPara 2."
        paragraphs = segment_paragraphs(text)

        assert len(paragraphs) == 2
        assert paragraphs[0] == "Para 1."
        assert paragraphs[1] == "Para 2."

    def test_single_paragraph(self):
        """Test text with no paragraph breaks."""
        text = "This is a single paragraph with no breaks."
        paragraphs = segment_paragraphs(text)

        assert len(paragraphs) == 1
        assert paragraphs[0] == text

    def test_indented_paragraphs(self):
        """Test paragraph detection with indentation."""
        text = "First paragraph.\n  Indented second paragraph."
        paragraphs = segment_paragraphs(text)

        # Should split on indentation change
        assert len(paragraphs) >= 1

    def test_empty_text(self):
        """Test empty string input."""
        assert segment_paragraphs("") == []
        assert segment_paragraphs("   ") == []

    def test_whitespace_only_lines(self):
        """Test paragraphs separated by whitespace-only lines."""
        text = "Para 1.\n  \t  \nPara 2."
        paragraphs = segment_paragraphs(text)

        assert len(paragraphs) == 2

    def test_preserve_empty_lines_false(self):
        """Test that empty lines are filtered by default."""
        text = "Para 1.\n\n\n\nPara 2."
        paragraphs = segment_paragraphs(text, preserve_empty_lines=False)

        # Should only have non-empty paragraphs
        assert all(p for p in paragraphs)
        assert len(paragraphs) == 2

    def test_multi_line_paragraphs(self):
        """Test paragraphs with multiple lines (no blank line breaks)."""
        text = "Line 1 of para 1.\nLine 2 of para 1.\n\nLine 1 of para 2."
        paragraphs = segment_paragraphs(text)

        assert len(paragraphs) == 2
        assert "Line 1 of para 1" in paragraphs[0]
        assert "Line 2 of para 1" in paragraphs[0]

    def test_leading_trailing_whitespace(self):
        """Test that leading/trailing whitespace is stripped."""
        text = "  Para 1.  \n\n  Para 2.  "
        paragraphs = segment_paragraphs(text)

        assert paragraphs[0] == "Para 1."
        assert paragraphs[1] == "Para 2."

    def test_mixed_paragraph_markers(self):
        """Test document with mixed paragraph indicators."""
        text = "Para 1.\n\nPara 2 line 1.\nPara 2 line 2.\n\n  Indented para 3."
        paragraphs = segment_paragraphs(text)

        assert len(paragraphs) >= 2


class TestGetParagraphIndices:
    """Tests for get_paragraph_indices function."""

    def test_basic_mapping(self):
        """Test mapping sentences to paragraphs."""
        text = "Sent 1. Sent 2.\n\nSent 3. Sent 4."
        sentences = ["Sent 1.", "Sent 2.", "Sent 3.", "Sent 4."]

        indices = get_paragraph_indices(text, sentences)

        assert len(indices) == 4
        # First two sentences in paragraph 0
        assert indices[0] == 0
        assert indices[1] == 0
        # Last two sentences in paragraph 1
        assert indices[2] == 1
        assert indices[3] == 1

    def test_single_paragraph(self):
        """Test all sentences in one paragraph."""
        text = "Sent 1. Sent 2. Sent 3."
        sentences = ["Sent 1.", "Sent 2.", "Sent 3."]

        indices = get_paragraph_indices(text, sentences)

        assert indices == [0, 0, 0]

    def test_empty_sentences(self):
        """Test with empty sentence list."""
        text = "Some text."
        indices = get_paragraph_indices(text, [])

        assert indices == []

    def test_one_sentence_per_paragraph(self):
        """Test document where each sentence is its own paragraph."""
        text = "Sent 1.\n\nSent 2.\n\nSent 3."
        sentences = ["Sent 1.", "Sent 2.", "Sent 3."]

        indices = get_paragraph_indices(text, sentences)

        assert indices == [0, 1, 2]

    def test_sentence_not_in_any_paragraph(self):
        """Test fallback when sentence not found."""
        text = "Para 1."
        sentences = ["Nonexistent sent.", "Para 1."]

        indices = get_paragraph_indices(text, sentences)

        # Nonexistent sentence defaults to paragraph 0
        assert indices[0] == 0
        assert indices[1] == 0


class TestGetParagraphSpans:
    """Tests for get_paragraph_spans function."""

    def test_basic_spans(self):
        """Test getting character spans for paragraphs."""
        text = "Para 1.\n\nPara 2."
        spans = get_paragraph_spans(text)

        assert len(spans) == 2
        # First paragraph
        assert spans[0][0] == 0
        assert text[spans[0][0] : spans[0][1]] == "Para 1."
        # Second paragraph
        assert text[spans[1][0] : spans[1][1]] == "Para 2."

    def test_empty_text(self):
        """Test empty text returns empty spans."""
        assert get_paragraph_spans("") == []
        assert get_paragraph_spans("   ") == []

    def test_single_paragraph_span(self):
        """Test span for single paragraph."""
        text = "Only one paragraph here."
        spans = get_paragraph_spans(text)

        assert len(spans) == 1
        assert spans[0][0] == 0
        assert spans[0][1] == len(text)

    def test_spans_non_overlapping(self):
        """Test that spans don't overlap."""
        text = "P1.\n\nP2.\n\nP3."
        spans = get_paragraph_spans(text)

        # Check each span is before the next
        for i in range(len(spans) - 1):
            assert spans[i][1] <= spans[i + 1][0]


class TestIntegration:
    """Integration tests with preprocessing pipeline."""

    def test_preprocess_with_paragraphs(self):
        """Test that preprocessing includes paragraph indices."""
        from src.concept_mapper.corpus.models import Document
        from src.concept_mapper.preprocessing import preprocess

        text = "First sentence.\n\nSecond sentence in new paragraph."
        doc = Document(text=text)
        processed = preprocess(doc)

        # Should have paragraph indices
        assert len(processed.paragraph_indices) == processed.num_sentences
        # Sentences should map to different paragraphs
        assert processed.paragraph_indices[0] == 0
        if processed.num_sentences > 1:
            assert processed.paragraph_indices[-1] in [0, 1]

    def test_serialization_includes_paragraphs(self):
        """Test that paragraph indices are preserved in serialization."""
        from src.concept_mapper.corpus.models import Document
        from src.concept_mapper.preprocessing import preprocess

        text = "P1.\n\nP2."
        doc = Document(text=text)
        processed = preprocess(doc)

        # Serialize
        data = processed.to_dict()
        assert "paragraph_indices" in data

        # Deserialize
        from src.concept_mapper.corpus.models import ProcessedDocument

        restored = ProcessedDocument.from_dict(data)
        assert restored.paragraph_indices == processed.paragraph_indices
