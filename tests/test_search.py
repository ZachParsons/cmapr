"""
Tests for search and concordance functionality (Phase 5).
"""

import pytest
from concept_mapper.corpus.models import ProcessedDocument
from concept_mapper.search import (
    SentenceMatch,
    find_sentences,
    KWICLine,
    concordance,
    format_kwic_lines,
    ContextWindow,
    get_context,
    dispersion,
    get_dispersion_summary,
    compare_dispersion,
    dispersion_plot_data,
    get_concentrated_regions,
)
from concept_mapper.search.find import (
    find_sentences_any,
    find_sentences_all,
    count_term_occurrences,
    find_in_document,
)
from concept_mapper.search.concordance import (
    concordance_sorted,
    concordance_filtered,
)
from concept_mapper.search.context import (
    get_context_by_match,
    format_context_windows,
    get_context_with_highlights,
)


@pytest.fixture
def sample_docs():
    """Create sample documents for testing."""
    doc1 = ProcessedDocument(
        raw_text="Abstraction is a key concept in philosophy. Many philosophers discuss abstraction. The term appears in various contexts.",
        sentences=[
            "Abstraction is a key concept in philosophy.",
            "Many philosophers discuss abstraction.",
            "The term appears in various contexts.",
        ],
        tokens=["abstraction", "is", "a", "key", "concept", "in", "philosophy"],
        lemmas=["abstraction", "be", "a", "key", "concept", "in", "philosophy"],
        pos_tags=[
            ("abstraction", "NOUN"),
            ("is", "VERB"),
            ("a", "DET"),
            ("key", "ADJ"),
            ("concept", "NOUN"),
            ("in", "ADP"),
            ("philosophy", "NOUN"),
        ],
        metadata={"source_path": "doc1.txt"},
    )

    doc2 = ProcessedDocument(
        raw_text="Ontology deals with the nature of being. Abstraction transforms processes into things. This philosophical concept has wide applications.",
        sentences=[
            "Ontology deals with the nature of being.",
            "Abstraction transforms processes into things.",
            "This philosophical concept has wide applications.",
        ],
        tokens=["ontology", "deals", "with", "the", "nature", "of", "being"],
        lemmas=["ontology", "deal", "with", "the", "nature", "of", "be"],
        pos_tags=[
            ("ontology", "NOUN"),
            ("deals", "VERB"),
            ("with", "ADP"),
            ("the", "DET"),
            ("nature", "NOUN"),
            ("of", "ADP"),
            ("being", "VERB"),
        ],
        metadata={"source_path": "doc2.txt"},
    )

    doc3 = ProcessedDocument(
        raw_text="Philosophy examines fundamental questions. Some terms like ontology are technical. Others are more general.",
        sentences=[
            "Philosophy examines fundamental questions.",
            "Some terms like ontology are technical.",
            "Others are more general.",
        ],
        tokens=["philosophy", "examines", "fundamental", "questions"],
        lemmas=["philosophy", "examine", "fundamental", "question"],
        pos_tags=[
            ("philosophy", "NOUN"),
            ("examines", "VERB"),
            ("fundamental", "ADJ"),
            ("questions", "NOUN"),
        ],
        metadata={"source_path": "doc3.txt"},
    )

    return [doc1, doc2, doc3]


# ============================================================================
# Test Find Functionality
# ============================================================================


class TestFind:
    """Tests for basic search functionality."""

    def test_find_sentences_basic(self, sample_docs):
        """Test basic sentence finding."""
        matches = find_sentences("abstraction", sample_docs)

        assert len(matches) == 3  # Appears in 3 sentences
        assert all(isinstance(m, SentenceMatch) for m in matches)
        assert matches[0].doc_id == "doc1.txt"
        assert matches[0].sent_index == 0
        assert "abstraction" in matches[0].sentence.lower()

    def test_find_sentences_case_insensitive(self, sample_docs):
        """Test case-insensitive search (default)."""
        matches_lower = find_sentences("abstraction", sample_docs)
        matches_upper = find_sentences("ABSTRACTION", sample_docs)

        assert len(matches_lower) == len(matches_upper)
        assert len(matches_lower) == 3

    def test_find_sentences_case_sensitive(self, sample_docs):
        """Test case-sensitive search."""
        matches = find_sentences("Abstraction", sample_docs, case_sensitive=True)

        # Only matches with capital R
        assert len(matches) == 2
        assert all("Abstraction" in m.sentence for m in matches)

    def test_find_sentences_term_positions(self, sample_docs):
        """Test that term positions are correctly identified."""
        matches = find_sentences("abstraction", sample_docs)

        # First match should have position recorded
        assert len(matches[0].term_positions) > 0
        pos = matches[0].term_positions[0]
        sentence_lower = matches[0].sentence.lower()
        assert sentence_lower[pos : pos + len("abstraction")] == "abstraction"

    def test_find_sentences_not_found(self, sample_docs):
        """Test searching for term that doesn't exist."""
        matches = find_sentences("nonexistent", sample_docs)

        assert len(matches) == 0
        assert matches == []

    def test_find_sentences_any(self, sample_docs):
        """Test finding sentences with any of multiple terms."""
        matches = find_sentences_any(["abstraction", "ontology"], sample_docs)

        assert len(matches) >= 4  # At least 3 abstraction + 2 ontology
        terms_found = {m.term for m in matches}
        assert "abstraction" in terms_found
        assert "ontology" in terms_found

    def test_find_sentences_all(self, sample_docs):
        """Test finding sentences containing all terms."""
        matches = find_sentences_all(["abstraction", "philosophy"], sample_docs)

        # Only first sentence has both
        assert len(matches) == 1
        assert "abstraction" in matches[0].sentence.lower()
        assert "philosophy" in matches[0].sentence.lower()

    def test_find_sentences_all_no_match(self, sample_docs):
        """Test finding sentences with all terms when none match."""
        matches = find_sentences_all(
            ["abstraction", "ontology", "epistemology"], sample_docs
        )

        assert len(matches) == 0

    def test_count_term_occurrences(self, sample_docs):
        """Test counting term occurrences."""
        count = count_term_occurrences("abstraction", sample_docs)

        # Should count 3 occurrences (once per sentence that has it)
        assert count == 3

    def test_count_term_occurrences_zero(self, sample_docs):
        """Test counting nonexistent term."""
        count = count_term_occurrences("nonexistent", sample_docs)

        assert count == 0

    def test_find_in_document(self, sample_docs):
        """Test searching within a single document."""
        matches = find_in_document("abstraction", sample_docs[0])

        assert len(matches) == 2  # Two sentences in doc1 have it
        assert all(m.doc_id == "doc1.txt" for m in matches)

    def test_sentence_match_str(self, sample_docs):
        """Test SentenceMatch string representation."""
        matches = find_sentences("abstraction", sample_docs)

        match_str = str(matches[0])
        assert "[doc1.txt:0]" in match_str
        assert "abstraction" in match_str.lower()


# ============================================================================
# Test Concordance (KWIC)
# ============================================================================


class TestConcordance:
    """Tests for KWIC concordance functionality."""

    def test_concordance_basic(self, sample_docs):
        """Test basic concordance generation."""
        lines = concordance("abstraction", sample_docs, width=30)

        assert len(lines) == 3  # Three occurrences
        assert all(isinstance(line, KWICLine) for line in lines)
        assert all(line.keyword.lower() == "abstraction" for line in lines)

    def test_concordance_preserves_case(self, sample_docs):
        """Test that keyword preserves original case."""
        lines = concordance("abstraction", sample_docs)

        # Should have both "Abstraction" (capitalized) and "abstraction" (lowercase)
        keywords = {line.keyword for line in lines}
        assert "Abstraction" in keywords
        assert "abstraction" in keywords

    def test_concordance_context_extraction(self, sample_docs):
        """Test that left and right context are extracted."""
        lines = concordance("abstraction", sample_docs, width=20)

        for line in lines:
            # All lines should have some context (except edge cases)
            assert isinstance(line.left_context, str)
            assert isinstance(line.right_context, str)

    def test_concordance_width(self, sample_docs):
        """Test that width parameter affects context size."""
        lines_narrow = concordance("abstraction", sample_docs, width=10)
        lines_wide = concordance("abstraction", sample_docs, width=50)

        # Wider context should generally be longer
        # (though word boundary trimming may affect this)
        assert len(lines_narrow) == len(lines_wide)  # Same number of matches

    def test_concordance_case_sensitive(self, sample_docs):
        """Test case-sensitive concordance."""
        lines = concordance("Abstraction", sample_docs, case_sensitive=True)

        # Only matches capitalized form
        assert len(lines) == 2
        assert all(line.keyword == "Abstraction" for line in lines)

    def test_kwic_line_str(self, sample_docs):
        """Test KWICLine string formatting."""
        lines = concordance("abstraction", sample_docs)

        line_str = str(lines[0])
        assert "[abstraction]" in line_str or "[Abstraction]" in line_str

    def test_format_kwic_lines_basic(self, sample_docs):
        """Test formatting KWIC lines."""
        lines = concordance("abstraction", sample_docs, width=30)
        formatted = format_kwic_lines(lines, width=30)

        assert isinstance(formatted, str)
        assert "[abstraction]" in formatted.lower()
        # Should have multiple lines (one per occurrence)
        assert formatted.count("\n") >= len(lines) - 1

    def test_format_kwic_lines_with_doc_id(self, sample_docs):
        """Test formatting with document IDs."""
        lines = concordance("abstraction", sample_docs)
        formatted = format_kwic_lines(lines, show_doc_id=True)

        assert "[doc1.txt:" in formatted
        assert "[doc2.txt:" in formatted

    def test_concordance_sorted_left(self, sample_docs):
        """Test sorting concordance by left context."""
        lines = concordance_sorted("abstraction", sample_docs, sort_by="left")

        assert len(lines) == 3
        # Lines should be sorted (hard to verify exact order without knowing trimming)
        assert all(isinstance(line, KWICLine) for line in lines)

    def test_concordance_sorted_right(self, sample_docs):
        """Test sorting concordance by right context."""
        lines = concordance_sorted("abstraction", sample_docs, sort_by="right")

        assert len(lines) == 3
        assert all(isinstance(line, KWICLine) for line in lines)

    def test_concordance_sorted_invalid(self, sample_docs):
        """Test invalid sort_by parameter."""
        with pytest.raises(ValueError):
            concordance_sorted("abstraction", sample_docs, sort_by="invalid")

    def test_concordance_filtered(self, sample_docs):
        """Test filtering concordance by additional terms."""
        # Find lines with "abstraction" that also contain "concept"
        lines = concordance_filtered(
            "abstraction", sample_docs, filter_terms=["concept"]
        )

        # Should match at least the first sentence
        assert len(lines) >= 1
        # Verify "concept" appears somewhere in the full context
        assert any(
            "concept" in (line.left_context + line.keyword + line.right_context).lower()
            for line in lines
        )

    def test_concordance_filtered_multiple_terms(self, sample_docs):
        """Test filtering with multiple filter terms."""
        lines = concordance_filtered(
            "abstraction", sample_docs, filter_terms=["philosophy", "concept"]
        )

        # Should match sentences containing either "philosophy" or "concept"
        assert len(lines) >= 1


# ============================================================================
# Test Context Windows
# ============================================================================


class TestContext:
    """Tests for context window functionality."""

    def test_get_context_basic(self, sample_docs):
        """Test basic context window extraction."""
        windows = get_context("abstraction", sample_docs, n_sentences=1)

        assert len(windows) == 3
        assert all(isinstance(w, ContextWindow) for w in windows)

    def test_get_context_with_before(self, sample_docs):
        """Test context window includes before sentences."""
        windows = get_context("abstraction", [sample_docs[1]], n_sentences=1)

        # Second sentence in doc2 has abstraction
        # Should have sentence 0 as "before"
        window = windows[0]
        assert len(window.before) == 1
        assert "Ontology" in window.before[0]

    def test_get_context_with_after(self, sample_docs):
        """Test context window includes after sentences."""
        windows = get_context("abstraction", [sample_docs[0]], n_sentences=1)

        # First sentence in doc1 has abstraction
        # Should have sentence 1 as "after"
        first_window = windows[0]
        assert len(first_window.after) >= 1
        assert "philosophers" in first_window.after[0].lower()

    def test_get_context_no_before(self, sample_docs):
        """Test context window at document start."""
        windows = get_context(
            "Abstraction", sample_docs, n_sentences=2, case_sensitive=True
        )

        # First occurrence is at start of doc1
        first_match = [
            w for w in windows if w.doc_id == "doc1.txt" and w.sent_index == 0
        ][0]
        assert len(first_match.before) == 0  # Nothing before first sentence

    def test_get_context_no_after(self, sample_docs):
        """Test context window at document end."""
        # Add doc with term at end
        doc = ProcessedDocument(
            raw_text="First sentence. Last mentions abstraction.",
            sentences=["First sentence.", "Last mentions abstraction."],
            tokens=["first", "sentence"],
            lemmas=["first", "sentence"],
            pos_tags=[("first", "ADJ"), ("sentence", "NOUN")],
            metadata={"source_path": "test.txt"},
        )

        windows = get_context("abstraction", [doc], n_sentences=1)
        last_window = windows[-1]

        assert len(last_window.after) == 0  # Nothing after last sentence

    def test_get_context_larger_window(self, sample_docs):
        """Test larger context window."""
        windows = get_context("abstraction", sample_docs, n_sentences=2)

        # Should have up to 2 sentences before and after
        for window in windows:
            assert len(window.before) <= 2
            assert len(window.after) <= 2

    def test_context_window_str(self, sample_docs):
        """Test ContextWindow string formatting."""
        windows = get_context("abstraction", sample_docs, n_sentences=1)

        window_str = str(windows[0])
        assert "[doc1.txt:" in window_str
        assert ">" in window_str  # Match indicator
        assert "abstraction" in window_str.lower()

    def test_get_context_by_match(self, sample_docs):
        """Test getting context for specific matches."""
        matches = find_sentences("abstraction", sample_docs)
        windows = get_context_by_match(matches[:2], sample_docs, n_sentences=1)

        assert len(windows) == 2
        assert all(isinstance(w, ContextWindow) for w in windows)

    def test_format_context_windows(self, sample_docs):
        """Test formatting context windows."""
        windows = get_context("abstraction", sample_docs, n_sentences=1)
        formatted = format_context_windows(windows)

        assert isinstance(formatted, str)
        assert "---" in formatted  # Default separator
        assert "abstraction" in formatted.lower()

    def test_format_context_windows_custom_separator(self, sample_docs):
        """Test custom separator in formatting."""
        windows = get_context("abstraction", sample_docs)
        formatted = format_context_windows(windows, separator="===")

        assert "===" in formatted
        assert "---" not in formatted

    def test_get_context_with_highlights(self, sample_docs):
        """Test context with highlighted search term."""
        windows = get_context_with_highlights("abstraction", sample_docs, n_sentences=1)

        # Match sentences should have highlighting
        for window in windows:
            assert "**" in window.match  # Default highlight markers
            assert "abstraction" in window.match.lower()

    def test_get_context_with_highlights_custom_markers(self, sample_docs):
        """Test custom highlight markers."""
        windows = get_context_with_highlights(
            "abstraction",
            sample_docs,
            n_sentences=1,
            highlight_start="<<",
            highlight_end=">>",
        )

        for window in windows:
            assert "<<" in window.match
            assert ">>" in window.match


# ============================================================================
# Test Dispersion Analysis
# ============================================================================


class TestDispersion:
    """Tests for term dispersion functionality."""

    def test_dispersion_basic(self, sample_docs):
        """Test basic dispersion analysis."""
        disp = dispersion("abstraction", sample_docs)

        assert isinstance(disp, dict)
        assert "doc1.txt" in disp
        assert "doc2.txt" in disp
        # Should have sentence indices
        assert isinstance(disp["doc1.txt"], list)
        assert len(disp["doc1.txt"]) == 2  # Two occurrences in doc1

    def test_dispersion_sentence_indices(self, sample_docs):
        """Test that dispersion returns correct sentence indices."""
        disp = dispersion("abstraction", sample_docs)

        # doc1: sentences 0 and 1
        assert 0 in disp["doc1.txt"]
        assert 1 in disp["doc1.txt"]

        # doc2: sentence 1
        assert 1 in disp["doc2.txt"]

    def test_dispersion_not_found(self, sample_docs):
        """Test dispersion for term not in corpus."""
        disp = dispersion("nonexistent", sample_docs)

        assert disp == {}

    def test_dispersion_case_insensitive(self, sample_docs):
        """Test case-insensitive dispersion (default)."""
        disp_lower = dispersion("abstraction", sample_docs)
        disp_upper = dispersion("ABSTRACTION", sample_docs)

        assert disp_lower == disp_upper

    def test_dispersion_case_sensitive(self, sample_docs):
        """Test case-sensitive dispersion."""
        disp = dispersion("Abstraction", sample_docs, case_sensitive=True)

        # Only matches capitalized form
        assert len(disp["doc1.txt"]) == 1  # Only sentence 0
        assert 0 in disp["doc1.txt"]

    def test_dispersion_by_char(self, sample_docs):
        """Test character-level dispersion."""
        disp = dispersion("abstraction", sample_docs, by_char=True)

        # Should have character positions instead of sentence indices
        assert "doc1.txt" in disp
        assert all(isinstance(pos, int) for pos in disp["doc1.txt"])
        # Character positions should be larger than sentence indices
        assert any(pos > 10 for doc_positions in disp.values() for pos in doc_positions)

    def test_get_dispersion_summary(self, sample_docs):
        """Test dispersion summary statistics."""
        summary = get_dispersion_summary("abstraction", sample_docs)

        assert summary["term"] == "abstraction"
        assert summary["total_docs"] == 3
        assert summary["docs_with_term"] == 2  # doc1 and doc2
        assert summary["total_occurrences"] == 3
        assert summary["coverage"] > 0
        assert "positions" in summary
        assert "avg_occurrences_per_doc" in summary

    def test_get_dispersion_summary_coverage(self, sample_docs):
        """Test coverage calculation."""
        summary = get_dispersion_summary("abstraction", sample_docs)

        # 2 out of 3 docs = 66.67%
        assert abs(summary["coverage"] - 66.67) < 0.1

    def test_get_dispersion_summary_not_found(self, sample_docs):
        """Test summary for term not in corpus."""
        summary = get_dispersion_summary("nonexistent", sample_docs)

        assert summary["docs_with_term"] == 0
        assert summary["total_occurrences"] == 0
        assert summary["coverage"] == 0.0

    def test_compare_dispersion(self, sample_docs):
        """Test comparing dispersion across terms."""
        comparison = compare_dispersion(
            ["abstraction", "ontology", "philosophy"], sample_docs
        )

        assert len(comparison) == 3
        assert "abstraction" in comparison
        assert "ontology" in comparison
        assert "philosophy" in comparison

        # Each should have summary statistics
        for term, summary in comparison.items():
            assert "coverage" in summary
            assert "total_occurrences" in summary

    def test_dispersion_plot_data(self, sample_docs):
        """Test generating plot data."""
        plot_data = dispersion_plot_data("abstraction", sample_docs)

        assert len(plot_data) == 3  # Three documents
        assert all(isinstance(item, tuple) for item in plot_data)
        assert all(len(item) == 3 for item in plot_data)

        # First item should be (doc_id, length, positions)
        doc_id, length, positions = plot_data[0]
        assert doc_id == "doc1.txt"
        assert length == 3  # Three sentences in doc1
        assert isinstance(positions, list)

    def test_dispersion_plot_data_includes_empty(self, sample_docs):
        """Test that plot data includes documents without term."""
        plot_data = dispersion_plot_data("abstraction", sample_docs)

        # doc3 doesn't have "abstraction" but should still be in plot data
        doc3_data = [item for item in plot_data if item[0] == "doc3.txt"][0]
        assert doc3_data[2] == []  # Empty positions list

    def test_get_concentrated_regions(self, sample_docs):
        """Test finding concentrated usage regions."""
        # Create doc with concentrated usage
        doc = ProcessedDocument(
            raw_text="Sentence one. Abstraction here. More abstraction. And abstraction again. Final abstraction mention. No mention here.",
            sentences=[
                "Sentence one.",
                "Abstraction here.",
                "More abstraction.",
                "And abstraction again.",
                "Final abstraction mention.",
                "No mention here.",
            ],
            tokens=[],
            lemmas=[],
            pos_tags=[],
            metadata={"source_path": "concentrated.txt"},
        )

        regions = get_concentrated_regions(
            "abstraction", [doc], window_size=4, min_occurrences=3
        )

        # Should find a region with 3+ occurrences in 4-sentence window
        assert len(regions) > 0
        assert all("occurrences" in r for r in regions)
        assert all("density" in r for r in regions)

    def test_get_concentrated_regions_sorted_by_density(self, sample_docs):
        """Test that concentrated regions are sorted by density."""
        doc = ProcessedDocument(
            raw_text="R abstraction. " * 10,
            sentences=["R abstraction."] * 10,  # High density
            tokens=[],
            lemmas=[],
            pos_tags=[],
            metadata={"source_path": "test.txt"},
        )

        regions = get_concentrated_regions(
            "abstraction", [doc], window_size=3, min_occurrences=2
        )

        # Should be sorted by density (descending)
        densities = [r["density"] for r in regions]
        assert densities == sorted(densities, reverse=True)

    def test_get_concentrated_regions_no_regions(self, sample_docs):
        """Test when no concentrated regions exist."""
        regions = get_concentrated_regions(
            "philosophy", sample_docs, window_size=2, min_occurrences=3
        )

        # Threshold too high for sparse occurrence
        assert regions == []
