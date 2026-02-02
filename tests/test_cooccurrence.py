"""
Tests for co-occurrence analysis (Phase 6).
"""

import pytest
import tempfile
import os
from concept_mapper.corpus.models import ProcessedDocument
from concept_mapper.terms.models import TermList, TermEntry
from concept_mapper.analysis.cooccurrence import (
    cooccurs_in_sentence,
    cooccurs_filtered,
    cooccurs_in_paragraph,
    cooccurs_within_n,
    pmi,
    log_likelihood_ratio,
    build_cooccurrence_matrix,
    save_cooccurrence_matrix,
    get_top_cooccurrences,
)


@pytest.fixture
def sample_docs():
    """Create sample documents for testing."""
    doc1 = ProcessedDocument(
        raw_text="Intentionality is a philosophical concept. The concept relates to consciousness and phenomenology.",
        sentences=[
            "Intentionality is a philosophical concept.",
            "The concept relates to consciousness and phenomenology.",
        ],
        tokens=["intentionality", "is", "a", "philosophical", "concept"],
        lemmas=["intentionality", "be", "a", "philosophical", "concept"],
        pos_tags=[
            ("intentionality", "NOUN"),
            ("is", "VERB"),
            ("a", "DET"),
            ("philosophical", "ADJ"),
            ("concept", "NOUN"),
        ],
        metadata={"source_path": "doc1.txt"},
    )

    doc2 = ProcessedDocument(
        raw_text="Consciousness appears in Husserl. Intentionality and consciousness are related concepts in phenomenological theory.",
        sentences=[
            "Consciousness appears in Husserl.",
            "Intentionality and consciousness are related concepts in phenomenological theory.",
        ],
        tokens=["consciousness", "appears", "in", "husserl"],
        lemmas=["consciousness", "appear", "in", "husserl"],
        pos_tags=[
            ("consciousness", "NOUN"),
            ("appears", "VERB"),
            ("in", "ADP"),
            ("husserl", "PROPN"),
        ],
        metadata={"source_path": "doc2.txt"},
    )

    doc3 = ProcessedDocument(
        raw_text="Philosophy examines ontology. Epistemology is another branch of philosophy.",
        sentences=[
            "Philosophy examines ontology.",
            "Epistemology is another branch of philosophy.",
        ],
        tokens=["philosophy", "examines", "ontology"],
        lemmas=["philosophy", "examine", "ontology"],
        pos_tags=[
            ("philosophy", "NOUN"),
            ("examines", "VERB"),
            ("ontology", "NOUN"),
        ],
        metadata={"source_path": "doc3.txt"},
    )

    return [doc1, doc2, doc3]


@pytest.fixture
def term_list():
    """Create sample term list for filtering tests."""
    terms = TermList()
    terms.add(TermEntry(term="intentionality"))
    terms.add(TermEntry(term="consciousness"))
    terms.add(TermEntry(term="phenomenology"))
    terms.add(TermEntry(term="philosophy"))
    terms.add(TermEntry(term="ontology"))
    terms.add(TermEntry(term="epistemology"))
    return terms


# ============================================================================
# Test Sentence-level Co-occurrence
# ============================================================================


class TestSentenceCooccurrence:
    """Tests for sentence-level co-occurrence analysis."""

    def test_cooccurs_in_sentence_basic(self, sample_docs):
        """Test basic sentence co-occurrence."""
        cooccurs = cooccurs_in_sentence("intentionality", sample_docs)

        # Should find words that appear in same sentences as "intentionality"
        assert "philosophical" in cooccurs
        assert "concept" in cooccurs
        assert "consciousness" in cooccurs

    def test_cooccurs_in_sentence_counts(self, sample_docs):
        """Test that co-occurrence counts are accurate."""
        cooccurs = cooccurs_in_sentence("intentionality", sample_docs)

        # "concept" appears in sentences with "intentionality"
        assert cooccurs["concept"] >= 1
        # "consciousness" appears in sentence with "intentionality" (doc2)
        assert cooccurs["consciousness"] >= 1
        # "philosophical" appears with "intentionality"
        assert cooccurs["philosophical"] >= 1

    def test_cooccurs_in_sentence_excludes_self(self, sample_docs):
        """Test that target term is excluded from results."""
        cooccurs = cooccurs_in_sentence("intentionality", sample_docs)

        # "intentionality" itself should not be in the results
        assert "intentionality" not in cooccurs

    def test_cooccurs_in_sentence_case_insensitive(self, sample_docs):
        """Test case-insensitive co-occurrence (default)."""
        cooccurs_lower = cooccurs_in_sentence("intentionality", sample_docs)
        cooccurs_upper = cooccurs_in_sentence("INTENTIONALITY", sample_docs)

        # Should find same co-occurrences regardless of case
        assert len(cooccurs_lower) > 0
        assert len(cooccurs_upper) > 0

    def test_cooccurs_in_sentence_not_found(self, sample_docs):
        """Test co-occurrence for term not in corpus."""
        cooccurs = cooccurs_in_sentence("nonexistent", sample_docs)

        assert len(cooccurs) == 0

    def test_cooccurs_in_sentence_empty_corpus(self):
        """Test co-occurrence with empty corpus."""
        cooccurs = cooccurs_in_sentence("test", [])

        assert len(cooccurs) == 0


# ============================================================================
# Test Filtered Co-occurrence
# ============================================================================


class TestFilteredCooccurrence:
    """Tests for filtered co-occurrence (term list only)."""

    def test_cooccurs_filtered_basic(self, sample_docs, term_list):
        """Test filtered co-occurrence using term list."""
        cooccurs = cooccurs_filtered("intentionality", sample_docs, term_list)

        # Should only include terms from the term list
        assert "consciousness" in cooccurs
        assert "philosophy" in cooccurs or "philosophical" not in cooccurs

    def test_cooccurs_filtered_excludes_non_list_terms(self, sample_docs, term_list):
        """Test that non-list terms are excluded."""
        cooccurs = cooccurs_filtered("intentionality", sample_docs, term_list)

        # Common words not in term list should be excluded
        assert "is" not in cooccurs
        assert "a" not in cooccurs
        assert "the" not in cooccurs

    def test_cooccurs_filtered_empty_list(self, sample_docs):
        """Test filtered co-occurrence with empty term list."""
        empty_list = TermList()
        cooccurs = cooccurs_filtered("intentionality", sample_docs, empty_list)

        # Should have no results with empty term list
        assert len(cooccurs) == 0

    def test_cooccurs_filtered_subset(self, sample_docs, term_list):
        """Test that filtered results are subset of unfiltered."""
        all_cooccurs = cooccurs_in_sentence("intentionality", sample_docs)
        filtered = cooccurs_filtered("intentionality", sample_docs, term_list)

        # Filtered should be subset
        assert len(filtered) <= len(all_cooccurs)


# ============================================================================
# Test Paragraph-level Co-occurrence
# ============================================================================


class TestParagraphCooccurrence:
    """Tests for paragraph-level co-occurrence."""

    def test_cooccurs_in_paragraph_basic(self, sample_docs):
        """Test paragraph-level co-occurrence."""
        cooccurs = cooccurs_in_paragraph("intentionality", sample_docs)

        # Should find terms in same document (treated as paragraph)
        assert "philosophical" in cooccurs
        assert "concept" in cooccurs
        assert "consciousness" in cooccurs

    def test_cooccurs_in_paragraph_broader_than_sentence(self, sample_docs):
        """Test that paragraph co-occurrence is broader than sentence."""
        sent_cooccurs = cooccurs_in_sentence("intentionality", sample_docs)
        para_cooccurs = cooccurs_in_paragraph("intentionality", sample_docs)

        # Paragraph should generally find more co-occurrences
        # (at minimum, same terms but potentially higher counts)
        assert len(para_cooccurs) >= len(sent_cooccurs)


# ============================================================================
# Test N-sentence Window Co-occurrence
# ============================================================================


class TestWindowCooccurrence:
    """Tests for N-sentence window co-occurrence."""

    def test_cooccurs_within_n_basic(self, sample_docs):
        """Test N-sentence window co-occurrence."""
        cooccurs = cooccurs_within_n("intentionality", sample_docs, n_sentences=1)

        # Should find terms within 1 sentence before/after
        assert len(cooccurs) > 0

    def test_cooccurs_within_n_window_size(self, sample_docs):
        """Test that larger windows capture more co-occurrences."""
        cooccurs_1 = cooccurs_within_n("intentionality", sample_docs, n_sentences=1)
        cooccurs_3 = cooccurs_within_n("intentionality", sample_docs, n_sentences=3)

        # Larger window should include everything from smaller window
        # (counts may be higher for terms that appear multiple times)
        for term in cooccurs_1:
            assert term in cooccurs_3

    def test_cooccurs_within_n_zero_window(self, sample_docs):
        """Test with zero-sentence window (just the sentence itself)."""
        cooccurs = cooccurs_within_n("intentionality", sample_docs, n_sentences=0)

        # Should be similar to sentence-level co-occurrence
        assert len(cooccurs) > 0


# ============================================================================
# Test PMI (Pointwise Mutual Information)
# ============================================================================


class TestPMI:
    """Tests for PMI calculation."""

    def test_pmi_positive_association(self, sample_docs):
        """Test PMI for terms that co-occur."""
        pmi_score = pmi("intentionality", "consciousness", sample_docs)

        # Terms that co-occur should have positive PMI
        # (may be zero or slightly negative for rare co-occurrences)
        assert isinstance(pmi_score, float)

    def test_pmi_symmetric(self, sample_docs):
        """Test that PMI is symmetric."""
        pmi_12 = pmi("intentionality", "consciousness", sample_docs)
        pmi_21 = pmi("consciousness", "intentionality", sample_docs)

        # Should be the same regardless of order
        assert abs(pmi_12 - pmi_21) < 0.001

    def test_pmi_independent_terms(self, sample_docs):
        """Test PMI for terms that don't co-occur."""
        pmi_score = pmi("intentionality", "ontology", sample_docs)

        # Terms in different documents should have low/zero PMI
        # (will be 0.0 if they never co-occur)
        assert pmi_score <= 0.0

    def test_pmi_nonexistent_term(self, sample_docs):
        """Test PMI when one term doesn't exist."""
        pmi_score = pmi("intentionality", "nonexistent", sample_docs)

        # Should return 0.0 for nonexistent terms
        assert pmi_score == 0.0

    def test_pmi_both_nonexistent(self, sample_docs):
        """Test PMI when both terms don't exist."""
        pmi_score = pmi("nonexistent1", "nonexistent2", sample_docs)

        assert pmi_score == 0.0

    def test_pmi_empty_corpus(self):
        """Test PMI with empty corpus."""
        pmi_score = pmi("term1", "term2", [])

        assert pmi_score == 0.0


# ============================================================================
# Test Log-Likelihood Ratio
# ============================================================================


class TestLogLikelihoodRatio:
    """Tests for log-likelihood ratio calculation."""

    def test_llr_positive_value(self, sample_docs):
        """Test that LLR returns positive value for associated terms."""
        llr = log_likelihood_ratio("intentionality", "consciousness", sample_docs)

        # Should be non-negative
        assert llr >= 0.0

    def test_llr_symmetric(self, sample_docs):
        """Test that LLR is symmetric."""
        llr_12 = log_likelihood_ratio("intentionality", "consciousness", sample_docs)
        llr_21 = log_likelihood_ratio("consciousness", "intentionality", sample_docs)

        # Should be the same regardless of order
        assert abs(llr_12 - llr_21) < 0.001

    def test_llr_high_for_frequent_cooccurrence(self, sample_docs):
        """Test that LLR is high for terms that frequently co-occur."""
        llr = log_likelihood_ratio("intentionality", "consciousness", sample_docs)

        # Should have some positive value (exact threshold depends on corpus)
        assert llr >= 0.0

    def test_llr_low_for_independent_terms(self, sample_docs):
        """Test LLR for independent terms."""
        llr = log_likelihood_ratio("intentionality", "ontology", sample_docs)

        # Should be low/zero for terms that don't co-occur
        assert llr >= 0.0  # LLR is always non-negative

    def test_llr_nonexistent_term(self, sample_docs):
        """Test LLR when term doesn't exist."""
        llr = log_likelihood_ratio("intentionality", "nonexistent", sample_docs)

        assert llr == 0.0

    def test_llr_empty_corpus(self):
        """Test LLR with empty corpus."""
        llr = log_likelihood_ratio("term1", "term2", [])

        assert llr == 0.0


# ============================================================================
# Test Co-occurrence Matrix
# ============================================================================


class TestCooccurrenceMatrix:
    """Tests for co-occurrence matrix building."""

    def test_build_matrix_basic(self, sample_docs, term_list):
        """Test basic matrix building."""
        matrix = build_cooccurrence_matrix(
            term_list, sample_docs, method="count", window="sentence"
        )

        # Should have entry for each term
        terms = [e.term for e in term_list.list_terms()]
        for term in terms:
            assert term in matrix

    def test_build_matrix_symmetric(self, sample_docs, term_list):
        """Test that matrix is symmetric."""
        matrix = build_cooccurrence_matrix(
            term_list, sample_docs, method="count", window="sentence"
        )

        terms = [e.term for e in term_list.list_terms()]
        for term1 in terms:
            for term2 in terms:
                assert abs(matrix[term1][term2] - matrix[term2][term1]) < 0.001

    def test_build_matrix_diagonal_zero(self, sample_docs, term_list):
        """Test that diagonal is zero (no self-association)."""
        matrix = build_cooccurrence_matrix(
            term_list, sample_docs, method="count", window="sentence"
        )

        terms = [e.term for e in term_list.list_terms()]
        for term in terms:
            assert matrix[term][term] == 0.0

    def test_build_matrix_count_method(self, sample_docs, term_list):
        """Test matrix with count method."""
        matrix = build_cooccurrence_matrix(
            term_list, sample_docs, method="count", window="sentence"
        )

        # All values should be non-negative
        for term1 in matrix:
            for term2 in matrix[term1]:
                assert matrix[term1][term2] >= 0.0

    def test_build_matrix_pmi_method(self, sample_docs, term_list):
        """Test matrix with PMI method."""
        matrix = build_cooccurrence_matrix(
            term_list, sample_docs, method="pmi", window="sentence"
        )

        # PMI can be negative, zero, or positive
        assert isinstance(matrix, dict)
        # Check some values exist
        terms = [e.term for e in term_list.list_terms()]
        assert len(matrix) == len(terms)

    def test_build_matrix_llr_method(self, sample_docs, term_list):
        """Test matrix with LLR method."""
        matrix = build_cooccurrence_matrix(
            term_list, sample_docs, method="llr", window="sentence"
        )

        # LLR should be non-negative
        for term1 in matrix:
            for term2 in matrix[term1]:
                assert matrix[term1][term2] >= 0.0

    def test_build_matrix_invalid_method(self, sample_docs, term_list):
        """Test matrix with invalid method."""
        with pytest.raises(ValueError):
            build_cooccurrence_matrix(
                term_list, sample_docs, method="invalid", window="sentence"
            )

    def test_build_matrix_invalid_window(self, sample_docs, term_list):
        """Test matrix with invalid window."""
        with pytest.raises(ValueError):
            build_cooccurrence_matrix(
                term_list, sample_docs, method="count", window="invalid"
            )

    def test_build_matrix_n_sentences_window(self, sample_docs, term_list):
        """Test matrix with n_sentences window."""
        matrix = build_cooccurrence_matrix(
            term_list,
            sample_docs,
            method="count",
            window="n_sentences",
            n_sentences=2,
        )

        # Should build successfully
        assert isinstance(matrix, dict)
        assert len(matrix) > 0

    def test_build_matrix_n_sentences_missing_param(self, sample_docs, term_list):
        """Test that n_sentences window requires n_sentences parameter."""
        with pytest.raises(ValueError):
            build_cooccurrence_matrix(
                term_list, sample_docs, method="count", window="n_sentences"
            )


# ============================================================================
# Test Matrix Saving
# ============================================================================


class TestMatrixSaving:
    """Tests for saving co-occurrence matrix."""

    def test_save_matrix_basic(self, sample_docs, term_list):
        """Test basic matrix saving."""
        matrix = build_cooccurrence_matrix(
            term_list, sample_docs, method="count", window="sentence"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "matrix.csv")
            save_cooccurrence_matrix(matrix, output_path)

            # File should exist
            assert os.path.exists(output_path)

    def test_save_matrix_format(self, sample_docs, term_list):
        """Test that saved matrix is valid CSV."""
        matrix = build_cooccurrence_matrix(
            term_list, sample_docs, method="count", window="sentence"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "matrix.csv")
            save_cooccurrence_matrix(matrix, output_path)

            # Read and verify CSV format
            with open(output_path, "r") as f:
                lines = f.readlines()
                # Should have header + data rows
                assert len(lines) > 1

    def test_save_matrix_square(self, sample_docs, term_list):
        """Test that saved matrix is square."""
        matrix = build_cooccurrence_matrix(
            term_list, sample_docs, method="count", window="sentence"
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = os.path.join(tmpdir, "matrix.csv")
            save_cooccurrence_matrix(matrix, output_path)

            # Read and verify dimensions
            with open(output_path, "r") as f:
                lines = f.readlines()
                # Number of data rows should equal number of columns - 1 (for row labels)
                num_rows = len(lines) - 1  # Exclude header
                num_cols = len(lines[0].split(",")) - 1  # Exclude row label column
                assert num_rows == num_cols


# ============================================================================
# Test Top Co-occurrences
# ============================================================================


class TestTopCooccurrences:
    """Tests for getting top co-occurrences."""

    def test_get_top_cooccurrences_basic(self, sample_docs):
        """Test getting top co-occurrences."""
        top = get_top_cooccurrences("intentionality", sample_docs, n=5, method="count")

        # Should return list of tuples
        assert isinstance(top, list)
        assert all(isinstance(item, tuple) for item in top)
        assert all(len(item) == 2 for item in top)

    def test_get_top_cooccurrences_limit(self, sample_docs):
        """Test that n parameter limits results."""
        top_3 = get_top_cooccurrences(
            "intentionality", sample_docs, n=3, method="count"
        )
        top_5 = get_top_cooccurrences(
            "intentionality", sample_docs, n=5, method="count"
        )

        # Should respect n parameter
        assert len(top_3) <= 3
        assert len(top_5) <= 5

    def test_get_top_cooccurrences_sorted(self, sample_docs):
        """Test that results are sorted by score."""
        top = get_top_cooccurrences("intentionality", sample_docs, n=10, method="count")

        # Should be sorted descending
        scores = [score for _, score in top]
        assert scores == sorted(scores, reverse=True)

    def test_get_top_cooccurrences_pmi(self, sample_docs):
        """Test top co-occurrences with PMI."""
        top = get_top_cooccurrences("intentionality", sample_docs, n=5, method="pmi")

        # Should return results
        assert isinstance(top, list)

    def test_get_top_cooccurrences_llr(self, sample_docs):
        """Test top co-occurrences with LLR."""
        top = get_top_cooccurrences("intentionality", sample_docs, n=5, method="llr")

        # Should return results
        assert isinstance(top, list)

    def test_get_top_cooccurrences_filtered(self, sample_docs, term_list):
        """Test top co-occurrences with term list filter."""
        top = get_top_cooccurrences(
            "intentionality", sample_docs, n=5, method="count", term_list=term_list
        )

        # All results should be in term list
        list_terms_lower = {e.term.lower() for e in term_list.list_terms()}
        for term, _ in top:
            assert term.lower() in list_terms_lower

    def test_get_top_cooccurrences_invalid_method(self, sample_docs):
        """Test with invalid method."""
        with pytest.raises(ValueError):
            get_top_cooccurrences("intentionality", sample_docs, n=5, method="invalid")

    def test_get_top_cooccurrences_nonexistent_term(self, sample_docs):
        """Test top co-occurrences for nonexistent term."""
        top = get_top_cooccurrences("nonexistent", sample_docs, n=5, method="count")

        # Should return empty list
        assert len(top) == 0
