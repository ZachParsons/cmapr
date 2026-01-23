"""
Test that the sample corpus matches its specification.

This test validates that the documented frequencies in data/sample/CORPUS_SPEC.md
match the actual frequencies found in the sample corpus files.
"""

import pytest
import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path to import pos_tagger
sys.path.insert(0, str(Path(__file__).parent.parent))
import pos_tagger as pt


class TestSampleCorpusFrequencies:
    """Test that documented frequencies match actual frequencies."""

    @pytest.fixture
    def sample_files(self):
        """Return paths to sample corpus files."""
        base = Path(__file__).parent.parent / "data" / "sample"
        return {
            "sample1": base / "sample1_dialectics.txt",
            "sample2": base / "sample2_epistemology.txt",
            "sample3": base / "sample3_ontology.txt",
        }

    @pytest.fixture
    def term_counts(self, sample_files):
        """Load and count terms in all sample files."""
        counts = {}
        for name, path in sample_files.items():
            text = pt.load_text(str(path))
            tokens = pt.tokenize_words(text)
            counts[name] = Counter([t.lower() for t in tokens])
        return counts

    def test_sample1_token_count(self, sample_files):
        """Test that sample1 has expected token count."""
        text = pt.load_text(str(sample_files["sample1"]))
        tokens = pt.tokenize_words(text)
        assert len(tokens) == 170, f"Expected 170 tokens, got {len(tokens)}"

    def test_sample2_token_count(self, sample_files):
        """Test that sample2 has expected token count."""
        text = pt.load_text(str(sample_files["sample2"]))
        tokens = pt.tokenize_words(text)
        assert len(tokens) == 172, f"Expected 172 tokens, got {len(tokens)}"

    def test_sample3_token_count(self, sample_files):
        """Test that sample3 has expected token count."""
        text = pt.load_text(str(sample_files["sample3"]))
        tokens = pt.tokenize_words(text)
        assert len(tokens) == 182, f"Expected 182 tokens, got {len(tokens)}"

    # Sample 1 term frequencies
    def test_sample1_dasein_flux(self, term_counts):
        """Test dasein-flux frequency in sample1."""
        assert term_counts["sample1"]["dasein-flux"] == 6

    def test_sample1_geist_praxis(self, term_counts):
        """Test geist-praxis frequency in sample1."""
        assert term_counts["sample1"]["geist-praxis"] == 7

    def test_sample1_abstraction(self, term_counts):
        """Test abstraction frequency in sample1."""
        assert term_counts["sample1"]["abstraction"] == 5

    def test_sample1_totality_consciousness(self, term_counts):
        """Test totality-consciousness frequency in sample1."""
        assert term_counts["sample1"]["totality-consciousness"] == 5

    # Sample 2 term frequencies
    def test_sample2_noetic_intuition(self, term_counts):
        """Test noetic-intuition frequency in sample2."""
        assert term_counts["sample2"]["noetic-intuition"] == 4

    def test_sample2_categorial(self, term_counts):
        """Test categorial frequency in sample2."""
        assert term_counts["sample2"]["categorial"] == 4

    def test_sample2_synthesis(self, term_counts):
        """Test synthesis frequency in sample2."""
        assert term_counts["sample2"]["synthesis"] == 4

    def test_sample2_intentionality_vectors(self, term_counts):
        """Test intentionality-vectors frequency in sample2."""
        assert term_counts["sample2"]["intentionality-vectors"] == 5

    def test_sample2_eidetic(self, term_counts):
        """Test eidetic frequency in sample2."""
        assert term_counts["sample2"]["eidetic"] == 5

    def test_sample2_reduction(self, term_counts):
        """Test reduction frequency in sample2."""
        assert term_counts["sample2"]["reduction"] == 5

    def test_sample2_lifeworld_horizons(self, term_counts):
        """Test lifeworld-horizons frequency in sample2."""
        assert term_counts["sample2"]["lifeworld-horizons"] == 4

    # Sample 3 term frequencies
    def test_sample3_being_toward_finitude(self, term_counts):
        """Test being-toward-finitude frequency in sample3."""
        assert term_counts["sample3"]["being-toward-finitude"] == 6

    def test_sample3_existential_thrownness(self, term_counts):
        """Test existential-thrownness frequency in sample3."""
        assert term_counts["sample3"]["existential-thrownness"] == 6

    def test_sample3_worldhood_disclosure(self, term_counts):
        """Test worldhood-disclosure frequency in sample3."""
        assert term_counts["sample3"]["worldhood-disclosure"] == 5

    def test_sample3_hermeneutic_circle(self, term_counts):
        """Test hermeneutic-circle frequency in sample3."""
        assert term_counts["sample3"]["hermeneutic-circle"] == 5

    def test_sample3_resolute(self, term_counts):
        """Test resolute frequency in sample3."""
        assert term_counts["sample3"]["resolute"] == 4

    def test_sample3_dasein_flux(self, term_counts):
        """Test dasein-flux frequency in sample3 (cross-file term)."""
        assert term_counts["sample3"]["dasein-flux"] == 1

    def test_sample3_geist_praxis(self, term_counts):
        """Test geist-praxis frequency in sample3 (cross-file term)."""
        assert term_counts["sample3"]["geist-praxis"] == 1

    # Cross-file totals
    def test_cross_file_dasein_flux_total(self, term_counts):
        """Test total dasein-flux frequency across all files."""
        total = (
            term_counts["sample1"]["dasein-flux"]
            + term_counts["sample2"]["dasein-flux"]
            + term_counts["sample3"]["dasein-flux"]
        )
        assert total == 7, f"Expected 7 total occurrences, got {total}"

    def test_cross_file_geist_praxis_total(self, term_counts):
        """Test total geist-praxis frequency across all files."""
        total = (
            term_counts["sample1"]["geist-praxis"]
            + term_counts["sample2"]["geist-praxis"]
            + term_counts["sample3"]["geist-praxis"]
        )
        assert total == 8, f"Expected 8 total occurrences, got {total}"


class TestSampleCorpusSearch:
    """Test search functionality on sample corpus."""

    @pytest.fixture
    def sample_files(self):
        """Return paths to sample corpus files."""
        base = Path(__file__).parent.parent / "data" / "sample"
        return {
            "sample1": str(base / "sample1_dialectics.txt"),
            "sample2": str(base / "sample2_epistemology.txt"),
            "sample3": str(base / "sample3_ontology.txt"),
        }

    def test_search_dasein_flux_sample1(self, sample_files):
        """Test searching for dasein-flux in sample1."""
        sentences = pt.search_term_in_file(sample_files["sample1"], "dasein-flux")
        assert (
            len(sentences) == 6
        ), f"Expected 6 sentences with dasein-flux, got {len(sentences)}"

    def test_search_geist_praxis_sample1(self, sample_files):
        """Test searching for geist-praxis in sample1."""
        sentences = pt.search_term_in_file(sample_files["sample1"], "geist-praxis")
        assert (
            len(sentences) == 7
        ), f"Expected 7 sentences with geist-praxis, got {len(sentences)}"

    def test_search_consciousness_sample1(self, sample_files):
        """Test searching for consciousness in sample1."""
        sentences = pt.search_term_in_file(sample_files["sample1"], "consciousness")
        # Should find both "consciousness" and "totality-consciousness"
        assert len(sentences) > 0, "Expected to find sentences with consciousness"

    def test_search_reduction_sample2(self, sample_files):
        """Test searching for reduction in sample2."""
        sentences = pt.search_term_in_file(sample_files["sample2"], "reduction")
        assert (
            len(sentences) == 5
        ), f"Expected 5 sentences with reduction, got {len(sentences)}"


class TestSampleCorpusAnalysis:
    """Test that analysis pipeline works on sample corpus."""

    @pytest.fixture
    def sample1_analysis(self):
        """Return analysis of sample1."""
        base = Path(__file__).parent.parent / "data" / "sample"
        return pt.run(str(base / "sample1_dialectics.txt"))

    def test_sample1_analysis_structure(self, sample1_analysis):
        """Test that analysis returns expected structure."""
        assert "tokens" in sample1_analysis
        assert "token_count" in sample1_analysis
        assert "all_verbs" in sample1_analysis
        assert "content_verbs" in sample1_analysis
        assert "nouns" in sample1_analysis
        assert "adjectives" in sample1_analysis

    def test_sample1_top_nouns_include_rare_terms(self, sample1_analysis):
        """Test that top nouns include our invented rare terms."""
        top_nouns = [noun for noun, count in sample1_analysis["nouns"][:10]]

        # These are our most frequent rare terms in sample1
        expected_terms = ["geist-praxis", "abstraction", "dasein-flux"]

        for term in expected_terms:
            assert (
                term in top_nouns
            ), f"Expected {term} in top 10 nouns, got {top_nouns}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
