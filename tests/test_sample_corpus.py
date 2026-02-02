"""
Test that the sample corpus matches its specification.

This test validates the new diverse philosophical corpus files:
- sample1_analytic_pragmatism.txt
- sample2_poststructural_political.txt
- sample3_mind_consciousness.txt
"""

import pytest
import sys
from pathlib import Path
from collections import Counter

# Add parent directory to path to import pos_tagger
sys.path.insert(0, str(Path(__file__).parent.parent))
import pos_tagger as pt


class TestSampleCorpusFiles:
    """Test that sample corpus files exist and are loadable."""

    @pytest.fixture
    def sample_files(self):
        """Return paths to sample corpus files."""
        base = Path(__file__).parent.parent / "samples"
        return {
            "sample1": base / "sample1_analytic_pragmatism.txt",
            "sample2": base / "sample2_poststructural_political.txt",
            "sample3": base / "sample3_mind_consciousness.txt",
        }

    def test_all_sample_files_exist(self, sample_files):
        """Test that all sample files exist."""
        for name, path in sample_files.items():
            assert path.exists(), f"{name} does not exist at {path}"

    def test_sample_files_not_empty(self, sample_files):
        """Test that sample files are not empty."""
        for name, path in sample_files.items():
            text = pt.load_text(str(path))
            assert len(text) > 0, f"{name} is empty"
            assert len(text) > 1000, f"{name} seems too short ({len(text)} chars)"

    def test_sample1_loadable(self, sample_files):
        """Test that sample1 can be loaded and analyzed."""
        result = pt.run(str(sample_files["sample1"]))
        assert "tokens" in result
        assert "nouns" in result
        assert result["token_count"] > 100

    def test_sample2_loadable(self, sample_files):
        """Test that sample2 can be loaded and analyzed."""
        result = pt.run(str(sample_files["sample2"]))
        assert "tokens" in result
        assert "nouns" in result
        assert result["token_count"] > 100

    def test_sample3_loadable(self, sample_files):
        """Test that sample3 can be loaded and analyzed."""
        result = pt.run(str(sample_files["sample3"]))
        assert "tokens" in result
        assert "nouns" in result
        assert result["token_count"] > 100


class TestSampleCorpusContent:
    """Test that sample corpus contains expected philosophical terms."""

    @pytest.fixture
    def term_counts(self):
        """Load and count terms in all sample files."""
        base = Path(__file__).parent.parent / "samples"
        files = {
            "sample1": base / "sample1_analytic_pragmatism.txt",
            "sample2": base / "sample2_poststructural_political.txt",
            "sample3": base / "sample3_mind_consciousness.txt",
        }

        counts = {}
        for name, path in files.items():
            text = pt.load_text(str(path))
            tokens = pt.tokenize_words(text)
            counts[name] = Counter([t.lower() for t in tokens])
        return counts

    # Sample 1: Analytic Philosophy & Pragmatism
    def test_sample1_contains_meaning_variance(self, term_counts):
        """Test that sample1 contains meaning-variance."""
        assert term_counts["sample1"]["meaning-variance"] > 0

    def test_sample1_contains_referential_opacity(self, term_counts):
        """Test that sample1 contains referential-opacity or opacity."""
        # May be tokenized as "referential-opacity" or "opacity"
        has_term = (
            term_counts["sample1"]["referential-opacity"] > 0
            or term_counts["sample1"]["opacity"] > 0
        )
        assert has_term, "Expected to find referential-opacity or opacity"

    def test_sample1_contains_pragmatic(self, term_counts):
        """Test that sample1 contains pragmatic-related terms."""
        has_term = (
            term_counts["sample1"]["pragmatic"] > 0
            or term_counts["sample1"]["pragmatism"] > 0
        )
        assert has_term, "Expected to find pragmatic or pragmatism"

    # Sample 2: Post-structuralism & Political Philosophy
    def test_sample2_contains_bio_regulation_terms(self, term_counts):
        """Test that sample2 contains bio-regulation terms."""
        # May be tokenized as "bio-regulation" or separate words
        has_term = term_counts["sample2"]["bio-regulation"] > 0 or (
            term_counts["sample2"]["bio"] > 0
            and term_counts["sample2"]["regulation"] > 0
        )
        assert has_term, "Expected to find bio-regulation"

    def test_sample2_contains_bio_regulation(self, term_counts):
        """Test that sample2 contains bio-regulation."""
        # May be tokenized as "bio-regulation" or separate words
        has_term = term_counts["sample2"]["bio-regulation"] > 0 or (
            term_counts["sample2"]["bio"] > 0
            and term_counts["sample2"]["regulation"] > 0
        )
        assert has_term, "Expected to find bio-regulation"

    def test_sample2_contains_deterritorialization(self, term_counts):
        """Test that sample2 contains deterritorialization."""
        assert term_counts["sample2"]["deterritorialization"] > 0

    def test_sample2_contains_rhizomatic(self, term_counts):
        """Test that sample2 contains rhizomatic."""
        has_term = (
            term_counts["sample2"]["rhizomatic-becoming"] > 0
            or term_counts["sample2"]["rhizomatic"] > 0
        )
        assert has_term, "Expected to find rhizomatic"

    # Sample 3: Philosophy of Mind & Consciousness
    def test_sample3_contains_phenomenal_character(self, term_counts):
        """Test that sample3 contains phenomenal-character."""
        has_term = (
            term_counts["sample3"]["phenomenal-character"] > 0
            or term_counts["sample3"]["phenomenal"] > 0
        )
        assert has_term, "Expected to find phenomenal-character or phenomenal"

    def test_sample3_contains_qualia(self, term_counts):
        """Test that sample3 contains qualia/quale terms."""
        has_term = (
            term_counts["sample3"]["quale"] > 0
            or term_counts["sample3"]["qualia"] > 0
            or term_counts["sample3"]["quale-inversion"] > 0
        )
        assert has_term, "Expected to find quale/qualia terms"

    def test_sample3_contains_zombie(self, term_counts):
        """Test that sample3 contains zombie-conceivability."""
        has_term = (
            term_counts["sample3"]["zombie-conceivability"] > 0
            or term_counts["sample3"]["zombie"] > 0
        )
        assert has_term, "Expected to find zombie terms"

    def test_sample3_contains_intentionality(self, term_counts):
        """Test that sample3 contains intentionality or intentional."""
        has_term = (
            term_counts["sample3"]["intentionality"] > 0
            or term_counts["sample3"]["intentional"] > 0
        )
        assert has_term, "Expected to find intentional terms"


class TestSampleCorpusAnalysis:
    """Test that analysis pipeline works on new sample corpus."""

    @pytest.fixture
    def sample1_analysis(self):
        """Return analysis of sample1."""
        base = Path(__file__).parent.parent / "samples"
        return pt.run(str(base / "sample1_analytic_pragmatism.txt"))

    @pytest.fixture
    def sample2_analysis(self):
        """Return analysis of sample2."""
        base = Path(__file__).parent.parent / "samples"
        return pt.run(str(base / "sample2_poststructural_political.txt"))

    @pytest.fixture
    def sample3_analysis(self):
        """Return analysis of sample3."""
        base = Path(__file__).parent.parent / "samples"
        return pt.run(str(base / "sample3_mind_consciousness.txt"))

    def test_sample1_analysis_structure(self, sample1_analysis):
        """Test that analysis returns expected structure."""
        assert "tokens" in sample1_analysis
        assert "token_count" in sample1_analysis
        assert "all_verbs" in sample1_analysis
        assert "content_verbs" in sample1_analysis
        assert "nouns" in sample1_analysis
        assert "adjectives" in sample1_analysis

    def test_sample2_analysis_structure(self, sample2_analysis):
        """Test that analysis returns expected structure."""
        assert "tokens" in sample2_analysis
        assert "token_count" in sample2_analysis
        assert "nouns" in sample2_analysis

    def test_sample3_analysis_structure(self, sample3_analysis):
        """Test that analysis returns expected structure."""
        assert "tokens" in sample3_analysis
        assert "token_count" in sample3_analysis
        assert "nouns" in sample3_analysis

    def test_sample1_has_philosophical_nouns(self, sample1_analysis):
        """Test that sample1 nouns include analytic philosophy terms."""
        top_nouns = [noun for noun, count in sample1_analysis["nouns"][:20]]
        # Should include some key terms (may be hyphenated or separate)
        has_analytic_terms = any(
            term in " ".join(top_nouns).lower()
            for term in ["meaning", "variance", "opacity", "scheme", "truth"]
        )
        assert has_analytic_terms, f"Expected analytic terms in top nouns: {top_nouns}"

    def test_sample2_has_philosophical_nouns(self, sample2_analysis):
        """Test that sample2 nouns include post-structural/political terms."""
        top_nouns = [noun for noun, count in sample2_analysis["nouns"][:20]]
        has_poststructural_terms = any(
            term in " ".join(top_nouns).lower()
            for term in ["différance", "bio", "regulation", "position", "capability"]
        )
        assert has_poststructural_terms, (
            f"Expected post-structural terms in top nouns: {top_nouns}"
        )

    def test_sample3_has_philosophical_nouns(self, sample3_analysis):
        """Test that sample3 nouns include philosophy of mind terms."""
        top_nouns = [noun for noun, count in sample3_analysis["nouns"][:20]]
        has_mind_terms = any(
            term in " ".join(top_nouns).lower()
            for term in [
                "phenomenal",
                "consciousness",
                "zombie",
                "quale",
                "mind",
                "intentional",
            ]
        )
        assert has_mind_terms, (
            f"Expected philosophy of mind terms in top nouns: {top_nouns}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
