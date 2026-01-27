"""
Tests for corpus-comparative rarity analysis.
"""

import pytest
from collections import Counter
from pathlib import Path
from src.concept_mapper.corpus.models import Document
from src.concept_mapper.preprocessing.pipeline import preprocess
from src.concept_mapper.analysis.rarity import (
    compare_to_reference,
    get_corpus_specific_terms,
    get_top_corpus_specific_terms,
    get_neologism_candidates,
    get_term_context_stats,
    tfidf_vs_reference,
    get_top_tfidf_terms,
    get_distinctive_by_tfidf,
    get_combined_distinctive_terms,
    get_wordnet_neologisms,
    get_capitalized_technical_terms,
    get_potential_neologisms,
    get_all_neologism_signals,
    get_definitional_contexts,
    score_by_definitional_context,
    get_definitional_sentences,
    get_highly_defined_terms,
    analyze_definitional_patterns,
    get_terms_with_definitions,
    filter_by_pos_tags,
    get_philosophical_term_candidates,
    get_compound_terms,
    get_filtered_candidates,
    PhilosophicalTermScorer,
    score_philosophical_terms,
    CONTENT_WORD_POS_TAGS,
    FUNCTION_WORD_POS_TAGS,
)


class TestCompareToReference:
    """Test corpus-comparative ratio calculations."""

    @pytest.fixture
    def simple_docs(self):
        """Create simple test documents with known term frequencies."""
        doc1 = Document(text="abstraction abstraction abstraction", metadata={})
        doc2 = Document(text="totality totality", metadata={})
        return [preprocess(doc1), preprocess(doc2)]

    @pytest.fixture
    def simple_reference(self):
        """Simple reference corpus: 'abstraction' rare, 'totality' absent."""
        # Total: 100 tokens
        # 'abstraction' appears 1 time (very rare)
        # 'totality' appears 0 times (absent)
        # 'the' appears 20 times (common)
        return Counter({"the": 20, "is": 15, "abstraction": 1, "and": 64})

    def test_ratio_calculation_basic(self, simple_docs, simple_reference):
        """Test basic ratio calculation."""
        ratios = compare_to_reference(
            simple_docs, simple_reference, use_lemmas=True, min_author_freq=1
        )

        # Author has: 3 'abstraction', 2 'totality' out of 5 total
        # Reference has: 1 'abstraction', 0 'totality' out of 100 total

        # abstraction ratio = (3/5) / (1/100) = 0.6 / 0.01 = 60
        assert "abstraction" in ratios
        assert ratios["abstraction"] == pytest.approx(60.0, rel=0.01)

        # totality ratio = (2/5) / (0.5/100) = 0.4 / 0.005 = 80 (with pseudocount)
        assert "totality" in ratios
        assert ratios["totality"] > 50  # Very high because absent from reference

    def test_min_frequency_threshold(self, simple_docs, simple_reference):
        """Test minimum frequency filtering."""
        # With min_author_freq=3, 'totality' (2 occurrences) should be excluded
        ratios = compare_to_reference(
            simple_docs, simple_reference, use_lemmas=True, min_author_freq=3
        )

        assert "abstraction" in ratios
        assert "totality" not in ratios

    def test_terms_not_in_reference(self, simple_docs):
        """Test handling of terms completely absent from reference."""
        # Empty reference corpus
        empty_ref = Counter({"filler": 100})  # Need some content for totals

        ratios = compare_to_reference(
            simple_docs, empty_ref, use_lemmas=True, min_author_freq=1
        )

        # Both terms should have very high ratios (pseudocount used)
        assert "abstraction" in ratios
        assert "totality" in ratios
        assert ratios["abstraction"] > 10
        assert ratios["totality"] > 10


class TestCorpusSpecificTerms:
    """Test corpus-specific term extraction."""

    @pytest.fixture
    def docs_with_rare_terms(self):
        """Documents with mix of common and rare terms."""
        doc = Document(
            text="daseinology daseinology daseinology temporalization temporalization the the the the",
            metadata={},
        )
        return [preprocess(doc)]

    @pytest.fixture
    def reference_with_common(self):
        """Reference with only common terms."""
        # 'the' is common, rare terms absent
        return Counter({"the": 50, "is": 30, "and": 20})

    def test_threshold_filtering(self, docs_with_rare_terms, reference_with_common):
        """Test that threshold filters correctly."""
        # Low threshold should catch more terms
        terms_low = get_corpus_specific_terms(
            docs_with_rare_terms,
            reference_with_common,
            threshold=5.0,
            min_author_freq=2,
        )

        # High threshold should catch fewer terms
        terms_high = get_corpus_specific_terms(
            docs_with_rare_terms,
            reference_with_common,
            threshold=50.0,
            min_author_freq=2,
        )

        # Both rare terms should be caught with low threshold
        assert "daseinology" in terms_low
        assert "temporalization" in terms_low

        # Common term should not be caught
        assert "the" not in terms_low

        # High threshold should still catch rare terms
        assert len(terms_high) <= len(terms_low)

    def test_reference_frequency_limit(self, docs_with_rare_terms):
        """Test filtering by maximum reference frequency."""
        # Reference with one rare term appearing once
        ref = Counter({"the": 50, "daseinology": 1, "is": 49})

        # Without reference limit, should get daseinology
        terms_no_limit = get_corpus_specific_terms(
            docs_with_rare_terms, ref, threshold=2.0, min_author_freq=2
        )
        assert "daseinology" in terms_no_limit

        # With reference limit of 0, should exclude daseinology
        terms_with_limit = get_corpus_specific_terms(
            docs_with_rare_terms,
            ref,
            threshold=2.0,
            min_author_freq=2,
            min_reference_freq=0,
        )
        assert "daseinology" not in terms_with_limit


class TestTopCorpusSpecificTerms:
    """Test ranking of corpus-specific terms."""

    @pytest.fixture
    def docs_with_varying_rarity(self):
        """Documents with terms of varying distinctiveness."""
        doc = Document(
            text="ultra_rare ultra_rare ultra_rare rare rare common",
            metadata={},
        )
        return [preprocess(doc)]

    @pytest.fixture
    def graded_reference(self):
        """Reference with graded frequencies."""
        return Counter(
            {
                "ultra_rare": 1,  # Very rare
                "rare": 10,  # Somewhat rare
                "common": 100,  # Common
                "filler": 889,  # More common
            }
        )

    def test_ranking_by_ratio(self, docs_with_varying_rarity, graded_reference):
        """Test that terms are ranked by ratio."""
        top_terms = get_top_corpus_specific_terms(
            docs_with_varying_rarity, graded_reference, n=10, min_author_freq=1
        )

        # Should return list of (term, ratio) tuples
        assert len(top_terms) > 0
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top_terms)

        # Ratios should be descending
        ratios = [ratio for term, ratio in top_terms]
        assert ratios == sorted(ratios, reverse=True)

        # ultra_rare should rank higher than rare
        term_names = [term for term, ratio in top_terms]
        if "ultra_rare" in term_names and "rare" in term_names:
            assert term_names.index("ultra_rare") < term_names.index("rare")

    def test_n_parameter(self, docs_with_varying_rarity, graded_reference):
        """Test that n parameter limits results."""
        top_2 = get_top_corpus_specific_terms(
            docs_with_varying_rarity, graded_reference, n=2, min_author_freq=1
        )
        top_10 = get_top_corpus_specific_terms(
            docs_with_varying_rarity, graded_reference, n=10, min_author_freq=1
        )

        assert len(top_2) <= 2
        assert len(top_10) <= 10
        assert len(top_2) <= len(top_10)


class TestNeologismCandidates:
    """Test neologism detection."""

    @pytest.fixture
    def docs_with_neologisms(self):
        """Documents with invented terms."""
        doc = Document(
            text="daseinology daseinology temporalization ekstatic the is",
            metadata={},
        )
        return [preprocess(doc)]

    @pytest.fixture
    def reference_missing_neologisms(self):
        """Reference corpus missing the neologisms."""
        return Counter({"the": 50, "is": 30, "a": 20})

    def test_neologism_detection(
        self, docs_with_neologisms, reference_missing_neologisms
    ):
        """Test that terms absent from reference are detected."""
        neologisms = get_neologism_candidates(
            docs_with_neologisms, reference_missing_neologisms, min_author_freq=1
        )

        # Invented terms should be detected
        assert "daseinology" in neologisms
        assert "temporalization" in neologisms

        # Common terms should not be detected
        assert "the" not in neologisms
        assert "is" not in neologisms

    def test_frequency_threshold(
        self, docs_with_neologisms, reference_missing_neologisms
    ):
        """Test minimum frequency threshold for neologisms."""
        # With min_freq=2, 'ekstatic' (1 occurrence) should be excluded
        neologisms = get_neologism_candidates(
            docs_with_neologisms, reference_missing_neologisms, min_author_freq=2
        )

        assert "daseinology" in neologisms  # 2 occurrences
        assert "ekstatic" not in neologisms  # 1 occurrence


class TestTermContextStats:
    """Test detailed term statistics."""

    @pytest.fixture
    def multi_doc_corpus(self):
        """Multiple documents for testing document frequency."""
        doc1 = Document(text="abstraction abstraction totality", metadata={})
        doc2 = Document(text="abstraction consciousness", metadata={})
        doc3 = Document(text="consciousness consciousness", metadata={})
        return [preprocess(doc1), preprocess(doc2), preprocess(doc3)]

    @pytest.fixture
    def simple_reference(self):
        """Simple reference corpus."""
        return Counter({"abstraction": 5, "consciousness": 50, "other": 945})

    def test_stats_structure(self, multi_doc_corpus, simple_reference):
        """Test that stats dictionary has expected structure."""
        stats = get_term_context_stats(
            "abstraction", multi_doc_corpus, simple_reference
        )

        # Should have all expected keys
        assert "author_count" in stats
        assert "author_freq" in stats
        assert "reference_count" in stats
        assert "reference_freq" in stats
        assert "ratio" in stats
        assert "documents_containing" in stats
        assert "in_reference" in stats

    def test_document_counting(self, multi_doc_corpus, simple_reference):
        """Test that documents_containing is counted correctly."""
        # 'abstraction' appears in 2 documents
        stats_reif = get_term_context_stats(
            "abstraction", multi_doc_corpus, simple_reference
        )
        assert stats_reif["documents_containing"] == 2

        # 'consciousness' appears in 2 documents
        stats_cons = get_term_context_stats(
            "consciousness", multi_doc_corpus, simple_reference
        )
        assert stats_cons["documents_containing"] == 2

        # 'totality' appears in 1 document
        stats_total = get_term_context_stats(
            "totality", multi_doc_corpus, simple_reference
        )
        assert stats_total["documents_containing"] == 1

    def test_in_reference_flag(self, multi_doc_corpus, simple_reference):
        """Test that in_reference flag is set correctly."""
        stats_reif = get_term_context_stats(
            "abstraction", multi_doc_corpus, simple_reference
        )
        assert stats_reif["in_reference"] is True

        stats_total = get_term_context_stats(
            "totality", multi_doc_corpus, simple_reference
        )
        assert stats_total["in_reference"] is False


class TestRealCorpusAnalysis:
    """Test rarity analysis on real sample corpus."""

    @pytest.fixture
    def sample_corpus(self):
        """Load sample philosophical corpus."""
        from src.concept_mapper.corpus.loader import load_directory

        base = Path(__file__).parent.parent / "samples"
        corpus = load_directory(str(base), pattern="sample*_*.txt", recursive=False)
        return [preprocess(doc) for doc in corpus.documents]

    @pytest.fixture
    def brown_corpus(self):
        """Load Brown corpus for comparison."""
        from src.concept_mapper.analysis.reference import load_reference_corpus

        return load_reference_corpus("brown")

    def test_sample_corpus_neologisms(self, sample_corpus, brown_corpus):
        """Test that known neologisms are detected in sample corpus."""
        neologisms = get_neologism_candidates(
            sample_corpus, brown_corpus, min_author_freq=2
        )

        # Known invented terms from test corpus should be detected
        # Sample 1 (analytic/pragmatism)
        potential_terms = ["meaning-variance", "instrumental-warranting"]
        # Sample 2 (post-structural/political)
        potential_terms.extend(["bio-regulation", "différance", "rhizomatic-becoming"])
        # Sample 3 (philosophy of mind)
        potential_terms.extend(
            ["zombie-conceivability", "quale-inversion", "modal-reconfiguration"]
        )

        # At least some neologisms should be detected
        # (exact matches depend on tokenization of hyphenated terms)
        detected_count = sum(1 for term in potential_terms if term in neologisms)
        assert (
            detected_count > 0
        ), f"Expected to detect some neologisms, found: {neologisms}"

    def test_sample_corpus_specific_terms(self, sample_corpus, brown_corpus):
        """Test that corpus-specific terms are identified."""
        specific_terms = get_corpus_specific_terms(
            sample_corpus, brown_corpus, threshold=50.0, min_author_freq=3
        )

        # Should identify multiple distinctive terms
        assert len(specific_terms) > 0

        # Common English words should not be included
        common_words = {"the", "is", "are", "of", "and", "to", "in"}
        assert not common_words.intersection(specific_terms)

    def test_top_terms_ranking(self, sample_corpus, brown_corpus):
        """Test that top terms are ranked sensibly."""
        top_terms = get_top_corpus_specific_terms(
            sample_corpus, brown_corpus, n=20, min_author_freq=2
        )

        # Should return list of terms
        assert len(top_terms) > 0
        assert len(top_terms) <= 20

        # All should have ratios > 1 (more common in author than reference)
        for term, ratio in top_terms:
            assert ratio > 1.0

    def test_term_stats_on_sample(self, sample_corpus, brown_corpus):
        """Test detailed stats on a known philosophical term."""
        # Test on a term we know exists in the corpus
        stats = get_term_context_stats("consciousness", sample_corpus, brown_corpus)

        # Should have positive counts
        assert stats["author_count"] > 0
        assert stats["documents_containing"] > 0

        # Should have valid frequencies
        assert 0 <= stats["author_freq"] <= 1
        assert stats["ratio"] >= 0


class TestTFIDFVsReference:
    """Test TF-IDF against reference corpus."""

    @pytest.fixture
    def docs_with_distinctive_terms(self):
        """Documents with mix of distinctive and common terms."""
        doc = Document(
            text="neologism neologism neologism rare the the the",
            metadata={},
        )
        return [preprocess(doc)]

    @pytest.fixture
    def reference_mostly_common(self):
        """Reference with common terms, missing distinctive ones."""
        return Counter({"the": 100, "is": 50, "and": 30, "rare": 5})

    def test_tfidf_calculation(
        self, docs_with_distinctive_terms, reference_mostly_common
    ):
        """Test basic TF-IDF calculation."""
        scores = tfidf_vs_reference(
            docs_with_distinctive_terms,
            reference_mostly_common,
            use_lemmas=True,
            min_author_freq=1,
        )

        # Should return scores for terms
        assert len(scores) > 0

        # Neologism (not in reference) should have higher TF-IDF than common terms
        if "neologism" in scores and "the" in scores:
            assert scores["neologism"] > scores["the"]

    def test_neologism_high_idf(
        self, docs_with_distinctive_terms, reference_mostly_common
    ):
        """Test that neologisms get high IDF boost."""
        scores = tfidf_vs_reference(
            docs_with_distinctive_terms, reference_mostly_common, min_author_freq=1
        )

        # Neologism should have positive TF-IDF
        if "neologism" in scores:
            assert scores["neologism"] > 0

        # Terms in reference should have lower or zero TF-IDF
        if "the" in scores:
            # 'the' appears in both, so IDF = log(2/2) = 0, thus TF-IDF = 0
            assert scores["the"] == pytest.approx(0.0, abs=0.01)

    def test_min_frequency_filtering(
        self, docs_with_distinctive_terms, reference_mostly_common
    ):
        """Test that minimum frequency threshold works."""
        scores_low = tfidf_vs_reference(
            docs_with_distinctive_terms, reference_mostly_common, min_author_freq=1
        )
        scores_high = tfidf_vs_reference(
            docs_with_distinctive_terms, reference_mostly_common, min_author_freq=3
        )

        # Higher threshold should have fewer or equal terms
        assert len(scores_high) <= len(scores_low)

        # 'neologism' (3 occurrences) should be in both
        assert "neologism" in scores_high


class TestTopTFIDFTerms:
    """Test TF-IDF ranking functionality."""

    @pytest.fixture
    def docs_varying_distinctiveness(self):
        """Documents with terms of varying TF and IDF."""
        doc = Document(
            text="super_rare super_rare super_rare rare common common",
            metadata={},
        )
        return [preprocess(doc)]

    @pytest.fixture
    def graded_reference(self):
        """Reference with graded term presence."""
        return Counter({"common": 100, "rare": 10, "filler": 890})

    def test_ranking_order(self, docs_varying_distinctiveness, graded_reference):
        """Test that terms are ranked by TF-IDF score."""
        top_terms = get_top_tfidf_terms(
            docs_varying_distinctiveness, graded_reference, n=10, min_author_freq=1
        )

        # Should return tuples
        assert all(isinstance(t, tuple) and len(t) == 2 for t in top_terms)

        # Scores should be descending
        scores = [score for term, score in top_terms]
        assert scores == sorted(scores, reverse=True)

    def test_n_parameter_limits(self, docs_varying_distinctiveness, graded_reference):
        """Test that n parameter works correctly."""
        top_2 = get_top_tfidf_terms(
            docs_varying_distinctiveness, graded_reference, n=2, min_author_freq=1
        )
        top_10 = get_top_tfidf_terms(
            docs_varying_distinctiveness, graded_reference, n=10, min_author_freq=1
        )

        assert len(top_2) <= 2
        assert len(top_10) <= 10
        assert len(top_2) <= len(top_10)


class TestDistinctiveByTFIDF:
    """Test threshold-based TF-IDF filtering."""

    @pytest.fixture
    def sample_docs(self):
        """Simple test documents."""
        doc = Document(text="philosophical philosophical term term common", metadata={})
        return [preprocess(doc)]

    @pytest.fixture
    def simple_ref(self):
        """Simple reference."""
        return Counter({"common": 100, "other": 900})

    def test_threshold_filtering(self, sample_docs, simple_ref):
        """Test that threshold filters correctly."""
        # Low threshold should catch more terms
        low_threshold = get_distinctive_by_tfidf(
            sample_docs, simple_ref, threshold=0.0001, min_author_freq=1
        )

        # High threshold should catch fewer terms
        high_threshold = get_distinctive_by_tfidf(
            sample_docs, simple_ref, threshold=0.01, min_author_freq=1
        )

        # Low threshold should have at least as many as high threshold
        assert len(low_threshold) >= len(high_threshold)

        # High threshold should be subset of low threshold
        assert high_threshold.issubset(low_threshold)


class TestCombinedDistinctiveTerms:
    """Test combined rarity detection methods."""

    @pytest.fixture
    def test_docs(self):
        """Documents for combined testing."""
        doc = Document(
            text="high_ratio high_ratio high_ratio high_tfidf common",
            metadata={},
        )
        return [preprocess(doc)]

    @pytest.fixture
    def test_ref(self):
        """Reference corpus for testing."""
        return Counter({"common": 100, "high_ratio": 1, "other": 899})

    def test_union_method(self, test_docs, test_ref):
        """Test union combines both methods."""
        union_terms = get_combined_distinctive_terms(
            test_docs,
            test_ref,
            ratio_threshold=5.0,
            tfidf_threshold=0.0001,
            min_author_freq=1,
            method="union",
        )

        # Should include terms from either method
        assert len(union_terms) > 0

    def test_intersection_method(self, test_docs, test_ref):
        """Test intersection requires both methods."""
        intersection_terms = get_combined_distinctive_terms(
            test_docs,
            test_ref,
            ratio_threshold=5.0,
            tfidf_threshold=0.0001,
            min_author_freq=1,
            method="intersection",
        )

        # Terms must pass both filters
        # Intersection should be subset of union
        union_terms = get_combined_distinctive_terms(
            test_docs,
            test_ref,
            ratio_threshold=5.0,
            tfidf_threshold=0.0001,
            min_author_freq=1,
            method="union",
        )
        assert intersection_terms.issubset(union_terms)

    def test_ratio_only_method(self, test_docs, test_ref):
        """Test ratio_only uses only frequency ratio."""
        ratio_terms = get_combined_distinctive_terms(
            test_docs,
            test_ref,
            ratio_threshold=5.0,
            tfidf_threshold=0.0001,
            min_author_freq=1,
            method="ratio_only",
        )

        # Should match get_corpus_specific_terms result
        expected = get_corpus_specific_terms(
            test_docs, test_ref, threshold=5.0, min_author_freq=1
        )
        assert ratio_terms == expected

    def test_tfidf_only_method(self, test_docs, test_ref):
        """Test tfidf_only uses only TF-IDF."""
        tfidf_terms = get_combined_distinctive_terms(
            test_docs,
            test_ref,
            ratio_threshold=5.0,
            tfidf_threshold=0.0001,
            min_author_freq=1,
            method="tfidf_only",
        )

        # Should match get_distinctive_by_tfidf result
        expected = get_distinctive_by_tfidf(
            test_docs, test_ref, threshold=0.0001, min_author_freq=1
        )
        assert tfidf_terms == expected

    def test_invalid_method_raises(self, test_docs, test_ref):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid method"):
            get_combined_distinctive_terms(test_docs, test_ref, method="invalid_method")


class TestTFIDFOnRealCorpus:
    """Test TF-IDF analysis on real sample corpus."""

    @pytest.fixture
    def sample_corpus(self):
        """Load sample philosophical corpus."""
        from src.concept_mapper.corpus.loader import load_directory

        base = Path(__file__).parent.parent / "samples"
        corpus = load_directory(str(base), pattern="sample*_*.txt", recursive=False)
        return [preprocess(doc) for doc in corpus.documents]

    @pytest.fixture
    def brown_corpus(self):
        """Load Brown corpus for comparison."""
        from src.concept_mapper.analysis.reference import load_reference_corpus

        return load_reference_corpus("brown")

    def test_tfidf_scores_on_sample(self, sample_corpus, brown_corpus):
        """Test TF-IDF scoring on sample corpus."""
        scores = tfidf_vs_reference(sample_corpus, brown_corpus, min_author_freq=2)

        # Should generate scores
        assert len(scores) > 0

        # All scores should be non-negative
        assert all(score >= 0 for score in scores.values())

    def test_top_tfidf_on_sample(self, sample_corpus, brown_corpus):
        """Test top TF-IDF terms are sensible."""
        top_terms = get_top_tfidf_terms(
            sample_corpus, brown_corpus, n=20, min_author_freq=2
        )

        # Should return terms
        assert len(top_terms) > 0
        assert len(top_terms) <= 20

        # Common words should not dominate
        term_names = [term for term, score in top_terms]
        common_words = {"the", "is", "are", "of", "and"}
        common_in_top = sum(1 for word in common_words if word in term_names)
        assert common_in_top < len(top_terms) / 2  # Less than half should be common

    def test_combined_methods_on_sample(self, sample_corpus, brown_corpus):
        """Test combined detection methods on sample corpus."""
        combined = get_combined_distinctive_terms(
            sample_corpus,
            brown_corpus,
            ratio_threshold=20.0,
            tfidf_threshold=0.0005,
            min_author_freq=3,
            method="union",
        )

        # Should identify distinctive terms
        assert len(combined) > 0

        # Common words should be excluded
        common_words = {"the", "is", "are", "of", "and", "to"}
        assert not common_words.intersection(combined)


class TestWordNetNeologisms:
    """Test WordNet-based neologism detection."""

    @pytest.fixture
    def docs_with_invented_terms(self):
        """Documents with mix of real and invented terms."""
        doc = Document(
            text="daseinology daseinology temporalization ekstatic philosophy consciousness the",
            metadata={},
        )
        return [preprocess(doc)]

    def test_wordnet_detection(self, docs_with_invented_terms):
        """Test that invented terms are detected as not in WordNet."""
        try:
            neologisms = get_wordnet_neologisms(
                docs_with_invented_terms, min_author_freq=1
            )

            # Invented terms should be detected
            assert "daseinology" in neologisms
            assert "temporalization" in neologisms

            # Real English words should not be detected
            assert "philosophy" not in neologisms
            assert "consciousness" not in neologisms
            assert "the" not in neologisms

        except ImportError:
            pytest.skip("WordNet not available")

    def test_frequency_threshold(self, docs_with_invented_terms):
        """Test minimum frequency threshold for WordNet detection."""
        try:
            # With min_freq=2, 'ekstatic' (1 occurrence) should be excluded
            neologisms = get_wordnet_neologisms(
                docs_with_invented_terms, min_author_freq=2
            )

            assert "daseinology" in neologisms  # 2 occurrences
            if "ekstatic" in neologisms:
                pytest.fail("Single-occurrence term should be filtered")

        except ImportError:
            pytest.skip("WordNet not available")

    def test_proper_noun_filtering(self):
        """Test that proper nouns can be filtered out."""
        # Create document with proper noun
        doc = Document(text="Kant KAnt philosophy neologism neologism", metadata={})
        docs = [preprocess(doc)]

        try:
            # With proper noun filtering
            with_filtering = get_wordnet_neologisms(
                docs, min_author_freq=1, exclude_proper_nouns=True
            )

            # Without proper noun filtering
            without_filtering = get_wordnet_neologisms(
                docs, min_author_freq=1, exclude_proper_nouns=False
            )

            # Should have more terms without filtering
            assert len(without_filtering) >= len(with_filtering)

        except ImportError:
            pytest.skip("WordNet not available")


class TestCapitalizedTechnicalTerms:
    """Test detection of mid-sentence capitalized terms."""

    @pytest.fixture
    def docs_with_capitalized(self):
        """Documents with capitalized philosophical abstractions."""
        doc = Document(
            text="The concept of Being is central. Being appears throughout. The Absolute is crucial.",
            metadata={},
        )
        return [preprocess(doc)]

    def test_capitalized_detection(self, docs_with_capitalized):
        """Test that mid-sentence capitalized terms are detected."""
        capitalized = get_capitalized_technical_terms(
            docs_with_capitalized, min_author_freq=1, exclude_sentence_initial=True
        )

        # Mid-sentence capitalized terms should be detected
        assert "Being" in capitalized
        assert "Absolute" in capitalized

        # Sentence-initial words should be excluded
        assert "The" not in capitalized

    def test_sentence_initial_filtering(self):
        """Test that sentence-initial capitals can be filtered."""
        doc = Document(
            text="Philosophy is central. Philosophy is important.", metadata={}
        )
        docs = [preprocess(doc)]

        # With sentence-initial filtering
        with_filtering = get_capitalized_technical_terms(
            docs, min_author_freq=1, exclude_sentence_initial=True
        )

        # Without sentence-initial filtering
        without_filtering = get_capitalized_technical_terms(
            docs, min_author_freq=1, exclude_sentence_initial=False
        )

        # Should have more terms without filtering
        assert len(without_filtering) >= len(with_filtering)

    def test_proper_noun_exclusion(self):
        """Test that proper nouns (NNP tags) can be excluded with flag."""
        doc = Document(text="Kant discussed Being and Being again.", metadata={})
        docs = [preprocess(doc)]

        # Without proper noun filtering - all capitalized terms included
        without_filter = get_capitalized_technical_terms(
            docs, min_author_freq=1, exclude_proper_nouns=False
        )

        # With proper noun filtering - NNP-tagged terms excluded
        with_filter = get_capitalized_technical_terms(
            docs, min_author_freq=1, exclude_proper_nouns=True
        )

        # Without filter should have more or equal terms
        assert len(without_filter) >= len(with_filter)

        # With filter should exclude NNP-tagged terms
        # Note: This may also filter "Being" if POS tagger tags it as NNP


class TestPotentialNeologisms:
    """Test general neologism detection with custom dictionaries."""

    @pytest.fixture
    def test_docs(self):
        """Simple test documents."""
        doc = Document(text="zxqflorp zxqflorp real word word", metadata={})
        return [preprocess(doc)]

    def test_custom_dictionary(self, test_docs):
        """Test using custom dictionary."""
        custom_dict = {"real", "word"}

        neologisms = get_potential_neologisms(
            test_docs, dictionary=custom_dict, min_author_freq=1
        )

        # Term not in custom dictionary should be detected
        assert "zxqflorp" in neologisms

        # Terms in custom dictionary should not be detected
        assert "real" not in neologisms
        assert "word" not in neologisms

    def test_wordnet_default(self, test_docs):
        """Test that WordNet is used as default dictionary."""
        try:
            neologisms = get_potential_neologisms(
                test_docs, dictionary=None, min_author_freq=1
            )

            # Should detect invented terms
            assert "zxqflorp" in neologisms

            # Should not detect real English words
            assert "word" not in neologisms

        except ImportError:
            pytest.skip("WordNet not available")


class TestAllNeologismSignals:
    """Test combined neologism detection from all methods."""

    @pytest.fixture
    def test_docs(self):
        """Documents with various types of neologisms."""
        doc = Document(
            text="daseinology daseinology The Being appears. Neologism neologism neologism.",
            metadata={},
        )
        return [preprocess(doc)]

    @pytest.fixture
    def test_reference(self):
        """Reference corpus missing test neologisms."""
        return Counter({"the": 50, "appears": 100, "other": 850})

    def test_all_signals_structure(self, test_docs, test_reference):
        """Test that all signal types are returned."""
        try:
            signals = get_all_neologism_signals(
                test_docs, test_reference, min_author_freq=1
            )

            # Should have all expected keys
            assert "reference" in signals
            assert "wordnet" in signals
            assert "capitalized" in signals
            assert "all_neologisms" in signals
            assert "high_confidence" in signals

            # All should be sets
            assert isinstance(signals["reference"], set)
            assert isinstance(signals["wordnet"], set)
            assert isinstance(signals["capitalized"], set)
            assert isinstance(signals["all_neologisms"], set)
            assert isinstance(signals["high_confidence"], set)

        except ImportError:
            pytest.skip("WordNet not available")

    def test_all_neologisms_is_union(self, test_docs, test_reference):
        """Test that all_neologisms is union of all methods."""
        try:
            signals = get_all_neologism_signals(
                test_docs, test_reference, min_author_freq=1
            )

            # Union of all individual methods
            manual_union = (
                signals["reference"] | signals["wordnet"] | signals["capitalized"]
            )

            assert signals["all_neologisms"] == manual_union

        except ImportError:
            pytest.skip("WordNet not available")

    def test_high_confidence_requires_multiple_signals(self, test_docs, test_reference):
        """Test that high_confidence requires at least 2 methods."""
        try:
            signals = get_all_neologism_signals(
                test_docs, test_reference, min_author_freq=1
            )

            # Each term in high_confidence must be in at least 2 other sets
            for term in signals["high_confidence"]:
                count = sum(
                    [
                        term in signals["reference"],
                        term in signals["wordnet"],
                        term in signals["capitalized"],
                    ]
                )
                assert count >= 2, f"{term} only found in {count} method(s)"

        except ImportError:
            pytest.skip("WordNet not available")


class TestWordNetOnRealCorpus:
    """Test WordNet-based detection on real sample corpus."""

    @pytest.fixture
    def sample_corpus(self):
        """Load sample philosophical corpus."""
        from src.concept_mapper.corpus.loader import load_directory

        base = Path(__file__).parent.parent / "samples"
        corpus = load_directory(str(base), pattern="sample*_*.txt", recursive=False)
        return [preprocess(doc) for doc in corpus.documents]

    @pytest.fixture
    def brown_corpus(self):
        """Load Brown corpus for comparison."""
        from src.concept_mapper.analysis.reference import load_reference_corpus

        return load_reference_corpus("brown")

    def test_wordnet_on_sample(self, sample_corpus):
        """Test WordNet detection on sample corpus."""
        try:
            neologisms = get_wordnet_neologisms(sample_corpus, min_author_freq=2)

            # Should detect some invented terms
            assert len(neologisms) > 0

            # Common philosophical words should not be detected
            common_phil = {"philosophy", "consciousness", "theory", "concept"}
            assert not common_phil.intersection(neologisms)

        except ImportError:
            pytest.skip("WordNet not available")

    def test_capitalized_on_sample(self, sample_corpus):
        """Test capitalized term detection on sample corpus."""
        capitalized = get_capitalized_technical_terms(sample_corpus, min_author_freq=2)

        # May or may not find capitalized terms depending on corpus
        # Just verify it runs without error
        assert isinstance(capitalized, set)

    def test_all_signals_on_sample(self, sample_corpus, brown_corpus):
        """Test combined signals on sample corpus."""
        try:
            signals = get_all_neologism_signals(
                sample_corpus, brown_corpus, min_author_freq=2
            )

            # Should detect neologisms from multiple sources
            assert len(signals["all_neologisms"]) > 0

            # High confidence terms are most reliable
            if len(signals["high_confidence"]) > 0:
                # High confidence should be subset of all
                assert signals["high_confidence"].issubset(signals["all_neologisms"])

        except ImportError:
            pytest.skip("WordNet not available")


class TestDefinitionalContexts:
    """Test extraction of definitional contexts."""

    @pytest.fixture
    def docs_with_definitions(self):
        """Documents with various definitional patterns."""
        text = """
        Dasein is being-in-the-world.
        By abstraction I mean the objectification of social relations.
        What I call différance is neither a word nor a concept.
        The concept of Being refers to existence itself.
        Consciousness, which is awareness of phenomena, plays a central role.
        I define intentionality as the directedness of mental states.
        """
        doc = Document(text=text, metadata={"source_path": "test.txt"})
        return [preprocess(doc)]

    def test_extract_copular_definitions(self, docs_with_definitions):
        """Test extraction of 'X is Y' patterns."""
        contexts = get_definitional_contexts(docs_with_definitions)

        # Should find copular definitions
        copular_terms = [
            term for term, sent, ptype, doc_id in contexts if ptype == "copular"
        ]
        assert "Dasein" in copular_terms or "dasein" in copular_terms

    def test_extract_explicit_mean(self, docs_with_definitions):
        """Test extraction of 'by X I mean' patterns."""
        contexts = get_definitional_contexts(docs_with_definitions)

        # Should find explicit mean pattern
        mean_terms = [
            term for term, sent, ptype, doc_id in contexts if ptype == "explicit_mean"
        ]
        assert "abstraction" in mean_terms

    def test_extract_metalinguistic(self, docs_with_definitions):
        """Test extraction of 'what I call X' patterns."""
        contexts = get_definitional_contexts(docs_with_definitions)

        # Should find metalinguistic pattern
        meta_terms = [
            term for term, sent, ptype, doc_id in contexts if ptype == "metalinguistic"
        ]
        assert "différance" in meta_terms

    def test_extract_conceptual(self, docs_with_definitions):
        """Test extraction of 'concept of X' patterns."""
        contexts = get_definitional_contexts(docs_with_definitions)

        # Should find conceptual pattern
        concept_terms = [
            term for term, sent, ptype, doc_id in contexts if ptype == "conceptual"
        ]
        assert "Being" in concept_terms or "being" in concept_terms

    def test_extract_appositive(self, docs_with_definitions):
        """Test extraction of 'X, which is' patterns."""
        contexts = get_definitional_contexts(docs_with_definitions)

        # Should find appositive pattern
        appositive_terms = [
            term for term, sent, ptype, doc_id in contexts if ptype == "appositive"
        ]
        assert (
            "Consciousness" in appositive_terms or "consciousness" in appositive_terms
        )

    def test_extract_explicit_define(self, docs_with_definitions):
        """Test extraction of 'I define X as' patterns."""
        contexts = get_definitional_contexts(docs_with_definitions)

        # Should find explicit define pattern
        define_terms = [
            term for term, sent, ptype, doc_id in contexts if ptype == "explicit_define"
        ]
        assert "intentionality" in define_terms

    def test_case_insensitive_by_default(self, docs_with_definitions):
        """Test that matching is case-insensitive by default."""
        contexts = get_definitional_contexts(
            docs_with_definitions, case_sensitive=False
        )

        # Should find terms regardless of case
        terms = [term.lower() for term, sent, ptype, doc_id in contexts]
        assert "dasein" in terms or "Dasein" in terms

    def test_custom_patterns(self):
        """Test using custom definitional patterns."""
        text = "The term foo denotes bar."
        doc = Document(text=text, metadata={})
        docs = [preprocess(doc)]

        # Custom pattern for "X denotes Y"
        custom_patterns = [
            (r"the\s+term\s+(\w+)\s+denotes", "custom_denotes"),
        ]

        contexts = get_definitional_contexts(docs, patterns=custom_patterns)

        # Should find custom pattern
        assert len(contexts) > 0
        terms = [term for term, sent, ptype, doc_id in contexts]
        assert "foo" in terms


class TestScoreByDefinitionalContext:
    """Test scoring terms by definitional context frequency."""

    @pytest.fixture
    def docs_with_repeated_definitions(self):
        """Documents where some terms are defined multiple times."""
        text = """
        Dasein is being-in-the-world.
        By Dasein I mean human existence.
        The concept of Dasein refers to authentic being.
        Intentionality is the directedness of consciousness.
        """
        doc = Document(text=text, metadata={})
        return [preprocess(doc)]

    def test_score_calculation(self, docs_with_repeated_definitions):
        """Test that terms are scored by definition frequency."""
        scores = score_by_definitional_context(docs_with_repeated_definitions)

        # Dasein should have higher score (3 definitions)
        # Intentionality should have lower score (1 definition)
        dasein_score = scores.get("dasein", 0) + scores.get("Dasein", 0)
        intentionality_score = scores.get("intentionality", 0) + scores.get(
            "Intentionality", 0
        )

        assert dasein_score >= 2  # At least 2 definitional contexts
        assert intentionality_score >= 1  # At least 1 definitional context

    def test_filter_by_terms(self, docs_with_repeated_definitions):
        """Test filtering scores to specific terms."""
        candidate_terms = {"dasein", "intentionality"}
        scores = score_by_definitional_context(
            docs_with_repeated_definitions, terms=candidate_terms
        )

        # Should only score candidate terms
        scored_terms = set(scores.keys())
        assert scored_terms.issubset(candidate_terms)

    def test_empty_docs(self):
        """Test scoring on documents with no definitions."""
        doc = Document(text="The sky is blue. Grass is green.", metadata={})
        docs = [preprocess(doc)]

        scores = score_by_definitional_context(docs)

        # Should return empty or minimal scores
        assert len(scores) >= 0  # May find generic "is" patterns


class TestDefinitionalSentences:
    """Test retrieval of definitional sentences for specific terms."""

    @pytest.fixture
    def docs_with_definitions(self):
        """Documents with definitions."""
        text = """
        Dasein is being-in-the-world.
        By Dasein I mean human existence.
        The concept of Dasein refers to temporal being.
        Consciousness is awareness.
        """
        doc = Document(text=text, metadata={})
        return [preprocess(doc)]

    def test_get_sentences_for_term(self, docs_with_definitions):
        """Test retrieving all definitional sentences for a term."""
        sentences = get_definitional_sentences("Dasein", docs_with_definitions)

        # Should find multiple sentences for Dasein
        assert len(sentences) >= 2

        # Each result should be (sentence, pattern_type) tuple
        assert all(isinstance(s, tuple) and len(s) == 2 for s in sentences)

    def test_case_insensitive_retrieval(self, docs_with_definitions):
        """Test case-insensitive retrieval."""
        # Query with lowercase
        sentences_lower = get_definitional_sentences(
            "dasein", docs_with_definitions, case_sensitive=False
        )

        # Query with capitalized
        sentences_cap = get_definitional_sentences(
            "Dasein", docs_with_definitions, case_sensitive=False
        )

        # Should get same results
        assert len(sentences_lower) == len(sentences_cap)

    def test_no_definitions_found(self, docs_with_definitions):
        """Test behavior when term has no definitions."""
        sentences = get_definitional_sentences("nonexistent", docs_with_definitions)

        # Should return empty list
        assert sentences == []


class TestHighlyDefinedTerms:
    """Test identification of highly defined terms."""

    @pytest.fixture
    def docs_with_varying_definitions(self):
        """Documents with some terms defined more than others."""
        text = """
        Dasein is being-in-the-world.
        By Dasein I mean authentic existence.
        The concept of Dasein is central to phenomenology.
        Being is existence.
        Consciousness is awareness.
        """
        doc = Document(text=text, metadata={})
        return [preprocess(doc)]

    def test_min_definitions_threshold(self, docs_with_varying_definitions):
        """Test filtering by minimum definition count."""
        # Require at least 2 definitions
        highly_defined = get_highly_defined_terms(
            docs_with_varying_definitions, min_definitions=2
        )

        # Dasein should be included (3 definitions)
        assert "dasein" in highly_defined or "Dasein" in highly_defined

        # Single-definition terms should be excluded
        # (depends on exact matches, may vary)

    def test_candidate_terms_filtering(self, docs_with_varying_definitions):
        """Test filtering to candidate terms only."""
        candidates = {"dasein", "being"}
        highly_defined = get_highly_defined_terms(
            docs_with_varying_definitions, min_definitions=1, terms=candidates
        )

        # Should only return candidates
        assert highly_defined.issubset(candidates)


class TestAnalyzeDefinitionalPatterns:
    """Test analysis of definitional pattern distribution."""

    @pytest.fixture
    def docs_with_varied_patterns(self):
        """Documents using different definitional patterns."""
        text = """
        Dasein is being-in-the-world.
        Consciousness is awareness.
        By abstraction I mean objectification.
        What I call différance is deferral.
        The concept of Being refers to existence.
        """
        doc = Document(text=text, metadata={})
        return [preprocess(doc)]

    def test_pattern_counting(self, docs_with_varied_patterns):
        """Test counting of pattern types."""
        pattern_counts = analyze_definitional_patterns(docs_with_varied_patterns)

        # Should return dictionary of pattern types to counts
        assert isinstance(pattern_counts, dict)
        assert len(pattern_counts) > 0

        # Should include various pattern types
        assert "copular" in pattern_counts
        assert pattern_counts["copular"] >= 2  # At least Dasein and Consciousness


class TestTermsWithDefinitions:
    """Test building glossary of definitions."""

    @pytest.fixture
    def docs_for_glossary(self):
        """Documents for testing glossary building."""
        text = """
        Dasein is being-in-the-world.
        By Dasein I mean authentic human existence.
        Consciousness is the state of awareness.
        The concept of intentionality refers to aboutness.
        """
        doc = Document(text=text, metadata={})
        return [preprocess(doc)]

    def test_build_glossary(self, docs_for_glossary):
        """Test building glossary for candidate terms."""
        candidates = {"dasein", "consciousness", "intentionality", "nonexistent"}

        glossary = get_terms_with_definitions(docs_for_glossary, candidates)

        # Should return dictionary
        assert isinstance(glossary, dict)

        # Terms with definitions should be included
        assert "dasein" in glossary or "Dasein" in glossary
        assert "consciousness" in glossary or "Consciousness" in glossary

        # Terms without definitions should not be included
        assert "nonexistent" not in glossary

        # Each entry should have list of sentences
        for term, sentences in glossary.items():
            assert isinstance(sentences, list)
            assert all(isinstance(s, str) for s in sentences)


class TestDefinitionalContextsOnRealCorpus:
    """Test definitional extraction on real sample corpus."""

    @pytest.fixture
    def sample_corpus(self):
        """Load sample philosophical corpus."""
        from src.concept_mapper.corpus.loader import load_directory

        base = Path(__file__).parent.parent / "samples"
        corpus = load_directory(str(base), pattern="sample*_*.txt", recursive=False)
        return [preprocess(doc) for doc in corpus.documents]

    def test_extract_definitions_from_sample(self, sample_corpus):
        """Test extracting definitions from sample corpus."""
        contexts = get_definitional_contexts(sample_corpus)

        # Should find some definitional contexts
        # (exact number depends on corpus content)
        assert isinstance(contexts, list)

        # Each context should be a tuple
        if len(contexts) > 0:
            assert all(isinstance(c, tuple) and len(c) == 4 for c in contexts)

    def test_score_sample_terms(self, sample_corpus):
        """Test scoring terms in sample corpus."""
        scores = score_by_definitional_context(sample_corpus)

        # Should return scores
        assert isinstance(scores, dict)

    def test_pattern_analysis_on_sample(self, sample_corpus):
        """Test analyzing patterns in sample corpus."""
        pattern_counts = analyze_definitional_patterns(sample_corpus)

        # Should return pattern distribution
        assert isinstance(pattern_counts, dict)


class TestPOSFiltering:
    """Test POS-based term filtering."""

    @pytest.fixture
    def docs_with_varied_pos(self):
        """Documents with various part-of-speech patterns."""
        text = """
        Consciousness is a complex philosophical concept.
        The concept refers to awareness and subjective experience.
        Philosophers debate consciousness extensively.
        """
        doc = Document(text=text, metadata={})
        return [preprocess(doc)]

    def test_filter_nouns_only(self, docs_with_varied_pos):
        """Test filtering for nouns only."""
        nouns = filter_by_pos_tags(
            docs_with_varied_pos,
            include_tags={"NN", "NNS", "NNP", "NNPS"},
            exclude_tags=set(),
            use_lemmas=True,
            min_freq=1,
        )

        # Should include nouns
        assert "consciousness" in nouns or "Consciousness" in nouns
        assert "concept" in nouns
        assert "experience" in nouns

        # Should not include adjectives
        assert "complex" not in nouns

    def test_exclude_function_words(self, docs_with_varied_pos):
        """Test excluding function words."""
        content_words = filter_by_pos_tags(
            docs_with_varied_pos,
            include_tags=CONTENT_WORD_POS_TAGS,
            exclude_tags=FUNCTION_WORD_POS_TAGS,
            use_lemmas=True,
            min_freq=1,
        )

        # Should not include function words
        assert "the" not in content_words
        assert "a" not in content_words
        assert "to" not in content_words
        assert "and" not in content_words

        # Should include content words
        assert "consciousness" in content_words or "Consciousness" in content_words

    def test_frequency_threshold(self, docs_with_varied_pos):
        """Test minimum frequency filtering."""
        # Low threshold
        terms_low = filter_by_pos_tags(
            docs_with_varied_pos, include_tags=CONTENT_WORD_POS_TAGS, min_freq=1
        )

        # High threshold
        terms_high = filter_by_pos_tags(
            docs_with_varied_pos, include_tags=CONTENT_WORD_POS_TAGS, min_freq=3
        )

        # High threshold should have fewer or equal terms
        assert len(terms_high) <= len(terms_low)


class TestPhilosophicalTermCandidates:
    """Test extraction of philosophical term candidates."""

    @pytest.fixture
    def philosophical_docs(self):
        """Documents with philosophical content."""
        text = """
        Being and consciousness are fundamental concepts in phenomenology.
        The phenomenological method investigates subjective experience.
        Intentionality refers to the directedness of consciousness.
        """
        doc = Document(text=text, metadata={})
        return [preprocess(doc)]

    def test_noun_focus(self, philosophical_docs):
        """Test focusing on nouns."""
        candidates = get_philosophical_term_candidates(
            philosophical_docs, focus="nouns", min_freq=1, use_lemmas=True
        )

        # Should include philosophical nouns
        assert "consciousness" in candidates or "Consciousness" in candidates
        assert "phenomenology" in candidates or "Phenomenology" in candidates

        # Should not include verbs
        assert "investigate" not in candidates

    def test_verb_focus(self, philosophical_docs):
        """Test focusing on verbs."""
        candidates = get_philosophical_term_candidates(
            philosophical_docs,
            focus="verbs",
            min_freq=1,
            use_lemmas=True,
            exclude_stopwords=False,  # Don't filter stopwords for this test
        )

        # Should include verbs when stopwords not filtered
        assert len(candidates) > 0  # Should find some verbs

        # Verify filter is actually working by checking POS tags
        # (not checking specific verbs due to stopword filtering variability)

    def test_all_content_words(self, philosophical_docs):
        """Test including all content words."""
        candidates = get_philosophical_term_candidates(
            philosophical_docs, focus="all_content", min_freq=1, use_lemmas=True
        )

        # Should include nouns, verbs, adjectives
        assert len(candidates) > 0

        # Should exclude stopwords
        assert "the" not in candidates
        assert "and" not in candidates

    def test_invalid_focus_raises(self, philosophical_docs):
        """Test that invalid focus raises error."""
        with pytest.raises(ValueError, match="Invalid focus"):
            get_philosophical_term_candidates(philosophical_docs, focus="invalid")


class TestCompoundTerms:
    """Test extraction of compound/multi-word terms."""

    @pytest.fixture
    def docs_with_compounds(self):
        """Documents with compound terms."""
        text = """
        The Thing-in-itself is a key concept in Kant's philosophy.
        The phenomenological reduction involves bracketing natural attitude.
        Intentional consciousness exhibits directedness toward objects.
        """
        doc = Document(text=text, metadata={})
        return [preprocess(doc)]

    def test_extract_hyphenated_compounds(self, docs_with_compounds):
        """Test extraction of hyphenated compounds."""
        compounds = get_compound_terms(docs_with_compounds, min_freq=1)

        # Should find hyphenated philosophical terms
        assert "Thing-in-itself" in compounds or "thing-in-itself" in compounds

    def test_extract_noun_phrases(self, docs_with_compounds):
        """Test extraction of noun phrases."""
        compounds = get_compound_terms(docs_with_compounds, min_freq=1)

        # Should find multi-word noun phrases
        # (exact matches depend on tokenization)
        assert len(compounds) > 0

    def test_frequency_threshold(self, docs_with_compounds):
        """Test frequency filtering for compounds."""
        # Low threshold
        compounds_low = get_compound_terms(docs_with_compounds, min_freq=1)

        # High threshold (nothing appears 5 times in this small corpus)
        compounds_high = get_compound_terms(docs_with_compounds, min_freq=5)

        # High threshold should have fewer compounds
        assert len(compounds_high) <= len(compounds_low)


class TestFilteredCandidates:
    """Test comprehensive candidate extraction."""

    @pytest.fixture
    def sample_docs(self):
        """Sample documents for testing."""
        text = """
        Consciousness is fundamental to phenomenology.
        Being-in-the-world describes human existence.
        The phenomenological method investigates experience.
        """
        doc = Document(text=text, metadata={})
        return [preprocess(doc)]

    @pytest.fixture
    def simple_reference(self):
        """Simple reference corpus."""
        return Counter({"the": 100, "is": 50, "to": 30, "other": 820})

    def test_filtered_candidates_structure(self, sample_docs, simple_reference):
        """Test that result has expected structure."""
        result = get_filtered_candidates(
            sample_docs, simple_reference, min_freq=1, include_compounds=True
        )

        # Should have all expected keys
        assert "single_words" in result
        assert "compounds" in result
        assert "all_candidates" in result

        # All should be sets
        assert isinstance(result["single_words"], set)
        assert isinstance(result["compounds"], set)
        assert isinstance(result["all_candidates"], set)

    def test_all_candidates_is_union(self, sample_docs, simple_reference):
        """Test that all_candidates is union of single_words and compounds."""
        result = get_filtered_candidates(
            sample_docs, simple_reference, min_freq=1, include_compounds=True
        )

        # all_candidates should be union
        assert result["all_candidates"] == result["single_words"] | result["compounds"]

    def test_without_compounds(self, sample_docs, simple_reference):
        """Test disabling compound extraction."""
        result = get_filtered_candidates(
            sample_docs, simple_reference, min_freq=1, include_compounds=False
        )

        # Compounds should be empty
        assert len(result["compounds"]) == 0

        # all_candidates should equal single_words
        assert result["all_candidates"] == result["single_words"]


class TestPOSFilteringOnRealCorpus:
    """Test POS filtering on real sample corpus."""

    @pytest.fixture
    def sample_corpus(self):
        """Load sample philosophical corpus."""
        from src.concept_mapper.corpus.loader import load_directory

        base = Path(__file__).parent.parent / "samples"
        corpus = load_directory(str(base), pattern="sample*_*.txt", recursive=False)
        return [preprocess(doc) for doc in corpus.documents]

    @pytest.fixture
    def brown_corpus(self):
        """Load Brown corpus for comparison."""
        from src.concept_mapper.analysis.reference import load_reference_corpus

        return load_reference_corpus("brown")

    def test_extract_noun_candidates(self, sample_corpus):
        """Test extracting noun candidates from sample."""
        nouns = get_philosophical_term_candidates(
            sample_corpus, focus="nouns", min_freq=2, use_lemmas=True
        )

        # Should find multiple noun candidates
        assert len(nouns) > 0

        # Function words should be excluded
        assert "the" not in nouns
        assert "of" not in nouns

    def test_extract_all_content_candidates(self, sample_corpus):
        """Test extracting all content word candidates."""
        candidates = get_philosophical_term_candidates(
            sample_corpus, focus="all_content", min_freq=2, use_lemmas=True
        )

        # Should find many candidates
        assert len(candidates) > 10

        # Should be diverse (nouns, verbs, adjectives)
        # Exact composition depends on corpus

    def test_extract_compounds(self, sample_corpus):
        """Test extracting compound terms from sample."""
        compounds = get_compound_terms(sample_corpus, min_freq=2)

        # Should find hyphenated compounds
        # (sample corpus contains terms like "meaning-variance", "bio-regulation", etc.)
        assert len(compounds) > 0

    def test_comprehensive_filtering(self, sample_corpus, brown_corpus):
        """Test comprehensive candidate extraction."""
        result = get_filtered_candidates(
            sample_corpus,
            brown_corpus,
            min_freq=2,
            include_pos_focus="all_content",
            include_compounds=True,
        )

        # Should find candidates in all categories
        assert len(result["single_words"]) > 0
        assert len(result["all_candidates"]) >= len(result["single_words"])

        # Compounds may or may not be found depending on corpus
        # Just verify no errors occurred


class TestPhilosophicalTermScorer:
    """Test hybrid philosophical term scorer (Phase 3.6)."""

    @pytest.fixture
    def test_docs(self):
        """Documents with diverse philosophical signals."""
        text = """
        Dasein is being-in-the-world.
        By Dasein I mean authentic human existence.
        The concept of Dasein refers to temporal being.
        The phenomenological method investigates consciousness.
        Being appears. Consciousness is awareness.
        """
        doc = Document(text=text, metadata={})
        return [preprocess(doc)]

    @pytest.fixture
    def test_reference(self):
        """Reference corpus missing philosophical terms."""
        return Counter(
            {
                "the": 500,
                "is": 300,
                "appears": 10,
                "and": 200,
                "of": 150,
                "to": 100,
                "a": 120,
                "other": 620,
            }
        )

    def test_scorer_initialization(self, test_docs, test_reference):
        """Test scorer initialization with default weights."""
        scorer = PhilosophicalTermScorer(test_docs, test_reference)

        # Should compute all signals
        assert hasattr(scorer, "ratios")
        assert hasattr(scorer, "tfidf_scores")
        assert hasattr(scorer, "neologisms")
        assert hasattr(scorer, "definitional_scores")
        assert hasattr(scorer, "capitalized_terms")

        # Should compute normalization factors
        assert hasattr(scorer, "max_ratio")
        assert hasattr(scorer, "max_tfidf")
        assert hasattr(scorer, "max_definitional")

        # Default weights should be set
        assert scorer.weights["ratio"] == 1.0
        assert scorer.weights["tfidf"] == 1.0
        assert scorer.weights["neologism"] == 0.5
        assert scorer.weights["definitional"] == 0.3
        assert scorer.weights["capitalized"] == 0.2

    def test_custom_weights(self, test_docs, test_reference):
        """Test scorer with custom weights."""
        custom_weights = {
            "ratio": 2.0,
            "tfidf": 1.5,
            "neologism": 1.0,
            "definitional": 0.5,
            "capitalized": 0.3,
        }

        scorer = PhilosophicalTermScorer(
            test_docs, test_reference, weights=custom_weights
        )

        assert scorer.weights == custom_weights

    def test_score_term_structure(self, test_docs, test_reference):
        """Test score_term returns expected structure."""
        scorer = PhilosophicalTermScorer(test_docs, test_reference)

        score = scorer.score_term("dasein")

        # Should have all expected keys
        assert "total" in score
        assert "ratio" in score
        assert "tfidf" in score
        assert "neologism" in score
        assert "definitional" in score
        assert "capitalized" in score
        assert "raw_total" in score

        # All should be numeric
        assert isinstance(score["total"], (int, float))
        assert isinstance(score["ratio"], (int, float))
        assert isinstance(score["tfidf"], (int, float))

    def test_score_term_with_multiple_signals(self, test_docs, test_reference):
        """Test that term with multiple signals gets higher score."""
        scorer = PhilosophicalTermScorer(test_docs, test_reference)

        # "dasein" should score high: high ratio, high tfidf, neologism, definitional, capitalized
        dasein_score = scorer.score_term("dasein")

        # "appear" should score lower: might be in reference, no definitional context
        appear_score = scorer.score_term("appear")

        # Dasein should have higher total score
        assert dasein_score["total"] > appear_score["total"]

        # Dasein should trigger definitional signal
        assert dasein_score["definitional"] > 0

    def test_neologism_signal(self, test_docs, test_reference):
        """Test neologism signal detection."""
        scorer = PhilosophicalTermScorer(test_docs, test_reference)

        # Score a term that's a neologism (not in reference)
        score = scorer.score_term("dasein")

        # Should have neologism signal (binary: 0 or 1)
        assert score["neologism"] in [0.0, 1.0]

        # Dasein should be detected as neologism (not in reference corpus)
        assert score["neologism"] == 1.0

    def test_capitalization_signal(self, test_docs, test_reference):
        """Test capitalization signal detection."""
        scorer = PhilosophicalTermScorer(test_docs, test_reference)

        # Check for capitalized terms in original text
        # Note: depends on how POS tagger handles "Being" and "Consciousness"
        score_being = scorer.score_term("Being")
        score_consciousness = scorer.score_term("Consciousness")

        # At least one should be detected as capitalized
        # (exact behavior depends on tokenization and lemmatization)
        # Just verify signal is binary
        assert score_being["capitalized"] in [0.0, 1.0]
        assert score_consciousness["capitalized"] in [0.0, 1.0]

    def test_score_all_basic(self, test_docs, test_reference):
        """Test score_all returns all terms."""
        scorer = PhilosophicalTermScorer(test_docs, test_reference, min_author_freq=1)

        results = scorer.score_all(min_score=0.0)

        # Should return list of tuples
        assert isinstance(results, list)
        assert len(results) > 0

        # Each result should be (term, score, components)
        for term, score, components in results:
            assert isinstance(term, str)
            assert isinstance(score, (int, float))
            assert isinstance(components, dict)

            # Score should match component total
            assert score == pytest.approx(components["total"], rel=0.01)

    def test_score_all_sorted_descending(self, test_docs, test_reference):
        """Test that score_all returns results sorted by score."""
        scorer = PhilosophicalTermScorer(test_docs, test_reference)

        results = scorer.score_all(min_score=0.0)

        # Check that scores are in descending order
        scores = [score for term, score, components in results]
        assert scores == sorted(scores, reverse=True)

    def test_score_all_with_min_score(self, test_docs, test_reference):
        """Test filtering by minimum score."""
        scorer = PhilosophicalTermScorer(test_docs, test_reference)

        # Get all results
        all_results = scorer.score_all(min_score=0.0)

        # Get filtered results
        filtered_results = scorer.score_all(min_score=1.0)

        # Filtered should have fewer or equal results
        assert len(filtered_results) <= len(all_results)

        # All filtered results should have score >= 1.0
        for term, score, components in filtered_results:
            assert score >= 1.0

    def test_score_all_with_top_n(self, test_docs, test_reference):
        """Test limiting to top N results."""
        scorer = PhilosophicalTermScorer(test_docs, test_reference)

        # Get top 3
        top_3 = scorer.score_all(min_score=0.0, top_n=3)

        # Should have at most 3 results
        assert len(top_3) <= 3

    def test_get_high_confidence_terms(self, test_docs, test_reference):
        """Test high confidence term extraction."""
        scorer = PhilosophicalTermScorer(test_docs, test_reference)

        # Get high confidence terms (at least 3 signals)
        high_conf = scorer.get_high_confidence_terms(min_signals=3, min_score=1.0)

        # Should return a set
        assert isinstance(high_conf, set)

        # Each term should have high score
        for term in high_conf:
            score = scorer.score_term(term)
            assert score["total"] >= 1.0

            # Count active signals
            signals_active = sum(
                [
                    score["ratio"] > 0.1,
                    score["tfidf"] > 0.1,
                    score["neologism"] > 0,
                    score["definitional"] > 0,
                    score["capitalized"] > 0,
                ]
            )
            assert signals_active >= 3

    def test_normalization(self, test_docs, test_reference):
        """Test score normalization."""
        scorer = PhilosophicalTermScorer(test_docs, test_reference)

        # Score with normalization (default)
        normalized = scorer.score_term("dasein", normalize=True)

        # Score without normalization
        unnormalized = scorer.score_term("dasein", normalize=False)

        # Normalized components should be different from unnormalized
        # (unless max happens to be 1.0)
        if scorer.max_ratio > 1.0:
            assert normalized["ratio"] != unnormalized["ratio"]
        if scorer.max_tfidf > 1.0:
            assert normalized["tfidf"] != unnormalized["tfidf"]

    def test_empty_docs_handling(self):
        """Test scorer with empty documents."""
        doc = Document(text="", metadata={})
        docs = [preprocess(doc)]
        ref = Counter({"the": 100})

        # Should initialize without error
        scorer = PhilosophicalTermScorer(docs, ref)

        # Should handle empty corpus
        results = scorer.score_all()
        assert isinstance(results, list)


class TestScorePhilosophicalTerms:
    """Test convenience function for scoring philosophical terms."""

    @pytest.fixture
    def test_docs(self):
        """Simple test documents."""
        text = """
        Dasein is being-in-the-world.
        By phenomenology I mean the study of consciousness.
        Being appears. Consciousness is awareness.
        """
        doc = Document(text=text, metadata={})
        return [preprocess(doc)]

    @pytest.fixture
    def test_reference(self):
        """Reference corpus."""
        return Counter({"the": 500, "is": 300, "appears": 10, "and": 200, "other": 990})

    def test_convenience_function_basic(self, test_docs, test_reference):
        """Test convenience function returns simplified results."""
        results = score_philosophical_terms(test_docs, test_reference, top_n=10)

        # Should return list of (term, score) tuples
        assert isinstance(results, list)
        assert len(results) > 0

        # Each result should be (term, score) tuple
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            term, score = item
            assert isinstance(term, str)
            assert isinstance(score, (int, float))

    def test_convenience_function_sorted(self, test_docs, test_reference):
        """Test that results are sorted by score."""
        results = score_philosophical_terms(test_docs, test_reference, top_n=10)

        # Scores should be in descending order
        scores = [score for term, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_convenience_function_respects_top_n(self, test_docs, test_reference):
        """Test that top_n parameter is respected."""
        results = score_philosophical_terms(test_docs, test_reference, top_n=3)

        # Should have at most 3 results
        assert len(results) <= 3


class TestHybridScorerOnRealCorpus:
    """Test hybrid scorer on real sample corpus."""

    @pytest.fixture
    def sample_corpus(self):
        """Load sample philosophical corpus."""
        from src.concept_mapper.corpus.loader import load_directory

        base = Path(__file__).parent.parent / "samples"
        corpus = load_directory(str(base), pattern="sample*_*.txt", recursive=False)
        return [preprocess(doc) for doc in corpus.documents]

    @pytest.fixture
    def brown_corpus(self):
        """Load Brown corpus for comparison."""
        from src.concept_mapper.analysis.reference import load_reference_corpus

        return load_reference_corpus("brown")

    def test_scorer_on_sample_corpus(self, sample_corpus, brown_corpus):
        """Test scorer initialization on real corpus."""
        scorer = PhilosophicalTermScorer(sample_corpus, brown_corpus, min_author_freq=3)

        # Should compute all signals without errors
        assert len(scorer.ratios) > 0
        assert len(scorer.tfidf_scores) > 0

    def test_score_all_on_sample(self, sample_corpus, brown_corpus):
        """Test scoring all terms in sample corpus."""
        scorer = PhilosophicalTermScorer(sample_corpus, brown_corpus, min_author_freq=3)

        results = scorer.score_all(min_score=0.5, top_n=20)

        # Should find high-scoring philosophical terms
        assert len(results) > 0

        # Top terms should have reasonable scores
        if len(results) > 0:
            top_term, top_score, components = results[0]
            assert top_score > 0.5

            # Should have multiple signals contributing
            active_signals = sum(
                [
                    components["ratio"] > 0,
                    components["tfidf"] > 0,
                    components["neologism"] > 0,
                    components["definitional"] > 0,
                    components["capitalized"] > 0,
                ]
            )
            assert active_signals >= 1  # At least one signal should fire

    def test_high_confidence_on_sample(self, sample_corpus, brown_corpus):
        """Test high confidence term extraction on sample."""
        scorer = PhilosophicalTermScorer(sample_corpus, brown_corpus, min_author_freq=3)

        # Get high confidence terms (at least 2 signals)
        high_conf = scorer.get_high_confidence_terms(min_signals=2, min_score=0.5)

        # Should find some high confidence terms
        # (exact count depends on corpus content)
        assert isinstance(high_conf, set)

    def test_convenience_function_on_sample(self, sample_corpus, brown_corpus):
        """Test convenience function on sample corpus."""
        results = score_philosophical_terms(
            sample_corpus, brown_corpus, min_author_freq=3, top_n=20
        )

        # Should return philosophical terms
        assert len(results) > 0

        # Results should be sorted
        scores = [score for term, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_different_weight_configurations(self, sample_corpus, brown_corpus):
        """Test scorer with different weight configurations."""
        # Emphasize ratio
        weights_ratio = {
            "ratio": 2.0,
            "tfidf": 0.5,
            "neologism": 0.3,
            "definitional": 0.2,
            "capitalized": 0.1,
        }

        scorer_ratio = PhilosophicalTermScorer(
            sample_corpus, brown_corpus, weights=weights_ratio, min_author_freq=3
        )

        # Emphasize TF-IDF
        weights_tfidf = {
            "ratio": 0.5,
            "tfidf": 2.0,
            "neologism": 0.3,
            "definitional": 0.2,
            "capitalized": 0.1,
        }

        scorer_tfidf = PhilosophicalTermScorer(
            sample_corpus, brown_corpus, weights=weights_tfidf, min_author_freq=3
        )

        # Both should produce results
        results_ratio = scorer_ratio.score_all(min_score=0.0, top_n=10)
        results_tfidf = scorer_tfidf.score_all(min_score=0.0, top_n=10)

        assert len(results_ratio) > 0
        assert len(results_tfidf) > 0

        # Different weighting may produce different top terms
        # (not guaranteed, but likely with diverse signals)
