"""
Tests for analysis modules: frequency, reference corpus, TF-IDF.
"""

import pytest
from collections import Counter

from src.concept_mapper.corpus import ProcessedDocument
from src.concept_mapper.analysis import (
    word_frequencies,
    pos_filtered_frequencies,
    corpus_frequencies,
    document_frequencies,
    get_vocabulary,
    load_reference_corpus,
    get_reference_vocabulary,
    get_reference_size,
    tf,
    idf,
    tfidf,
    corpus_tfidf_scores,
    document_tfidf_scores,
)


class TestFrequency:
    """Test frequency distribution functions."""

    @pytest.fixture
    def sample_doc(self):
        """Create a sample processed document."""
        return ProcessedDocument(
            raw_text="The cat sat. The cat ran.",
            tokens=["The", "cat", "sat", ".", "The", "cat", "ran", "."],
            pos_tags=[
                ("The", "DT"),
                ("cat", "NN"),
                ("sat", "VBD"),
                (".", "."),
                ("The", "DT"),
                ("cat", "NN"),
                ("ran", "VBD"),
                (".", "."),
            ],
            lemmas=["the", "cat", "sit", ".", "the", "cat", "run", "."],
        )

    def test_word_frequencies_tokens(self, sample_doc):
        """Test word frequency counting with tokens."""
        freq = word_frequencies(sample_doc, use_lemmas=False)

        assert freq["the"] == 2
        assert freq["cat"] == 2
        assert freq["sat"] == 1
        assert freq["ran"] == 1

    def test_word_frequencies_lemmas(self, sample_doc):
        """Test word frequency counting with lemmas."""
        freq = word_frequencies(sample_doc, use_lemmas=True)

        assert freq["the"] == 2
        assert freq["cat"] == 2
        assert freq["sit"] == 1  # sat -> sit
        assert freq["run"] == 1  # ran -> run

    def test_word_frequencies_case_sensitivity(self):
        """Test case handling in frequency counting."""
        doc = ProcessedDocument(
            raw_text="Cat CAT cat",
            tokens=["Cat", "CAT", "cat"],
            lemmas=["cat", "cat", "cat"],
        )

        # Lowercase (default)
        freq = word_frequencies(doc, lowercase=True)
        assert freq["cat"] == 3

        # Case-sensitive
        freq = word_frequencies(doc, lowercase=False)
        assert freq["Cat"] == 1
        assert freq["CAT"] == 1
        assert freq["cat"] == 1

    def test_pos_filtered_frequencies(self, sample_doc):
        """Test filtering frequencies by POS tag."""
        # Get only nouns
        noun_freq = pos_filtered_frequencies(sample_doc, {"NN", "NNS"})
        assert noun_freq["cat"] == 2
        assert "the" not in noun_freq

        # Get only verbs
        verb_freq = pos_filtered_frequencies(sample_doc, {"VBD"})
        assert "sat" in verb_freq or "sit" in verb_freq

    def test_corpus_frequencies(self):
        """Test aggregating frequencies across corpus."""
        docs = [
            ProcessedDocument(
                raw_text="cat dog", tokens=["cat", "dog"], lemmas=["cat", "dog"]
            ),
            ProcessedDocument(
                raw_text="cat bird", tokens=["cat", "bird"], lemmas=["cat", "bird"]
            ),
        ]

        freq = corpus_frequencies(docs)
        assert freq["cat"] == 2
        assert freq["dog"] == 1
        assert freq["bird"] == 1

    def test_document_frequencies(self):
        """Test counting documents containing terms."""
        docs = [
            ProcessedDocument(
                raw_text="cat cat", tokens=["cat", "cat"], lemmas=["cat", "cat"]
            ),  # cat appears 2x but in 1 doc
            ProcessedDocument(
                raw_text="cat dog", tokens=["cat", "dog"], lemmas=["cat", "dog"]
            ),
            ProcessedDocument(
                raw_text="dog bird", tokens=["dog", "bird"], lemmas=["dog", "bird"]
            ),
        ]

        doc_freq = document_frequencies(docs)
        assert doc_freq["cat"] == 2  # in 2 documents
        assert doc_freq["dog"] == 2  # in 2 documents
        assert doc_freq["bird"] == 1  # in 1 document

    def test_get_vocabulary(self):
        """Test extracting unique vocabulary."""
        docs = [
            ProcessedDocument(
                raw_text="cat dog", tokens=["cat", "dog"], lemmas=["cat", "dog"]
            ),
            ProcessedDocument(
                raw_text="cat bird", tokens=["cat", "bird"], lemmas=["cat", "bird"]
            ),
        ]

        vocab = get_vocabulary(docs)
        assert vocab == {"cat", "dog", "bird"}


class TestReferenceCorpus:
    """Test reference corpus loading."""

    def test_load_reference_corpus(self, tmp_path):
        """Test loading Brown corpus."""
        # This will compute and cache
        freq = load_reference_corpus("brown", cache=True, cache_dir=tmp_path)

        assert isinstance(freq, Counter)
        assert len(freq) > 10000  # Brown has many unique words
        assert freq["the"] > freq["philosophy"]  # "the" is more common

    def test_load_reference_corpus_caching(self, tmp_path):
        """Test that reference corpus is cached."""
        # First load
        freq1 = load_reference_corpus("brown", cache=True, cache_dir=tmp_path)

        # Second load should use cache
        freq2 = load_reference_corpus("brown", cache=True, cache_dir=tmp_path)

        assert freq1 == freq2

        # Check cache file exists
        cache_file = tmp_path / "cache" / "brown_corpus_freqs.json"
        assert cache_file.exists()

    def test_load_reference_corpus_invalid_name(self):
        """Test loading unsupported corpus."""
        with pytest.raises(ValueError, match="Unsupported reference corpus"):
            load_reference_corpus("invalid_corpus")

    def test_get_reference_vocabulary(self):
        """Test getting reference vocabulary."""
        vocab = get_reference_vocabulary("brown")

        assert isinstance(vocab, set)
        assert len(vocab) > 10000
        assert "the" in vocab
        assert "philosophy" in vocab

    def test_get_reference_size(self):
        """Test getting reference corpus size."""
        size = get_reference_size("brown")

        assert isinstance(size, int)
        assert size > 1000000  # Brown has over 1M words


class TestTFIDF:
    """Test TF-IDF calculations."""

    @pytest.fixture
    def sample_corpus(self):
        """Create a sample corpus for TF-IDF tests."""
        return [
            ProcessedDocument(
                raw_text="cat cat dog",
                tokens=["cat", "cat", "dog"],
                lemmas=["cat", "cat", "dog"],
            ),
            ProcessedDocument(
                raw_text="dog bird", tokens=["dog", "bird"], lemmas=["dog", "bird"]
            ),
            ProcessedDocument(
                raw_text="bird bird fish",
                tokens=["bird", "bird", "fish"],
                lemmas=["bird", "bird", "fish"],
            ),
        ]

    def test_tf_calculation(self, sample_corpus):
        """Test term frequency calculation."""
        doc = sample_corpus[0]  # "cat cat dog"

        # cat appears 2/3 times
        assert tf("cat", doc) == pytest.approx(2 / 3)
        # dog appears 1/3 times
        assert tf("dog", doc) == pytest.approx(1 / 3)
        # bird doesn't appear
        assert tf("bird", doc) == 0.0

    def test_idf_calculation(self, sample_corpus):
        """Test inverse document frequency calculation."""
        # cat appears in 1/3 docs
        cat_idf = idf("cat", sample_corpus)
        assert cat_idf == pytest.approx(1.0986122886681098)  # log(3/1)

        # dog appears in 2/3 docs
        dog_idf = idf("dog", sample_corpus)
        assert dog_idf == pytest.approx(0.4054651081081644)  # log(3/2)

        # bird appears in 2/3 docs
        bird_idf = idf("bird", sample_corpus)
        assert bird_idf == pytest.approx(0.4054651081081644)  # log(3/2)

        # nonexistent term
        assert idf("nonexistent", sample_corpus) == 0.0

    def test_tfidf_calculation(self, sample_corpus):
        """Test TF-IDF calculation."""
        doc = sample_corpus[0]  # "cat cat dog"

        # cat: high TF (2/3), high IDF (rare)
        cat_tfidf = tfidf("cat", doc, sample_corpus)
        assert cat_tfidf > 0.7

        # dog: low TF (1/3), medium IDF
        dog_tfidf = tfidf("dog", doc, sample_corpus)
        assert dog_tfidf < cat_tfidf

        # bird: TF=0 (not in doc)
        assert tfidf("bird", doc, sample_corpus) == 0.0

    def test_corpus_tfidf_scores(self, sample_corpus):
        """Test corpus-wide TF-IDF scoring."""
        scores = corpus_tfidf_scores(sample_corpus)

        assert "cat" in scores
        assert "dog" in scores
        assert "bird" in scores
        assert "fish" in scores

        # cat should have high score (frequent in one doc, rare overall)
        assert scores["cat"] > scores["dog"]

    def test_corpus_tfidf_with_min_score(self, sample_corpus):
        """Test filtering TF-IDF scores by minimum threshold."""
        scores = corpus_tfidf_scores(sample_corpus, min_score=0.5)

        # Only high-scoring terms included
        assert len(scores) < len(corpus_frequencies(sample_corpus))

    def test_document_tfidf_scores(self, sample_corpus):
        """Test TF-IDF scores for specific document."""
        doc = sample_corpus[0]  # "cat cat dog"
        scores = document_tfidf_scores(doc, sample_corpus)

        assert "cat" in scores
        assert "dog" in scores
        assert "bird" not in scores  # not in this document
        assert scores["cat"] > scores["dog"]

    def test_tfidf_with_lemmas(self):
        """Test TF-IDF using lemmas instead of surface forms."""
        docs = [
            ProcessedDocument(
                raw_text="running runs",
                tokens=["running", "runs"],
                lemmas=["run", "run"],
            ),
            ProcessedDocument(raw_text="jumping", tokens=["jumping"], lemmas=["jump"]),
        ]

        # Using lemmas, both "running" and "runs" become "run"
        freq = word_frequencies(docs[0], use_lemmas=True)
        assert freq["run"] == 2

        # TF should reflect lemmatized counts
        tf_score = tf("run", docs[0], use_lemmas=True)
        assert tf_score == 1.0  # run appears 2/2 times


class TestPhilosophicalTerms:
    """Test frequency analysis on philosophical corpus."""

    @pytest.fixture
    def philosophical_doc(self):
        """Create a document with philosophical terms."""
        return ProcessedDocument(
            raw_text="Daseinology examines being. Daseinology reveals temporalization. Temporalization structures existence.",
            tokens=[
                "Daseinology",
                "examines",
                "being",
                ".",
                "Daseinology",
                "reveals",
                "temporalization",
                ".",
                "Temporalization",
                "structures",
                "existence",
                ".",
            ],
            pos_tags=[
                ("Daseinology", "NN"),
                ("examines", "VBZ"),
                ("being", "NN"),
                (".", "."),
                ("Daseinology", "NN"),
                ("reveals", "VBZ"),
                ("temporalization", "NN"),
                (".", "."),
                ("Temporalization", "NN"),
                ("structures", "VBZ"),
                ("existence", "NN"),
                (".", "."),
            ],
            lemmas=[
                "daseinology",
                "examine",
                "being",
                ".",
                "daseinology",
                "reveal",
                "temporalization",
                ".",
                "temporalization",
                "structure",
                "existence",
                ".",
            ],
        )

    def test_philosophical_term_frequencies(self, philosophical_doc):
        """Test counting philosophical neologisms."""
        freq = word_frequencies(philosophical_doc, use_lemmas=True)

        assert freq["daseinology"] == 2
        assert freq["temporalization"] == 2
        assert freq["being"] == 1
        assert freq["existence"] == 1

    def test_philosophical_term_filtering(self, philosophical_doc):
        """Test extracting only nouns (philosophical concepts)."""
        noun_freq = pos_filtered_frequencies(
            philosophical_doc, {"NN", "NNS"}, use_lemmas=True
        )

        # Should include philosophical terms
        assert "daseinology" in noun_freq
        assert "temporalization" in noun_freq

        # Should exclude verbs
        assert "examine" not in noun_freq
        assert "reveal" not in noun_freq
