"""
Tests for pos_tagger module.

Demonstrates that pure, composable functions are easy to test.
"""

import pytest
import pos_tagger as pt


class TestCoreFunctions:
    """Test core pure functions."""

    def test_tokenize_words(self):
        text = "Hello, world! How are you?"
        tokens = pt.tokenize_words(text)
        assert isinstance(tokens, list)
        assert len(tokens) == 8  # Punctuation counted separately
        assert tokens[0] == "Hello"
        assert "world" in tokens

    def test_tokenize_sentences(self):
        text = "First sentence. Second sentence. Third sentence."
        sentences = pt.tokenize_sentences(text)
        assert len(sentences) == 3
        assert "First" in sentences[0]

    def test_tag_parts_of_speech(self):
        tokens = ["The", "cat", "runs"]
        tagged = pt.tag_parts_of_speech(tokens)
        assert isinstance(tagged, list)
        assert len(tagged) == 3
        assert all(isinstance(item, tuple) for item in tagged)
        assert all(len(item) == 2 for item in tagged)

    def test_extract_words_by_pos(self):
        tagged = [("cat", "NN"), ("runs", "VBZ"), ("quickly", "RB")]
        verbs = pt.extract_words_by_pos(tagged, "V")
        assert verbs == ["runs"]

        nouns = pt.extract_words_by_pos(tagged, "N")
        assert nouns == ["cat"]

        adverbs = pt.extract_words_by_pos(tagged, "R")
        assert adverbs == ["quickly"]


class TestFilteringFunctions:
    """Test filtering and frequency functions."""

    def test_get_common_verbs(self):
        common = pt.get_common_verbs()
        assert isinstance(common, set)
        assert "be" in common
        assert "have" in common
        assert len(common) > 0

    def test_get_stopwords_set(self):
        stopwords = pt.get_stopwords_set()
        assert isinstance(stopwords, set)
        assert "the" in stopwords
        assert "a" in stopwords

    def test_filter_common_words(self):
        words = ["run", "be", "jump", "the", "have"]
        common = {"be", "have"}
        stop_words = {"the"}

        filtered = pt.filter_common_words(words, common, stop_words)
        assert filtered == ["run", "jump"]

    def test_calculate_frequency_distribution(self):
        words = ["cat", "dog", "cat", "bird", "cat"]
        freq_dist = pt.calculate_frequency_distribution(words)
        assert freq_dist["cat"] == 3
        assert freq_dist["dog"] == 1
        assert freq_dist["bird"] == 1

    def test_get_most_common(self):
        words = ["a", "b", "a", "c", "a", "b"]
        freq_dist = pt.calculate_frequency_distribution(words)
        most_common = pt.get_most_common(freq_dist, 2)

        assert len(most_common) == 2
        assert most_common[0] == ("a", 3)
        assert most_common[1] == ("b", 2)


class TestSearchFunctions:
    """Test search functionality."""

    def test_find_sentences_with_term(self):
        sentences = [
            "The cat runs quickly.",
            "The dog barks loudly.",
            "The cat sleeps peacefully.",
        ]
        results = pt.find_sentences_with_term(sentences, "cat")
        assert len(results) == 2
        assert "cat" in results[0].lower()
        assert "cat" in results[1].lower()

    def test_find_sentences_case_insensitive(self):
        sentences = ["The CAT runs.", "The dog barks."]
        results = pt.find_sentences_with_term(sentences, "cat")
        assert len(results) == 1


class TestComposedPipelines:
    """Test composed pipeline functions."""

    def test_get_most_common_verbs(self):
        tokens = ["run", "jump", "run", "swim"]
        tagged = pt.tag_parts_of_speech(tokens)
        verbs = pt.get_most_common_verbs(tagged, 2)

        assert isinstance(verbs, list)
        assert len(verbs) <= 2
        assert all(isinstance(item, tuple) for item in verbs)

    def test_get_content_rich_verbs_filters_common_verbs(self):
        # Create a sample with common verbs that should be filtered
        tokens = ["be", "run", "have", "jump", "be", "run"]
        tagged = pt.tag_parts_of_speech(tokens)
        verbs = pt.get_content_rich_verbs(tagged, 10)

        # "be" and "have" should be filtered out
        verb_words = [v[0] for v in verbs]
        assert "be" not in verb_words
        assert "have" not in verb_words


class TestMainPipeline:
    """Test main pipeline functions."""

    def test_run_pipeline_basic(self):
        text = "The cat runs. The dog jumps."
        result = pt.run_pipeline(text, top_n=5)

        assert isinstance(result, dict)
        assert "tokens" in result
        assert "token_count" in result
        assert "tagged" in result
        assert "all_verbs" in result
        assert "content_verbs" in result
        assert "nouns" in result
        assert "adjectives" in result
        assert "adverbs" in result

    def test_run_pipeline_token_count(self):
        text = "One two three four five."
        result = pt.run_pipeline(text, top_n=5)
        assert result["token_count"] == 6  # 5 words + 1 period

    def test_run_pipeline_returns_correct_structure(self):
        text = "The quick brown fox jumps."
        result = pt.run_pipeline(text, top_n=3)

        # Check that all verb-related fields are lists of tuples
        assert isinstance(result["all_verbs"], list)
        assert isinstance(result["content_verbs"], list)
        assert isinstance(result["nouns"], list)
        assert isinstance(result["adjectives"], list)


class TestIntegration:
    """Integration tests using actual text."""

    def test_full_pipeline_on_sample_text(self):
        sample = """
        The proletariat represents the universal class in capitalist society.
        Workers are alienated from their labor and from each other.
        Class consciousness develops through historical struggle.
        """

        result = pt.run_pipeline(sample, top_n=5)

        assert result["token_count"] > 0
        assert len(result["all_verbs"]) > 0
        assert len(result["nouns"]) > 0

        # Check that key terms appear
        noun_words = [n[0] for n in result["nouns"]]
        assert any(
            word in noun_words for word in ["proletariat", "class", "workers", "labor"]
        )


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
