"""
Tests for text reconstruction with proper spacing and punctuation.
"""

from src.concept_mapper.transformations.text_reconstruction import TextReconstructor


class TestTextReconstructor:
    """Tests for TextReconstructor class."""

    def test_rebuild_basic(self):
        """Test basic word replacement."""
        reconstructor = TextReconstructor()
        tokens = ["The", "cat", "ran", "quickly", "."]
        replacements = [(2, "sprinted"), (3, "swiftly")]

        result = reconstructor.rebuild(tokens, replacements)
        assert result == "The cat sprinted swiftly."

    def test_rebuild_no_replacements(self):
        """Test rebuild with no replacements."""
        reconstructor = TextReconstructor()
        tokens = ["The", "cat", "sat", "."]
        replacements = []

        result = reconstructor.rebuild(tokens, replacements)
        assert result == "The cat sat."

    def test_rebuild_all_tokens(self):
        """Test replacing all tokens."""
        reconstructor = TextReconstructor()
        tokens = ["The", "cat", "sat"]
        replacements = [(0, "A"), (1, "dog"), (2, "stood")]

        result = reconstructor.rebuild(tokens, replacements)
        assert result == "A dog stood"

    def test_punctuation_attachment(self):
        """Test punctuation attaches correctly."""
        reconstructor = TextReconstructor()
        tokens = ["Hello", ",", "world", "!", "How", "are", "you", "?"]
        replacements = []

        result = reconstructor.rebuild(tokens, replacements)
        assert result == "Hello, world! How are you?"

    def test_opening_punctuation(self):
        """Test opening punctuation (no space after)."""
        reconstructor = TextReconstructor()
        tokens = ["The", "test", "(", "example", ")", "works", "."]
        replacements = []

        result = reconstructor.rebuild(tokens, replacements)
        assert result == "The test (example) works."

    def test_contractions(self):
        """Test contraction handling."""
        reconstructor = TextReconstructor()
        tokens = ["It", "'s", "great", ",", "is", "n't", "it", "?"]
        replacements = []

        result = reconstructor.rebuild(tokens, replacements)
        assert result == "It's great, isn't it?"

    def test_brackets_and_braces(self):
        """Test various bracket types."""
        reconstructor = TextReconstructor()
        tokens = ["Items", "[", "a", ",", "b", "]", "and", "{", "x", "}", "."]
        replacements = []

        result = reconstructor.rebuild(tokens, replacements)
        assert result == "Items [a, b] and {x}."

    def test_multiple_punctuation(self):
        """Test multiple punctuation marks."""
        reconstructor = TextReconstructor()
        tokens = ["What", "?", "!", "Really", "?"]
        replacements = []

        result = reconstructor.rebuild(tokens, replacements)
        # Multiple punctuation should all attach
        assert result == "What?! Really?"

    def test_empty_tokens(self):
        """Test empty token list."""
        reconstructor = TextReconstructor()
        result = reconstructor.rebuild([], [])
        assert result == ""

    def test_single_token(self):
        """Test single token."""
        reconstructor = TextReconstructor()
        result = reconstructor.rebuild(["Hello"], [])
        assert result == "Hello"

    def test_semicolon_and_colon(self):
        """Test semicolon and colon attachment."""
        reconstructor = TextReconstructor()
        tokens = ["First", "part", ";", "second", "part", ":", "conclusion", "."]
        replacements = []

        result = reconstructor.rebuild(tokens, replacements)
        assert result == "First part; second part: conclusion."

    def test_rebuild_phrases_basic(self):
        """Test basic phrase replacement."""
        reconstructor = TextReconstructor()
        tokens = ["The", "body", "without", "organs", "resists", "."]
        phrase_replacements = [(1, 4, "medium")]

        result = reconstructor.rebuild_phrases(tokens, phrase_replacements)
        assert result == "The medium resists."

    def test_rebuild_phrases_multiple(self):
        """Test multiple phrase replacements."""
        reconstructor = TextReconstructor()
        tokens = [
            "The",
            "body",
            "without",
            "organs",
            "and",
            "the",
            "body",
            "without",
            "organs",
            "differ",
            ".",
        ]
        phrase_replacements = [(1, 4, "medium"), (6, 9, "entity")]

        result = reconstructor.rebuild_phrases(tokens, phrase_replacements)
        assert result == "The medium and the entity differ."

    def test_rebuild_phrases_overlapping_order(self):
        """Test that replacements are applied in correct order."""
        reconstructor = TextReconstructor()
        tokens = ["a", "b", "c", "d", "e"]
        # Replace from right to left to avoid index issues
        phrase_replacements = [(1, 3, "X"), (3, 5, "Y")]

        result = reconstructor.rebuild_phrases(tokens, phrase_replacements)
        # With proper ordering: [a, b, c, d, e] -> [a, b, c, Y] -> [a, X, Y]
        assert result == "a X Y"

    def test_rebuild_phrases_entire_sentence(self):
        """Test replacing entire sentence."""
        reconstructor = TextReconstructor()
        tokens = ["The", "quick", "brown", "fox"]
        phrase_replacements = [(0, 4, "Animal")]

        result = reconstructor.rebuild_phrases(tokens, phrase_replacements)
        assert result == "Animal"

    def test_rebuild_phrases_with_punctuation(self):
        """Test phrase replacement preserves punctuation."""
        reconstructor = TextReconstructor()
        tokens = [
            "The",
            "body",
            "without",
            "organs",
            ",",
            "according",
            "to",
            "Deleuze",
            ",",
            "resists",
            ".",
        ]
        phrase_replacements = [(1, 4, "medium")]

        result = reconstructor.rebuild_phrases(tokens, phrase_replacements)
        assert result == "The medium, according to Deleuze, resists."

    def test_replacement_with_capitalization(self):
        """Test replacement preserves surrounding capitalization context."""
        reconstructor = TextReconstructor()
        tokens = ["The", "CAT", "sat", "down", "."]
        replacements = [(1, "DOG")]  # Keep all caps

        result = reconstructor.rebuild(tokens, replacements)
        assert result == "The DOG sat down."

    def test_phrase_replacement_sentence_initial(self):
        """Test phrase replacement at sentence start."""
        reconstructor = TextReconstructor()
        tokens = ["Body", "without", "organs", "is", "a", "concept", "."]
        phrase_replacements = [(0, 3, "Medium")]

        result = reconstructor.rebuild_phrases(tokens, phrase_replacements)
        assert result == "Medium is a concept."
