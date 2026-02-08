"""
Tests for synonym replacement with inflection preservation.
"""

from src.concept_mapper.corpus.models import Document
from src.concept_mapper.preprocessing import preprocess
from src.concept_mapper.transformations.replacement import (
    ReplacementSpec,
    SynonymReplacer,
)


class TestReplacementSpec:
    """Tests for ReplacementSpec dataclass."""

    def test_single_word_spec(self):
        """Test specification for single-word replacement."""
        spec = ReplacementSpec("run", "sprint")

        assert not spec.is_phrase
        assert not spec.target_is_phrase
        assert spec.source_lemma == "run"
        assert spec.target_lemma == "sprint"

    def test_phrase_to_single_spec(self):
        """Test specification for phrase→single replacement."""
        spec = ReplacementSpec(["body", "without", "organ"], "medium")

        assert spec.is_phrase
        assert not spec.target_is_phrase

    def test_phrase_to_phrase_spec(self):
        """Test specification for phrase→phrase replacement."""
        spec = ReplacementSpec(
            ["body", "without", "organ"], ["blank", "resistant", "field"]
        )

        assert spec.is_phrase
        assert spec.target_is_phrase


class TestSynonymReplacer:
    """Tests for SynonymReplacer class."""

    def test_replace_single_word_basic(self):
        """Test basic single-word replacement."""
        doc = Document(text="The cat runs quickly.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("cat", "dog")
        result = replacer.replace_in_document(spec, processed)

        assert "dog" in result
        assert "cat" not in result

    def test_replace_with_inflection_past_tense(self):
        """Test replacement preserves past tense."""
        doc = Document(text="The cat ran quickly.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("run", "sprint")
        result = replacer.replace_in_document(spec, processed)

        # "ran" should become "sprinted"
        assert "sprinted" in result
        assert "ran" not in result

    def test_replace_with_inflection_present_progressive(self):
        """Test replacement preserves present progressive (-ing)."""
        doc = Document(text="The cat is running.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("run", "sprint")
        result = replacer.replace_in_document(spec, processed)

        # "running" should become "sprinting"
        assert "sprinting" in result
        assert "running" not in result

    def test_replace_plural_noun(self):
        """Test replacement preserves plural form."""
        doc = Document(text="The cats are here.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("cat", "dog")
        result = replacer.replace_in_document(spec, processed)

        # "cats" should become "dogs"
        assert "dogs" in result
        assert "cats" not in result

    def test_replace_comparative_adjective(self):
        """Test replacement preserves comparative form."""
        doc = Document(text="This is quicker than that.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("quick", "fast")
        result = replacer.replace_in_document(spec, processed)

        # "quicker" should become "faster"
        assert "faster" in result or "more fast" in result
        assert "quicker" not in result

    def test_capitalization_sentence_initial(self):
        """Test capitalization preserved at sentence start."""
        doc = Document(text="Running is good.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("run", "sprint")
        result = replacer.replace_in_document(spec, processed)

        # Should capitalize "Sprinting"
        assert result.startswith("Sprinting")

    def test_capitalization_all_caps(self):
        """Test all-caps preservation."""
        doc = Document(text="The RUN command works.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("run", "sprint")
        result = replacer.replace_in_document(spec, processed)

        # Should preserve all-caps
        assert "SPRINT" in result

    def test_multiple_replacements_same_word(self):
        """Test replacing multiple occurrences."""
        doc = Document(text="The cat ran. The cat jumped.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("cat", "dog")
        result = replacer.replace_in_document(spec, processed)

        # Both "cat" occurrences should be replaced
        assert result.count("dog") == 2
        assert "cat" not in result

    def test_no_replacement_when_not_found(self):
        """Test text unchanged when lemma not found."""
        doc = Document(text="The bird flies.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("cat", "dog")
        result = replacer.replace_in_document(spec, processed)

        # Should be unchanged
        assert result == "The bird flies."

    def test_punctuation_preserved(self):
        """Test punctuation is preserved correctly."""
        doc = Document(text="The cat runs, jumps, and plays!")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("cat", "dog")
        result = replacer.replace_in_document(spec, processed)

        # Punctuation should be preserved
        assert "dog runs, jumps, and plays!" in result

    def test_replace_phrase_to_single_word(self):
        """Test replacing multi-word phrase with single word."""
        doc = Document(text="The body without organs is a concept.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec(["body", "without", "organ"], "medium")
        result = replacer.replace_in_document(spec, processed)

        # "body without organs" should become "medium"
        assert "medium" in result
        assert "body without organs" not in result

    def test_replace_phrase_preserves_plurality(self):
        """Test phrase replacement preserves plurality of head word."""
        doc = Document(text="The bodies without organs are important.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec(["body", "without", "organ"], "medium")
        result = replacer.replace_in_document(spec, processed)

        # "bodies" (plural) should make "medium" → "mediums"
        assert "mediums" in result or "media" in result  # Both are valid plurals

    def test_replace_phrase_to_phrase(self):
        """Test replacing multi-word phrase with another phrase."""
        doc = Document(text="The body without organs is a concept.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec(
            ["body", "without", "organ"], ["blank", "resistant", "field"]
        )
        result = replacer.replace_in_document(spec, processed)

        # Should replace with new phrase
        assert "blank resistant" in result
        assert "field" in result
        assert "body without organs" not in result

    def test_replace_phrase_multiple_occurrences(self):
        """Test replacing multiple phrase occurrences."""
        doc = Document(
            text="The body without organs differs from the body without organs."
        )
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec(["body", "without", "organ"], "medium")
        result = replacer.replace_in_document(spec, processed)

        # Both occurrences should be replaced
        assert result.count("medium") == 2

    def test_phrase_replacement_sentence_initial_capitalization(self):
        """Test phrase replacement capitalizes first word of sentence."""
        doc = Document(text="Body without organs is a concept.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec(["body", "without", "organ"], "medium")
        result = replacer.replace_in_document(spec, processed)

        # Should capitalize "Medium"
        assert result.startswith("Medium")

    def test_case_insensitive_matching_default(self):
        """Test case-insensitive matching by default."""
        doc = Document(text="The CAT runs.")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("cat", "dog")
        result = replacer.replace_in_document(spec, processed)

        # Should match "CAT" and preserve all-caps
        assert "DOG" in result

    def test_complex_sentence_structure(self):
        """Test replacement in complex sentence with multiple clauses."""
        doc = Document(
            text="When the cat runs quickly, the dog watches, and the bird flies away."
        )
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("cat", "mouse")
        result = replacer.replace_in_document(spec, processed)

        assert "mouse" in result
        assert "cat" not in result
        # Other words should be preserved
        assert "dog" in result
        assert "bird" in result

    def test_empty_document(self):
        """Test replacement on empty document."""
        doc = Document(text="")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("cat", "dog")
        result = replacer.replace_in_document(spec, processed)

        assert result == ""

    def test_only_punctuation(self):
        """Test document with only punctuation."""
        doc = Document(text="...")
        processed = preprocess(doc)

        replacer = SynonymReplacer()
        spec = ReplacementSpec("cat", "dog")
        result = replacer.replace_in_document(spec, processed)

        assert result == "..."
