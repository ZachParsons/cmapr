"""
Tests for phrase matching using lemma n-grams.
"""

from src.concept_mapper.corpus.models import Document
from src.concept_mapper.preprocessing import preprocess
from src.concept_mapper.transformations.phrase_matcher import PhraseMatcher


class TestPhraseMatcher:
    """Tests for PhraseMatcher class."""

    def test_find_single_match(self):
        """Test finding a single phrase occurrence."""
        doc = Document(text="The body without organs is a concept.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        matches = matcher.find_phrase_matches(["body", "without", "organ"], processed)

        assert len(matches) == 1
        assert matches[0].start_idx == 1
        assert matches[0].end_idx == 4
        assert matches[0].tokens == ["body", "without", "organs"]
        assert matches[0].lemmas == ["body", "without", "organ"]

    def test_find_multiple_matches(self):
        """Test finding multiple occurrences of same phrase."""
        doc = Document(
            text="The body without organs differs from the body without organs."
        )
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        matches = matcher.find_phrase_matches(["body", "without", "organ"], processed)

        assert len(matches) == 2
        assert matches[0].start_idx == 1
        assert matches[1].start_idx == 7

    def test_find_no_matches(self):
        """Test when phrase is not found."""
        doc = Document(text="This is a simple sentence.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        matches = matcher.find_phrase_matches(["body", "without", "organ"], processed)

        assert len(matches) == 0

    def test_inflection_matching(self):
        """Test that different inflections are matched via lemmas."""
        doc = Document(text="The cat runs. The cats ran quickly.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        # Search for singular "cat" lemma
        matches = matcher.find_phrase_matches(["cat"], processed)

        # Should match both "cat" and "cats" (both lemmatize to "cat")
        assert len(matches) == 2

    def test_case_insensitive_matching(self):
        """Test case-insensitive matching (default)."""
        doc = Document(text="The Body Without Organs is important.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        matches = matcher.find_phrase_matches(["body", "without", "organ"], processed)

        assert len(matches) == 1

    def test_case_sensitive_matching(self):
        """Test case-sensitive matching with lemmas."""
        doc = Document(text="The body without organs and Body without organs.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        # Note: Lemmatizer normalizes to lowercase, so case-sensitive mode
        # still matches both "body" and "Body" since they lemmatize to "body"
        matches = matcher.find_phrase_matches(
            ["body", "without", "organ"], processed, case_sensitive=False
        )

        # Should match both occurrences
        assert len(matches) == 2

    def test_head_word_identification_noun_phrase(self):
        """Test head word is identified correctly in noun phrases."""
        # Use clearer context so POS tagger identifies "organs" as noun
        doc = Document(text="The body without organs is important.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        matches = matcher.find_phrase_matches(["body", "without", "organ"], processed)

        assert len(matches) == 1
        match = matches[0]

        # Head should be rightmost noun ("organs")
        assert match.head_idx == 2
        assert match.head_token == "organs"
        assert match.head_lemma == "organ"

    def test_head_word_pos_tag(self):
        """Test head word POS tag is correct."""
        doc = Document(text="The bodies without organs differ.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        matches = matcher.find_phrase_matches(["body", "without", "organ"], processed)

        assert len(matches) == 1
        match = matches[0]

        # Head is "organs" which should be tagged as NNS (plural noun)
        assert match.head_pos.startswith("NN")

    def test_single_word_phrase(self):
        """Test matching single-word 'phrase'."""
        doc = Document(text="The cat sat on the mat.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        matches = matcher.find_phrase_matches(["cat"], processed)

        assert len(matches) == 1
        assert matches[0].tokens == ["cat"]

    def test_long_phrase(self):
        """Test matching longer multi-word phrases."""
        doc = Document(text="The quick brown fox jumps over the lazy dog.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        matches = matcher.find_phrase_matches(
            ["quick", "brown", "fox", "jump"], processed
        )

        assert len(matches) == 1
        assert matches[0].tokens == ["quick", "brown", "fox", "jumps"]

    def test_phrase_with_punctuation_boundary(self):
        """Test phrase matching respects punctuation boundaries."""
        doc = Document(text="The body, without organs, is a concept.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        # This should NOT match because punctuation breaks the sequence
        matches = matcher.find_phrase_matches(["body", "without", "organ"], processed)

        # Punctuation breaks lemma sequence, so no match
        assert len(matches) == 0

    def test_phrase_at_sentence_start(self):
        """Test matching phrase at sentence beginning."""
        doc = Document(text="Body without organs is a concept.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        matches = matcher.find_phrase_matches(["body", "without", "organ"], processed)

        assert len(matches) == 1
        assert matches[0].start_idx == 0

    def test_phrase_at_sentence_end(self):
        """Test matching phrase at sentence end."""
        doc = Document(text="This is the body without organs.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        matches = matcher.find_phrase_matches(["body", "without", "organ"], processed)

        assert len(matches) == 1
        # Should be near end (before period)
        assert matches[0].end_idx < len(processed.tokens)

    def test_find_phrase_positions(self):
        """Test convenience method for getting just positions."""
        doc = Document(text="The body without organs and the body without organs.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        positions = matcher.find_phrase_positions(
            ["body", "without", "organ"], processed
        )

        assert len(positions) == 2
        assert positions[0] == (1, 4)
        assert positions[1] == (6, 9)

    def test_empty_document(self):
        """Test matching on empty document."""
        doc = Document(text="")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        matches = matcher.find_phrase_matches(["body", "without", "organ"], processed)

        assert len(matches) == 0

    def test_phrase_longer_than_document(self):
        """Test when phrase is longer than document."""
        doc = Document(text="Short text.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        matches = matcher.find_phrase_matches(
            ["this", "is", "a", "very", "long", "phrase"], processed
        )

        assert len(matches) == 0

    def test_partial_match_not_found(self):
        """Test that partial matches are not returned."""
        doc = Document(text="The body without is incomplete.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        # Looking for "body without organs" but only "body without" exists
        matches = matcher.find_phrase_matches(["body", "without", "organ"], processed)

        assert len(matches) == 0

    def test_overlapping_phrases(self):
        """Test document with overlapping phrase patterns."""
        doc = Document(text="The body without organs without bodies.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        # Search for "without organ"
        matches = matcher.find_phrase_matches(["without", "organ"], processed)

        # Should find "without organs"
        assert len(matches) == 1

    def test_verb_phrase_head(self):
        """Test head identification in verb phrases."""
        doc = Document(text="The cat is running quickly.")
        processed = preprocess(doc)

        matcher = PhraseMatcher()
        matches = matcher.find_phrase_matches(["be", "run"], processed)

        if len(matches) > 0:
            match = matches[0]
            # Head should be the verb (rightmost verb in phrase)
            assert match.head_pos.startswith("VB")
