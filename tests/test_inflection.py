"""
Tests for inflection generation module.

Tests POS-aware inflection for nouns, verbs, adjectives, adverbs,
including irregular forms.
"""

import pytest
from src.concept_mapper.transformations.inflection import InflectionGenerator


class TestInflectionGenerator:
    """Tests for InflectionGenerator class."""

    @pytest.fixture
    def generator(self):
        """Fixture providing an InflectionGenerator instance."""
        return InflectionGenerator()

    # Noun tests
    def test_noun_singular_unchanged(self, generator):
        """Base form noun should remain unchanged."""
        assert generator.inflect("cat", "NN") == "cat"

    def test_noun_plural(self, generator):
        """Singular noun lemma should inflect to plural."""
        assert generator.inflect("cat", "NNS") == "cats"
        assert generator.inflect("dog", "NNS") == "dogs"

    def test_noun_irregular_plural(self, generator):
        """Irregular plurals should be handled correctly."""
        assert generator.inflect("child", "NNS") == "children"
        assert generator.inflect("person", "NNS") == "people"
        assert generator.inflect("mouse", "NNS") == "mice"

    # Verb tests
    def test_verb_base_form(self, generator):
        """Base form verb should remain unchanged."""
        assert generator.inflect("run", "VB") == "run"

    def test_verb_past_tense(self, generator):
        """Verb lemma should inflect to past tense."""
        result = generator.inflect("run", "VBD")
        assert result == "ran"

    def test_verb_gerund(self, generator):
        """Verb lemma should inflect to gerund (-ing form)."""
        assert generator.inflect("run", "VBG") == "running"
        assert generator.inflect("sprint", "VBG") == "sprinting"

    def test_verb_past_participle(self, generator):
        """Verb lemma should inflect to past participle."""
        assert generator.inflect("run", "VBN") == "run"
        assert generator.inflect("eat", "VBN") == "eaten"

    def test_verb_present_third_person(self, generator):
        """Verb should inflect to 3rd person singular present."""
        assert generator.inflect("run", "VBZ") == "runs"
        assert generator.inflect("go", "VBZ") == "goes"

    def test_verb_irregular_past(self, generator):
        """Irregular past tenses should be handled."""
        assert generator.inflect("go", "VBD") == "went"
        assert generator.inflect("be", "VBD") in ["was", "were"]

    # Adjective tests
    def test_adjective_base(self, generator):
        """Base form adjective should remain unchanged."""
        assert generator.inflect("quick", "JJ") == "quick"

    def test_adjective_comparative(self, generator):
        """Adjective should inflect to comparative."""
        assert generator.inflect("quick", "JJR") == "quicker"
        assert generator.inflect("fast", "JJR") == "faster"

    def test_adjective_superlative(self, generator):
        """Adjective should inflect to superlative."""
        assert generator.inflect("quick", "JJS") == "quickest"
        assert generator.inflect("fast", "JJS") == "fastest"

    def test_adjective_irregular(self, generator):
        """Irregular adjective comparisons should be handled."""
        assert generator.inflect("good", "JJR") == "better"
        assert generator.inflect("good", "JJS") == "best"
        assert generator.inflect("bad", "JJR") == "worse"
        assert generator.inflect("bad", "JJS") == "worst"

    # Adverb tests
    def test_adverb_base(self, generator):
        """Base form adverb should remain unchanged."""
        result = generator.inflect("quickly", "RB")
        assert result in ["quickly", "quick"]  # lemminflect may vary

    def test_adverb_comparative(self, generator):
        """Adverb comparative forms."""
        # Note: Many adverbs use "more" instead of inflection
        result = generator.inflect("quickly", "RBR")
        # Accept either inflected form or base (lemminflect may not have all adverbs)
        assert result in ["quicker", "quickly", "more quickly"]

    # Edge cases
    def test_unknown_pos_tag(self, generator):
        """Unknown POS tags should return lemma unchanged."""
        assert generator.inflect("test", "XX") == "test"

    def test_empty_string(self, generator):
        """Empty string should return empty."""
        assert generator.inflect("", "NN") == ""

    def test_can_inflect_valid(self, generator):
        """can_inflect should return True for valid inflections."""
        assert generator.can_inflect("cat", "NNS") is True
        assert generator.can_inflect("run", "VBD") is True

    def test_can_inflect_invalid(self, generator):
        """can_inflect should return False for invalid POS tags."""
        assert generator.can_inflect("test", "XX") is False

    def test_get_all_forms(self, generator):
        """get_all_forms should return all inflections."""
        forms = generator.get_all_forms("run")
        assert isinstance(forms, dict)
        # Should have verb forms
        assert 'VBD' in forms or 'VB' in forms

    # Multi-syllable words
    def test_multi_syllable_verb(self, generator):
        """Multi-syllable verbs should inflect correctly."""
        assert generator.inflect("consider", "VBG") == "considering"
        assert generator.inflect("analyze", "VBZ") == "analyzes"

    def test_philosophical_terms(self, generator):
        """Common philosophical terms should inflect correctly."""
        # Noun plurals
        assert generator.inflect("concept", "NNS") == "concepts"
        assert generator.inflect("consciousness", "NNS") == "consciousnesses"

        # Verb forms
        assert generator.inflect("reify", "VBD") == "reified"
        assert generator.inflect("reify", "VBG") == "reifying"
