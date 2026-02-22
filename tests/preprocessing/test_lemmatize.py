"""
Tests for lemmatization with inflect fallback.
"""

from concept_mapper.preprocessing.lemmatize import (
    lemmatize,
    lemmatize_tagged,
    get_wordnet_pos,
)
from nltk.corpus import wordnet


class TestLemmatize:
    """Test basic lemmatization function."""

    def test_standard_verbs(self):
        """Test standard verb lemmatization."""
        assert lemmatize("running", wordnet.VERB) == "run"
        assert lemmatize("ran", wordnet.VERB) == "run"
        assert lemmatize("goes", wordnet.VERB) == "go"

    def test_standard_nouns(self):
        """Test standard noun lemmatization with WordNet."""
        assert lemmatize("cats", wordnet.NOUN) == "cat"
        assert lemmatize("dogs", wordnet.NOUN) == "dog"
        assert lemmatize("children", wordnet.NOUN) == "child"

    def test_specialized_plural_nouns(self):
        """Test that specialized terms use inflect fallback."""
        # Terms not in WordNet should use inflect for pluralization
        assert lemmatize("semiotics", wordnet.NOUN) == "semiotic"
        assert lemmatize("isotopies", wordnet.NOUN) == "isotopy"

    def test_already_singular(self):
        """Test that singular forms are unchanged."""
        assert lemmatize("semiotic", wordnet.NOUN) == "semiotic"
        assert lemmatize("isotopy", wordnet.NOUN) == "isotopy"
        assert lemmatize("philosophy", wordnet.NOUN) == "philosophy"

    def test_adjectives(self):
        """Test adjective lemmatization."""
        assert lemmatize("better", wordnet.ADJ) == "good"
        # Note: WordNet doesn't lemmatize all superlatives perfectly
        assert lemmatize("larger", wordnet.ADJ) == "large"

    def test_case_insensitivity(self):
        """Test that lemmatization is case-insensitive."""
        assert lemmatize("Cats", wordnet.NOUN) == "cat"
        assert lemmatize("RUNNING", wordnet.VERB) == "run"
        assert lemmatize("Semiotics", wordnet.NOUN) == "semiotic"


class TestLemmatizeTagged:
    """Test POS-tagged lemmatization."""

    def test_standard_tagged_tokens(self):
        """Test lemmatization with POS tags."""
        tagged = [
            ("The", "DT"),
            ("cats", "NNS"),
            ("were", "VBD"),
            ("running", "VBG"),
        ]
        lemmas = lemmatize_tagged(tagged)
        assert lemmas == ["the", "cat", "be", "run"]

    def test_specialized_plural_tagged(self):
        """Test that specialized plurals use inflect fallback."""
        tagged = [
            ("semiotics", "NNS"),
            ("isotopies", "NNS"),
            ("semiotic", "NN"),
        ]
        lemmas = lemmatize_tagged(tagged)
        assert lemmas == ["semiotic", "isotopy", "semiotic"]

    def test_proper_noun_plurals(self):
        """Test plural proper nouns (NNPS) with inflect fallback."""
        tagged = [
            ("Semiotics", "NNPS"),  # Proper plural - will use inflect
            ("semiotics", "NNS"),  # Plural noun - will use inflect
            ("Isotopies", "NNPS"),  # Proper plural - will use inflect
        ]
        lemmas = lemmatize_tagged(tagged)
        assert lemmas == ["semiotic", "semiotic", "isotopy"]

    def test_singular_nouns_unchanged(self):
        """Test that singular nouns are not incorrectly pluralized."""
        tagged = [
            ("semiosis", "NN"),  # Singular, should stay as-is
            ("process", "NN"),  # Singular, should stay as-is
            ("Paris", "NNP"),  # Proper noun, should stay as-is
        ]
        lemmas = lemmatize_tagged(tagged)
        # These should NOT be changed to "semiosi", "proces", "pari"
        assert lemmas == ["semiosis", "process", "paris"]  # lowercase by WordNet

    def test_mixed_pos_tags(self):
        """Test lemmatization with various POS tags."""
        tagged = [
            ("Philosophy", "NN"),
            ("investigates", "VBZ"),
            ("consciousness", "NN"),
            ("and", "CC"),
            ("semiotics", "NNS"),
        ]
        lemmas = lemmatize_tagged(tagged)
        assert "semiotic" in lemmas  # plural normalized
        assert "investigate" in lemmas  # verb lemmatized


class TestGetWordnetPos:
    """Test POS tag conversion."""

    def test_noun_tags(self):
        """Test noun tag conversion."""
        assert get_wordnet_pos("NN") == wordnet.NOUN
        assert get_wordnet_pos("NNS") == wordnet.NOUN
        assert get_wordnet_pos("NNP") == wordnet.NOUN
        assert get_wordnet_pos("NNPS") == wordnet.NOUN

    def test_verb_tags(self):
        """Test verb tag conversion."""
        assert get_wordnet_pos("VB") == wordnet.VERB
        assert get_wordnet_pos("VBD") == wordnet.VERB
        assert get_wordnet_pos("VBG") == wordnet.VERB
        assert get_wordnet_pos("VBN") == wordnet.VERB
        assert get_wordnet_pos("VBP") == wordnet.VERB
        assert get_wordnet_pos("VBZ") == wordnet.VERB

    def test_adjective_tags(self):
        """Test adjective tag conversion."""
        assert get_wordnet_pos("JJ") == wordnet.ADJ
        assert get_wordnet_pos("JJR") == wordnet.ADJ
        assert get_wordnet_pos("JJS") == wordnet.ADJ

    def test_adverb_tags(self):
        """Test adverb tag conversion."""
        assert get_wordnet_pos("RB") == wordnet.ADV
        assert get_wordnet_pos("RBR") == wordnet.ADV
        assert get_wordnet_pos("RBS") == wordnet.ADV

    def test_default_to_noun(self):
        """Test that unknown tags default to noun."""
        assert get_wordnet_pos("CC") == wordnet.NOUN
        assert get_wordnet_pos("DT") == wordnet.NOUN
        assert get_wordnet_pos(".") == wordnet.NOUN


class TestRealWorldScenarios:
    """Test lemmatization on real philosophical texts."""

    def test_philosophical_terms(self):
        """Test specialized philosophical terminology."""
        philosophical_tagged = [
            ("phenomenologies", "NNS"),
            ("ontologies", "NNS"),
            ("epistemologies", "NNS"),
            ("hermeneutics", "NNS"),
        ]
        lemmas = lemmatize_tagged(philosophical_tagged)

        # Should normalize to singular forms
        assert "phenomenology" in lemmas
        assert "ontology" in lemmas
        assert "epistemology" in lemmas
        # Note: "hermeneutics" is typically used in singular form

    def test_semiotic_terminology(self):
        """Test semiotic/linguistic terms from Eco's text."""
        semiotic_tagged = [
            ("signifiers", "NNS"),
            ("interpretants", "NNS"),
            ("isotopies", "NNS"),
            ("synecdoches", "NNS"),
        ]
        lemmas = lemmatize_tagged(semiotic_tagged)

        assert "signifier" in lemmas
        assert "interpretant" in lemmas
        assert "isotopy" in lemmas
        assert "synecdoche" in lemmas
