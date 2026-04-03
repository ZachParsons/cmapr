"""
Tests for preprocessing modules: tokenization, POS tagging, lemmatization, pipeline.
"""

import pytest
from nltk.corpus import wordnet
from src.concept_mapper.corpus import Document
from src.concept_mapper.preprocessing import (
    filter_by_pos,
    get_wordnet_pos,
    lemmatize,
    lemmatize_tagged,
    lemmatize_words,
    preprocess,
    preprocess_corpus,
    tag_sentences,
    tag_tokens,
    tokenize_sentences,
    tokenize_words,
    tokenize_words_preserve_case,
)


class TestTokenization:
    """Test tokenization functions."""

    def test_tokenize_words_basic(self):
        """Test basic word tokenization."""
        text = "The cat sat on the mat."
        tokens = tokenize_words(text)

        assert len(tokens) == 7
        assert tokens[0] == "The"
        assert tokens[-1] == "."

    def test_tokenize_words_with_punctuation(self):
        """Test tokenization handles punctuation correctly."""
        text = "It's here! What's that?"
        tokens = tokenize_words(text)

        assert "It" in tokens
        assert "'s" in tokens
        assert "!" in tokens
        assert "?" in tokens

    def test_tokenize_sentences_basic(self):
        """Test basic sentence tokenization."""
        text = "The cat sat. It ran away."
        sentences = tokenize_sentences(text)

        assert len(sentences) == 2
        assert sentences[0] == "The cat sat."
        assert sentences[1] == "It ran away."

    def test_tokenize_sentences_with_abbreviations(self):
        """Test sentence tokenization handles abbreviations."""
        text = "Dr. Smith lives in the U.S.A. He works there."
        sentences = tokenize_sentences(text)

        # Should not split on Dr. or U.S.A.
        assert len(sentences) == 2

    def test_tokenize_words_preserve_case(self):
        """Test tokenization with case preservation."""
        text = "The CAT sat"
        original, lowercased = tokenize_words_preserve_case(text)

        assert original == ["The", "CAT", "sat"]
        assert lowercased == ["the", "cat", "sat"]


class TestPOSTagging:
    """Test POS tagging functions."""

    def test_tag_tokens_basic(self):
        """Test basic POS tagging."""
        tokens = ["The", "cat", "sat"]
        tagged = tag_tokens(tokens)

        assert len(tagged) == 3
        assert tagged[0] == ("The", "DT")  # Determiner
        assert tagged[1][1] == "NN"  # Noun
        assert tagged[2][1].startswith("VB")  # Verb

    def test_tag_sentences(self):
        """Test tagging multiple sentences."""
        sentences = ["The cat sat.", "It ran."]
        tagged_sents = tag_sentences(sentences)

        assert len(tagged_sents) == 2
        assert isinstance(tagged_sents[0], list)
        assert isinstance(tagged_sents[0][0], tuple)

    def test_filter_by_pos_nouns(self):
        """Test filtering tokens by POS tag."""
        tagged = [("The", "DT"), ("cat", "NN"), ("sat", "VBD")]
        nouns = filter_by_pos(tagged, {"NN", "NNS"})

        assert nouns == ["cat"]

    def test_filter_by_pos_multiple_tags(self):
        """Test filtering with multiple POS tags."""
        tagged = [
            ("The", "DT"),
            ("quick", "JJ"),
            ("cat", "NN"),
            ("ran", "VBD"),
            ("quickly", "RB"),
        ]
        # Get nouns and verbs
        words = filter_by_pos(tagged, {"NN", "NNS", "VB", "VBD"})

        assert "cat" in words
        assert "ran" in words
        assert "quick" not in words


class TestLemmatization:
    """Test lemmatization functions."""

    def test_get_wordnet_pos_verb(self):
        """Test mapping verb POS tags."""
        assert get_wordnet_pos("VB") == wordnet.VERB
        assert get_wordnet_pos("VBD") == wordnet.VERB
        assert get_wordnet_pos("VBG") == wordnet.VERB

    def test_get_wordnet_pos_noun(self):
        """Test mapping noun POS tags."""
        assert get_wordnet_pos("NN") == wordnet.NOUN
        assert get_wordnet_pos("NNS") == wordnet.NOUN
        assert get_wordnet_pos("NNP") == wordnet.NOUN

    def test_get_wordnet_pos_adjective(self):
        """Test mapping adjective POS tags."""
        assert get_wordnet_pos("JJ") == wordnet.ADJ
        assert get_wordnet_pos("JJR") == wordnet.ADJ

    def test_get_wordnet_pos_adverb(self):
        """Test mapping adverb POS tags."""
        assert get_wordnet_pos("RB") == wordnet.ADV
        assert get_wordnet_pos("RBR") == wordnet.ADV

    def test_lemmatize_verb(self):
        """Test lemmatizing verbs."""
        assert lemmatize("running", wordnet.VERB) == "run"
        assert lemmatize("ran", wordnet.VERB) == "run"

    def test_lemmatize_noun(self):
        """Test lemmatizing nouns."""
        assert lemmatize("cats", wordnet.NOUN) == "cat"
        assert lemmatize("mice", wordnet.NOUN) == "mouse"

    def test_lemmatize_adjective(self):
        """Test lemmatizing adjectives."""
        assert lemmatize("better", wordnet.ADJ) == "good"
        assert (
            lemmatize("best", wordnet.ADJ) == "best"
        )  # Note: "best" doesn't lemmatize to "good"

    def test_lemmatize_tagged(self):
        """Test lemmatizing POS-tagged tokens."""
        tagged = [("The", "DT"), ("cats", "NNS"), ("were", "VBD"), ("running", "VBG")]
        lemmas = lemmatize_tagged(tagged)

        assert "cat" in lemmas
        assert "be" in lemmas
        assert "run" in lemmas

    def test_lemmatize_words_default(self):
        """Test lemmatizing without POS tags."""
        words = ["cats", "dogs", "mice"]
        lemmas = lemmatize_words(words)

        assert lemmas == ["cat", "dog", "mouse"]


class TestPipeline:
    """Test preprocessing pipeline."""

    def test_preprocess_simple_document(self):
        """Test preprocessing a simple document."""
        doc = Document(text="The cat sat. It ran.", metadata={"title": "Test"})
        processed = preprocess(doc)

        assert processed.raw_text == "The cat sat. It ran."
        assert processed.num_sentences == 2
        assert processed.num_tokens > 0
        assert len(processed.pos_tags) == len(processed.tokens)
        assert len(processed.lemmas) == len(processed.tokens)
        assert processed.title == "Test"

    def test_preprocess_preserves_metadata(self):
        """Test that preprocessing preserves document metadata."""
        doc = Document(
            text="Test text.",
            metadata={"title": "Test", "author": "John Doe"},
        )
        processed = preprocess(doc)

        assert processed.metadata["title"] == "Test"
        assert processed.metadata["author"] == "John Doe"

    def test_preprocess_lemmatization_accuracy(self):
        """Test that pipeline produces correct lemmas."""
        doc = Document(text="The cats were running quickly.")
        processed = preprocess(doc)

        lemmas = processed.lemmas
        assert "cat" in lemmas  # cats → cat
        assert "be" in lemmas  # were → be
        assert "run" in lemmas  # running → run

    def test_preprocess_corpus(self):
        """Test preprocessing multiple documents."""
        docs = [
            Document(text="First document."),
            Document(text="Second document."),
        ]
        processed = preprocess_corpus(docs)

        assert len(processed) == 2
        assert all(p.num_tokens > 0 for p in processed)

    def test_preprocess_empty_text(self):
        """Test preprocessing empty document."""
        doc = Document(text="")
        processed = preprocess(doc)

        assert processed.raw_text == ""
        assert processed.num_sentences == 0
        assert processed.num_tokens == 0

    def test_preprocess_philosophical_text(self):
        """Test preprocessing with philosophical vocabulary."""
        doc = Document(
            text="Daseinology examines the structures of being-in-the-world. "
            "Temporalization is the process by which dasein projects itself."
        )
        processed = preprocess(doc)

        # Check that philosophical terms are tokenized
        tokens_lower = [t.lower() for t in processed.tokens]
        assert "daseinology" in tokens_lower
        assert "temporalization" in tokens_lower

        # Check sentences are split correctly
        assert processed.num_sentences == 2

    def test_preprocess_with_ocr_cleaning(self):
        """Test preprocessing with OCR cleaning enabled."""
        # Text with OCR artifacts: split words, spacing issues, page numbers
        doc = Document(
            text="1 . 5. The obsti nacy of the signif icant.\n42\nMore content here."
        )
        processed = preprocess(doc, clean_ocr=True)

        # Check that text was cleaned
        # Page number (42) should be removed
        assert "42" not in processed.raw_text or processed.raw_text.count("42") == 0

        # Split words should be rejoined in the cleaned text
        # The cleaned text is stored in raw_text after cleaning
        assert (
            "obstinacy" in processed.raw_text.lower()
            or "obsti nacy" not in processed.raw_text.lower()
        )

        # Spacing should be fixed (1 . 5. → 1.5.)
        assert "1.5." in processed.raw_text or "1 . 5." not in processed.raw_text

        # Tokenization should work on cleaned text
        assert processed.num_tokens > 0
        assert processed.num_sentences > 0

    def test_preprocess_without_ocr_cleaning(self):
        """Test that OCR cleaning is not applied by default."""
        doc = Document(text="1 . 5. The obsti nacy.\n42\n")
        processed = preprocess(doc, clean_ocr=False)

        # Without cleaning, OCR artifacts should remain
        # (This verifies the default behavior)
        assert processed.raw_text == doc.text


# ============================================================================
# spaCy availability guard (Phase 13)
# ============================================================================


def _spacy_available():
    try:
        import spacy

        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False


requires_spacy = pytest.mark.skipif(
    not _spacy_available(),
    reason="spaCy en_core_web_sm not installed",
)


# ============================================================================
# Test NounChunkExtraction (Phase 13)
# ============================================================================


@requires_spacy
class TestNounChunkExtraction:
    """_extract_noun_chunks returns lowercased multi-word noun phrases without leading determiners."""

    def test_multi_word_chunks_extracted(self):
        """A text with known multi-word noun phrases produces at least one chunk with a space."""
        from src.concept_mapper.preprocessing.pipeline import _extract_noun_chunks

        text = (
            "The sign vehicle is a triadic relation. "
            "Unlimited semiosis involves sign processes."
        )
        chunks = _extract_noun_chunks(text)
        assert any(" " in c for c in chunks), (
            "Expected at least one multi-word chunk from a text with compound noun phrases"
        )

    def test_leading_determiners_stripped(self):
        """Chunks returned do not start with a determiner such as 'the'."""
        from src.concept_mapper.preprocessing.pipeline import _extract_noun_chunks

        chunks = _extract_noun_chunks("The triadic relation is fundamental.")
        assert "triadic relation" in chunks, (
            "Expected 'triadic relation' (without 'the') to appear in chunks"
        )
        assert not any(c.startswith("the ") for c in chunks), (
            "No chunk should start with a leading determiner 'the'"
        )

    def test_single_word_chunks_excluded(self):
        """Short sentences that yield only single-word noun phrases produce no results."""
        from src.concept_mapper.preprocessing.pipeline import _extract_noun_chunks

        chunks = _extract_noun_chunks("Signs exist. Codes function.")
        assert all(" " in c for c in chunks), (
            "Expected single-word chunks to be excluded; only multi-word chunks allowed"
        )

    def test_known_multi_word_phrases_found(self):
        """A semiotics-domain text yields at least one recognised multi-word phrase."""
        from src.concept_mapper.preprocessing.pipeline import _extract_noun_chunks

        text = (
            "The sign vehicle is the concrete bearer of meaning. "
            "A triadic relation connects representamen, object, and interpretant. "
            "The sign function maps expression to content in the sign system."
        )
        chunks = _extract_noun_chunks(text)
        known_phrases = {"sign vehicle", "triadic relation", "sign function"}
        assert any(p in chunks for p in known_phrases), (
            f"Expected at least one of {known_phrases} in extracted chunks: {chunks}"
        )


# ============================================================================
# Test PreprocessSpacyFlag (Phase 13)
# ============================================================================


class TestPreprocessSpacyFlag:
    """preprocess(use_spacy=...) controls whether noun_chunks appear in metadata."""

    def test_preprocess_without_spacy_has_no_noun_chunks(self):
        """preprocess with use_spacy=False does not add noun_chunks to metadata."""
        from src.concept_mapper.corpus import Document

        doc = Document(
            text="The sign vehicle conveys meaning. Semiosis is a triadic process."
        )
        processed = preprocess(doc, use_spacy=False)
        assert "noun_chunks" not in processed.metadata, (
            "Expected no 'noun_chunks' in metadata when use_spacy=False"
        )

    @requires_spacy
    def test_preprocess_with_spacy_stores_list(self):
        """preprocess with use_spacy=True stores a list under metadata['noun_chunks']."""
        from src.concept_mapper.corpus import Document

        doc = Document(
            text="The sign vehicle conveys meaning. Semiosis is a triadic process."
        )
        processed = preprocess(doc, use_spacy=True)
        assert "noun_chunks" in processed.metadata, (
            "Expected 'noun_chunks' in metadata when use_spacy=True"
        )
        assert isinstance(processed.metadata["noun_chunks"], list), (
            "Expected metadata['noun_chunks'] to be a list"
        )

    @requires_spacy
    def test_preprocess_with_spacy_finds_multiword(self):
        """preprocess with use_spacy=True finds at least one multi-word chunk."""
        from src.concept_mapper.corpus import Document

        doc = Document(
            text="The sign vehicle is a triadic relation in Peircean semiotic theory."
        )
        processed = preprocess(doc, use_spacy=True)
        chunks = processed.metadata.get("noun_chunks", [])
        assert any(" " in c for c in chunks), (
            f"Expected at least one multi-word chunk in metadata, got: {chunks}"
        )

    @requires_spacy
    def test_noun_chunks_survive_to_dict_round_trip(self):
        """noun_chunks in metadata survive a to_dict() → from_dict() round trip."""
        from src.concept_mapper.corpus import Document
        from src.concept_mapper.corpus.models import ProcessedDocument

        doc = Document(
            text="The sign vehicle is a triadic relation involving representamen and object."
        )
        processed = preprocess(doc, use_spacy=True)
        as_dict = processed.to_dict()
        restored = ProcessedDocument.from_dict(as_dict)
        assert "noun_chunks" in restored.metadata, (
            "Expected 'noun_chunks' to survive a to_dict/from_dict round trip"
        )
