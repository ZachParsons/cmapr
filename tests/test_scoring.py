"""
Tests for the post-scoring filter chain in concept_mapper.terms.scoring.

Covers the term-review cleanup filters: edge-punctuation stripping, stopword
removal, WordNet derivational dedup (noun-preferred), and the conservative
OCR-artifact heuristics. See docs/plans/term-review-cleanup.md.
"""

from src.concept_mapper.terms.scoring import (
    filter_ocr_artifacts,
    filter_stopwords,
    merge_derivational_variants,
    strip_stray_quotes,
)


def _cands(*pairs):
    """Build candidate 3-tuples from (term, score) pairs."""
    return [(term, float(score), {"total": float(score)}) for term, score in pairs]


def _terms(candidates):
    return [t for t, _, _ in candidates]


class TestStripStrayQuotes:
    def test_strips_slashes_and_quotes(self):
        out = strip_stray_quotes(_cands(("/man/", 9), ("'animal'", 8)))
        assert _terms(out) == ["man", "animal"]

    def test_preserves_internal_punctuation(self):
        out = strip_stray_quotes(_cands(("sign-function", 7), ("co-text", 6)))
        assert _terms(out) == ["sign-function", "co-text"]

    def test_drops_terms_that_empty_out(self):
        out = strip_stray_quotes(_cands(("///", 1), ("''", 1)))
        assert out == []


class TestFilterStopwords:
    def test_drops_function_words(self):
        out = filter_stopwords(
            _cands(
                ("since", 9),
                ("whether", 8),
                ("although", 7),
                ("else", 6),
                ("onto", 5),
                ("versa", 4),
            )
        )
        assert out == []

    def test_keeps_content_words(self):
        out = filter_stopwords(_cands(("semiosis", 5), ("interpretant", 4)))
        assert _terms(out) == ["semiosis", "interpretant"]

    def test_multiword_phrases_bypass(self):
        # "content plane" contains a stopword token but is a real phrase.
        out = filter_stopwords(_cands(("content plane", 4)))
        assert _terms(out) == ["content plane"]


class TestDerivationalDedup:
    def test_prefers_noun_form(self):
        # taxonomic (adj) collapses onto taxonomy (noun); cluster max score kept.
        out = merge_derivational_variants(_cands(("taxonomic", 5), ("taxonomy", 3)))
        assert _terms(out) == ["taxonomy"]
        assert out[0][1] == 5.0  # carries the higher of the two scores

    def test_collapses_icon_iconic(self):
        out = merge_derivational_variants(_cands(("iconic", 4), ("icon", 2)))
        assert _terms(out) == ["icon"]

    def test_unrelated_terms_untouched(self):
        out = merge_derivational_variants(_cands(("signifier", 6), ("sign", 5)))
        assert set(_terms(out)) >= {"signifier"}

    def test_phrases_never_merge(self):
        out = merge_derivational_variants(_cands(("iconic sign", 4), ("icon", 2)))
        assert set(_terms(out)) == {"iconic sign", "icon"}


class TestFilterOcrArtifacts:
    def test_drops_function_word_merges(self):
        out = filter_ocr_artifacts(
            _cands(
                ("thesecase", 9),
                ("thesesign", 8),
                ("mybody", 7),
                ("intoplay", 6),
            )
        )
        assert out == []

    def test_drops_leading_char_drops(self):
        out = filter_ocr_artifacts(_cands(("ictionary", 6), ("ther", 5)))
        assert out == []

    def test_keeps_neologisms(self):
        keep = _cands(
            ("interpretant", 4),
            ("biconditional", 3),
            ("metasemiotic", 2),
            ("co-text", 1),
        )
        assert _terms(filter_ocr_artifacts(keep)) == _terms(keep)

    def test_keeps_real_wordnet_words(self):
        out = filter_ocr_artifacts(_cands(("semiosis", 5), ("language", 4)))
        assert _terms(out) == ["semiosis", "language"]
