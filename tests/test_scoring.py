"""
Tests for the post-scoring filter chain in concept_mapper.terms.scoring.

Covers the term-review cleanup filters: edge-punctuation stripping, stopword
removal, WordNet derivational dedup (noun-preferred), and the conservative
OCR-artifact heuristics. See docs/plans/term-review-cleanup.md.
"""

from src.concept_mapper.terms.scoring import (
    apply_aliases,
    filter_ocr_artifacts,
    filter_stopwords,
    infer_canonical,
    lemma_and_derivational_merge,
    load_aliases,
    load_learned_stopwords,
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

    def test_extra_learned_stopwords(self):
        # Learned (atopic) stopwords drop in addition to the built-in set.
        out = filter_stopwords(
            _cands(("sign", 5), ("abduction", 4)), extra={"abduction"}
        )
        assert _terms(out) == ["sign"]


class TestLearnedFilters:
    def test_load_missing_files_empty(self, tmp_path):
        assert load_learned_stopwords(tmp_path / "nope.json") == set()
        assert load_aliases(tmp_path / "nope.json") == {}

    def test_load_learned_stopwords(self, tmp_path):
        p = tmp_path / "stopwords_extra.json"
        p.write_text('{"stopwords": ["However", "abduction"]}')
        assert load_learned_stopwords(p) == {"however", "abduction"}

    def test_load_aliases(self, tmp_path):
        p = tmp_path / "aliases.json"
        p.write_text('{"aliases": {"Taxonomic": "Taxonomy"}}')
        assert load_aliases(p) == {"taxonomic": "taxonomy"}

    def test_apply_aliases_drops_when_canonical_present(self):
        kept, dropped = apply_aliases(
            _cands(("taxonomic", 5), ("taxonomy", 3), ("semiosis", 2)),
            {"taxonomic": "taxonomy"},
        )
        assert _terms(kept) == ["taxonomy", "semiosis"]
        assert dropped == [("taxonomic", "taxonomy")]

    def test_apply_aliases_keeps_when_canonical_absent(self):
        # Cross-work alias must not erase a term that stands alone here.
        kept, dropped = apply_aliases(
            _cands(("taxonomic", 5), ("semiosis", 2)), {"taxonomic": "taxonomy"}
        )
        assert _terms(kept) == ["taxonomic", "semiosis"]
        assert dropped == []

    def test_infer_canonical_derivational(self):
        assert infer_canonical("taxonomic", ["taxonomy", "sign", "code"]) == "taxonomy"

    def test_infer_canonical_stem_prefix(self):
        assert infer_canonical("signification", ["signify", "code"]) == "signify"

    def test_infer_canonical_none(self):
        assert infer_canonical("zzqx", ["sign", "code"]) is None


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

    def test_return_groups_maps_canonical_to_cluster(self):
        out, groups = merge_derivational_variants(
            _cands(("taxonomic", 5), ("taxonomy", 3)), return_groups=True
        )
        assert _terms(out) == ["taxonomy"]
        assert set(groups["taxonomy"]) == {"taxonomic", "taxonomy"}


class TestMergeProvenance:
    def test_provenance_records_absorbed_variants(self):
        out, prov = lemma_and_derivational_merge(
            _cands(("taxonomic", 5), ("taxonomy", 3), ("iconic", 4), ("icon", 2)),
            return_provenance=True,
        )
        assert set(_terms(out)) == {"taxonomy", "icon"}
        # Each surviving canonical lists the derivational form it absorbed.
        assert "taxonomic" in prov["taxonomy"]
        assert "iconic" in prov["icon"]

    def test_no_provenance_for_standalone_terms(self):
        _out, prov = lemma_and_derivational_merge(
            _cands(("semiosis", 5), ("interpretant", 4)), return_provenance=True
        )
        # Unmerged terms carry no variants.
        assert prov == {}


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
