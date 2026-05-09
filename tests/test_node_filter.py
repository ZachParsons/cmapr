"""Tests for NodeFilter — node inclusion criteria for the concept graph.

Covers every rejection criterion independently, then confirms valid
philosophical terms pass and known noise terms from the Eco corpus fail.
"""

from collections import Counter


from concept_mapper.graph.node_filter import NodeFilter


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def make_filter(vocab=None, freqs=None, min_freq=3):
    """Build a NodeFilter with controllable inputs."""
    vocab = vocab or set()
    freqs = Counter(freqs or {})
    return NodeFilter(corpus_vocab=vocab, term_freqs=freqs, min_freq=min_freq)


# A minimal corpus that contains 'structure', 'semiosis', 'signal'
# so we can test the fragment check without a real corpus.
SAMPLE_VOCAB = {"structure", "semiosis", "signal", "sign", "interpretant", "code"}
SAMPLE_FREQS = {
    "sign": 20,
    "semiosis": 15,
    "interpretant": 10,
    "signal": 8,
    "structure": 7,
    "code": 6,
    "structu": 3,  # artificial fragment entry
}


# ---------------------------------------------------------------------------
# Individual criterion tests
# ---------------------------------------------------------------------------


class TestPOSCriterion:
    def test_noun_passes(self):
        f = make_filter(freqs={"sign": 5})
        ok, _ = f.is_valid("sign", pos="NN")
        assert ok

    def test_verb_passes(self):
        f = make_filter(freqs={"produce": 5})
        ok, _ = f.is_valid("produce", pos="VB")
        assert ok

    def test_adjective_passes(self):
        f = make_filter(freqs={"intensional": 5})
        ok, _ = f.is_valid("intensional", pos="JJ")
        assert ok

    def test_adverb_passes(self):
        f = make_filter(freqs={"directly": 5})
        ok, _ = f.is_valid("directly", pos="RB")
        assert ok

    def test_determiner_rejected(self):
        f = make_filter(freqs={"that": 10})
        ok, reason = f.is_valid("that", pos="DT")
        assert not ok
        assert "POS" in reason

    def test_preposition_rejected(self):
        f = make_filter(freqs={"upon": 5})
        ok, reason = f.is_valid("upon", pos="IN")
        assert not ok
        assert "POS" in reason

    def test_no_pos_skips_check(self):
        """When pos is not supplied the POS criterion is not applied."""
        f = make_filter(freqs={"upon": 5})
        ok, _ = f.is_valid("upon", pos=None)
        # Should pass POS check but may still fail stopword/length etc.
        # 'upon' is not in STOPWORDS and len 4 so should pass all
        assert ok


class TestLengthCriterion:
    def test_exactly_four_passes(self):
        f = make_filter(freqs={"sign": 5})
        ok, _ = f.is_valid("sign")
        assert ok

    def test_three_chars_rejected(self):
        f = make_filter(freqs={"the": 50})
        ok, reason = f.is_valid("the")
        assert not ok
        assert "short" in reason

    def test_two_chars_rejected(self):
        f = make_filter(freqs={"nn": 5})
        ok, reason = f.is_valid("nn")
        assert not ok
        assert "short" in reason


class TestAbbreviationCriterion:
    def test_allcaps_four_rejected(self):
        f = make_filter(freqs={"sems": 5})
        ok, reason = f.is_valid("SEMS")
        assert not ok
        assert "abbreviation" in reason

    def test_allcaps_five_passes(self):
        """All-caps with 5+ chars is not caught by the abbreviation rule."""
        f = make_filter(freqs={"index": 5})
        ok, _ = f.is_valid("INDEX")
        assert ok

    def test_mixed_case_four_passes(self):
        """'Sign' is not all-caps so the abbreviation rule doesn't apply."""
        f = make_filter(freqs={"sign": 5})
        ok, _ = f.is_valid("Sign")
        assert ok


class TestStopwordCriterion:
    def test_common_stopword_rejected(self):
        f = make_filter(freqs={"that": 100})
        ok, reason = f.is_valid("that")
        assert not ok
        assert "stopword" in reason

    def test_philosophical_term_not_stopword(self):
        f = make_filter(freqs={"semiosis": 10})
        ok, _ = f.is_valid("semiosis")
        assert ok


class TestFrequencyCriterion:
    def test_below_min_freq_rejected(self):
        f = make_filter(freqs={"lekton": 2}, min_freq=3)
        ok, reason = f.is_valid("lekton")
        assert not ok
        assert "frequency" in reason

    def test_exactly_min_freq_passes(self):
        f = make_filter(freqs={"lekton": 3}, min_freq=3)
        ok, _ = f.is_valid("lekton")
        assert ok

    def test_absent_term_rejected(self):
        f = make_filter(freqs={}, min_freq=3)
        ok, reason = f.is_valid("ghost")
        assert not ok
        assert "frequency" in reason


class TestFragmentCriterion:
    def test_corpus_fragment_rejected(self):
        """'structu' is not in WordNet and 'structure' is in corpus → fragment."""
        f = NodeFilter(
            corpus_vocab={"structure"},
            term_freqs=Counter({"structu": 5}),
        )
        ok, reason = f.is_valid("structu")
        assert not ok
        assert "fragment" in reason

    def test_wordnet_word_not_fragment(self):
        """'sign' is in WordNet so it passes even though 'signal' is in corpus."""
        f = NodeFilter(
            corpus_vocab={"signal", "signs", "signifier"},
            term_freqs=Counter({"sign": 20}),
        )
        ok, _ = f.is_valid("sign")
        assert ok

    def test_neologism_not_fragment(self):
        """'semiosi' is not in WordNet. If no longer corpus word starts with it,
        it is NOT a fragment (it may be a genuine neologism or foreign term)."""
        f = NodeFilter(
            corpus_vocab={"sign", "code"},  # no word starting with 'semiosi'
            term_freqs=Counter({"semiosi": 5}),
        )
        ok, _ = f.is_valid("semiosi")
        assert ok

    def test_neologism_with_extension_is_fragment(self):
        """'semiosi' IS a fragment when 'semiosis' appears in the corpus."""
        f = NodeFilter(
            corpus_vocab={"semiosis"},
            term_freqs=Counter({"semiosi": 5}),
        )
        ok, reason = f.is_valid("semiosi")
        assert not ok
        assert "fragment" in reason

    def test_suffix_fragment_rejected(self):
        """'tion' is not in WordNet and 'proposition' ends with it → suffix fragment."""
        f = NodeFilter(
            corpus_vocab={"proposition"},
            term_freqs=Counter({"tion": 10}),
        )
        ok, reason = f.is_valid("tion")
        assert not ok
        assert "fragment" in reason

    def test_suffix_fragment_ence_rejected(self):
        """'ence' is not in WordNet and 'evidence' ends with it → suffix fragment."""
        f = NodeFilter(
            corpus_vocab={"evidence"},
            term_freqs=Counter({"ence": 8}),
        )
        ok, reason = f.is_valid("ence")
        assert not ok
        assert "fragment" in reason

    def test_wordnet_suffix_not_fragment(self):
        """'form' is in WordNet so it passes even though 'information' ends with it."""
        f = NodeFilter(
            corpus_vocab={"information", "formation"},
            term_freqs=Counter({"form": 10}),
        )
        ok, _ = f.is_valid("form")
        assert ok


class TestInvalidCharacters:
    def test_slash_rejected(self):
        f = make_filter(freqs={"/man/": 5})
        ok, reason = f.is_valid("/man/")
        assert not ok
        assert "invalid" in reason

    def test_bracket_rejected(self):
        f = make_filter(freqs={"[14]": 5})
        ok, reason = f.is_valid("[14]")
        assert not ok
        assert "invalid" in reason

    def test_hyphenated_term_passes(self):
        """Hyphens are valid — sign-function, co-text, content-plane."""
        f = make_filter(freqs={"sign-function": 5})
        ok, _ = f.is_valid("sign-function")
        assert ok


# ---------------------------------------------------------------------------
# Bulk API tests
# ---------------------------------------------------------------------------


class TestFilterAndRejected:
    def setup_method(self):
        self.f = NodeFilter(
            corpus_vocab=SAMPLE_VOCAB,
            term_freqs=Counter(SAMPLE_FREQS),
            min_freq=3,
        )

    def test_filter_keeps_valid(self):
        valid = self.f.filter(["sign", "interpretant", "semiosis"])
        assert set(valid) == {"sign", "interpretant", "semiosis"}

    def test_filter_removes_noise(self):
        # 'nn' too short, 'structu' fragment (structure in vocab), '/man/' invalid chars
        valid = self.f.filter(["sign", "nn", "structu", "/man/", "interpretant"])
        assert set(valid) == {"sign", "interpretant"}

    def test_rejected_returns_reasons(self):
        reasons = self.f.rejected(["sign", "nn", "structu"])
        assert "sign" not in reasons
        assert "nn" in reasons
        assert "structu" in reasons


# ---------------------------------------------------------------------------
# Acceptance test: known good and bad terms from the Eco corpus
# ---------------------------------------------------------------------------


class TestEcoCorpusTerms:
    """
    Verify that terms from the eco_spl1 rarities output behave as expected.
    Uses a minimal stand-in for corpus_vocab and realistic frequencies.
    """

    def setup_method(self):
        vocab = {
            "semiosis",
            "semiotic",
            "semiotics",
            "sign",
            "signs",
            "signifier",
            "signification",
            "signify",
            "interpretant",
            "interpretation",
            "structure",
            "structuralism",
            "lekton",
            "aliquid",
        }
        freqs = Counter(
            {
                "semiotic": 30,
                "signifier": 12,
                "signification": 10,
                "interpretant": 9,
                "sign": 25,
                "lekton": 5,
                "aliquid": 4,
                "semiosi": 8,  # 'semiosis' is in vocab → should be fragment
                "/man/": 6,  # invalid chars
                "tion": 4,  # too short? no, 4 chars — but fragment of 'tion*'?
                "structu": 3,  # fragment of 'structure'
            }
        )
        self.f = NodeFilter(corpus_vocab=vocab, term_freqs=freqs, min_freq=3)

    def test_core_philosophical_terms_pass(self):
        for term in [
            "semiotic",
            "signifier",
            "signification",
            "interpretant",
            "lekton",
            "aliquid",
        ]:
            ok, reason = self.f.is_valid(term)
            assert ok, f"Expected {term!r} to pass but got: {reason}"

    def test_sign_passes_despite_longer_forms(self):
        """'sign' is in WordNet so the fragment check never fires."""
        ok, reason = self.f.is_valid("sign")
        assert ok, f"Expected 'sign' to pass but got: {reason}"

    def test_semiosi_rejected_as_fragment(self):
        """'semiosi' not in WordNet and 'semiosis' is in corpus → fragment."""
        ok, reason = self.f.is_valid("semiosi")
        assert not ok
        assert "fragment" in reason

    def test_slash_notation_rejected(self):
        ok, reason = self.f.is_valid("/man/")
        assert not ok
        assert "invalid" in reason

    def test_corpus_fragment_rejected(self):
        ok, reason = self.f.is_valid("structu")
        assert not ok
        assert "fragment" in reason


# ---------------------------------------------------------------------------
# Multi-word terms (spaCy noun chunks)
# ---------------------------------------------------------------------------


class TestMultiWordTerms:
    """Phrases from spaCy noun-chunk extraction bypass single-token guards."""

    def test_phrase_passes_with_zero_token_freq(self):
        """A phrase whose components have no entry in term_freqs still passes —
        phrase frequency is enforced upstream by the rarities chunk scorer."""
        f = make_filter(freqs={}, min_freq=3)
        ok, reason = f.is_valid("sign vehicle")
        assert ok, f"Expected 'sign vehicle' to pass but got: {reason}"

    def test_phrase_skips_fragment_check(self):
        """A phrase isn't compared against the corpus fragment heuristic."""
        f = NodeFilter(
            corpus_vocab={"signifier", "signification"},
            term_freqs=Counter(),
            min_freq=3,
        )
        # 'sign' alone would pass via WordNet, but the fragment check is
        # the relevant one — confirm phrases bypass it entirely.
        ok, _ = f.is_valid("sign function")
        assert ok

    def test_phrase_skips_short_token_check(self):
        """Multi-word terms with short components still pass."""
        f = make_filter(freqs={})
        ok, _ = f.is_valid("ad hoc reasoning")
        assert ok

    def test_phrase_pos_check_still_applies(self):
        """POS rejection works for phrases too (when supplied)."""
        f = make_filter(freqs={})
        ok, reason = f.is_valid("sign vehicle", pos="DT")
        assert not ok
        assert "POS" in reason

    def test_phrase_pos_noun_passes(self):
        f = make_filter(freqs={})
        ok, _ = f.is_valid("triadic relation", pos="NN")
        assert ok

    def test_phrase_with_stopword_components_passes(self):
        """Stopword check is single-token-only — a real phrase will not
        match a single-word stopword anyway, so phrases bypass the check."""
        f = make_filter(freqs={})
        ok, _ = f.is_valid("the sign and the meaning")
        assert ok

    def test_filter_keeps_phrases(self):
        """Bulk filter() keeps phrases alongside valid single tokens."""
        f = NodeFilter(
            corpus_vocab={"sign", "interpretant"},
            term_freqs=Counter({"sign": 20, "interpretant": 10}),
            min_freq=3,
        )
        kept = f.filter(["sign", "sign vehicle", "interpretant", "triadic relation"])
        assert set(kept) == {
            "sign",
            "sign vehicle",
            "interpretant",
            "triadic relation",
        }
