"""
Tests for coreference resolution at ingest (preprocessing/coref.py).

Skipped when the optional ``coref`` extra (fastcoref) is not installed.
The model downloads on first use, so these are also network-dependent the
first time they run in a fresh environment.
"""

import pytest

pytest.importorskip("fastcoref")

from concept_mapper.preprocessing.coref import (  # noqa: E402
    resolve_coreferences,
)


class TestResolveCoreferences:
    def test_pronoun_rewritten_with_antecedent(self):
        text = "Semiosis is unlimited. It generates interpretants."
        resolved = resolve_coreferences(text)
        assert "Semiosis generates interpretants" in resolved

    def test_possessive_pronoun_gets_genitive(self):
        text = "Semiosis never halts. Its interpretants multiply."
        resolved = resolve_coreferences(text)
        assert "Semiosis's interpretants" in resolved

    def test_text_without_anaphora_unchanged(self):
        text = "A sign is a correlate of expression and content."
        assert resolve_coreferences(text) == text

    def test_empty_text_unchanged(self):
        assert resolve_coreferences("") == ""

    def test_non_pronoun_mentions_left_verbatim(self):
        # Conservative policy: only pronouns are rewritten — a nominal
        # rephrasing like "this process" must stay as the author wrote it.
        text = "Semiosis is unlimited. This process generates interpretants."
        resolved = resolve_coreferences(text)
        assert "This process generates" in resolved


class TestPipelineIntegration:
    def test_preprocess_resolves_when_enabled(self):
        from concept_mapper.corpus.models import Document
        from concept_mapper.preprocessing.pipeline import preprocess

        doc = Document(
            text="Semiosis is unlimited. It generates interpretants.",
            metadata={},
        )
        processed = preprocess(doc, detect_structure=False, resolve_coref=True)
        assert any("Semiosis generates" in s for s in processed.sentences), (
            processed.sentences
        )
