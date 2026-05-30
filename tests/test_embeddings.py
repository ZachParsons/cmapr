"""
Tests for the definitional-sentence ranker.

Skipped when the optional ``embeddings`` extra is not installed, since this
module depends on sentence-transformers / torch.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

pytest.importorskip("sentence_transformers")

from concept_mapper.analysis.embeddings import (  # noqa: E402
    DefinitionRanker,
    enrich_graph_with_definitions,
)
from concept_mapper.graph.model import ConceptGraph  # noqa: E402


@pytest.fixture(scope="module")
def ranker() -> DefinitionRanker:
    """Single ranker instance — model load is expensive (~5–15s first time)."""
    return DefinitionRanker()


@pytest.fixture
def docs():
    """Two minimal docs with one definitional and several non-definitional
    sentences mentioning the term 'abduction'."""
    sentences = [
        "Abduction is defined as a kind of inference distinct from deduction.",
        "We discussed abduction yesterday at the seminar.",
        "There were many people at the abduction conference.",
        "By abduction we mean a form of hypothetical inference.",
        "The room was cold during the abduction talk.",
    ]
    doc = MagicMock()
    doc.sentences = sentences
    return [doc]


def test_ranker_prefers_definitional_sentence(ranker):
    sentences = [
        "We discussed sign at the conference yesterday.",
        "A sign is defined as a correlate of expression and content.",
        "The book about signs is on the third shelf.",
    ]
    ranked = ranker.rank(sentences)
    # The "is defined as" sentence should be ranked first
    assert ranked[0][0].startswith("A sign is defined as")
    # Score should be a real cosine in [-1, 1]
    assert -1.0 <= ranked[0][1] <= 1.0


def test_ranker_empty_input(ranker):
    assert ranker.rank([]) == []


def test_ranker_preserves_all_input_sentences(ranker):
    sentences = [
        "First sentence.",
        "A concept is defined as something specific.",
        "Third sentence.",
    ]
    ranked = ranker.rank(sentences)
    assert {s for s, _ in ranked} == set(sentences)


def test_ranker_uses_disk_cache(tmp_path, ranker):
    sentences = ["A sign is defined as something.", "We met at the conference."]
    ranker_with_cache = DefinitionRanker(cache_dir=tmp_path)
    ranker_with_cache.rank(sentences)
    cache_files = list(tmp_path.glob("*.npy"))
    assert len(cache_files) == 1, "expected one cache file written"
    # Second call should not error and should hit the cache
    ranker_with_cache.rank(sentences)
    assert len(list(tmp_path.glob("*.npy"))) == 1


def test_enrich_graph_attaches_definition(docs):
    graph = ConceptGraph(directed=True)
    graph.add_node("abduction", label="abduction")
    graph.add_node("seminar", label="seminar")

    n_added = enrich_graph_with_definitions(graph, docs, threshold=0.0)
    # `abduction` appears in definitional sentences → should be enriched.
    # `seminar` appears only in one non-definitional sentence; threshold=0
    # means it gets *some* sentence regardless.
    assert n_added >= 1
    abd_attrs = graph.get_node("abduction")
    assert "definition" in abd_attrs
    # The chosen sentence must mention abduction
    assert "abduction" in abd_attrs["definition"].lower()


def test_enrich_respects_existing_definition(docs):
    graph = ConceptGraph(directed=True)
    graph.add_node("abduction", label="abduction", definition="pre-existing definition")
    enrich_graph_with_definitions(graph, docs, threshold=0.0)
    assert graph.get_node("abduction")["definition"] == "pre-existing definition"


def test_enrich_skips_when_no_candidates():
    """Nodes whose term never appears in the corpus stay untouched."""
    graph = ConceptGraph(directed=True)
    graph.add_node("nonexistent_term", label="nonexistent_term")
    doc = MagicMock()
    doc.sentences = ["This corpus does not mention that node at all."]
    n_added = enrich_graph_with_definitions(graph, [doc], threshold=0.0)
    assert n_added == 0
    assert "definition" not in graph.get_node("nonexistent_term")


def test_enrich_threshold_filters_low_quality(docs):
    """A very high threshold should suppress all enrichments."""
    graph = ConceptGraph(directed=True)
    graph.add_node("abduction", label="abduction")
    n_added = enrich_graph_with_definitions(graph, docs, threshold=0.999)
    assert n_added == 0
    assert "definition" not in graph.get_node("abduction")
