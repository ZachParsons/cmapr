"""
Derive a definition for every graph node from the input text.

Definitions are **mandatory** (every node gets one) and **extractive** (a real
sentence from the corpus, never a dictionary gloss). The evidence pool is the
term's *concordance* — every sentence it occurs in — ranked by a composite
"how definitional is this sentence?" score that blends:

* explicit definitional patterns (``analysis.rarity.DEFINITIONAL_PATTERNS``),
* optional sentence-embedding similarity to a definitional prototype
  (``analysis.embeddings.DefinitionRanker``) when the ``embeddings`` extra is
  installed — a booster, never required,
* the definiendum appearing early / in subject position,
* a small bonus for the term's first occurrences,
* a soft length penalty (drop fragments and run-ons).

When a term has no full-sentence occurrence, the coverage fallback chain is
``best concordance sentence → edge-derived gloss → first substring occurrence``
so a node is never left undefined.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from ..corpus.models import ProcessedDocument
from ..search.concordance import build_concordance

# Per-pattern confidence (keys match DEFINITIONAL_PATTERNS' pattern_type).
_PATTERN_WEIGHTS = {
    "explicit_define": 1.0,
    "explicit_mean": 1.0,
    "metalinguistic": 0.9,
    "appositive": 0.8,
    "referential": 0.8,
    "interpretive": 0.8,
    "copular": 0.6,
    "conceptual": 0.5,
}

# Definition-specific cues missing from rarity.DEFINITIONAL_PATTERNS (which is
# tuned for term *importance*, not extraction). Kept local so the rarity signal
# is untouched. Each is (regex with the definiendum as group 1, pattern_type).
_EXTRA_DEFINITION_PATTERNS = [
    (r"\b(\w+(?:-\w+)*)\s+(?:is|are|was|were)\s+defined\s+as", "explicit_define"),
    (r"\b(\w+(?:-\w+)*)\s+(?:denotes|designates|signifies|names)\b", "referential"),
    (r"\b(\w+(?:-\w+)*)\s+is\s+(?:understood|conceived|taken)\s+as", "interpretive"),
]

# Composite signal weights. embed's weight folds into pattern when no ranker.
_W_PATTERN = 0.45
_W_EMBED = 0.30
_W_SUBJECT = 0.15
_W_INTRO = 0.10

# Edge types that can characterize a node, best-first (for the gloss fallback).
_GLOSS_EDGE_PRIORITY = ("definition", "kind-of", "property", "relation")
_GLOSS_PREFIX = {
    "definition": "",
    "kind-of": "a kind of ",
    "property": "described as ",
    "relation": "related to ",
}


def _pattern_score(sentence: str, term: str) -> float:
    """Best definitional-pattern weight whose captured term matches ``term``."""
    from .rarity import DEFINITIONAL_PATTERNS  # noqa: PLC0415

    tl = term.lower()
    last = tl.split()[-1] if " " in tl else tl
    best = 0.0
    for pattern, ptype in (*DEFINITIONAL_PATTERNS, *_EXTRA_DEFINITION_PATTERNS):
        for match in re.finditer(pattern, sentence, re.IGNORECASE):
            captured = match.group(1).lower()
            if captured == tl or captured == last:
                best = max(best, _PATTERN_WEIGHTS.get(ptype, 0.5))
    return best


def _subject_position(sentence: str, term: str) -> float:
    """1.0 when the term leads the sentence, decaying over the first few words."""
    tokens = sentence.lower().split()
    head = term.lower().split()[0]
    for idx, tok in enumerate(tokens[:6]):
        if tok.strip(".,;:'\"()").startswith(head):
            return max(0.0, 1.0 - idx * 0.18)
    return 0.0


def _length_penalty(sentence: str) -> float:
    """Soft penalty for sentences outside a readable definitional length."""
    n = len(sentence.split())
    if 8 <= n <= 35:
        return 0.0
    if n < 8:
        return min(0.3, (8 - n) * 0.05)
    return min(0.3, (n - 35) * 0.01)


def _composite_score(
    sentence: str, term: str, *, sim: Optional[float], is_intro: bool
) -> float:
    """Blend the definitional signals into a single 0–1-ish score."""
    pattern = _pattern_score(sentence, term)
    subject = _subject_position(sentence, term)
    intro = 1.0 if is_intro else 0.0
    if sim is None:
        # No embedding signal — fold its weight into the pattern signal.
        score = (_W_PATTERN + _W_EMBED) * pattern
    else:
        score = _W_PATTERN * pattern + _W_EMBED * max(0.0, min(1.0, sim))
    score += _W_SUBJECT * subject + _W_INTRO * intro
    return score - _length_penalty(sentence)


def _edge_gloss(graph, node_id: str) -> Optional[str]:
    """Characterize a node from its outgoing typed edges (fallback only)."""
    nx_graph = graph.graph
    edge_iter = (
        nx_graph.out_edges(node_id, data=True)
        if graph.directed
        else nx_graph.edges(node_id, data=True)
    )
    by_type: Dict[str, str] = {}
    for src, tgt, data in edge_iter:
        other = tgt if src == node_id else src
        by_type.setdefault(data.get("type", "relation"), other)
    for etype in _GLOSS_EDGE_PRIORITY:
        if etype in by_type:
            return f"{_GLOSS_PREFIX[etype]}{by_type[etype]}".strip() or None
    return None


def _first_occurrence_sentence(docs: List[ProcessedDocument], term: str) -> Optional[str]:
    """Last-resort: the first sentence whose text contains the term substring."""
    tl = term.lower()
    for doc in docs:
        for sentence in doc.sentences or []:
            if tl in sentence.lower():
                return sentence.strip()
    return None


def derive_definitions(
    graph,
    docs: List[ProcessedDocument],
    *,
    ranker=None,
) -> int:
    """Attach a text-derived ``definition`` to every node; return the count.

    Nodes already carrying a ``definition`` (e.g. from a curated term list) are
    left untouched. ``ranker`` is an optional
    :class:`analysis.embeddings.DefinitionRanker` whose similarity becomes the
    ``embed`` signal; without it the composite is fully offline.
    """
    node_terms = [graph.get_node(n).get("term") or n for n in graph.nodes()]
    concord = build_concordance(docs, node_terms)

    if ranker is not None:
        all_sentences = [s for doc in docs for s in (doc.sentences or [])]
        if all_sentences:
            ranker.precompute(all_sentences)

    nx_graph = graph.graph
    defined = 0
    for node_id in graph.nodes():
        attrs = nx_graph.nodes[node_id]
        if attrs.get("definition"):
            continue
        term = attrs.get("term") or node_id

        records = [r for r in concord.get(term, []) if "text" in r]
        definition: Optional[str] = None
        source: Optional[str] = None

        if records:
            sims: Optional[Dict[str, float]] = None
            if ranker is not None:
                sims = dict(ranker.rank([r["text"] for r in records]))
            best: Optional[Tuple[float, dict]] = None
            for idx, rec in enumerate(records):
                sim = sims.get(rec["text"]) if sims is not None else None
                s = _composite_score(
                    rec["text"], term, sim=sim, is_intro=idx < 2
                )
                if best is None or s > best[0]:
                    best = (s, rec)
            definition = best[1]["text"]
            source = best[1].get("loc") or None
        if not definition:
            definition = _edge_gloss(graph, node_id)
        if not definition:
            definition = _first_occurrence_sentence(docs, term)

        if definition:
            # Collapse line-wrap newlines / runs of whitespace for clean display.
            nx_graph.nodes[node_id]["definition"] = " ".join(definition.split())
            if source:
                nx_graph.nodes[node_id]["definition_source"] = source
            defined += 1

    return defined
