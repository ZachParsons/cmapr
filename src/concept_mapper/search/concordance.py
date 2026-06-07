"""
Per-term sentence concordance for the graph visualization.

When a graph node is clicked in the HTML viz, a sidebar lists every sentence
the node's (lemmatized) term appears in, in document order, with the structural
location and the matched word forms flagged for highlighting.

``build_concordance`` is the entry point. It pre-lemmatizes each sentence once
and reuses that index across all requested terms, rather than calling
``find_sentences`` per term (which would re-lemmatize the whole corpus N times).

Output shape — ``{term: [record, ...]}`` where each record is::

    {
        "text":  "...full sentence text...",
        "marks": ["sign", "signs"],   # surface forms to <mark> in the UI
        "loc":   "Ch. 3 — The Sign › §2.1 Definitions",  # may be ""
    }

A term whose list is capped carries a final sentinel record
``{"truncated": shown, "total": total}`` (the UI renders a "showing first N of
M" note); callers that don't care can ignore it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..corpus.models import ProcessedDocument
from ..preprocessing.tokenize import tokenize_words
from ..preprocessing.tagging import tag_tokens
from ..preprocessing.lemmatize import lemmatize, lemmatize_tagged
from nltk.corpus import wordnet as _wn


def _loc_get(loc: Any, key: str) -> Optional[Any]:
    """Read a field from a SentenceLocation or its raw-dict equivalent."""
    if loc is None:
        return None
    if isinstance(loc, dict):
        return loc.get(key)
    return getattr(loc, key, None)


def _format_location(loc: Any) -> str:
    """Build a breadcrumb like ``Ch. 3 — The Sign › §2.1 Definitions``.

    Uses whichever structural fields are present; falls back to the paragraph
    number, then to an empty string. Pages are not tracked at ingest, so they
    never appear here.
    """
    if loc is None:
        return ""

    parts: List[str] = []

    chapter = _loc_get(loc, "chapter")
    chapter_title = _loc_get(loc, "chapter_title")
    if chapter or chapter_title:
        label = f"Ch. {chapter}" if chapter else "Ch."
        if chapter_title:
            label = f"{label} — {chapter_title}" if chapter else chapter_title
        parts.append(label)

    section = _loc_get(loc, "section")
    section_title = _loc_get(loc, "section_title")
    if section or section_title:
        label = f"§{section}" if section else "§"
        if section_title:
            label = f"{label} {section_title}" if section else section_title
        parts.append(label)

    subsection = _loc_get(loc, "subsection")
    subsection_title = _loc_get(loc, "subsection_title")
    if subsection or subsection_title:
        parts.append(subsection_title or f"§§{subsection}")

    if parts:
        return " › ".join(parts)

    paragraph = _loc_get(loc, "paragraph")
    if paragraph is not None:
        return f"¶ {paragraph}"
    return ""


def build_concordance(
    docs: List[ProcessedDocument],
    terms: List[str],
    *,
    per_term_cap: int = 400,
) -> Dict[str, List[Dict[str, Any]]]:
    """Map each term to its located, highlight-ready sentence occurrences.

    Args:
        docs: Preprocessed documents (sentences + sentence_locations).
        terms: Node terms to build concordances for (single words or phrases).
        per_term_cap: Max sentences kept per term; extras are dropped and a
            ``{"truncated": shown, "total": total}`` sentinel is appended.

    Returns:
        ``{term: [record, ...]}``. Terms with no occurrences map to ``[]``.
    """
    # Pre-lemmatize every sentence once; keep doc order.
    indexed: List[Dict[str, Any]] = []
    for doc in docs:
        sentences = doc.sentences or []
        locations = getattr(doc, "sentence_locations", None) or []
        for sent_idx, sentence in enumerate(sentences):
            tokens = tokenize_words(sentence)
            lemmas = lemmatize_tagged(tag_tokens(tokens)) if tokens else []
            loc = locations[sent_idx] if sent_idx < len(locations) else None
            indexed.append(
                {
                    "text": sentence.strip(),
                    "lower": sentence.lower(),
                    "tokens": tokens,
                    "lemmas": lemmas,
                    "lemma_set": set(lemmas),
                    "loc": _format_location(loc),
                }
            )

    result: Dict[str, List[Dict[str, Any]]] = {}

    for term in terms:
        if term in result:
            continue
        records: List[Dict[str, Any]] = []
        total = 0

        if " " in term:
            # Phrase: case-insensitive substring match (lemma search only keys
            # off the first token, so it would be wrong for phrases).
            needle = term.lower()
            for sent in indexed:
                if needle in sent["lower"]:
                    total += 1
                    if len(records) < per_term_cap:
                        records.append(
                            {
                                "text": sent["text"],
                                "marks": [term],
                                "loc": sent["loc"],
                            }
                        )
        else:
            term_lemma = lemmatize(term, _wn.NOUN)
            for sent in indexed:
                if term_lemma not in sent["lemma_set"]:
                    continue
                total += 1
                if len(records) >= per_term_cap:
                    continue
                # Surface forms in this sentence whose lemma matches.
                marks = sorted(
                    {
                        tok
                        for tok, lem in zip(sent["tokens"], sent["lemmas"])
                        if lem == term_lemma
                    }
                )
                records.append(
                    {
                        "text": sent["text"],
                        "marks": marks or [term],
                        "loc": sent["loc"],
                    }
                )

        if total > len(records):
            records.append({"truncated": len(records), "total": total})

        result[term] = records

    return result
