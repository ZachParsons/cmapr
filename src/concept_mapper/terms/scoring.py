"""
Post-scoring filter chain for the rarities pipeline.

`PhilosophicalTermScorer` in :mod:`concept_mapper.analysis.rarity`
produces ranked term candidates. Before they become a `TermList`, a
sequence of filters runs over them: quote-stripping, multi-word noun
chunk injection, proper-name removal, lemma + derivational-suffix
collapse with top-N trim, fragment removal, POS filtering, vetting.

That chain used to live inline in ``cli.py:rarities`` *and* in
``cli.py:run`` (duplicated). This module is the single source of truth.

Each helper is independently composable; the CLI handlers wire them in
order and emit their own user-facing echoes between steps.

A candidate is a 3-tuple ``(term, score, components_dict)`` matching the
shape `PhilosophicalTermScorer.score_all` returns.
"""

from __future__ import annotations

import json
from collections import Counter
from math import log
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

Candidate = Tuple[str, float, Dict[str, Any]]

# Curly + straight single quotes that frequently attach to OCR'd tokens
_QUOTE_CHARS = "'‘’‚‛"

# Derivational suffixes used by lemma_and_derivational_merge to collapse
# adjective/noun variants whose base is already a candidate.
_DERIVATIONAL_SUFFIXES = (
    "ual",
    "ial",
    "ical",
    "ic",
    "ive",
    "ous",
    "ity",
    "ism",
    "ist",
    "ness",
    "ary",
    "ory",
    "al",
)

# WordNet suffixes used by filter_fragments to detect prefix-of-word
# completions (e.g. 'tion' is a fragment of 'proposition' + 's').
_COMPLETION_SUFFIXES = ("s", "y", "es", "ed", "er", "al", "ic", "is", "sis")

# Coarse POS categories used by filter_by_pos_categories.
_POS_CATEGORY_MAP: Dict[str, Set[str]] = {
    "noun": {"NN", "NNS", "NNP", "NNPS"},
    "verb": {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"},
    "adj": {"JJ", "JJR", "JJS"},
    "adv": {"RB", "RBR", "RBS"},
}


# ---------------------------------------------------------------------------
# Filter step 1 — quote stripping
# ---------------------------------------------------------------------------


def strip_stray_quotes(candidates: List[Candidate]) -> List[Candidate]:
    """Strip leading/trailing quote characters from terms; drop empties.

    OCR'd PDFs frequently leave stray apostrophes or curly quotes attached
    to tokens (e.g. ``"'animal'"`` → ``"animal"``).
    """
    return [
        (term.strip(_QUOTE_CHARS), score, components)
        for term, score, components in candidates
        if term.strip(_QUOTE_CHARS)
    ]


# ---------------------------------------------------------------------------
# Filter step 2 — multi-word noun chunks (only when spaCy ran upstream)
# ---------------------------------------------------------------------------


def score_multi_word_chunks(docs: list) -> List[Candidate]:
    """Score multi-word noun chunks via TF-IDF.

    Reads ``doc.metadata["noun_chunks"]`` (populated by
    ``preprocessing/noun_chunks.py`` when ``cmapr ingest --spacy`` ran).
    Phrases appearing in fewer than ``min(2, n_docs)`` documents are
    dropped.

    Returns a candidate list ready to merge into the scorer's output.
    Scoring formula: ``log(1 + freq) * (1 + log((n_docs + 1) / (df + 1)))``
    — chosen to land in the 1–4 range comparable with
    `PhilosophicalTermScorer`'s top terms.
    """
    chunks: list = []
    for doc in docs:
        chunks.extend(getattr(doc, "metadata", {}).get("noun_chunks", []) or [])
    if not chunks:
        return []

    chunk_freq = Counter(chunks)
    doc_chunk_sets = [
        set(getattr(d, "metadata", {}).get("noun_chunks", []) or []) for d in docs
    ]
    n_docs = max(len(docs), 1)
    min_freq = min(2, n_docs)

    out: List[Candidate] = []
    for chunk, freq in chunk_freq.items():
        if freq < min_freq:
            continue
        df = sum(1 for dc in doc_chunk_sets if chunk in dc)
        idf = log((n_docs + 1) / (df + 1))
        score = log(1 + freq) * (1 + idf)
        out.append((chunk, score, {"tfidf": score, "total": score}))
    return out


def merge_extra_candidates(
    candidates: List[Candidate], extra: List[Candidate]
) -> List[Candidate]:
    """Merge an extra candidate list into the main one and re-sort by score."""
    if not extra:
        return candidates
    return sorted(candidates + extra, key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# Filter step 3 — proper-name removal
# ---------------------------------------------------------------------------


def filter_proper_names(
    candidates: List[Candidate],
    docs: list,
    reference: Dict[str, int],
) -> List[Candidate]:
    """Drop terms that look like proper names.

    Heuristic: a term is a proper name if (a) it is tagged as a proper
    noun in ≥ 30 % of its corpus occurrences AND (b) it appears < 25 ppm
    in the Brown reference corpus. The Brown floor keeps common English
    words like "language" or "symbol" from being filtered just because
    they happen to be title-cased in this corpus.
    """
    from concept_mapper.analysis.rarity import proper_noun_ratios  # noqa: PLC0415

    pn_ratios = proper_noun_ratios(docs)
    ref_total = sum(reference.values())

    def _is_proper_name(term: str) -> bool:
        if pn_ratios.get(term, 0) < 0.3:
            return False
        ref_ppm = reference.get(term, 0) / ref_total * 1_000_000
        return ref_ppm < 25

    return [
        (term, score, components)
        for term, score, components in candidates
        if not _is_proper_name(term)
    ]


# ---------------------------------------------------------------------------
# Filter step 4 — lemma + derivational-suffix merge with top-N trim
# ---------------------------------------------------------------------------


def lemma_and_derivational_merge(
    candidates: List[Candidate], top_n: Optional[int] = None
) -> List[Candidate]:
    """Collapse inflected and derivational variants to a canonical base.

    Pass 1: merge inflected noun forms via WordNet lemma (``semiotics`` →
    ``semiotic``).

    Pass 2: collapse derivational adjective/noun variants whose shorter
    base is already a candidate (``co-textual`` → ``co-text``). Longer
    suffixes tried first to avoid partial matches.

    Optional ``top_n`` trim — applied after the final sort. The top-n cut
    happens *here*, not at the end of the pipeline, because subsequent
    filters (fragments, POS, vetting) should refine a bounded list rather
    than a long one.
    """
    from concept_mapper.preprocessing.lemmatize import lemmatize  # noqa: PLC0415
    from nltk.corpus import wordnet as wn  # noqa: PLC0415

    # Pass 1
    lemma_best: Dict[str, Candidate] = {}
    for term, score, components in candidates:
        base = lemmatize(term, wn.NOUN)
        if base not in lemma_best or score > lemma_best[base][1]:
            lemma_best[base] = (base, score, components)

    # Pass 2
    merged: Dict[str, Candidate] = {}
    for base_form, entry in lemma_best.items():
        canonical = base_form
        for suffix in _DERIVATIONAL_SUFFIXES:
            if base_form.endswith(suffix) and len(base_form) - len(suffix) >= 3:
                shorter = base_form[: -len(suffix)]
                if shorter in lemma_best:
                    canonical = shorter
                    break
        _, score, components = entry
        if canonical not in merged or score > merged[canonical][1]:
            merged[canonical] = (canonical, score, components)

    out = sorted(merged.values(), key=lambda x: x[1], reverse=True)
    if top_n is not None:
        out = out[:top_n]
    return out


# ---------------------------------------------------------------------------
# Filter step 5 — fragment removal
# ---------------------------------------------------------------------------


def filter_fragments(candidates: List[Candidate]) -> List[Candidate]:
    """Drop fragments: terms < 4 chars or prefix-of-WordNet-word artifacts.

    A term is a fragment if it is below 4 characters, OR (not in WordNet
    AND completion-suffix-prefixes a WordNet word). Multi-word phrases
    pass on length.
    """
    from nltk.corpus import wordnet as wn  # noqa: PLC0415

    wn_words = set(wn.words())

    def _is_fragment(term: str) -> bool:
        if len(term) < 4:
            return True
        if term in wn_words:
            return False
        return any((term + s) in wn_words for s in _COMPLETION_SUFFIXES)

    return [
        (term, score, components)
        for term, score, components in candidates
        if not _is_fragment(term)
    ]


# ---------------------------------------------------------------------------
# Filter step 6 — POS category restriction
# ---------------------------------------------------------------------------


def filter_by_pos_categories(
    candidates: List[Candidate],
    docs: list,
    pos_arg: str,
) -> Tuple[List[Candidate], List[str]]:
    """Restrict candidates to one or more POS categories.

    ``pos_arg`` is a comma-separated string of ``noun``, ``verb``,
    ``adj``, and/or ``adv``. Multi-word phrases (terms containing a
    space) always pass.

    Returns ``(filtered_candidates, unknown_categories)``. The caller
    decides how to warn about ``unknown_categories``.
    """
    from concept_mapper.analysis.rarity import filter_by_pos_tags  # noqa: PLC0415

    requested_tags: Set[str] = set()
    unknown: List[str] = []
    for cat in pos_arg.split(","):
        cat = cat.strip().lower()
        if cat not in _POS_CATEGORY_MAP:
            unknown.append(cat)
            continue
        requested_tags.update(_POS_CATEGORY_MAP[cat])

    if not requested_tags:
        return candidates, unknown

    allowed = filter_by_pos_tags(docs, include_tags=requested_tags, exclude_tags=None)
    out = [
        (term, score, components)
        for term, score, components in candidates
        # multi-word noun chunks (contain space) always pass
        if " " in term or term.lower() in allowed
    ]
    return out, unknown


# ---------------------------------------------------------------------------
# Filter step 7 — vetting (load/apply/save)
# ---------------------------------------------------------------------------


def load_vetting(path: Path) -> Tuple[Set[str], Set[str], bool]:
    """Read accept/reject sets from ``vetting.json``.

    Returns ``(accepted, rejected, found)``. When the file does not
    exist, returns empty sets and ``found=False``. Term keys are
    lowercased on read.
    """
    accepted: Set[str] = set()
    rejected: Set[str] = set()
    p = Path(path)
    if not p.exists():
        return accepted, rejected, False
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    accepted = {t.lower() for t in data.get("accept", [])}
    rejected = {t.lower() for t in data.get("reject", [])}
    return accepted, rejected, True


def save_vetting(path: Path, accepted: Set[str], rejected: Set[str]) -> None:
    """Write accept/reject sets to ``vetting.json`` (sorted, indented)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(
            {"accept": sorted(accepted), "reject": sorted(rejected)},
            f,
            indent=2,
            ensure_ascii=False,
        )


def apply_vetting(
    candidates: List[Candidate],
    raw_candidates: List[Candidate],
    accepted: Set[str],
    rejected: Set[str],
) -> List[Candidate]:
    """Drop rejected terms; re-include accepted terms cut by earlier filters.

    ``raw_candidates`` is the pre-filter snapshot (taken before
    proper-name / fragment / top-n trims). Accepted terms can be
    re-injected from there if they were dropped along the way.

    Returns a freshly sorted list.
    """
    if rejected:
        candidates = [(t, s, c) for t, s, c in candidates if t.lower() not in rejected]

    if accepted:
        current = {t.lower() for t, _, _ in candidates}
        for term, score, components in raw_candidates:
            if term.lower() in accepted and term.lower() not in current:
                candidates.append((term, score, components))
                current.add(term.lower())
        candidates = sorted(candidates, key=lambda x: x[1], reverse=True)

    return candidates


# ---------------------------------------------------------------------------
# High-level convenience — the chain used by `cmapr run`
# ---------------------------------------------------------------------------


def apply_run_pipeline(
    candidates: List[Candidate],
    docs: list,
    reference: Dict[str, int],
    *,
    top_n: Optional[int] = None,
    no_filter_names: bool = False,
    no_lemmatize: bool = False,
    no_filter_fragments: bool = False,
) -> List[Candidate]:
    """The simplified filter chain used by ``cmapr run``.

    Order: quote-strip → proper-name → lemma+suffix merge (top-N) →
    fragment. Does *not* include multi-word noun chunks, POS filter, or
    vetting — those are rarities-only steps.
    """
    candidates = strip_stray_quotes(candidates)
    if not no_filter_names:
        candidates = filter_proper_names(candidates, docs, reference)
    if not no_lemmatize:
        candidates = lemma_and_derivational_merge(candidates, top_n=top_n)
    if not no_filter_fragments:
        candidates = filter_fragments(candidates)
    return candidates
