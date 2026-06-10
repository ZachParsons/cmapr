"""
Coreference resolution (optional, ``[coref]`` extra → fastcoref).

Philosophical prose is anaphora-heavy: "Semiosis is unlimited. *It*
generates interpretants…" — the second sentence carries a production
relation that is invisible to extraction because the concept appears only
as a pronoun. Resolving pronouns to their antecedents *before*
tokenization multiplies the sentences in which each concept is visible to
the concordance, the definitions ranker, and the proposition extractors.

Policy note: coreference resolution is classification over the text's own
mention spans — it selects an antecedent, it does not generate content.
Only pronominal mentions are rewritten, and only with the verbatim text of
their cluster's representative mention, so the result stays span-grounded.

Loaded lazily so the heavy dependency chain (fastcoref → torch +
transformers) is only required when the user opts in
(``cmapr ingest --coref``). The model (``biu-nlp/f-coref``) downloads on
first use.
"""

from __future__ import annotations

from typing import List, Tuple

# Mentions we are willing to rewrite. Conservative: pronouns only — noun
# rephrasings ("this process") are left alone since replacing them can
# change meaning.
_PRONOUNS = frozenset(
    {
        "it",
        "they",
        "them",
        "he",
        "she",
        "him",
        "her",
        "itself",
        "themselves",
        "himself",
        "herself",
    }
)
_POSSESSIVES = frozenset({"its", "their", "theirs", "his", "hers"})

_MODEL = None


def _get_model():
    """Load and cache the fastcoref model."""
    global _MODEL
    if _MODEL is None:
        from fastcoref import FCoref  # noqa: PLC0415

        _MODEL = FCoref(device="cpu")
    return _MODEL


def coref_available() -> bool:
    """True when the coref extra is installed (model loads lazily later)."""
    try:
        import fastcoref  # noqa: F401, PLC0415

        return True
    except ImportError:
        return False


def _representative(text: str, spans: List[Tuple[int, int]]) -> str:
    """The cluster's anchor mention: first non-pronoun span, else empty."""
    for start, end in spans:
        mention = text[start:end]
        head = mention.lower().strip()
        if head not in _PRONOUNS and head not in _POSSESSIVES:
            return mention.strip()
    return ""


def resolve_coreferences(text: str) -> str:
    """Rewrite pronominal mentions with their antecedent's verbatim text."""
    if not text.strip():
        return text
    model = _get_model()
    prediction = model.predict(texts=[text])[0]
    clusters = prediction.get_clusters(as_strings=False)

    replacements: List[Tuple[int, int, str]] = []
    for spans in clusters:
        spans = sorted(spans)
        rep = _representative(text, spans)
        # Anchors longer than a clause make unreadable substitutions.
        if not rep or len(rep.split()) > 6:
            continue
        for start, end in spans:
            mention = text[start:end].lower().strip()
            if mention in _PRONOUNS:
                replacements.append((start, end, rep))
            elif mention in _POSSESSIVES:
                replacements.append((start, end, rep + "'s"))

    # Apply right-to-left so earlier offsets stay valid.
    for start, end, rep in sorted(replacements, key=lambda r: -r[0]):
        text = text[:start] + rep + text[end:]
    return text
