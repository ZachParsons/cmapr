"""
Sentence-centric proposition extraction via spaCy dependency parsing.

Replaces the regex chain's pairwise scan (which only reads sentences
containing *two* seed terms and matches surface patterns with wide
wildcards) with one parse per sentence: every verb's actual argument
structure is read off the dependency tree, so

* arguments are the verb's real subject/object — no `.{0,80}` gap false
  positives,
* passives, copulars, appositives, and prepositional objects resolve
  correctly,
* coordinated arguments expand ("produces X and Y" → two propositions),
* a sentence containing one seed term still yields propositions against
  other nouns in the sentence (gated downstream by the NodeFilter).

Stays deterministic and extractive: every proposition's arguments are
tokens of the parsed sentence, and the sentence itself is the evidence.
Requires the ``[spacy]`` extra + ``en_core_web_sm`` (lazy-loaded via
``preprocessing/noun_chunks``); callers should check
:func:`dependency_available` and fall back to the regex chain.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .proposition_extractor import Proposition, _score_sentence

# Nominal heads that signal hyponymy when followed by "of" ("a kind of X").
_KIND_WORDS = frozenset(
    {
        "kind",
        "type",
        "sort",
        "species",
        "form",
        "class",
        "subtype",
        "variety",
        "mode",
        "instance",
        "example",
        "case",
    }
)

# Verb lemma → proposition type. Mirrors the regex chain's wordlists
# (_PRODUCTION_VERBS / _DEPENDENCE_PHRASES / _COMPOSITION_VERBS / opposition),
# but keyed on lemmas since the parse hands us the verb directly.
_VERB_TYPE: Dict[str, str] = {
    # definition
    "define": "definition",
    "mean": "definition",
    "denote": "definition",
    "signify": "definition",
    "call": "definition",  # "this process is called semiosis"
    # production
    "produce": "production",
    "generate": "production",
    "create": "production",
    "cause": "production",
    "yield": "production",
    "entail": "production",
    "imply": "production",
    "determine": "production",
    "express": "production",
    "represent": "production",
    "encode": "production",
    "trigger": "production",
    "activate": "production",
    "introduce": "production",
    "indicate": "production",
    # dependence (subject depends on / derives from object)
    "presuppose": "dependence",
    "require": "dependence",
    "need": "dependence",
    "depend": "dependence",
    "rest": "dependence",
    "derive": "dependence",
    "arise": "dependence",
    "emerge": "dependence",
    # component
    "comprise": "component",
    "constitute": "component",
    "compose": "component",
    "consist": "component",
    # opposition
    "oppose": "opposition",
    "contrast": "opposition",
    # generic relation (explicit relational verbs)
    "relate": "relation",
    "connect": "relation",
    "link": "relation",
    "correspond": "relation",
    "involve": "relation",
    "contain": "relation",
    "include": "relation",
    "associate": "relation",
    "interact": "relation",
    "instantiate": "relation",
    "realize": "relation",
    "implement": "relation",
    "apply": "relation",
    "modify": "relation",
    "restrict": "relation",
    "combine": "relation",
}

# (verb lemma, preposition) → type, for verbs whose meaning lives in the
# verb+prep combination. These fire only with their preposition — "stand"
# alone is not definitional, "stands for" is. Mined from the eco_spl1
# corpus (top unmapped verbs connecting seed-term pairs, 2026-06-09).
_VERB_PREP_TYPE: Dict[Tuple[str, str], str] = {
    ("stand", "for"): "definition",
    ("refer", "to"): "production",
    ("lead", "to"): "production",
    ("base", "on"): "dependence",
    ("rest", "on"): "dependence",
    ("belong", "to"): "kind-of",
    ("serve", "as"): "production",
    ("function", "as"): "production",
}

# Prepositions through which a mapped verb's object may arrive
# ("depends on", "derives from", "is defined as", "contrasts with").
_OBJECT_PREPS = frozenset(
    {"on", "upon", "from", "as", "to", "with", "of", "in", "for"}
)

# Semantically empty copular targets: "a sign is *something* which …" must
# not become kind-of(sign, something). When such a target carries a relative
# clause, the relation is extracted *through* it instead.
_VACUOUS_NOMINALS = frozenset(
    {
        "something",
        "anything",
        "nothing",
        "everything",
        "someone",
        "somebody",
        "anyone",
        "anybody",
        "one",
        "thing",
        "what",
        "way",
        "matter",
        "fact",
        "question",
        "problem",
        "point",
        "case",
        "topic",
    }
)


def dependency_available() -> bool:
    """True when spaCy + en_core_web_sm can be loaded in this environment."""
    try:
        from ..preprocessing.noun_chunks import _get_spacy_nlp  # noqa: PLC0415

        _get_spacy_nlp()
        return True
    except Exception:
        return False


class DependencyExtractor:
    """Harvest typed propositions from every sentence in one parsed pass.

    Args:
        sentences: corpus sentences (document order).
        known_terms: seed terms (lowercased); always valid argument nodes.
        include_new_nodes: also emit propositions whose non-seed argument is
            a noun in the sentence — the graph builder's NodeFilter decides
            whether it becomes a node. Only *typed* verbs may introduce new
            nodes; the generic relation fallback requires both arguments to
            be known terms (recall there is not worth the noise).
    """

    def __init__(
        self,
        sentences: List[str],
        known_terms: List[str],
        include_new_nodes: bool = True,
    ):
        from ..preprocessing.noun_chunks import _get_spacy_nlp  # noqa: PLC0415

        self._nlp = _get_spacy_nlp()
        self._sentences = sentences
        self._known = {t.lower() for t in known_terms}
        self._known_multi = {t for t in self._known if " " in t}
        self._include_new = include_new_nodes

    # ------------------------------------------------------------------
    # Argument resolution
    # ------------------------------------------------------------------

    def _resolve_term(self, tok, allow_new: bool) -> Optional[str]:
        """Map an argument token to a node term, or None to drop it."""
        # Semantically empty arguments never become nodes ("related to
        # something"), regardless of construction.
        if tok.lemma_.lower() in _VACUOUS_NOMINALS:
            return None
        # All-caps tokens are running headers / OCR shouting fused into the
        # sentence ("SEMIOTICS AND THE PHILOSOPHY OF LANGUAGE (b) There is…"),
        # not authorial arguments.
        if len(tok.text) > 1 and tok.text.isupper():
            return None
        # Multi-word known term: the token's noun chunk, determiner stripped.
        # Hyphenated seeds tokenize as `sign - function`; renormalize.
        if self._known_multi:
            for chunk in tok.doc.noun_chunks:
                if chunk.start <= tok.i < chunk.end:
                    text = " ".join(w.lower_ for w in chunk if w.dep_ != "det").strip()
                    text = re.sub(r"\s*-\s*", "-", text)
                    if text in self._known_multi:
                        return text
                    break
        # Hyphenated single seeds ("sign-function") split into three tokens;
        # rejoin the token with its immediate hyphen-compound neighbours.
        if tok.i + 2 < len(tok.doc) and tok.doc[tok.i + 1].text == "-":
            joined = f"{tok.lower_}-{tok.doc[tok.i + 2].lower_}"
            if joined in self._known:
                return joined
        if tok.i >= 2 and tok.doc[tok.i - 1].text == "-":
            joined = f"{tok.doc[tok.i - 2].lower_}-{tok.lower_}"
            if joined in self._known:
                return joined
        if tok.lemma_.lower() in self._known:
            return tok.lemma_.lower()
        if tok.lower_ in self._known:
            return tok.lower_
        # Compound anchoring: "a kind of signification process" — the head
        # noun is "process", but the seed term is its compound modifier.
        # Prefer a known term inside the compound over minting a new node.
        # (amod is deliberately excluded: adjective seeds get their own
        # property emission, and anchoring to them collapses source==target.)
        for child in tok.children:
            if child.dep_ == "compound":
                if child.lemma_.lower() in self._known:
                    return child.lemma_.lower()
                if child.lower_ in self._known:
                    return child.lower_
        if allow_new and self._include_new and tok.pos_ in ("NOUN", "PROPN"):
            return tok.lemma_.lower()
        return None

    @staticmethod
    def _with_conjuncts(tok) -> List:
        return [tok, *tok.conjuncts]

    @staticmethod
    def _kind_target(nominal):
        """For 'a kind of X' nominals return X, else the nominal itself."""
        if nominal.lemma_.lower() in _KIND_WORDS:
            for prep in nominal.children:
                if prep.dep_ == "prep" and prep.lower_ == "of":
                    for pobj in prep.children:
                        if pobj.dep_ == "pobj":
                            return pobj
        return nominal

    # ------------------------------------------------------------------
    # Per-construction extraction
    # ------------------------------------------------------------------

    def _collect_objects(self, verb) -> Tuple[List[Tuple], List]:
        """The verb's objects as (token, via_prep) plus passive agents."""
        objects: List[Tuple] = []
        agents: List = []
        for c in verb.children:
            if c.dep_ in ("dobj", "obj", "oprd"):
                objects += [(t, None) for t in self._with_conjuncts(c)]
            elif c.dep_ == "agent":
                for pobj in c.children:
                    if pobj.dep_ == "pobj":
                        agents += self._with_conjuncts(pobj)
            elif c.dep_ == "prep" and c.lower_ in _OBJECT_PREPS:
                for pobj in c.children:
                    if pobj.dep_ == "pobj":
                        objects += [(t, c.lower_) for t in self._with_conjuncts(pobj)]
        return objects, agents

    def _typed_emissions(self, verb, subjects) -> List[Tuple[str, str, str, str]]:
        """Per-object typed tuples for a non-copular verb and given subjects.

        The type is resolved per object: a (verb, preposition) pairing wins
        ("stands *for*" → definition), then the verb's own mapping; an object
        whose preposition carries no meaning for an unmapped verb is dropped.
        """
        passive = any(c.dep_ == "nsubjpass" for c in verb.children)
        vtype = _VERB_TYPE.get(verb.lemma_)
        objects, agents = self._collect_objects(verb)

        out: List[Tuple[str, str, str, str]] = []
        if passive and agents:
            # "Y is produced by X": actor = agent, patient = subject slot.
            etype = vtype or "relation"
            out += [(a, s, etype, verb.lemma_) for a in agents for s in subjects]
            return out

        typed_any = False
        for obj, prep in objects:
            etype = (
                _VERB_PREP_TYPE.get((verb.lemma_, prep)) if prep else None
            ) or vtype
            if etype is None:
                continue
            typed_any = True
            label = f"{verb.lemma_} {prep}".strip() if prep else verb.lemma_
            out += [(s, obj, etype, label) for s in subjects]
        if not typed_any and not passive and not verb.is_stop and verb.pos_ == "VERB":
            # Unmapped content verb: generic relation, direct objects only.
            out += [
                (s, o, "relation", verb.lemma_)
                for s in subjects
                for o, prep in objects
                if prep is None
            ]
        return out

    def _from_verb(self, verb) -> List[Tuple[str, str, str, str]]:
        """(source, target, type, label) tuples from one verb's arguments."""
        subjects = [c for c in verb.children if c.dep_ in ("nsubj", "nsubjpass")]
        if not subjects:
            return []
        subjects = [s for subj in subjects for s in self._with_conjuncts(subj)]

        # Copular: "X is (a kind of) Y" / "X is unlimited"
        if verb.lemma_ == "be":
            out = []
            for c in verb.children:
                if c.dep_ == "attr":
                    target = self._kind_target(c)
                    if target.lemma_.lower() in _VACUOUS_NOMINALS:
                        # "a sign is something which stands for X": the claim
                        # lives in the relative clause — extract through it
                        # with the sentence subject, never kind-of(something).
                        for rel in target.children:
                            if rel.dep_ == "relcl" and rel.pos_ == "VERB":
                                out += self._typed_emissions(rel, subjects)
                        continue
                    for t in self._with_conjuncts(target):
                        out += [(s, t, "kind-of", "is a kind of") for s in subjects]
                elif c.dep_ == "acomp":
                    out += [
                        (s, a, "property", "is")
                        for s in subjects
                        for a in self._with_conjuncts(c)
                    ]
            return out

        return self._typed_emissions(verb, subjects)

    def _from_appos(self, appos) -> List[Tuple[str, str, str, str]]:
        """Appositive: 'semiosis, a species of inference' → kind-of."""
        target = self._kind_target(appos)
        if target.lemma_.lower() in _VACUOUS_NOMINALS:
            return []
        return [(appos.head, target, "kind-of", "is a kind of")]

    # ------------------------------------------------------------------
    # Corpus pass
    # ------------------------------------------------------------------

    def extract_all(self) -> List[Proposition]:
        """One parse per sentence; merged, evidence-ranked propositions."""
        merged: Dict[tuple, Proposition] = {}
        scored: Dict[tuple, List[Tuple[float, str]]] = {}
        n_sents = max(len(self._sentences), 1)

        for sent_idx, doc in enumerate(self._nlp.pipe(self._sentences, batch_size=64)):
            sentence = self._sentences[sent_idx]
            raw: List[Tuple] = []
            for tok in doc:
                if tok.pos_ in ("VERB", "AUX"):
                    raw += self._from_verb(tok)
                elif tok.dep_ == "appos":
                    raw += self._from_appos(tok)
                elif (
                    tok.dep_ == "amod"
                    and tok.lemma_.lower() in self._known
                ):
                    # Adjectival seed terms ("contextual selection",
                    # "extensional device") never head an argument, so verb
                    # extraction can't see them — the modification itself is
                    # the relation: head characterized-as adjective.
                    raw.append((tok.head, tok, "property", "modifies"))

            for src_tok, tgt_tok, etype, label in raw:
                # Generic relations never mint new nodes — typed ones may.
                allow_new = etype != "relation"
                source = self._resolve_term(src_tok, allow_new)
                target = self._resolve_term(tgt_tok, allow_new)
                if not source or not target or source == target:
                    continue
                # At least one argument must be a known term.
                if source not in self._known and target not in self._known:
                    continue
                key = (source, target, etype)
                if key in merged:
                    merged[key].weight += 1
                else:
                    merged[key] = Proposition(
                        source=source,
                        target=target,
                        label=label,
                        type=etype,
                        evidence=[],
                        directed=True,
                        weight=1,
                    )
                    scored[key] = []
                scored[key].append(
                    (
                        _score_sentence(sentence, source, target, sent_idx, n_sents),
                        sentence,
                    )
                )

        for key, prop in merged.items():
            ranked = sorted(scored[key], key=lambda x: -x[0])
            seen = set()
            prop.evidence = []
            for _, s in ranked:
                if s not in seen:
                    seen.add(s)
                    prop.evidence.append(s)
                if len(prop.evidence) == 3:
                    break
        return list(merged.values())
