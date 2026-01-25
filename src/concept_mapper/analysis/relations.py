"""
Relation extraction for discovering term relationships.

Uses pattern-based extraction and POS tagging to identify grammatical
relationships between terms (Subject-Verb-Object, copular definitions,
prepositional phrases).

Note: Currently uses pattern-based extraction with NLTK. SpaCy integration
is pending Python 3.14 compatibility resolution.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from collections import defaultdict
from ..corpus.models import ProcessedDocument
from ..preprocessing.tagging import tag_tokens
from ..preprocessing.tokenize import tokenize_words


@dataclass
class SVOTriple:
    """
    Subject-Verb-Object triple extracted from text.

    Captures who does what to whom.

    Attributes:
        subject: The subject (doer)
        verb: The action
        object: The object (receiver)
        sentence: The source sentence
        doc_id: Document identifier
    """

    subject: str
    verb: str
    object: str
    sentence: str
    doc_id: str = ""

    def __str__(self) -> str:
        return f"({self.subject}, {self.verb}, {self.object})"


@dataclass
class CopularRelation:
    """
    Copular relation (X is Y) expressing definition or identity.

    Attributes:
        subject: The term being defined
        complement: What it is/are
        copula: The linking verb (is, are, was, etc.)
        sentence: The source sentence
        doc_id: Document identifier
    """

    subject: str
    complement: str
    copula: str
    sentence: str
    doc_id: str = ""

    def __str__(self) -> str:
        return f"({self.subject} {self.copula} {self.complement})"


@dataclass
class PrepRelation:
    """
    Prepositional relation capturing noun phrases linked by prepositions.

    Examples: "consciousness of objects", "freedom from necessity"

    Attributes:
        head: The head noun
        prep: The preposition (of, from, to, for, etc.)
        object: The object of the preposition
        sentence: The source sentence
        doc_id: Document identifier
    """

    head: str
    prep: str
    object: str
    sentence: str
    doc_id: str = ""

    def __str__(self) -> str:
        return f"({self.head} {self.prep} {self.object})"


@dataclass
class Relation:
    """
    Aggregated relation with evidence sentences.

    Attributes:
        source: Source term
        relation_type: Type of relation ("svo", "copular", "prep")
        target: Target term
        evidence: List of example sentences
        metadata: Additional information (verb, preposition, etc.)
    """

    source: str
    relation_type: str
    target: str
    evidence: List[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        meta_str = f" ({self.metadata})" if self.metadata else ""
        return f"{self.source} --[{self.relation_type}]--> {self.target}{meta_str}"


def parse_sentence(sentence: str) -> List[Tuple[str, str]]:
    """
    Parse sentence and return POS-tagged tokens.

    Note: Basic POS tagging. SpaCy dependency parsing will provide
    more sophisticated analysis when Python 3.14 compatibility is resolved.

    Args:
        sentence: Input sentence

    Returns:
        List of (word, pos_tag) tuples

    Example:
        >>> parse_sentence("The cat chases the mouse.")
        [('The', 'DT'), ('cat', 'NN'), ('chases', 'VBZ'), ...]
    """
    tokens = tokenize_words(sentence)
    tagged = tag_tokens(tokens)
    return tagged


def extract_svo(sentence: str, doc_id: str = "") -> List[SVOTriple]:
    """
    Extract Subject-Verb-Object triples from a sentence.

    Uses pattern matching on POS tags to identify basic SVO structures.
    Pattern: NOUN/PRON + VERB + NOUN (with determiners/adjectives allowed)

    Args:
        sentence: Input sentence
        doc_id: Document identifier (optional)

    Returns:
        List of SVOTriple objects

    Example:
        >>> extract_svo("The dog bites the man.")
        [SVOTriple(subject='dog', verb='bites', object='man', ...)]
    """
    tagged = parse_sentence(sentence)
    triples = []

    # Pattern: Look for sequences like:
    # (DT)? (JJ)* NN/NNS/NNP + VB* + (DT)? (JJ)* NN/NNS/NNP

    i = 0
    while i < len(tagged):
        # Find subject (noun potentially with determiners/adjectives)
        if tagged[i][1] in ["NN", "NNS", "NNP", "NNPS", "PRP"]:
            subject = tagged[i][0]

            # Look for verb after subject
            j = i + 1
            while j < len(tagged) and tagged[j][1] in [
                "RB",
                "MD",
            ]:  # Skip adverbs, modals
                j += 1

            if j < len(tagged) and tagged[j][1].startswith("VB"):
                verb = tagged[j][0]

                # Look for object after verb
                k = j + 1
                while k < len(tagged) and tagged[k][1] in [
                    "DT",
                    "JJ",
                    "RB",
                    "IN",
                ]:  # Skip determiners, adjectives, adverbs, prepositions
                    k += 1

                if k < len(tagged) and tagged[k][1] in [
                    "NN",
                    "NNS",
                    "NNP",
                    "NNPS",
                    "PRP",
                ]:
                    obj = tagged[k][0]

                    triple = SVOTriple(
                        subject=subject,
                        verb=verb,
                        object=obj,
                        sentence=sentence,
                        doc_id=doc_id,
                    )
                    triples.append(triple)

        i += 1

    return triples


def extract_svo_for_term(
    term: str,
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
) -> List[SVOTriple]:
    """
    Extract SVO triples involving a specific term.

    Finds all triples where the term appears as subject, verb, or object.

    Args:
        term: Term to search for
        docs: List of preprocessed documents
        case_sensitive: Whether matching is case-sensitive (default: False)

    Returns:
        List of SVOTriple objects involving the term

    Example:
        >>> triples = extract_svo_for_term("abstraction", docs)
        >>> for t in triples:
        ...     print(t)
    """
    search_term = term if case_sensitive else term.lower()
    results = []

    for doc_idx, doc in enumerate(docs):
        doc_id = doc.metadata.get("source_path", f"doc_{doc_idx}")

        for sentence in doc.sentences:
            # Extract all triples from sentence
            triples = extract_svo(sentence, doc_id)

            # Filter to triples involving the term
            for triple in triples:
                subj_match = (
                    search_term in triple.subject.lower()
                    if not case_sensitive
                    else search_term in triple.subject
                )
                verb_match = (
                    search_term in triple.verb.lower()
                    if not case_sensitive
                    else search_term in triple.verb
                )
                obj_match = (
                    search_term in triple.object.lower()
                    if not case_sensitive
                    else search_term in triple.object
                )

                if subj_match or verb_match or obj_match:
                    results.append(triple)

    return results


def extract_copular(
    term: str,
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
) -> List[CopularRelation]:
    """
    Extract copular relations (X is/are/was Y) for a term.

    Identifies sentences where the term is linked to another concept
    via a copular verb (be, become, seem, etc.).

    Pattern: X {is|are|was|were|be|being|been|becomes|seems} Y

    Args:
        term: Term to find definitions/identities for
        docs: List of preprocessed documents
        case_sensitive: Whether matching is case-sensitive (default: False)

    Returns:
        List of CopularRelation objects

    Example:
        >>> relations = extract_copular("Being", docs)
        >>> # Finds: "Being is presence", "Being was conceived as..."
    """
    search_term = term if case_sensitive else term.lower()
    copular_verbs = {
        "is",
        "are",
        "was",
        "were",
        "be",
        "being",
        "been",
        "becomes",
        "become",
        "became",
        "seems",
        "seem",
        "appeared",
        "appears",
    }

    results = []

    for doc_idx, doc in enumerate(docs):
        doc_id = doc.metadata.get("source_path", f"doc_{doc_idx}")

        for sentence in doc.sentences:
            sentence_compare = sentence if case_sensitive else sentence.lower()

            # Check if term appears in sentence
            if search_term not in sentence_compare:
                continue

            # Parse sentence
            tagged = parse_sentence(sentence)

            # Look for pattern: TERM + COPULA + COMPLEMENT
            for i in range(len(tagged)):
                word_compare = tagged[i][0] if case_sensitive else tagged[i][0].lower()

                if search_term in word_compare:
                    # Found the term, look for copular verb nearby
                    for j in range(i + 1, min(i + 4, len(tagged))):
                        if tagged[j][0].lower() in copular_verbs:
                            copula = tagged[j][0]

                            # Look for complement after copula
                            for k in range(j + 1, min(j + 6, len(tagged))):
                                if tagged[k][1] in ["NN", "NNS", "NNP", "NNPS", "JJ"]:
                                    complement = tagged[k][0]

                                    # Extract multi-word complement if possible
                                    complement_words = [complement]
                                    for m in range(k + 1, min(k + 4, len(tagged))):
                                        if tagged[m][1] in [
                                            "NN",
                                            "NNS",
                                            "NNP",
                                            "NNPS",
                                            "JJ",
                                            "IN",
                                        ]:
                                            complement_words.append(tagged[m][0])
                                        else:
                                            break

                                    full_complement = " ".join(complement_words)

                                    relation = CopularRelation(
                                        subject=tagged[i][0],
                                        complement=full_complement,
                                        copula=copula,
                                        sentence=sentence,
                                        doc_id=doc_id,
                                    )
                                    results.append(relation)
                                    break
                            break

    return results


def extract_prepositional(
    term: str,
    docs: List[ProcessedDocument],
    case_sensitive: bool = False,
) -> List[PrepRelation]:
    """
    Extract prepositional relations involving a term.

    Finds patterns like: "X of Y", "X from Y", "X to Y"

    Common philosophical patterns:
    - "consciousness of objects"
    - "freedom from necessity"
    - "relation to being"

    Args:
        term: Term to find prepositional relations for
        docs: List of preprocessed documents
        case_sensitive: Whether matching is case-sensitive (default: False)

    Returns:
        List of PrepRelation objects

    Example:
        >>> relations = extract_prepositional("consciousness", docs)
        >>> # Finds: "consciousness of objects", "consciousness in time"
    """
    search_term = term if case_sensitive else term.lower()
    prepositions = {
        "of",
        "from",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "for",
        "through",
        "against",
        "between",
        "among",
        "within",
    }

    results = []

    for doc_idx, doc in enumerate(docs):
        doc_id = doc.metadata.get("source_path", f"doc_{doc_idx}")

        for sentence in doc.sentences:
            sentence_compare = sentence if case_sensitive else sentence.lower()

            # Check if term appears in sentence
            if search_term not in sentence_compare:
                continue

            # Parse sentence
            tagged = parse_sentence(sentence)

            # Look for pattern: TERM + PREP + NOUN
            for i in range(len(tagged)):
                word_compare = tagged[i][0] if case_sensitive else tagged[i][0].lower()

                if search_term in word_compare and tagged[i][1] in [
                    "NN",
                    "NNS",
                    "NNP",
                    "NNPS",
                ]:
                    head = tagged[i][0]

                    # Look for preposition after the head noun
                    if i + 1 < len(tagged) and tagged[i + 1][0].lower() in prepositions:
                        prep = tagged[i + 1][0]

                        # Look for object of preposition
                        for j in range(i + 2, min(i + 6, len(tagged))):
                            if tagged[j][1] in ["NN", "NNS", "NNP", "NNPS"]:
                                # Extract multi-word object if possible
                                obj_words = [tagged[j][0]]
                                for k in range(j + 1, min(j + 3, len(tagged))):
                                    if tagged[k][1] in [
                                        "NN",
                                        "NNS",
                                        "NNP",
                                        "NNPS",
                                        "JJ",
                                    ]:
                                        obj_words.append(tagged[k][0])
                                    else:
                                        break

                                obj = " ".join(obj_words)

                                relation = PrepRelation(
                                    head=head,
                                    prep=prep,
                                    object=obj,
                                    sentence=sentence,
                                    doc_id=doc_id,
                                )
                                results.append(relation)
                                break
                        break

    return results


def get_relations(
    term: str,
    docs: List[ProcessedDocument],
    types: Optional[List[str]] = None,
    case_sensitive: bool = False,
) -> List[Relation]:
    """
    Get all relations involving a term, aggregated with evidence.

    Extracts and aggregates SVO, copular, and prepositional relations,
    grouping multiple occurrences of the same relation together.

    Args:
        term: Term to find relations for
        docs: List of preprocessed documents
        types: List of relation types to extract ("svo", "copular", "prep")
               If None, extracts all types (default: None)
        case_sensitive: Whether matching is case-sensitive (default: False)

    Returns:
        List of Relation objects with aggregated evidence

    Example:
        >>> relations = get_relations("consciousness", docs)
        >>> for rel in relations:
        ...     print(rel)
        ...     print(f"  Evidence: {len(rel.evidence)} sentences")
    """
    if types is None:
        types = ["svo", "copular", "prep"]

    # Storage for aggregating relations
    relation_map = defaultdict(lambda: {"evidence": [], "metadata": {}})

    # Extract SVO relations
    if "svo" in types:
        svo_triples = extract_svo_for_term(term, docs, case_sensitive)

        for triple in svo_triples:
            # Create keys for different relation directions
            if term.lower() in triple.subject.lower():
                key = ("svo", triple.subject, triple.object, triple.verb)
                relation_map[key]["evidence"].append(triple.sentence)
                relation_map[key]["metadata"]["verb"] = triple.verb
                relation_map[key]["metadata"]["role"] = "subject"

            if term.lower() in triple.object.lower():
                key = ("svo", triple.subject, triple.object, triple.verb)
                relation_map[key]["evidence"].append(triple.sentence)
                relation_map[key]["metadata"]["verb"] = triple.verb
                relation_map[key]["metadata"]["role"] = "object"

    # Extract copular relations
    if "copular" in types:
        copular_rels = extract_copular(term, docs, case_sensitive)

        for cop in copular_rels:
            key = ("copular", cop.subject, cop.complement, cop.copula)
            relation_map[key]["evidence"].append(cop.sentence)
            relation_map[key]["metadata"]["copula"] = cop.copula

    # Extract prepositional relations
    if "prep" in types:
        prep_rels = extract_prepositional(term, docs, case_sensitive)

        for prep_rel in prep_rels:
            key = ("prep", prep_rel.head, prep_rel.object, prep_rel.prep)
            relation_map[key]["evidence"].append(prep_rel.sentence)
            relation_map[key]["metadata"]["preposition"] = prep_rel.prep

    # Convert to Relation objects
    results = []
    for (rel_type, source, target, _), data in relation_map.items():
        relation = Relation(
            source=source,
            relation_type=rel_type,
            target=target,
            evidence=data["evidence"],
            metadata=data["metadata"],
        )
        results.append(relation)

    return results
