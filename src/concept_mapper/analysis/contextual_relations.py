"""
Contextual relation extraction workflow.

Combines search, term extraction, and relation detection to analyze
how a search term relates to significant terms in its context.
"""

from dataclasses import dataclass
from typing import List, Optional, Literal, TYPE_CHECKING
from collections import defaultdict

from concept_mapper.corpus.models import ProcessedDocument

if TYPE_CHECKING:
    from concept_mapper.search.find import SentenceMatch
    from concept_mapper.search.extract import SignificantTermsResult
    from concept_mapper.analysis.relations import Relation


@dataclass
class ContextualRelation:
    """
    A relation between search term and a significant context term.

    Attributes:
        source: Search term
        relation_type: Type of relation (svo, copular, prep, or cooccurrence)
        target: Related term found in context
        score: Significance score of the target term
        evidence: List of sentences where this relation appears
        metadata: Additional information (e.g., verb, preposition)
    """
    source: str
    relation_type: str
    target: str
    score: float
    evidence: List[str]
    metadata: dict


class ContextualRelationExtractor:
    """
    Extract and analyze relationships between a search term and
    significant terms in its surrounding context.
    """

    def __init__(
        self,
        docs: List[ProcessedDocument],
        scope: Literal["sentence", "paragraph"] = "sentence",
        significance_threshold: float = 0.1,
        pos_types: Optional[List[str]] = None,
        scoring_mode: Literal["corpus_frequency", "hybrid"] = "corpus_frequency",
    ):
        """
        Initialize contextual relation extractor.

        Args:
            docs: Preprocessed documents to analyze
            scope: Context scope (sentence or paragraph)
            significance_threshold: Minimum score for term significance
            pos_types: POS types to extract (nouns, verbs, adjectives, adverbs)
            scoring_mode: Method for scoring term significance
        """
        self.docs = docs
        self.scope = scope
        self.significance_threshold = significance_threshold
        self.pos_types = pos_types or ["nouns", "verbs"]
        self.scoring_mode = scoring_mode

    def extract_for_term(
        self,
        search_term: str,
        match_lemma: bool = False,
        extract_relations: bool = True,
        top_n: Optional[int] = None,
    ) -> List[ContextualRelation]:
        """
        Extract contextual relations for a single search term.

        Args:
            search_term: Term to search for
            match_lemma: Match lemmatized forms (e.g., "run" matches "running")
            extract_relations: Extract grammatical relations (SVO, copular, prep)
            top_n: Limit to top N most significant terms per context

        Returns:
            List of ContextualRelation objects
        """
        # Lazy imports to avoid circular dependencies
        from concept_mapper.search.find import find_sentences
        from concept_mapper.search.extract import extract_significant_terms
        from concept_mapper.analysis.relations import get_relations

        # Step 1: Find all occurrences of the search term
        matches = find_sentences(search_term, self.docs, match_lemma=match_lemma)

        if not matches:
            return []

        # Step 2: Extract significant terms from contexts
        significant_results = extract_significant_terms(
            search_term,
            self.docs,
            threshold=self.significance_threshold,
            pos_types=self.pos_types,
            top_n=top_n,
            match_lemma=match_lemma,
            scoring_mode=self.scoring_mode,
        )

        # Step 3: Build co-occurrence relations from significant terms
        relations = []

        # Group significant terms by sentence for aggregation
        term_scores = defaultdict(lambda: {"scores": [], "sentences": []})

        for result in significant_results:
            sentence = result.sentence_match.sentence
            for term, score, _ in result.significant_terms:
                term_scores[term]["scores"].append(score)
                term_scores[term]["sentences"].append(sentence)

        # Create co-occurrence relations
        for term, data in term_scores.items():
            avg_score = sum(data["scores"]) / len(data["scores"])
            relations.append(
                ContextualRelation(
                    source=search_term,
                    relation_type="cooccurrence",
                    target=term,
                    score=avg_score,
                    evidence=data["sentences"],
                    metadata={"occurrences": len(data["sentences"])},
                )
            )

        # Step 4: Extract grammatical relations if requested
        if extract_relations:
            # Extract SVO, copular, and prepositional relations
            grammatical_rels = get_relations(
                search_term,
                self.docs,
                types=["svo", "copular", "prep"],
                case_sensitive=False,
            )

            # Convert to ContextualRelation format
            for rel in grammatical_rels:
                # Check if target is in our significant terms
                if rel.target in term_scores:
                    score = sum(term_scores[rel.target]["scores"]) / len(
                        term_scores[rel.target]["scores"]
                    )
                else:
                    # Assign default score for grammatically related terms
                    score = 0.5

                relations.append(
                    ContextualRelation(
                        source=rel.source,
                        relation_type=rel.relation_type,
                        target=rel.target,
                        score=score,
                        evidence=rel.evidence,
                        metadata=rel.metadata,
                    )
                )

        return relations

    def extract_batch(
        self,
        search_terms: List[str],
        match_lemma: bool = False,
        extract_relations: bool = True,
        top_n: Optional[int] = None,
    ) -> dict[str, List[ContextualRelation]]:
        """
        Extract contextual relations for multiple search terms.

        Args:
            search_terms: Terms to search for
            match_lemma: Match lemmatized forms
            extract_relations: Extract grammatical relations
            top_n: Limit to top N most significant terms per context

        Returns:
            Dictionary mapping search terms to their relations
        """
        results = {}
        for term in search_terms:
            results[term] = self.extract_for_term(
                term,
                match_lemma=match_lemma,
                extract_relations=extract_relations,
                top_n=top_n,
            )
        return results

    def to_dict(self, relations: List[ContextualRelation]) -> List[dict]:
        """
        Convert relations to dictionary format for JSON export.

        Args:
            relations: List of ContextualRelation objects

        Returns:
            List of dictionaries
        """
        return [
            {
                "source": rel.source,
                "relation_type": rel.relation_type,
                "target": rel.target,
                "score": rel.score,
                "evidence": rel.evidence,
                "metadata": rel.metadata,
            }
            for rel in relations
        ]

    def to_graph_data(self, relations: List[ContextualRelation]) -> dict:
        """
        Convert relations to graph format (nodes and edges).

        Args:
            relations: List of ContextualRelation objects

        Returns:
            Dictionary with 'nodes' and 'edges' keys
        """
        # Collect unique terms
        terms = set()
        for rel in relations:
            terms.add(rel.source)
            terms.add(rel.target)

        # Create nodes
        nodes = [{"id": term, "label": term} for term in terms]

        # Create edges
        edges = [
            {
                "source": rel.source,
                "target": rel.target,
                "relation_type": rel.relation_type,
                "score": rel.score,
                "evidence_count": len(rel.evidence),
            }
            for rel in relations
        ]

        return {"nodes": nodes, "edges": edges}


def analyze_context(
    search_term: str,
    docs: List[ProcessedDocument],
    scope: Literal["sentence", "paragraph"] = "sentence",
    significance_threshold: float = 0.1,
    pos_types: Optional[List[str]] = None,
    match_lemma: bool = False,
    extract_relations: bool = True,
    top_n: Optional[int] = None,
) -> List[ContextualRelation]:
    """
    Convenience function for contextual relation extraction.

    Args:
        search_term: Term to search for
        docs: Preprocessed documents
        scope: Context scope (sentence or paragraph)
        significance_threshold: Minimum score for term significance
        pos_types: POS types to extract (nouns, verbs, adjectives, adverbs)
        match_lemma: Match lemmatized forms
        extract_relations: Extract grammatical relations (SVO, copular, prep)
        top_n: Limit to top N most significant terms per context

    Returns:
        List of ContextualRelation objects

    Examples:
        >>> from concept_mapper.corpus.loader import load_file
        >>> from concept_mapper.preprocessing.pipeline import preprocess
        >>>
        >>> doc = load_file("samples/eco_spl.txt")
        >>> processed = preprocess(doc)
        >>>
        >>> relations = analyze_context("sign", [processed], top_n=10)
        >>> for rel in relations[:5]:
        ...     print(f"{rel.source} --{rel.relation_type}--> {rel.target} ({rel.score:.2f})")
    """
    extractor = ContextualRelationExtractor(
        docs=docs,
        scope=scope,
        significance_threshold=significance_threshold,
        pos_types=pos_types,
    )

    return extractor.extract_for_term(
        search_term=search_term,
        match_lemma=match_lemma,
        extract_relations=extract_relations,
        top_n=top_n,
    )
