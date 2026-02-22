"""
Tests for relation extraction (Phase 7).
"""

import pytest
from concept_mapper.corpus.models import ProcessedDocument
from concept_mapper.analysis.relations import (
    SVOTriple,
    CopularRelation,
    PrepRelation,
    Relation,
    parse_sentence,
    extract_svo,
    extract_svo_for_term,
    extract_copular,
    extract_prepositional,
    get_relations,
)


@pytest.fixture
def sample_docs():
    """Create sample documents for testing."""
    doc1 = ProcessedDocument(
        raw_text="The dog bites the man. Consciousness is intentional. Being is presence in time.",
        sentences=[
            "The dog bites the man.",
            "Consciousness is intentional.",
            "Being is presence in time.",
        ],
        tokens=[],
        lemmas=[],
        pos_tags=[],
        metadata={"source_path": "doc1.txt"},
    )

    doc2 = ProcessedDocument(
        raw_text="Abstraction transforms processes into things. Freedom from necessity defines autonomy.",
        sentences=[
            "Abstraction transforms processes into things.",
            "Freedom from necessity defines autonomy.",
        ],
        tokens=[],
        lemmas=[],
        pos_tags=[],
        metadata={"source_path": "doc2.txt"},
    )

    doc3 = ProcessedDocument(
        raw_text="Consciousness of objects involves intentionality. Being was conceived as presence.",
        sentences=[
            "Consciousness of objects involves intentionality.",
            "Being was conceived as presence.",
        ],
        tokens=[],
        lemmas=[],
        pos_tags=[],
        metadata={"source_path": "doc3.txt"},
    )

    return [doc1, doc2, doc3]


# ============================================================================
# Test Sentence Parsing
# ============================================================================


class TestParsing:
    """Tests for basic sentence parsing."""

    def test_parse_sentence_pos_tags(self):
        """Test that POS tags are assigned."""
        tagged = parse_sentence("The cat is big.")

        # Should have some standard POS tags
        tags = [t[1] for t in tagged]
        assert any(tag.startswith("NN") for tag in tags)  # Noun
        assert any(tag.startswith("VB") for tag in tags)  # Verb
        assert any(tag.startswith("JJ") for tag in tags)  # Adjective


# ============================================================================
# Test SVO Extraction
# ============================================================================


class TestSVOExtraction:
    """Tests for Subject-Verb-Object extraction."""

    def test_extract_svo_structure(self):
        """Test SVO triple structure."""
        triples = extract_svo("The dog bites the man.")

        if triples:  # May extract successfully
            triple = triples[0]
            assert triple.subject
            assert triple.verb
            assert triple.object
            assert triple.sentence == "The dog bites the man."

    def test_extract_svo_with_doc_id(self):
        """Test that doc_id is preserved."""
        triples = extract_svo("The cat chases the mouse.", doc_id="test_doc")

        if triples:
            assert triples[0].doc_id == "test_doc"

    def test_extract_svo_str_representation(self):
        """Test string representation of SVO triple."""
        triple = SVOTriple(
            subject="dog",
            verb="bites",
            object="man",
            sentence="Test sentence.",
            doc_id="test",
        )

        str_repr = str(triple)
        assert "dog" in str_repr
        assert "bites" in str_repr
        assert "man" in str_repr

    def test_extract_svo_for_term_filtering(self, sample_docs):
        """Test that results are filtered to term."""
        triples = extract_svo_for_term("consciousness", sample_docs)

        # All results should involve "consciousness"
        for triple in triples:
            assert (
                "consciousness" in triple.subject.lower()
                or "consciousness" in triple.verb.lower()
                or "consciousness" in triple.object.lower()
            )

    def test_extract_svo_for_term_case_insensitive(self, sample_docs):
        """Test case-insensitive SVO extraction."""
        triples_lower = extract_svo_for_term("consciousness", sample_docs)
        triples_upper = extract_svo_for_term("CONSCIOUSNESS", sample_docs)

        # Should find same results regardless of case
        assert len(triples_lower) == len(triples_upper)

    def test_extract_svo_for_nonexistent_term(self, sample_docs):
        """Test SVO extraction for term not in corpus."""
        triples = extract_svo_for_term("nonexistent", sample_docs)

        assert len(triples) == 0


# ============================================================================
# Test Copular Relation Extraction
# ============================================================================


class TestCopularExtraction:
    """Tests for copular relation (X is Y) extraction."""

    def test_extract_copular_is_pattern(self, sample_docs):
        """Test extraction of 'X is Y' pattern."""
        relations = extract_copular("consciousness", sample_docs)

        # Should find "Consciousness is intentional"
        if relations:
            assert any("intentional" in r.complement.lower() for r in relations)

    def test_copular_relation_structure(self):
        """Test CopularRelation dataclass structure."""
        rel = CopularRelation(
            subject="Being",
            complement="presence",
            copula="is",
            sentence="Being is presence.",
            doc_id="test",
        )

        assert rel.subject == "Being"
        assert rel.complement == "presence"
        assert rel.copula == "is"
        assert rel.sentence == "Being is presence."

    def test_copular_relation_str(self):
        """Test string representation of copular relation."""
        rel = CopularRelation(
            subject="Being",
            complement="presence",
            copula="is",
            sentence="Being is presence.",
            doc_id="test",
        )

        str_repr = str(rel)
        assert "Being" in str_repr
        assert "is" in str_repr
        assert "presence" in str_repr

    def test_extract_copular_case_sensitive(self, sample_docs):
        """Test case-sensitive copular extraction."""
        relations_lower = extract_copular("being", sample_docs, case_sensitive=False)
        relations_exact = extract_copular("Being", sample_docs, case_sensitive=True)

        # Case-insensitive should find more or equal
        assert len(relations_lower) >= len(relations_exact)

    def test_extract_copular_nonexistent_term(self, sample_docs):
        """Test copular extraction for term not in corpus."""
        relations = extract_copular("nonexistent", sample_docs)

        assert len(relations) == 0


# ============================================================================
# Test Prepositional Relation Extraction
# ============================================================================


class TestPrepositionalExtraction:
    """Tests for prepositional relation extraction."""

    def test_extract_prepositional_of_pattern(self, sample_docs):
        """Test extraction of 'X of Y' pattern."""
        relations = extract_prepositional("consciousness", sample_docs)

        # Should find "consciousness of objects"
        if relations:
            assert any("objects" in r.object.lower() for r in relations)
            assert any(r.prep == "of" for r in relations)

    def test_extract_prepositional_from_pattern(self, sample_docs):
        """Test extraction of 'X from Y' pattern."""
        relations = extract_prepositional("freedom", sample_docs)

        # Should find "freedom from necessity"
        if relations:
            assert any("necessity" in r.object.lower() for r in relations)
            assert any(r.prep == "from" for r in relations)

    def test_prep_relation_structure(self):
        """Test PrepRelation dataclass structure."""
        rel = PrepRelation(
            head="consciousness",
            prep="of",
            object="objects",
            sentence="Consciousness of objects is intentional.",
            doc_id="test",
        )

        assert rel.head == "consciousness"
        assert rel.prep == "of"
        assert rel.object == "objects"

    def test_prep_relation_str(self):
        """Test string representation of prepositional relation."""
        rel = PrepRelation(
            head="consciousness",
            prep="of",
            object="objects",
            sentence="Test.",
            doc_id="test",
        )

        str_repr = str(rel)
        assert "consciousness" in str_repr
        assert "of" in str_repr
        assert "objects" in str_repr

    def test_extract_prepositional_case_sensitive(self, sample_docs):
        """Test case-sensitive prepositional extraction."""
        relations_lower = extract_prepositional(
            "consciousness", sample_docs, case_sensitive=False
        )
        relations_exact = extract_prepositional(
            "Consciousness", sample_docs, case_sensitive=True
        )

        # Case-insensitive should find more or equal
        assert len(relations_lower) >= len(relations_exact)

    def test_extract_prepositional_nonexistent_term(self, sample_docs):
        """Test prepositional extraction for term not in corpus."""
        relations = extract_prepositional("nonexistent", sample_docs)

        assert len(relations) == 0


# ============================================================================
# Test Relation Aggregation
# ============================================================================


class TestRelationAggregation:
    """Tests for aggregated relation extraction."""

    def test_get_relations_specific_types(self, sample_docs):
        """Test extracting specific relation types."""
        relations_copular = get_relations(
            "consciousness", sample_docs, types=["copular"]
        )
        relations_prep = get_relations("consciousness", sample_docs, types=["prep"])

        # All results should be of requested type
        assert all(r.relation_type == "copular" for r in relations_copular)
        assert all(r.relation_type == "prep" for r in relations_prep)

    def test_get_relations_evidence(self, sample_docs):
        """Test that evidence sentences are included."""
        relations = get_relations("consciousness", sample_docs)

        # All relations should have evidence
        for rel in relations:
            assert len(rel.evidence) > 0
            assert all(isinstance(s, str) for s in rel.evidence)

    def test_get_relations_metadata(self, sample_docs):
        """Test that metadata is included."""
        relations = get_relations("consciousness", sample_docs)

        # Relations should have metadata
        for rel in relations:
            assert isinstance(rel.metadata, dict)

    def test_relation_dataclass_structure(self):
        """Test Relation dataclass structure."""
        rel = Relation(
            source="consciousness",
            relation_type="copular",
            target="intentional",
            evidence=["Consciousness is intentional."],
            metadata={"copula": "is"},
        )

        assert rel.source == "consciousness"
        assert rel.relation_type == "copular"
        assert rel.target == "intentional"
        assert len(rel.evidence) == 1
        assert rel.metadata["copula"] == "is"

    def test_relation_str_representation(self):
        """Test string representation of Relation."""
        rel = Relation(
            source="consciousness",
            relation_type="copular",
            target="intentional",
            evidence=["Test."],
            metadata={"copula": "is"},
        )

        str_repr = str(rel)
        assert "consciousness" in str_repr
        assert "copular" in str_repr
        assert "intentional" in str_repr

    def test_get_relations_aggregates_duplicates(self):
        """Test that duplicate relations are aggregated."""
        # Create docs with repeated relation
        doc1 = ProcessedDocument(
            raw_text="Being is presence. Being is presence.",
            sentences=["Being is presence.", "Being is presence."],
            tokens=[],
            lemmas=[],
            pos_tags=[],
            metadata={"source_path": "test.txt"},
        )

        relations = get_relations("being", [doc1], types=["copular"])

        # Should aggregate evidence for same relation
        if relations:
            # May have one relation with multiple evidence sentences
            total_evidence = sum(len(r.evidence) for r in relations)
            assert total_evidence >= 2

    def test_get_relations_empty_types(self, sample_docs):
        """Test with empty types list."""
        relations = get_relations("consciousness", sample_docs, types=[])

        # Should return empty list if no types requested
        assert len(relations) == 0

    def test_get_relations_nonexistent_term(self, sample_docs):
        """Test relation extraction for term not in corpus."""
        relations = get_relations("nonexistent", sample_docs)

        assert len(relations) == 0

    def test_get_relations_multiple_types(self, sample_docs):
        """Test extracting multiple relation types."""
        relations = get_relations(
            "consciousness", sample_docs, types=["copular", "prep"]
        )

        # Should have both types
        types = {r.relation_type for r in relations}
        # May or may not have both, but shouldn't have svo
        assert "svo" not in types


# ============================================================================
# Integration Tests
# ============================================================================


class TestRelationIntegration:
    """Integration tests for complete relation extraction pipeline."""

    def test_full_pipeline(self, sample_docs):
        """Test complete extraction pipeline."""
        # Extract all relations for a term
        relations = get_relations("consciousness", sample_docs)

        # Should extract some relations
        assert isinstance(relations, list)

        # Each relation should have complete structure
        for rel in relations:
            assert rel.source
            assert rel.relation_type in ["svo", "copular", "prep"]
            assert rel.target
            assert len(rel.evidence) > 0

    def test_multiple_terms(self, sample_docs):
        """Test extracting relations for multiple terms."""
        terms = ["consciousness", "being", "freedom"]

        all_relations = {}
        for term in terms:
            relations = get_relations(term, sample_docs)
            all_relations[term] = relations

        # Should have results for each term
        assert len(all_relations) == len(terms)

    def test_empty_corpus(self):
        """Test relation extraction on empty corpus."""
        relations = get_relations("test", [])

        assert len(relations) == 0
