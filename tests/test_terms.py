"""
Tests for term list management (Phase 4).
"""

import pytest
import tempfile
from pathlib import Path
from collections import Counter

from src.concept_mapper.terms.models import TermEntry, TermList
from src.concept_mapper.terms.manager import TermManager
from src.concept_mapper.terms.suggester import suggest_terms_from_analysis
from src.concept_mapper.corpus.models import Document
from src.concept_mapper.preprocessing.pipeline import preprocess


class TestTermEntry:
    """Test TermEntry dataclass."""

    def test_create_minimal_entry(self):
        """Test creating entry with just a term."""
        entry = TermEntry(term="abstraction")

        assert entry.term == "abstraction"
        assert entry.lemma is None
        assert entry.pos is None
        assert entry.definition is None
        assert entry.notes is None
        assert entry.examples == []
        assert entry.metadata == {}

    def test_create_full_entry(self):
        """Test creating entry with all fields."""
        entry = TermEntry(
            term="abstraction",
            lemma="reify",
            pos="NN",
            definition="The objectification of social relations",
            notes="Central to Philosopher' theory",
            examples=["Example sentence 1", "Example sentence 2"],
            metadata={"score": 2.5, "source": "analysis"},
        )

        assert entry.term == "abstraction"
        assert entry.lemma == "reify"
        assert entry.pos == "NN"
        assert entry.definition == "The objectification of social relations"
        assert entry.notes == "Central to Philosopher' theory"
        assert len(entry.examples) == 2
        assert entry.metadata["score"] == 2.5

    def test_to_dict(self):
        """Test serialization to dictionary."""
        entry = TermEntry(
            term="dasein", lemma="dasein", pos="NN", definition="Being-there"
        )

        data = entry.to_dict()

        assert data["term"] == "dasein"
        assert data["lemma"] == "dasein"
        assert data["pos"] == "NN"
        assert data["definition"] == "Being-there"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "term": "totality",
            "lemma": "totality",
            "pos": "NN",
            "definition": "The whole system",
            "notes": "Thinkerist concept",
            "examples": ["Example"],
            "metadata": {},
        }

        entry = TermEntry.from_dict(data)

        assert entry.term == "totality"
        assert entry.definition == "The whole system"
        assert entry.notes == "Thinkerist concept"

    def test_str_representation(self):
        """Test string representation."""
        entry = TermEntry(term="abstraction", definition="Objectification")
        assert str(entry) == "abstraction: Objectification"

        entry_no_def = TermEntry(term="dasein")
        assert str(entry_no_def) == "dasein"


class TestTermList:
    """Test TermList collection."""

    def test_create_empty_list(self):
        """Test creating empty term list."""
        terms = TermList()

        assert terms.name == "Untitled Term List"
        assert terms.description == ""
        assert len(terms) == 0

    def test_create_named_list(self):
        """Test creating list with name and description."""
        terms = TermList(name="Philosopher Terms", description="Key concepts from Philosopher")

        assert terms.name == "Philosopher Terms"
        assert terms.description == "Key concepts from Philosopher"

    def test_add_term(self):
        """Test adding terms."""
        terms = TermList()
        entry = TermEntry(term="abstraction")

        terms.add(entry)

        assert len(terms) == 1
        assert "abstraction" in terms

    def test_add_duplicate_raises(self):
        """Test that adding duplicate raises error."""
        terms = TermList()
        entry = TermEntry(term="abstraction")

        terms.add(entry)

        with pytest.raises(ValueError, match="already exists"):
            terms.add(entry)

    def test_remove_term(self):
        """Test removing terms."""
        terms = TermList()
        entry = TermEntry(term="abstraction")
        terms.add(entry)

        terms.remove("abstraction")

        assert len(terms) == 0
        assert "abstraction" not in terms

    def test_remove_nonexistent_raises(self):
        """Test that removing nonexistent term raises error."""
        terms = TermList()

        with pytest.raises(KeyError, match="not found"):
            terms.remove("nonexistent")

    def test_update_term(self):
        """Test updating term fields."""
        terms = TermList()
        entry = TermEntry(term="abstraction")
        terms.add(entry)

        terms.update("abstraction", definition="New definition", notes="New notes")

        updated = terms.get("abstraction")
        assert updated.definition == "New definition"
        assert updated.notes == "New notes"

    def test_update_nonexistent_raises(self):
        """Test updating nonexistent term raises error."""
        terms = TermList()

        with pytest.raises(KeyError, match="not found"):
            terms.update("nonexistent", definition="test")

    def test_update_invalid_field_raises(self):
        """Test updating invalid field raises error."""
        terms = TermList()
        entry = TermEntry(term="abstraction")
        terms.add(entry)

        with pytest.raises(ValueError, match="Invalid field"):
            terms.update("abstraction", invalid_field="value")

    def test_get_term(self):
        """Test getting a term."""
        terms = TermList()
        entry = TermEntry(term="abstraction", definition="Test")
        terms.add(entry)

        retrieved = terms.get("abstraction")

        assert retrieved is not None
        assert retrieved.term == "abstraction"
        assert retrieved.definition == "Test"

    def test_get_nonexistent_returns_none(self):
        """Test getting nonexistent term returns None."""
        terms = TermList()

        result = terms.get("nonexistent")

        assert result is None

    def test_contains(self):
        """Test contains method."""
        terms = TermList()
        entry = TermEntry(term="abstraction")
        terms.add(entry)

        assert terms.contains("abstraction")
        assert not terms.contains("nonexistent")

    def test_list_terms(self):
        """Test listing all terms."""
        terms = TermList()
        terms.add(TermEntry(term="zebra"))
        terms.add(TermEntry(term="aardvark"))
        terms.add(TermEntry(term="monkey"))

        all_terms = terms.list_terms()

        assert len(all_terms) == 3
        # Should be sorted alphabetically
        assert all_terms[0].term == "aardvark"
        assert all_terms[1].term == "monkey"
        assert all_terms[2].term == "zebra"

    def test_list_term_names(self):
        """Test listing term names only."""
        terms = TermList()
        terms.add(TermEntry(term="abstraction"))
        terms.add(TermEntry(term="totality"))

        names = terms.list_term_names()

        assert names == ["abstraction", "totality"]

    def test_iteration(self):
        """Test iterating over terms."""
        terms = TermList()
        terms.add(TermEntry(term="a"))
        terms.add(TermEntry(term="b"))

        term_list = list(terms)

        assert len(term_list) == 2

    def test_len(self):
        """Test length."""
        terms = TermList()
        assert len(terms) == 0

        terms.add(TermEntry(term="a"))
        assert len(terms) == 1

        terms.add(TermEntry(term="b"))
        assert len(terms) == 2

    def test_to_dict(self):
        """Test serialization to dictionary."""
        terms = TermList(name="Test", description="Testing")
        terms.add(TermEntry(term="abstraction"))

        data = terms.to_dict()

        assert data["name"] == "Test"
        assert data["description"] == "Testing"
        assert len(data["terms"]) == 1
        assert data["terms"][0]["term"] == "abstraction"

    def test_from_dict(self):
        """Test deserialization from dictionary."""
        data = {
            "name": "Test List",
            "description": "Test",
            "terms": [{"term": "abstraction", "lemma": None, "pos": None}],
        }

        terms = TermList.from_dict(data)

        assert terms.name == "Test List"
        assert len(terms) == 1
        assert "abstraction" in terms


class TestTermListPersistence:
    """Test saving and loading term lists."""

    def test_save_and_load(self):
        """Test round-trip save and load."""
        terms = TermList(name="Test Terms", description="Testing persistence")
        terms.add(TermEntry(term="abstraction", definition="Test definition"))
        terms.add(TermEntry(term="totality", notes="Test notes"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "terms.json"

            # Save
            terms.save(path)
            assert path.exists()

            # Load
            loaded = TermList.load(path)

            assert loaded.name == "Test Terms"
            assert loaded.description == "Testing persistence"
            assert len(loaded) == 2
            assert "abstraction" in loaded
            assert "totality" in loaded

            # Check term details preserved
            reif = loaded.get("abstraction")
            assert reif.definition == "Test definition"

    def test_save_creates_parent_directories(self):
        """Test that save creates parent directories."""
        terms = TermList()
        terms.add(TermEntry(term="test"))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "terms.json"

            terms.save(path)

            assert path.exists()
            assert path.parent.exists()

    def test_load_nonexistent_raises(self):
        """Test loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            TermList.load(Path("/nonexistent/path.json"))


class TestTermListMerge:
    """Test merging term lists."""

    def test_merge_no_conflicts(self):
        """Test merging lists with no conflicts."""
        list1 = TermList(name="List 1")
        list1.add(TermEntry(term="a"))
        list1.add(TermEntry(term="b"))

        list2 = TermList(name="List 2")
        list2.add(TermEntry(term="c"))
        list2.add(TermEntry(term="d"))

        merged = list1.merge(list2)

        assert len(merged) == 4
        assert "a" in merged
        assert "b" in merged
        assert "c" in merged
        assert "d" in merged

    def test_merge_with_conflicts_no_overwrite(self):
        """Test merging with conflicts, don't overwrite."""
        list1 = TermList(name="List 1")
        list1.add(TermEntry(term="abstraction", definition="Original"))

        list2 = TermList(name="List 2")
        list2.add(TermEntry(term="abstraction", definition="New"))
        list2.add(TermEntry(term="totality"))

        merged = list1.merge(list2, overwrite=False)

        # Should keep original definition
        assert len(merged) == 2
        reif = merged.get("abstraction")
        assert reif.definition == "Original"

    def test_merge_with_conflicts_overwrite(self):
        """Test merging with conflicts, overwrite."""
        list1 = TermList(name="List 1")
        list1.add(TermEntry(term="abstraction", definition="Original"))

        list2 = TermList(name="List 2")
        list2.add(TermEntry(term="abstraction", definition="New"))

        merged = list1.merge(list2, overwrite=True)

        # Should have new definition
        reif = merged.get("abstraction")
        assert reif.definition == "New"


class TestTermManager:
    """Test TermManager bulk operations."""

    def test_create_manager(self):
        """Test creating manager."""
        manager = TermManager()
        assert manager.term_list is not None
        assert len(manager.term_list) == 0

    def test_create_manager_with_list(self):
        """Test creating manager with existing list."""
        terms = TermList()
        terms.add(TermEntry(term="test"))

        manager = TermManager(terms)

        assert len(manager.term_list) == 1

    def test_import_from_txt(self):
        """Test importing from text file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test file
            txt_path = Path(tmpdir) / "terms.txt"
            txt_path.write_text("abstraction\ntotality\ncommodification\n")

            manager = TermManager()
            count = manager.import_from_txt(txt_path)

            assert count == 3
            assert "abstraction" in manager.term_list
            assert "totality" in manager.term_list
            assert "commodification" in manager.term_list

    def test_import_from_txt_skips_duplicates(self):
        """Test that import skips existing terms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "terms.txt"
            txt_path.write_text("abstraction\ntotality\n")

            manager = TermManager()
            manager.term_list.add(TermEntry(term="abstraction"))

            count = manager.import_from_txt(txt_path)

            # Should only add "totality"
            assert count == 1
            assert len(manager.term_list) == 2

    def test_export_to_txt(self):
        """Test exporting to text file."""
        manager = TermManager()
        manager.term_list.add(TermEntry(term="abstraction"))
        manager.term_list.add(TermEntry(term="totality"))

        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = Path(tmpdir) / "output.txt"

            count = manager.export_to_txt(txt_path)

            assert count == 2
            assert txt_path.exists()

            content = txt_path.read_text()
            assert "abstraction" in content
            assert "totality" in content

    def test_import_export_txt_round_trip(self):
        """Test import/export round trip preserves terms."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Export
            manager1 = TermManager()
            manager1.term_list.add(TermEntry(term="a"))
            manager1.term_list.add(TermEntry(term="b"))

            export_path = Path(tmpdir) / "export.txt"
            manager1.export_to_txt(export_path)

            # Import
            manager2 = TermManager()
            manager2.import_from_txt(export_path)

            assert len(manager2.term_list) == 2
            assert "a" in manager2.term_list
            assert "b" in manager2.term_list

    def test_export_to_csv(self):
        """Test exporting to CSV."""
        manager = TermManager()
        manager.term_list.add(
            TermEntry(
                term="abstraction",
                lemma="reify",
                pos="NN",
                definition="Objectification",
                notes="Philosopher",
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "terms.csv"

            count = manager.export_to_csv(csv_path)

            assert count == 1
            assert csv_path.exists()

            content = csv_path.read_text()
            assert "term,lemma,pos,definition,notes" in content
            assert "abstraction,reify,NN,Objectification,Philosopher" in content

    def test_import_from_csv(self):
        """Test importing from CSV."""
        csv_content = """term,lemma,pos,definition,notes
abstraction,reify,NN,Objectification,Philosopher
totality,totality,NN,The whole,Thinker
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "terms.csv"
            csv_path.write_text(csv_content)

            manager = TermManager()
            count = manager.import_from_csv(csv_path)

            assert count == 2
            assert "abstraction" in manager.term_list
            assert "totality" in manager.term_list

            # Check fields preserved
            reif = manager.term_list.get("abstraction")
            assert reif.lemma == "reify"
            assert reif.pos == "NN"
            assert reif.definition == "Objectification"

    def test_clear(self):
        """Test clearing term list."""
        manager = TermManager()
        manager.term_list.add(TermEntry(term="a"))
        manager.term_list.add(TermEntry(term="b"))

        manager.clear()

        assert len(manager.term_list) == 0

    def test_filter_by_pos(self):
        """Test filtering by POS tags."""
        manager = TermManager()
        manager.term_list.add(TermEntry(term="abstraction", pos="NN"))
        manager.term_list.add(TermEntry(term="totality", pos="NN"))
        manager.term_list.add(TermEntry(term="reify", pos="VB"))

        filtered = manager.filter_by_pos(["NN"])

        assert len(filtered) == 2
        assert "abstraction" in filtered
        assert "totality" in filtered
        assert "reify" not in filtered

    def test_get_statistics(self):
        """Test getting statistics."""
        manager = TermManager()
        manager.term_list.add(
            TermEntry(term="a", pos="NN", definition="Test", examples=["ex"])
        )
        manager.term_list.add(TermEntry(term="b", pos="VB"))
        manager.term_list.add(TermEntry(term="c"))

        stats = manager.get_statistics()

        assert stats["total_terms"] == 3
        assert stats["terms_with_definitions"] == 1
        assert stats["terms_with_examples"] == 1
        assert stats["terms_with_pos"] == 2
        assert stats["pos_distribution"]["NN"] == 1
        assert stats["pos_distribution"]["VB"] == 1


class TestSuggester:
    """Test auto-population from analysis."""

    @pytest.fixture
    def test_docs(self):
        """Create test documents with philosophical terms."""
        text = """
        Dasein is being-in-the-world.
        By Dasein I mean authentic existence.
        Abstraction transforms social relations into things.
        The concept of totality refers to the whole system.
        """
        doc = Document(text=text, metadata={})
        return [preprocess(doc)]

    @pytest.fixture
    def test_reference(self):
        """Simple reference corpus."""
        return Counter({"the": 500, "is": 300, "of": 200, "other": 1000})

    def test_suggest_terms_from_analysis(self, test_docs, test_reference):
        """Test suggesting terms from analysis."""
        term_list = suggest_terms_from_analysis(
            test_docs, test_reference, min_score=0.5, top_n=10, max_examples=2
        )

        # Should create term list
        assert isinstance(term_list, TermList)
        assert len(term_list) > 0

        # Terms should have examples
        for entry in term_list.list_terms():
            # Most terms should have examples (some may not if sentences are short)
            if len(entry.examples) > 0:
                assert isinstance(entry.examples[0], str)

    def test_suggest_terms_respects_top_n(self, test_docs, test_reference):
        """Test that top_n parameter limits results."""
        term_list = suggest_terms_from_analysis(
            test_docs, test_reference, min_score=0.0, top_n=5
        )

        assert len(term_list) <= 5

    def test_suggest_terms_respects_min_score(self, test_docs, test_reference):
        """Test that min_score filters results."""
        # Very high threshold should return few or no terms
        term_list = suggest_terms_from_analysis(
            test_docs, test_reference, min_score=10.0, top_n=50
        )

        # With such a high threshold, should have fewer terms
        assert len(term_list) < 50

    def test_suggested_terms_have_metadata(self, test_docs, test_reference):
        """Test that suggested terms include score metadata."""
        term_list = suggest_terms_from_analysis(
            test_docs, test_reference, min_score=0.5, top_n=5
        )

        if len(term_list) > 0:
            entry = term_list.list_terms()[0]
            assert "score" in entry.metadata
            assert isinstance(entry.metadata["score"], (int, float))


class TestSuggesterByMethod:
    """Test suggestion by specific methods."""

    @pytest.fixture
    def philosophical_docs(self):
        """Documents with clear philosophical signals."""
        text = """
        Dasein is being-in-the-world. By Dasein I mean authentic existence.
        Abstraction transforms relations into things.
        The concept of totality is central to dialectics.
        """
        doc = Document(text=text, metadata={})
        return [preprocess(doc)]

    @pytest.fixture
    def brown_corpus_sample(self):
        """Sample reference corpus."""
        return Counter({"the": 1000, "is": 500, "to": 300, "other": 5000})

    def test_suggest_by_ratio_method(self, philosophical_docs, brown_corpus_sample):
        """Test suggestion using ratio method."""
        from src.concept_mapper.terms.suggester import suggest_terms_by_method

        term_list = suggest_terms_by_method(
            philosophical_docs, brown_corpus_sample, method="ratio", top_n=10
        )

        assert isinstance(term_list, TermList)
        assert "Corpus-Comparative Ratio" in term_list.name

    def test_suggest_by_tfidf_method(self, philosophical_docs, brown_corpus_sample):
        """Test suggestion using TF-IDF method."""
        from src.concept_mapper.terms.suggester import suggest_terms_by_method

        term_list = suggest_terms_by_method(
            philosophical_docs, brown_corpus_sample, method="tfidf", top_n=10
        )

        assert isinstance(term_list, TermList)
        assert "TF-IDF" in term_list.name

    def test_suggest_by_invalid_method_raises(
        self, philosophical_docs, brown_corpus_sample
    ):
        """Test that invalid method raises error."""
        from src.concept_mapper.terms.suggester import suggest_terms_by_method

        with pytest.raises(ValueError, match="Unknown method"):
            suggest_terms_by_method(
                philosophical_docs, brown_corpus_sample, method="invalid"
            )
