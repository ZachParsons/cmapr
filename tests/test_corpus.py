"""
Tests for corpus data structures and loading.
"""

from pathlib import Path

import pytest

from src.concept_mapper.corpus import (
    Corpus,
    Document,
    ProcessedDocument,
    load_directory,
    load_file,
    load_text,
)


class TestDocument:
    """Test Document dataclass."""

    def test_create_document(self):
        """Test creating a document."""
        doc = Document(text="Sample text", metadata={"title": "Test"})

        assert doc.text == "Sample text"
        assert doc.title == "Test"
        assert doc.metadata["title"] == "Test"

    def test_document_properties(self):
        """Test document property accessors."""
        metadata = {
            "title": "Philosophy of Mind",
            "author": "John Searle",
            "date": "1992",
            "source_path": "/path/to/file.txt",
        }
        doc = Document(text="Content", metadata=metadata)

        assert doc.title == "Philosophy of Mind"
        assert doc.author == "John Searle"
        assert doc.date == "1992"
        assert doc.source_path == Path("/path/to/file.txt")

    def test_document_to_dict(self):
        """Test serializing document to dict."""
        doc = Document(text="Text", metadata={"title": "Test"})
        data = doc.to_dict()

        assert data["text"] == "Text"
        assert data["metadata"]["title"] == "Test"

    def test_document_from_dict(self):
        """Test deserializing document from dict."""
        data = {"text": "Text", "metadata": {"title": "Test"}}
        doc = Document.from_dict(data)

        assert doc.text == "Text"
        assert doc.title == "Test"


class TestProcessedDocument:
    """Test ProcessedDocument dataclass."""

    def test_create_processed_document(self):
        """Test creating a processed document."""
        doc = ProcessedDocument(
            raw_text="The cat sat.",
            sentences=["The cat sat."],
            tokens=["The", "cat", "sat", "."],
            pos_tags=[("The", "DT"), ("cat", "NN"), ("sat", "VBD"), (".", ".")],
            lemmas=["the", "cat", "sit", "."],
        )

        assert doc.raw_text == "The cat sat."
        assert doc.num_sentences == 1
        assert doc.num_tokens == 4

    def test_processed_document_to_dict(self):
        """Test serializing processed document."""
        doc = ProcessedDocument(
            raw_text="Text",
            sentences=["Text"],
            tokens=["Text"],
            pos_tags=[("Text", "NN")],
            lemmas=["text"],
            metadata={"title": "Test"},
        )
        data = doc.to_dict()

        assert data["raw_text"] == "Text"
        assert data["sentences"] == ["Text"]
        assert data["metadata"]["title"] == "Test"

    def test_processed_document_from_dict(self):
        """Test deserializing processed document."""
        data = {
            "raw_text": "Text",
            "sentences": ["Text"],
            "tokens": ["Text"],
            "pos_tags": [("Text", "NN")],
            "lemmas": ["text"],
            "metadata": {"title": "Test"},
        }
        doc = ProcessedDocument.from_dict(data)

        assert doc.raw_text == "Text"
        assert doc.num_sentences == 1
        assert doc.title == "Test"


class TestCorpus:
    """Test Corpus class."""

    def test_create_empty_corpus(self):
        """Test creating an empty corpus."""
        corpus = Corpus()
        assert len(corpus) == 0

    def test_create_corpus_with_documents(self):
        """Test creating corpus with documents."""
        docs = [
            Document(text="Doc 1", metadata={"title": "First"}),
            Document(text="Doc 2", metadata={"title": "Second"}),
        ]
        corpus = Corpus(documents=docs)

        assert len(corpus) == 2
        assert corpus[0].title == "First"
        assert corpus[1].title == "Second"

    def test_add_document(self):
        """Test adding documents to corpus."""
        corpus = Corpus()
        corpus.add_document(Document(text="Doc 1"))

        assert len(corpus) == 1

    def test_corpus_iteration(self):
        """Test iterating over corpus."""
        docs = [Document(text=f"Doc {i}") for i in range(3)]
        corpus = Corpus(documents=docs)

        count = 0
        for doc in corpus:
            assert doc.text.startswith("Doc")
            count += 1

        assert count == 3

    def test_corpus_to_dict(self):
        """Test serializing corpus to dict."""
        docs = [
            Document(text="Doc 1", metadata={"title": "First"}),
            Document(text="Doc 2", metadata={"title": "Second"}),
        ]
        corpus = Corpus(documents=docs)
        data = corpus.to_dict()

        assert data["num_documents"] == 2
        assert len(data["documents"]) == 2
        assert data["documents"][0]["metadata"]["title"] == "First"

    def test_corpus_from_dict(self):
        """Test deserializing corpus from dict."""
        data = {
            "num_documents": 2,
            "documents": [
                {"text": "Doc 1", "metadata": {"title": "First"}},
                {"text": "Doc 2", "metadata": {"title": "Second"}},
            ],
        }
        corpus = Corpus.from_dict(data)

        assert len(corpus) == 2
        assert corpus[0].title == "First"


class TestLoader:
    """Test file loading functions."""

    @pytest.fixture
    def sample_file(self, tmp_path):
        """Create a sample text file."""
        file_path = tmp_path / "sample.txt"
        file_path.write_text("This is a sample text file.\nWith multiple lines.")
        return file_path

    @pytest.fixture
    def sample_directory(self, tmp_path):
        """Create a directory with multiple text files."""
        dir_path = tmp_path / "corpus"
        dir_path.mkdir()

        (dir_path / "doc1.txt").write_text("First document.")
        (dir_path / "doc2.txt").write_text("Second document.")
        (dir_path / "doc3.txt").write_text("Third document.")

        return dir_path

    def test_load_text(self, sample_file):
        """Test loading raw text from file."""
        text = load_text(sample_file)

        assert "sample text file" in text
        assert "multiple lines" in text

    def test_load_text_with_encoding_fallback(self, tmp_path):
        """Test encoding fallback for files with special characters."""
        file_path = tmp_path / "special.txt"
        # Write with UTF-8
        file_path.write_text("Café résumé", encoding="utf-8")

        text = load_text(file_path)
        assert "Café" in text

    def test_load_file(self, sample_file):
        """Test loading file into Document object."""
        doc = load_file(sample_file)

        assert isinstance(doc, Document)
        assert "sample text file" in doc.text
        assert doc.title == "sample"
        assert doc.metadata["filename"] == "sample.txt"
        assert "source_path" in doc.metadata

    def test_load_file_with_metadata(self, sample_file):
        """Test loading file with custom metadata."""
        metadata = {"title": "Custom Title", "author": "Test Author"}
        doc = load_file(sample_file, metadata=metadata)

        assert doc.title == "Custom Title"
        assert doc.author == "Test Author"
        assert "source_path" in doc.metadata  # Should be added automatically

    def test_load_directory(self, sample_directory):
        """Test loading all files from directory."""
        corpus = load_directory(sample_directory)

        assert isinstance(corpus, Corpus)
        assert len(corpus) == 3

        # Check documents are sorted by filename
        titles = [doc.title for doc in corpus]
        assert titles == ["doc1", "doc2", "doc3"]

    def test_load_directory_with_pattern(self, tmp_path):
        """Test loading directory with file pattern."""
        dir_path = tmp_path / "mixed"
        dir_path.mkdir()

        (dir_path / "doc1.txt").write_text("Text file 1")
        (dir_path / "doc2.txt").write_text("Text file 2")
        (dir_path / "data.md").write_text("Markdown file")

        # Load only .txt files
        corpus = load_directory(dir_path, pattern="*.txt")
        assert len(corpus) == 2

        # Load only .md files
        corpus = load_directory(dir_path, pattern="*.md")
        assert len(corpus) == 1

    def test_load_directory_recursive(self, tmp_path):
        """Test loading directory recursively."""
        # Create nested structure
        (tmp_path / "level1").mkdir()
        (tmp_path / "level1" / "level2").mkdir()

        (tmp_path / "doc0.txt").write_text("Root doc")
        (tmp_path / "level1" / "doc1.txt").write_text("Level 1 doc")
        (tmp_path / "level1" / "level2" / "doc2.txt").write_text("Level 2 doc")

        # Non-recursive (default)
        corpus = load_directory(tmp_path, pattern="*.txt", recursive=False)
        assert len(corpus) == 1  # Only root doc

        # Recursive
        corpus = load_directory(tmp_path, pattern="*.txt", recursive=True)
        assert len(corpus) == 3  # All docs

    def test_load_directory_not_found(self, tmp_path):
        """Test loading non-existent directory."""
        with pytest.raises(FileNotFoundError):
            load_directory(tmp_path / "nonexistent")

    def test_load_directory_not_a_directory(self, sample_file):
        """Test loading file as directory."""
        with pytest.raises(NotADirectoryError):
            load_directory(sample_file)
