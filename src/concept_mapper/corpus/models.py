"""
Core data structures for corpus representation.

Defines the data models used throughout the preprocessing pipeline:
- Document: Raw text with metadata
- Corpus: Collection of documents
- ProcessedDocument: Fully preprocessed document with linguistic annotations
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class Document:
    """
    A single text document with metadata.

    Represents raw, unprocessed text along with optional metadata
    like title, author, date, and source path.
    """

    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> Optional[str]:
        """Get document title from metadata."""
        return self.metadata.get("title")

    @property
    def author(self) -> Optional[str]:
        """Get document author from metadata."""
        return self.metadata.get("author")

    @property
    def date(self) -> Optional[str]:
        """Get document date from metadata."""
        return self.metadata.get("date")

    @property
    def source_path(self) -> Optional[Path]:
        """Get source file path from metadata."""
        path = self.metadata.get("source_path")
        return Path(path) if path else None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Document":
        """Create Document from dictionary."""
        return cls(text=data["text"], metadata=data.get("metadata", {}))


@dataclass
class ProcessedDocument:
    """
    A document with full linguistic preprocessing.

    Contains the raw text along with all derived linguistic annotations:
    - Sentences (segmented)
    - Tokens (word-level)
    - POS tags (part-of-speech)
    - Lemmas (base forms)
    """

    raw_text: str
    sentences: List[str] = field(default_factory=list)
    tokens: List[str] = field(default_factory=list)
    pos_tags: List[tuple[str, str]] = field(default_factory=list)
    lemmas: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def title(self) -> Optional[str]:
        """Get document title from metadata."""
        return self.metadata.get("title")

    @property
    def num_sentences(self) -> int:
        """Get number of sentences."""
        return len(self.sentences)

    @property
    def num_tokens(self) -> int:
        """Get number of tokens."""
        return len(self.tokens)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "raw_text": self.raw_text,
            "sentences": self.sentences,
            "tokens": self.tokens,
            "pos_tags": self.pos_tags,
            "lemmas": self.lemmas,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ProcessedDocument":
        """Create ProcessedDocument from dictionary."""
        return cls(
            raw_text=data["raw_text"],
            sentences=data.get("sentences", []),
            tokens=data.get("tokens", []),
            pos_tags=data.get("pos_tags", []),
            lemmas=data.get("lemmas", []),
            metadata=data.get("metadata", {}),
        )


class Corpus:
    """
    A collection of documents.

    Manages multiple documents and provides convenience methods
    for corpus-level operations.
    """

    def __init__(self, documents: Optional[List[Document]] = None):
        """
        Initialize corpus.

        Args:
            documents: List of Document objects (optional)
        """
        self.documents: List[Document] = documents or []

    def add_document(self, document: Document) -> None:
        """Add a document to the corpus."""
        self.documents.append(document)

    def __len__(self) -> int:
        """Get number of documents in corpus."""
        return len(self.documents)

    def __getitem__(self, index: int) -> Document:
        """Get document by index."""
        return self.documents[index]

    def __iter__(self):
        """Iterate over documents."""
        return iter(self.documents)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "documents": [doc.to_dict() for doc in self.documents],
            "num_documents": len(self.documents),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Corpus":
        """Create Corpus from dictionary."""
        documents = [Document.from_dict(doc_data) for doc_data in data["documents"]]
        return cls(documents=documents)
