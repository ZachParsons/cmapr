"""
Data models for term list management.

Provides TermEntry for individual terms and TermList for managing collections
of curated philosophical terms.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from pathlib import Path
import json
from concept_mapper.validation import validate_term_list


@dataclass
class TermEntry:
    """
    A single curated philosophical term with metadata.

    Attributes:
        term: The term as it appears in text (e.g., "abstraction")
        lemma: Lemmatized form (e.g., "reify")
        pos: Part of speech (e.g., "NN", "VB")
        definition: Human-provided definition or explanation
        notes: Additional scholarly notes or context
        examples: List of example sentences from corpus
        metadata: Additional custom metadata
    """

    term: str
    lemma: Optional[str] = None
    pos: Optional[str] = None
    definition: Optional[str] = None
    notes: Optional[str] = None
    examples: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TermEntry":
        """Create TermEntry from dictionary."""
        return cls(**data)

    def __str__(self) -> str:
        """String representation showing term and definition."""
        if self.definition:
            return f"{self.term}: {self.definition}"
        return self.term

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"TermEntry(term='{self.term}', lemma='{self.lemma}', pos='{self.pos}')"


class TermList:
    """
    A curated collection of philosophical terms.

    Manages a glossary of key concepts with lookup, CRUD operations,
    and persistence.

    Example:
        >>> terms = TermList()
        >>> terms.add(TermEntry(term="abstraction", definition="..."))
        >>> entry = terms.get("abstraction")
        >>> terms.save("output/terms.json")
    """

    def __init__(self, name: Optional[str] = None, description: Optional[str] = None):
        """
        Initialize term list.

        Args:
            name: Optional name for this term list
            description: Optional description of the term list's purpose
        """
        self.name = name or "Untitled Term List"
        self.description = description or ""
        self._terms: Dict[str, TermEntry] = {}

    def add(self, entry: TermEntry) -> None:
        """
        Add a term entry to the list.

        Args:
            entry: TermEntry to add

        Raises:
            ValueError: If term already exists
        """
        if entry.term in self._terms:
            raise ValueError(f"Term '{entry.term}' already exists in list")
        self._terms[entry.term] = entry

    def remove(self, term: str) -> None:
        """
        Remove a term from the list.

        Args:
            term: Term to remove

        Raises:
            KeyError: If term not found
        """
        if term not in self._terms:
            raise KeyError(f"Term '{term}' not found in list")
        del self._terms[term]

    def update(self, term: str, **kwargs) -> None:
        """
        Update fields of an existing term.

        Args:
            term: Term to update
            **kwargs: Fields to update (definition, notes, examples, etc.)

        Raises:
            KeyError: If term not found
        """
        if term not in self._terms:
            raise KeyError(f"Term '{term}' not found in list")

        entry = self._terms[term]
        for key, value in kwargs.items():
            if hasattr(entry, key):
                setattr(entry, key, value)
            else:
                raise ValueError(f"Invalid field: {key}")

    def get(self, term: str) -> Optional[TermEntry]:
        """
        Get a term entry.

        Args:
            term: Term to retrieve

        Returns:
            TermEntry if found, None otherwise
        """
        return self._terms.get(term)

    def contains(self, term: str) -> bool:
        """
        Check if term exists in list.

        Args:
            term: Term to check

        Returns:
            True if term exists, False otherwise
        """
        return term in self._terms

    def list_terms(self) -> List[TermEntry]:
        """
        Get all terms in the list.

        Returns:
            List of all TermEntry objects, sorted by term
        """
        return sorted(self._terms.values(), key=lambda e: e.term)

    def list_term_names(self) -> List[str]:
        """
        Get all term names.

        Returns:
            List of term strings, sorted alphabetically
        """
        return sorted(self._terms.keys())

    def __len__(self) -> int:
        """Return number of terms in list."""
        return len(self._terms)

    def __contains__(self, term: str) -> bool:
        """Support 'in' operator."""
        return term in self._terms

    def __iter__(self):
        """Iterate over term entries."""
        return iter(self._terms.values())

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns:
            Dictionary with name, description, and terms
        """
        return {
            "name": self.name,
            "description": self.description,
            "terms": [entry.to_dict() for entry in self.list_terms()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TermList":
        """
        Create TermList from dictionary.

        Args:
            data: Dictionary with name, description, and terms

        Returns:
            New TermList instance
        """
        term_list = cls(name=data.get("name"), description=data.get("description"))

        for term_data in data.get("terms", []):
            entry = TermEntry.from_dict(term_data)
            term_list.add(entry)

        return term_list

    def save(self, path: Path) -> None:
        """
        Save term list to JSON file.

        Args:
            path: Path to save to
        """
        # Validate term list is not empty
        data = self.to_dict()
        validate_term_list(data)

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path: Path) -> "TermList":
        """
        Load term list from JSON file.

        Args:
            path: Path to load from

        Returns:
            Loaded TermList instance

        Raises:
            FileNotFoundError: If file doesn't exist
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Term list file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def merge(self, other: "TermList", overwrite: bool = False) -> "TermList":
        """
        Merge another term list into a new list.

        Args:
            other: Another TermList to merge
            overwrite: If True, other's entries overwrite conflicts

        Returns:
            New TermList with merged entries
        """
        merged = TermList(
            name=f"{self.name} + {other.name}", description="Merged from two lists"
        )

        # Add terms from self
        for entry in self._terms.values():
            merged.add(entry)

        # Add terms from other
        for entry in other._terms.values():
            if entry.term in merged:
                if overwrite:
                    merged.remove(entry.term)
                    merged.add(entry)
                # else: skip conflicting terms
            else:
                merged.add(entry)

        return merged

    def __str__(self) -> str:
        """String representation."""
        return f"TermList('{self.name}', {len(self)} terms)"

    def __repr__(self) -> str:
        """Detailed representation."""
        return f"TermList(name='{self.name}', terms={len(self)})"
