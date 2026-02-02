"""
Term list manager with bulk operations and import/export.

Provides high-level operations for managing term lists including
importing from various formats and exporting for sharing.
"""

from pathlib import Path
from typing import Optional, List
from .models import TermList, TermEntry
import csv
import json
from concept_mapper.validation import validate_term_list, validate_csv_data


class TermManager:
    """
    Manager for bulk term list operations.

    Handles importing from text files, exporting to various formats,
    and managing term list workflows.

    Example:
        >>> manager = TermManager(term_list)
        >>> manager.import_from_txt("terms.txt")
        >>> manager.export_to_csv("output/terms.csv")
    """

    def __init__(self, term_list: Optional[TermList] = None):
        """
        Initialize manager.

        Args:
            term_list: Optional existing TermList to manage
        """
        self.term_list = term_list or TermList()

    def import_from_txt(
        self, path: Path, delimiter: str = "\n", encoding: str = "utf-8"
    ) -> int:
        """
        Import terms from plain text file.

        File format: One term per line (or custom delimiter).

        Args:
            path: Path to text file
            delimiter: Line delimiter (default: newline)
            encoding: File encoding (default: utf-8)

        Returns:
            Number of terms imported

        Example:
            File contents:
            ```
            intentionality
            totality
            commodification
            ```
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        with open(path, "r", encoding=encoding) as f:
            content = f.read()

        terms = [line.strip() for line in content.split(delimiter) if line.strip()]

        count = 0
        for term in terms:
            # Skip if already exists
            if term not in self.term_list:
                entry = TermEntry(term=term)
                self.term_list.add(entry)
                count += 1

        return count

    def export_to_txt(
        self, path: Path, delimiter: str = "\n", encoding: str = "utf-8"
    ) -> int:
        """
        Export terms to plain text file.

        Args:
            path: Path to output file
            delimiter: Line delimiter (default: newline)
            encoding: File encoding (default: utf-8)

        Returns:
            Number of terms exported
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        terms = self.term_list.list_term_names()

        # Validate terms list is not empty
        validate_term_list(terms)

        with open(path, "w", encoding=encoding) as f:
            f.write(delimiter.join(terms))
            if delimiter == "\n":
                f.write("\n")  # Trailing newline for text files

        return len(terms)

    def export_to_csv(
        self,
        path: Path,
        fields: Optional[List[str]] = None,
        encoding: str = "utf-8",
    ) -> int:
        """
        Export terms to CSV file.

        Args:
            path: Path to output CSV
            fields: Optional list of fields to include
                    (default: all fields except metadata)
            encoding: File encoding (default: utf-8)

        Returns:
            Number of terms exported
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Default fields
        if fields is None:
            fields = ["term", "lemma", "pos", "definition", "notes"]

        terms = self.term_list.list_terms()

        # Validate terms list is not empty
        validate_csv_data(terms, file_type="terms CSV")

        with open(path, "w", encoding=encoding, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()

            for entry in terms:
                row = {}
                for field in fields:
                    value = getattr(entry, field, None)
                    # Handle list fields (examples)
                    if isinstance(value, list):
                        row[field] = "; ".join(value)
                    else:
                        row[field] = value or ""
                writer.writerow(row)

        return len(terms)

    def import_from_csv(
        self,
        path: Path,
        term_column: str = "term",
        encoding: str = "utf-8",
    ) -> int:
        """
        Import terms from CSV file.

        CSV must have at least a 'term' column. Additional columns
        map to TermEntry fields.

        Args:
            path: Path to CSV file
            term_column: Name of column containing terms (default: "term")
            encoding: File encoding (default: utf-8)

        Returns:
            Number of terms imported
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        count = 0
        with open(path, "r", encoding=encoding) as f:
            reader = csv.DictReader(f)

            for row in reader:
                term = row.get(term_column)
                if not term or term in self.term_list:
                    continue

                # Build entry from CSV columns
                entry_data = {"term": term}

                # Map CSV columns to TermEntry fields
                field_mapping = {
                    "lemma": "lemma",
                    "pos": "pos",
                    "definition": "definition",
                    "notes": "notes",
                    "examples": "examples",
                }

                for csv_col, entry_field in field_mapping.items():
                    if csv_col in row and row[csv_col]:
                        value = row[csv_col]
                        # Handle semicolon-separated lists
                        if entry_field == "examples" and value:
                            entry_data[entry_field] = [
                                ex.strip() for ex in value.split(";") if ex.strip()
                            ]
                        else:
                            entry_data[entry_field] = value

                entry = TermEntry(**entry_data)
                self.term_list.add(entry)
                count += 1

        return count

    def merge_from_file(
        self, path: Path, overwrite: bool = False, format: str = "json"
    ) -> int:
        """
        Merge terms from another term list file.

        Args:
            path: Path to term list file (JSON, TXT, or CSV)
            overwrite: If True, imported entries overwrite conflicts
            format: File format ('json', 'txt', 'csv')

        Returns:
            Number of new terms added
        """
        if format == "json":
            other = TermList.load(path)
            self.term_list = self.term_list.merge(other, overwrite=overwrite)
            return len(other)
        elif format == "txt":
            return self.import_from_txt(path)
        elif format == "csv":
            return self.import_from_csv(path)
        else:
            raise ValueError(f"Unsupported format: {format}")

    def clear(self) -> None:
        """Clear all terms from the list."""
        self.term_list = TermList(
            name=self.term_list.name, description=self.term_list.description
        )

    def filter_by_pos(self, pos_tags: List[str]) -> TermList:
        """
        Create new TermList containing only terms with specified POS tags.

        Args:
            pos_tags: List of POS tags to include (e.g., ['NN', 'NNS'])

        Returns:
            New TermList with filtered terms
        """
        filtered = TermList(
            name=f"{self.term_list.name} (filtered)",
            description=f"Filtered by POS: {', '.join(pos_tags)}",
        )

        for entry in self.term_list:
            if entry.pos and entry.pos in pos_tags:
                filtered.add(entry)

        return filtered

    def get_statistics(self) -> dict:
        """
        Get statistics about the term list.

        Returns:
            Dictionary with counts and summaries
        """
        terms = self.term_list.list_terms()

        stats = {
            "total_terms": len(terms),
            "terms_with_definitions": sum(1 for t in terms if t.definition),
            "terms_with_examples": sum(1 for t in terms if t.examples),
            "terms_with_pos": sum(1 for t in terms if t.pos),
            "terms_with_lemma": sum(1 for t in terms if t.lemma),
        }

        # POS distribution
        pos_counts = {}
        for term in terms:
            if term.pos:
                pos_counts[term.pos] = pos_counts.get(term.pos, 0) + 1
        stats["pos_distribution"] = pos_counts

        return stats

    def export_to_json(
        self,
        path: Path,
        encoding: str = "utf-8",
        indent: int = 2,
    ) -> int:
        """
        Export terms to JSON file.

        Args:
            path: Path to output JSON
            encoding: File encoding (default: utf-8)
            indent: JSON indentation (default: 2)

        Returns:
            Number of terms exported
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        terms = self.term_list.list_terms()

        # Convert to list of dicts
        data = [
            {
                "term": entry.term,
                "lemma": entry.lemma,
                "pos": entry.pos,
                "definition": entry.definition,
                "notes": entry.notes,
                "examples": entry.examples,
                "metadata": entry.metadata,
            }
            for entry in terms
        ]

        # Validate terms list is not empty
        validate_term_list(data)

        with open(path, "w", encoding=encoding) as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)

        return len(terms)

    def import_from_json(
        self,
        path: Path,
        encoding: str = "utf-8",
    ) -> int:
        """
        Import terms from JSON file.

        Args:
            path: Path to JSON file
            encoding: File encoding (default: utf-8)

        Returns:
            Number of terms imported
        """
        path = Path(path)

        with open(path, "r", encoding=encoding) as f:
            data = json.load(f)

        # Convert to TermEntry objects
        from .models import TermEntry

        count = 0
        for item in data:
            entry = TermEntry.from_dict(item)
            self.term_list.add(entry)
            count += 1

        return count
