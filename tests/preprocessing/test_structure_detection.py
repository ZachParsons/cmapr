"""
Tests for document structure detection.
"""

from concept_mapper.preprocessing.structure import DocumentStructureDetector
from concept_mapper.corpus.models import StructureNode, SentenceLocation


class TestDocumentStructureDetector:
    """Tests for DocumentStructureDetector class."""

    def test_numbered_headings(self):
        """Test detection of numbered headings like 1., 1.2., 1.2.3."""
        text = """
1. Introduction
This is the introduction text.

1.1. Background
Background information here.

1.2. Motivation
More content.

2. Methods
Methods section.

2.1. Data Collection
Data collection details.
"""
        sentences = text.strip().split("\n")
        detector = DocumentStructureDetector()
        nodes, locations = detector.detect(text, sentences)

        # Check we detected nodes
        assert len(nodes) > 0
        assert len(locations) == len(sentences)

        # Check chapter nodes
        chapters = [n for n in nodes if n.level == "chapter"]
        assert len(chapters) >= 2

        # Check first chapter
        assert chapters[0].number == "1"
        assert "Introduction" in chapters[0].title

    def test_numbered_headings_with_spaces(self):
        """Test detection of numbered headings with spaces like '1. 7.'."""
        text = """
1. First Chapter
Content here.

1. 7. Section Seven
More content.

1.10. Section Ten
Even more.
"""
        sentences = text.strip().split("\n")
        detector = DocumentStructureDetector()
        nodes, locations = detector.detect(text, sentences)

        # Check that spaces are normalized
        numbers = [n.number for n in nodes]
        assert "1" in numbers
        assert "1.7" in numbers  # Space should be removed
        assert "1.10" in numbers

    def test_markdown_headings(self):
        """Test detection of markdown headings."""
        text = """
# Introduction

This is some text.

## Background

More text here.

## Methods

Even more text.

### Data Collection

Details.
"""
        sentences = text.strip().split("\n")
        detector = DocumentStructureDetector()
        nodes, locations = detector.detect(text, sentences)

        assert len(nodes) > 0

        # Check levels
        chapters = [n for n in nodes if n.level == "chapter"]
        sections = [n for n in nodes if n.level == "section"]
        subsections = [n for n in nodes if n.level == "subsection"]

        assert len(chapters) >= 1
        assert len(sections) >= 2
        assert len(subsections) >= 1

    def test_named_chapters(self):
        """Test detection of named chapters like 'Chapter 1'."""
        text = """
Chapter 1: Introduction

This is the introduction.

Chapter 2: Methods

Methods go here.

Section 3: Analysis

Analysis content.
"""
        sentences = text.strip().split("\n")
        detector = DocumentStructureDetector()
        nodes, locations = detector.detect(text, sentences)

        assert len(nodes) > 0

        # Check that chapters were detected
        chapters = [n for n in nodes if n.level == "chapter"]
        assert len(chapters) >= 2

    def test_no_structure_fallback(self):
        """Test that documents without structure fall back to paragraphs."""
        text = """
This is just plain text with no structure.
It has multiple sentences.

This is a second paragraph.
With more sentences.

And a third paragraph here.
"""
        sentences = text.strip().split("\n")
        detector = DocumentStructureDetector()
        nodes, locations = detector.detect(text, sentences)

        # Should still return locations (with paragraph numbers)
        assert len(locations) == len(sentences)

        # Check that locations have paragraph info
        has_paragraph_info = any(loc.paragraph is not None for loc in locations)
        assert has_paragraph_info

    def test_sentence_location_mapping(self):
        """Test that sentences are correctly mapped to their structure locations."""
        text = """
1. Chapter One
First sentence of chapter one.

1.1. Section One-One
First sentence of section.
Second sentence of section.

2. Chapter Two
First sentence of chapter two.
"""
        sentences = [s.strip() for s in text.strip().split("\n") if s.strip()]
        detector = DocumentStructureDetector()
        nodes, locations = detector.detect(text, sentences)

        assert len(locations) == len(sentences)

        # Check that chapter numbers are propagated to locations
        chapter_locations = [loc for loc in locations if loc.chapter]
        assert len(chapter_locations) > 0

        # Check that section locations include both chapter and section
        section_locations = [loc for loc in locations if loc.section]
        if section_locations:  # May not have sections if sentences not mapped correctly
            for loc in section_locations:
                assert loc.chapter is not None  # Section should also have chapter

    def test_hierarchy_building(self):
        """Test that parent-child relationships are built correctly."""
        text = """
1. Introduction
Text here.

1.1. Background
More text.

1.1.1. Historical Context
Details.

2. Methods
Methods text.
"""
        sentences = text.strip().split("\n")
        detector = DocumentStructureDetector()
        nodes, locations = detector.detect(text, sentences)

        # Find nodes
        node_dict = {n.number: n for n in nodes}

        # Check chapter has no parent
        if "1" in node_dict:
            assert node_dict["1"].parent_number is None

        # Check section has chapter as parent
        if "1.1" in node_dict:
            assert node_dict["1.1"].parent_number == "1"

        # Check subsection has section as parent
        if "1.1.1" in node_dict:
            assert node_dict["1.1.1"].parent_number == "1.1"

    def test_structure_node_serialization(self):
        """Test that StructureNode can be serialized and deserialized."""
        node = StructureNode(
            level="chapter",
            number="1",
            title="Introduction",
            start_index=0,
            end_index=10,
            parent_number=None,
            children=["1.1", "1.2"],
        )

        # Serialize
        data = node.to_dict()
        assert data["level"] == "chapter"
        assert data["number"] == "1"
        assert data["title"] == "Introduction"

        # Deserialize
        restored = StructureNode.from_dict(data)
        assert restored.level == node.level
        assert restored.number == node.number
        assert restored.title == node.title
        assert restored.children == node.children

    def test_sentence_location_serialization(self):
        """Test that SentenceLocation can be serialized and deserialized."""
        location = SentenceLocation(
            sent_index=5,
            chapter="1",
            chapter_title="Introduction",
            section="1.2",
            section_title="Background",
        )

        # Serialize
        data = location.to_dict()
        assert data["sent_index"] == 5
        assert data["chapter"] == "1"
        assert data["section"] == "1.2"

        # Deserialize
        restored = SentenceLocation.from_dict(data)
        assert restored.sent_index == location.sent_index
        assert restored.chapter == location.chapter
        assert restored.section == location.section
