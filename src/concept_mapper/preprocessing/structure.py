"""
Document structure detection module.

Automatically identifies hierarchical organization in documents (chapters, sections, subsections)
to preserve semantic context during analysis.
"""

import re
from typing import Dict, List, Optional, Tuple

from concept_mapper.corpus.models import SentenceLocation, StructureNode


class DocumentStructureDetector:
    """
    Detects and extracts hierarchical structure from documents.

    Supports multiple detection patterns in priority order:
    1. Numbered headings: 1., 1.1., 1.1.1.
    2. Named chapters: "Chapter 1", "Part I", "Section 3"
    3. Markdown headings: #, ##, ###
    4. All-caps headings: INTRODUCTION, CONCLUSION
    5. Paragraph boundaries: Blank lines (fallback)
    """

    # Pattern priority order
    NUMBERED_HEADING = re.compile(
        r"^(\d+(?:\.\s?\d+)*)\.\s+([A-Za-z].{3,})$", re.MULTILINE
    )  # 1. Title, 1.2. Title, 1. 7. Title (allow spaces in numbers, require text after)
    NAMED_CHAPTER = re.compile(
        r"^(?:(Chapter|Part|Section)\s+(\d+|[IVXLCDM]+))(?:\s*[:\-\.]?\s*(.+))?$",
        re.IGNORECASE | re.MULTILINE,
    )
    MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    ALLCAPS_HEADING = re.compile(r"^([A-Z][A-Z\s]{3,})$", re.MULTILINE)

    def detect(
        self, text: str, sentences: List[str]
    ) -> Tuple[List[StructureNode], List[SentenceLocation]]:
        """
        Detect document structure and map sentences to locations.

        Args:
            text: Full document text
            sentences: List of sentence strings

        Returns:
            Tuple of (structure_nodes, sentence_locations)
        """
        # Try detection methods in priority order
        nodes = self._detect_numbered_headings(text, sentences)
        if nodes:
            return self._build_structure(nodes, sentences)

        nodes = self._detect_named_chapters(text, sentences)
        if nodes:
            return self._build_structure(nodes, sentences)

        nodes = self._detect_markdown_headings(text, sentences)
        if nodes:
            return self._build_structure(nodes, sentences)

        nodes = self._detect_allcaps_headings(text, sentences)
        if nodes:
            return self._build_structure(nodes, sentences)

        # Fallback: paragraph boundaries
        return self._detect_paragraphs(text, sentences)

    def _detect_numbered_headings(
        self, text: str, sentences: List[str]
    ) -> Optional[List[Dict]]:
        """Detect numbered headings like '1.', '1.2.', '1.2.3.'."""
        matches = list(self.NUMBERED_HEADING.finditer(text))
        if not matches:
            return None

        nodes = []
        for match in matches:
            number = match.group(1)
            title = match.group(2).strip()
            position = match.start()

            # Normalize number: remove spaces (e.g., "1. 7" -> "1.7")
            number = number.replace(" ", "")

            # Determine level from number of dots
            level_depth = number.count(".")
            if level_depth == 0:
                level = "chapter"
            elif level_depth == 1:
                level = "section"
            else:
                level = "subsection"

            nodes.append(
                {
                    "level": level,
                    "number": number,
                    "title": title,
                    "position": position,
                }
            )

        return nodes if len(nodes) >= 2 else None  # Require at least 2 headings

    def _detect_named_chapters(
        self, text: str, sentences: List[str]
    ) -> Optional[List[Dict]]:
        """Detect named chapters like 'Chapter 1', 'Part I', 'Section 3'."""
        matches = list(self.NAMED_CHAPTER.finditer(text))
        if not matches:
            return None

        nodes = []
        for match in matches:
            prefix = match.group(1)  # "Chapter", "Part", "Section"
            number = match.group(2)  # "1", "I", etc.
            title = match.group(3) or ""  # Optional title after number

            # Convert roman numerals to arabic if needed
            if number.upper() in [
                "I",
                "II",
                "III",
                "IV",
                "V",
                "VI",
                "VII",
                "VIII",
                "IX",
                "X",
            ]:
                number = str(self._roman_to_int(number.upper()))

            position = match.start()

            # Determine level from prefix
            if prefix.lower() == "part":
                level = "chapter"
            elif prefix.lower() == "chapter":
                level = "chapter"
            else:
                level = "section"

            nodes.append(
                {
                    "level": level,
                    "number": number,
                    "title": title.strip() if title else f"{prefix} {number}",
                    "position": position,
                }
            )

        return nodes if len(nodes) >= 2 else None

    def _detect_markdown_headings(
        self, text: str, sentences: List[str]
    ) -> Optional[List[Dict]]:
        """Detect markdown headings like '#', '##', '###'."""
        matches = list(self.MARKDOWN_HEADING.finditer(text))
        if not matches:
            return None

        nodes = []
        chapter_count = 0
        section_count = 0
        subsection_count = 0

        for match in matches:
            hashes = match.group(1)
            title = match.group(2).strip()
            position = match.start()
            depth = len(hashes)

            # Map hash count to level
            if depth == 1:
                level = "chapter"
                chapter_count += 1
                section_count = 0
                subsection_count = 0
                number = str(chapter_count)
            elif depth == 2:
                level = "section"
                section_count += 1
                subsection_count = 0
                number = (
                    f"{chapter_count}.{section_count}"
                    if chapter_count > 0
                    else str(section_count)
                )
            else:
                level = "subsection"
                subsection_count += 1
                number = (
                    f"{chapter_count}.{section_count}.{subsection_count}"
                    if chapter_count > 0
                    else str(subsection_count)
                )

            nodes.append(
                {
                    "level": level,
                    "number": number,
                    "title": title,
                    "position": position,
                }
            )

        return nodes if len(nodes) >= 2 else None

    def _detect_allcaps_headings(
        self, text: str, sentences: List[str]
    ) -> Optional[List[Dict]]:
        """Detect all-caps headings like 'INTRODUCTION', 'CONCLUSION'."""
        matches = list(self.ALLCAPS_HEADING.finditer(text))
        if not matches:
            return None

        # Filter out likely false positives (acronyms, short strings)
        valid_matches = []
        for match in matches:
            heading = match.group(1).strip()
            # Require at least 8 characters and not just acronym-like
            if len(heading) >= 8 and " " in heading:
                valid_matches.append(match)

        if not valid_matches:
            return None

        nodes = []
        for idx, match in enumerate(valid_matches, 1):
            title = match.group(1).strip()
            position = match.start()

            nodes.append(
                {
                    "level": "chapter",
                    "number": str(idx),
                    "title": title.title(),  # Convert to title case
                    "position": position,
                }
            )

        return nodes if len(nodes) >= 2 else None

    def _detect_paragraphs(
        self, text: str, sentences: List[str]
    ) -> Tuple[List[StructureNode], List[SentenceLocation]]:
        """Fallback: detect paragraph boundaries using blank lines."""
        # Split text into paragraphs (separated by blank lines)
        paragraphs = re.split(r"\n\s*\n", text)

        # Map sentences to paragraphs
        sentence_to_para: Dict[int, int] = {}

        for para_idx, para in enumerate(paragraphs):
            para_text = para.strip()
            if not para_text:
                continue

            # Find sentences that belong to this paragraph
            for sent_idx, sentence in enumerate(sentences):
                if sentence.strip() in para_text:
                    sentence_to_para[sent_idx] = para_idx + 1

        # Create minimal structure nodes (one per paragraph)
        nodes = []
        locations = []

        for sent_idx in range(len(sentences)):
            para_num = sentence_to_para.get(sent_idx, 1)
            locations.append(
                SentenceLocation(
                    sent_index=sent_idx,
                    paragraph=para_num,
                )
            )

        return nodes, locations

    def _build_structure(
        self, node_dicts: List[Dict], sentences: List[str]
    ) -> Tuple[List[StructureNode], List[SentenceLocation]]:
        """
        Build hierarchical structure and assign sentences to nodes.

        Args:
            node_dicts: List of detected heading dictionaries
            sentences: List of sentence strings

        Returns:
            Tuple of (structure_nodes, sentence_locations)
        """
        # Sort nodes by position in text
        node_dicts = sorted(node_dicts, key=lambda x: x["position"])

        # Map sentences to nodes by finding which heading precedes each sentence
        sentence_to_node: Dict[int, int] = {}
        text_positions = self._get_sentence_positions(sentences)

        for sent_idx, sent_pos in enumerate(text_positions):
            # Find the last heading before this sentence
            for node_idx, node in enumerate(node_dicts):
                if node["position"] <= sent_pos:
                    sentence_to_node[sent_idx] = node_idx

        # Build structure nodes with start/end indices
        structure_nodes = []
        for idx, node_dict in enumerate(node_dicts):
            # Find sentences in this section
            start_idx = min(
                (
                    sent_idx
                    for sent_idx, node_idx in sentence_to_node.items()
                    if node_idx == idx
                ),
                default=0,
            )
            # End is the start of next section or end of document
            if idx + 1 < len(node_dicts):
                end_idx = min(
                    (
                        sent_idx
                        for sent_idx, node_idx in sentence_to_node.items()
                        if node_idx == idx + 1
                    ),
                    default=len(sentences),
                )
            else:
                end_idx = len(sentences)

            structure_nodes.append(
                StructureNode(
                    level=node_dict["level"],
                    number=node_dict["number"],
                    title=node_dict["title"],
                    start_index=start_idx,
                    end_index=end_idx,
                )
            )

        # Build parent-child relationships
        self._build_hierarchy(structure_nodes)

        # Create sentence locations
        sentence_locations = self._create_sentence_locations(
            structure_nodes, len(sentences)
        )

        return structure_nodes, sentence_locations

    def _get_sentence_positions(self, sentences: List[str]) -> List[int]:
        """Get approximate positions of sentences in original text."""
        # This is approximate - in real usage we'd track exact positions during tokenization
        positions = []
        current_pos = 0
        for sentence in sentences:
            positions.append(current_pos)
            current_pos += len(sentence) + 1  # +1 for space
        return positions

    def _build_hierarchy(self, nodes: List[StructureNode]) -> None:
        """Build parent-child relationships between nodes."""
        for i, node in enumerate(nodes):
            # Find parent by looking backwards for a less specific number
            for j in range(i - 1, -1, -1):
                potential_parent = nodes[j]
                if self._is_parent(potential_parent.number, node.number):
                    node.parent_number = potential_parent.number
                    potential_parent.children.append(node.number)
                    break

    def _is_parent(self, parent_num: str, child_num: str) -> bool:
        """Check if parent_num is a parent of child_num in numbering hierarchy."""
        # For numbered headings like "1" and "1.2"
        if "." in child_num:
            parent_part = child_num.rsplit(".", 1)[0]
            return parent_num == parent_part
        return False

    def _create_sentence_locations(
        self, nodes: List[StructureNode], num_sentences: int
    ) -> List[SentenceLocation]:
        """Create flattened location lookup for all sentences."""
        locations = []

        # Build a lookup for node hierarchy
        node_dict = {n.number: n for n in nodes}

        for sent_idx in range(num_sentences):
            # Find all containing nodes (chapter, section, subsection)
            chapter = None
            chapter_title = None
            section = None
            section_title = None
            subsection = None
            subsection_title = None

            # Find the most specific node containing this sentence
            containing_nodes = [
                node for node in nodes if node.start_index <= sent_idx < node.end_index
            ]

            # Process from most specific to least specific
            for node in containing_nodes:
                if node.level == "subsection":
                    subsection = node.number
                    subsection_title = node.title
                    # Get parent section and chapter
                    if node.parent_number and node.parent_number in node_dict:
                        parent = node_dict[node.parent_number]
                        if parent.level == "section":
                            section = parent.number
                            section_title = parent.title
                            if (
                                parent.parent_number
                                and parent.parent_number in node_dict
                            ):
                                grandparent = node_dict[parent.parent_number]
                                chapter = grandparent.number
                                chapter_title = grandparent.title
                elif node.level == "section":
                    section = node.number
                    section_title = node.title
                    # Get parent chapter
                    if node.parent_number and node.parent_number in node_dict:
                        parent = node_dict[node.parent_number]
                        if parent.level == "chapter":
                            chapter = parent.number
                            chapter_title = parent.title
                elif node.level == "chapter":
                    chapter = node.number
                    chapter_title = node.title

            locations.append(
                SentenceLocation(
                    sent_index=sent_idx,
                    chapter=chapter,
                    chapter_title=chapter_title,
                    section=section,
                    section_title=section_title,
                    subsection=subsection,
                    subsection_title=subsection_title,
                )
            )

        return locations

    def _roman_to_int(self, s: str) -> int:
        """Convert Roman numeral to integer."""
        roman_values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
        result = 0
        prev_value = 0

        for char in reversed(s):
            value = roman_values.get(char, 0)
            if value < prev_value:
                result -= value
            else:
                result += value
            prev_value = value

        return result
