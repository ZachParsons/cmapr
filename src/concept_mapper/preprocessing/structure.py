"""
Document structure detection module.

Automatically identifies hierarchical organization in documents (chapters, sections, subsections)
to preserve semantic context during analysis.
"""

import re
from pathlib import Path
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
        self, text: str, sentences: List[str], toc_file: Optional[Path] = None
    ) -> Tuple[List[StructureNode], List[SentenceLocation]]:
        """
        Detect document structure and map sentences to locations.

        Args:
            text: Full document text
            sentences: List of sentence strings
            toc_file: Optional path to table of contents file for guided detection

        Returns:
            Tuple of (structure_nodes, sentence_locations)
        """
        # Try TOC-based detection first (if provided)
        if toc_file:
            nodes = self._detect_from_toc(toc_file, text, sentences)
            if nodes:
                return self._build_structure(nodes, sentences, text)

        # Try automatic detection methods in priority order
        nodes = self._detect_numbered_headings(text, sentences)
        if nodes:
            return self._build_structure(nodes, sentences, text)

        nodes = self._detect_named_chapters(text, sentences)
        if nodes:
            return self._build_structure(nodes, sentences, text)

        nodes = self._detect_markdown_headings(text, sentences)
        if nodes:
            return self._build_structure(nodes, sentences, text)

        nodes = self._detect_allcaps_headings(text, sentences)
        if nodes:
            return self._build_structure(nodes, sentences, text)

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

            # Determine level from number of dots (generic depth-based naming)
            level_depth = number.count(".")
            level = f"level_{level_depth}"

            nodes.append(
                {
                    "level": level,
                    "number": number,
                    "title": title,
                    "position": position,
                }
            )

        # Filter out table of contents / index entries
        nodes = self._filter_toc_entries(nodes, text)

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

            # Determine level from prefix (treat "Part" and "Chapter" as top level)
            if prefix.lower() in ("part", "chapter"):
                level = "level_0"
            else:
                level = "level_1"

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

            # Map hash count to level (generic depth-based)
            level_depth = depth - 1  # # = depth 0, ## = depth 1, ### = depth 2
            level = f"level_{level_depth}"

            if depth == 1:
                chapter_count += 1
                section_count = 0
                subsection_count = 0
                number = str(chapter_count)
            elif depth == 2:
                section_count += 1
                subsection_count = 0
                number = (
                    f"{chapter_count}.{section_count}"
                    if chapter_count > 0
                    else str(section_count)
                )
            else:
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
                    "level": "level_0",
                    "number": str(idx),
                    "title": title.title(),  # Convert to title case
                    "position": position,
                }
            )

        return nodes if len(nodes) >= 2 else None

    def _detect_from_toc(
        self, toc_file: Path, text: str, sentences: List[str]
    ) -> Optional[List[Dict]]:
        """
        Detect structure using a manually-created table of contents file.

        The TOC file should be in markdown format with hierarchy indicated by:
        - ## N. Title — page  (chapters)
        - - N.N. Title — page  (sections, with bullet)
        -   - N.N.N. Title — page  (subsections, with indent + bullet)

        Args:
            toc_file: Path to TOC file
            text: Source document text
            sentences: List of sentences from source

        Returns:
            List of node dictionaries with level, number, title, position
        """
        if not toc_file.exists():
            return None

        # Read and parse TOC file
        toc_content = toc_file.read_text(encoding="utf-8")
        toc_entries = self._parse_toc_markdown(toc_content)

        if not toc_entries:
            return None

        # Match TOC entries to positions in source text
        nodes = []
        for entry in toc_entries:
            position = self._find_heading_in_text(entry["title"], text)
            if position is not None:
                nodes.append(
                    {
                        "level": entry["level"],
                        "number": entry["number"],
                        "title": entry["title"],
                        "position": position,
                    }
                )

        # Ensure chapters appear before their sections
        nodes = self._ensure_chapter_positions(nodes, toc_entries)

        return nodes if len(nodes) >= 2 else None

    def _ensure_chapter_positions(
        self, nodes: List[Dict], toc_entries: List[Dict]
    ) -> List[Dict]:
        """
        Ensure chapter nodes appear before their first section.

        For academic texts where chapter titles may be found after their sections,
        or not found at all, this ensures chapters are positioned correctly.
        """
        chapter_entries = {
            entry["number"]: entry
            for entry in toc_entries
            if entry["level"] == "level_0"
        }

        # Find existing chapter nodes and their positions
        chapter_nodes = {
            node["number"]: node for node in nodes if node["level"] == "level_0"
        }

        for chapter_num, chapter_entry in chapter_entries.items():
            # Find first section of this chapter
            section_prefix = f"{chapter_num}."
            first_section = None
            first_section_pos = float("inf")

            for node in nodes:
                if node["number"].startswith(section_prefix) and "." in node["number"]:
                    if node["position"] < first_section_pos:
                        first_section = node
                        first_section_pos = node["position"]

            if first_section:
                if chapter_num in chapter_nodes:
                    # Chapter exists but may be in wrong position
                    chapter_node = chapter_nodes[chapter_num]
                    if chapter_node["position"] > first_section["position"]:
                        # Reposition chapter before first section
                        chapter_node["position"] = first_section["position"] - 1
                else:
                    # Chapter not found, add it before first section
                    nodes.append(
                        {
                            "level": "level_0",
                            "number": chapter_num,
                            "title": chapter_entry["title"],
                            "position": first_section["position"] - 1,
                        }
                    )

        return nodes

    def _parse_toc_markdown(self, toc_content: str) -> List[Dict]:
        """
        Parse markdown-formatted TOC into structured entries.

        Expected format:
        - ## N. Title — page  (chapter)
        - - N.N. Title — page  (section)
        -   - N.N.N. Title — page  (subsection)

        Returns:
            List of dicts with keys: level, number, title
        """
        entries = []
        lines = toc_content.split("\n")

        for line in lines:
            # Skip empty lines, horizontal rules, and header line
            if not line.strip() or line.strip() == "---" or line.startswith("# "):
                continue

            # Chapter: ## N. Title — page
            chapter_match = re.match(r"^##\s+(\d+)\.\s+(.+?)\s+—\s+\d+$", line)
            if chapter_match:
                number = chapter_match.group(1)
                title = chapter_match.group(2).strip()
                entries.append({"level": "level_0", "number": number, "title": title})
                continue

            # Chapter without number: ## Title — page (e.g., Introduction)
            chapter_no_num_match = re.match(r"^##\s+([A-Z][^—]+?)\s+—\s+\d+$", line)
            if chapter_no_num_match:
                title = chapter_no_num_match.group(1).strip()
                # Generate a number for unnumbered chapters
                entries.append({"level": "level_0", "number": "0", "title": title})
                continue

            # Unnumbered standalone heading (e.g., Introduction without ##)
            standalone_match = re.match(r"^([A-Z][^—\-]+?)\s+—\s+\d+$", line)
            if (
                standalone_match
                and not line.startswith("-")
                and not line.startswith(" ")
            ):
                title = standalone_match.group(1).strip()
                entries.append({"level": "level_0", "number": "0", "title": title})
                continue

            # Section: - N.N. Title — page (with bullet, no extra indent)
            section_match = re.match(r"^-\s+(\d+\.\d+)\.\s+(.+?)\s+—\s+\d+$", line)
            if section_match:
                number = section_match.group(1)
                title = section_match.group(2).strip()
                entries.append({"level": "level_1", "number": number, "title": title})
                continue

            # Subsection: - N.N.N. Title — page (with indent + bullet)
            subsection_match = re.match(
                r"^\s+-\s+(\d+\.\d+\.\d+)\.\s+(.+?)\s+—\s+\d+$", line
            )
            if subsection_match:
                number = subsection_match.group(1)
                title = subsection_match.group(2).strip()
                entries.append({"level": "level_2", "number": number, "title": title})
                continue

        return entries

    def _find_heading_in_text(self, title: str, text: str) -> Optional[int]:
        """
        Find the position of a heading title in the source text.

        Uses fuzzy matching to handle minor variations in punctuation,
        spacing, and case. Skips the first 10% of the document to avoid
        matching TOC entries.

        Args:
            title: The heading title to find
            text: The source document text

        Returns:
            Character position in text, or None if not found
        """
        # Normalize the search title
        title_normalized = self._normalize_heading(title)

        # Skip first 5% of document to avoid matching TOC entries
        toc_skip_threshold = int(len(text) * 0.05)

        # First try exact case-insensitive match
        text_lower = text.lower()
        title_lower = title.lower()

        # Pattern 1: Try to find as a standalone line (skip TOC region)
        pattern = re.compile(
            rf"^{re.escape(title_lower)}$", re.MULTILINE | re.IGNORECASE
        )
        for match in pattern.finditer(text):
            if match.start() >= toc_skip_threshold:
                return match.start()

        # Pattern 2: Try with section number prefix (e.g., "1.2. Title")
        # Extract number from title if present
        num_match = re.match(r"^(\d+(?:\.\d+)*)\.\s+(.+)$", title)
        if num_match:
            number = num_match.group(1)
            title_part = num_match.group(2)
            # Try to find "N.N. Title" or "N.N.Title" patterns
            pattern = re.compile(
                rf"^{re.escape(number)}\.\s*{re.escape(title_part.lower())}",
                re.MULTILINE | re.IGNORECASE,
            )
            for match in pattern.finditer(text):
                if match.start() >= toc_skip_threshold:
                    return match.start()

        # Pattern 3: Just find the title text (without number)
        # This is more lenient for cases where numbering differs
        if title_normalized:
            pattern = re.compile(rf"\b{re.escape(title_normalized)}\b", re.IGNORECASE)
            for match in pattern.finditer(text_lower):
                if match.start() >= toc_skip_threshold:
                    return match.start()

        # Not found
        return None

    def _normalize_heading(self, heading: str) -> str:
        """
        Normalize a heading for fuzzy matching.

        Removes section numbers, excess whitespace, and standardizes punctuation.
        """
        # Remove section numbers at start (e.g., "1.2. Title" -> "Title")
        heading = re.sub(r"^\d+(?:\.\d+)*\.\s*", "", heading)

        # Remove page numbers (e.g., "Title — 42" -> "Title")
        heading = re.sub(r"\s+—\s+\d+$", "", heading)

        # Normalize whitespace
        heading = " ".join(heading.split())

        # Remove certain punctuation for better matching
        heading = heading.replace("'", "'")  # Normalize quotes
        heading = heading.replace(""", '"').replace(""", '"')

        return heading.strip()

    def _filter_toc_entries(self, nodes: List[Dict], text: str) -> List[Dict]:
        """
        Filter out table of contents and index entries.

        Strategy: Detect TOC region (high density of numbered entries with page numbers
        in first 15% of document) and skip all entries from that region.
        """
        if not nodes:
            return nodes

        text_length = len(text)
        toc_threshold = int(text_length * 0.15)  # First 15% might be TOC

        # Find nodes with page numbers in early part of document
        early_nodes_with_pages = []
        for node in nodes:
            if node["position"] < toc_threshold:
                title_parts = node["title"].rsplit(None, 1)
                has_page_number = (
                    len(title_parts) == 2
                    and title_parts[-1].isdigit()
                    and len(title_parts[-1]) <= 4
                )
                if has_page_number:
                    early_nodes_with_pages.append(node)

        # If many nodes with page numbers found in early part, likely a TOC section
        if len(early_nodes_with_pages) > 10:
            # Find the end of TOC region (position of last early node with page num)
            toc_end_position = max(n["position"] for n in early_nodes_with_pages)

            # Filter out all nodes before TOC end
            filtered = [n for n in nodes if n["position"] > toc_end_position]

            # Clean page numbers from remaining titles
            for node in filtered:
                title_parts = node["title"].rsplit(None, 1)
                if (
                    len(title_parts) == 2
                    and title_parts[-1].isdigit()
                    and len(title_parts[-1]) <= 4
                ):
                    node["title"] = title_parts[0]

            # Also remove junk (very high numbers, short titles)
            cleaned = []
            for node in filtered:
                num = node["number"]
                title = node["title"]

                # Skip if chapter number is > 20
                if "." not in num:
                    try:
                        if int(num) > 20:
                            continue
                    except ValueError:
                        continue

                # Skip if title is too short
                if len(title) < 10:
                    continue

                cleaned.append(node)

            # Final de-duplication: keep first occurrence of each number
            seen = set()
            final = []
            for node in cleaned:
                if node["number"] not in seen:
                    seen.add(node["number"])
                    final.append(node)

            return final
        else:
            # No clear TOC detected - just clean page numbers and de-duplicate
            for node in nodes:
                title_parts = node["title"].rsplit(None, 1)
                if (
                    len(title_parts) == 2
                    and title_parts[-1].isdigit()
                    and len(title_parts[-1]) <= 4
                ):
                    node["title"] = title_parts[0]

            # De-duplicate
            seen = set()
            final = []
            for node in nodes:
                if node["number"] not in seen:
                    seen.add(node["number"])
                    final.append(node)

            return final

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
        self, node_dicts: List[Dict], sentences: List[str], text: str = None
    ) -> Tuple[List[StructureNode], List[SentenceLocation]]:
        """
        Build hierarchical structure and assign sentences to nodes.

        Args:
            node_dicts: List of detected heading dictionaries
            sentences: List of sentence strings
            text: Full document text (used to scale sentence positions accurately)

        Returns:
            Tuple of (structure_nodes, sentence_locations)
        """
        # Sort nodes by position in text
        node_dicts = sorted(node_dicts, key=lambda x: x["position"])

        # Map sentences to nodes by finding which heading precedes each sentence.
        # Sentence positions are approximate (sum of lengths) but heading positions
        # are real character offsets in the full text.  Scale sentence positions up
        # to the same coordinate space so the comparison is meaningful.
        sentence_to_node: Dict[int, int] = {}
        text_positions = self._get_sentence_positions(sentences)

        if text and text_positions:
            text_len = len(text)
            # Total approx length = position of last sentence + its length
            total_approx = text_positions[-1] + len(sentences[-1]) if sentences else 1
            scale = text_len / total_approx if total_approx > 0 else 1.0
            text_positions = [pos * scale for pos in text_positions]

        for sent_idx, sent_pos in enumerate(text_positions):
            # Find the last heading before this sentence
            for node_idx, node in enumerate(node_dicts):
                if node["position"] <= sent_pos:
                    sentence_to_node[sent_idx] = node_idx

        # Build structure nodes with start/end indices
        structure_nodes = []
        for idx, node_dict in enumerate(node_dicts):
            # Find first sentence in this section (None if no sentences map here)
            start_idx = min(
                (
                    sent_idx
                    for sent_idx, node_idx in sentence_to_node.items()
                    if node_idx == idx
                ),
                default=None,
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

            # If no sentences mapped to this node, place it as an empty range at
            # the boundary (start == end) so it doesn't steal sentences from others
            if start_idx is None:
                start_idx = end_idx

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
                if node.level == "level_2":
                    subsection = node.number
                    subsection_title = node.title
                    # Get parent section and chapter
                    if node.parent_number and node.parent_number in node_dict:
                        parent = node_dict[node.parent_number]
                        if parent.level == "level_1":
                            section = parent.number
                            section_title = parent.title
                            if (
                                parent.parent_number
                                and parent.parent_number in node_dict
                            ):
                                grandparent = node_dict[parent.parent_number]
                                chapter = grandparent.number
                                chapter_title = grandparent.title
                elif node.level == "level_1":
                    section = node.number
                    section_title = node.title
                    # Get parent chapter
                    if node.parent_number and node.parent_number in node_dict:
                        parent = node_dict[node.parent_number]
                        if parent.level == "level_0":
                            chapter = parent.number
                            chapter_title = parent.title
                elif node.level == "level_0":
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
