"""
Tests for front/back-matter detection (preprocessing/matter.py) —
structured-ingestion Phase C.1/C.2.
"""

from concept_mapper.preprocessing.matter import (
    detect_matter,
    trim_matter,
)


def _book_text():
    front = (
        "SEMIOTICS AND THE PHILOSOPHY OF LANGUAGE\n"
        "Umberto Eco\n"
        "© 1984 by Umberto Eco. All rights reserved.\n"
        "ISBN 0-253-35168-5\n"
        "Contents\n"
        "Introduction — 1\n"
        "1. Signs — 14\n"
        "1.1. Crisis of a concept — 14\n"
        "1.2. The signs of an obstinacy — 15\n"
        "2. Dictionary vs. Encyclopedia — 46\n"
    )
    body_lines = [
        f"Body sentence number {i}: a sign stands for something else."
        for i in range(60)
    ]
    back = (
        "BIBLIOGRAPHY\n"
        "Eco, Umberto. 1976. A Theory of Semiotics.\n"
        "Peirce, Charles S. 1931. Collected Papers.\n"
    )
    return front + "\n".join(body_lines) + "\n" + back


def _index_text():
    body_lines = [
        f"Body sentence number {i}: semiosis is a process." for i in range(60)
    ]
    index_lines = [
        "abduction, 23, 40, 131",
        "code, 14, 19, 164-88",
        "interpretant, 1, 43, 68",
        "metaphor, 87-129",
        "Peirce, Charles S., 1, 4, 15n",
        "semiosis, 1, 2, 26",
        "sign, 14-45",
        "sign-function, 21, 169",
        "symbol, 130-63",
        "synecdoche, 92, 116",
        "text, 24, 78",
        "Wolff, Christian, 26",
    ]
    return "\n".join(body_lines) + "\n" + "\n".join(index_lines)


class TestBackMatter:
    def test_bibliography_heading_trimmed(self):
        body, report = trim_matter(_book_text())
        assert "BIBLIOGRAPHY" not in body
        assert "Collected Papers" not in body
        assert "bibliography" in report["back"].lower()

    def test_index_shape_without_heading_trimmed(self):
        body, report = trim_matter(_index_text())
        assert "abduction, 23" not in body
        assert "index-shaped" in report["back"]
        assert "Body sentence number 59" in body

    def test_heading_in_first_half_ignored(self):
        # "Index" as a discussed topic early in a text is not back-matter.
        text = "Index\n" + "\n".join(
            f"Sentence {i} about the index as a sign type." for i in range(40)
        )
        body, report = trim_matter(text)
        assert report["back_lines"] == 0
        assert body == text


class TestFrontMatter:
    def test_copyright_and_contents_trimmed(self):
        body, report = trim_matter(_book_text())
        assert "All rights reserved" not in body
        assert "ISBN" not in body
        # TOC lines gone, body intact
        assert "Crisis of a concept — 14" not in body
        assert "Body sentence number 0" in body
        assert "copyright page" in report["front"]
        assert "contents block" in report["front"]

    def test_clean_chapter_text_untouched(self):
        text = "\n".join(
            f"Sentence {i}: semiosis produces an interpretant." for i in range(50)
        )
        body, report = trim_matter(text)
        assert body == text
        assert report["front_lines"] == 0 and report["back_lines"] == 0


class TestDetectMatter:
    def test_detection_reports_without_modifying(self):
        result = detect_matter(_book_text())
        assert result["front_end"] is not None
        assert result["back_start"] is not None


class TestPipelineIntegration:
    def test_preprocess_trim_matter_flag(self):
        from concept_mapper.corpus.models import Document
        from concept_mapper.preprocessing.pipeline import preprocess

        doc = Document(text=_book_text(), metadata={})
        processed = preprocess(doc, detect_structure=False, trim_matter=True)
        assert "matter_trimmed" in processed.metadata
        assert processed.metadata["matter_trimmed"]["back_lines"] > 0
        assert not any("ISBN" in s for s in processed.sentences)

    def test_preprocess_default_leaves_matter(self):
        from concept_mapper.corpus.models import Document
        from concept_mapper.preprocessing.pipeline import preprocess

        doc = Document(text=_book_text(), metadata={})
        processed = preprocess(doc, detect_structure=False)
        assert "matter_trimmed" not in processed.metadata
        assert any("BIBLIOGRAPHY" in s for s in processed.sentences)
