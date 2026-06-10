"""
Front-matter / back-matter detection (structured-ingestion Phase C.1–C.2).

Book-shaped inputs carry non-body text at both ends: title page, copyright
page, and table of contents up front; bibliography, notes, appendices, and
the index at the back. Left in, they poison downstream stages — an index
is *term-dense* (it inflates rarity frequencies and mints garbage
concordance entries), and TOC lines duplicate every heading.

Detection is heuristic and position-gated so trimmed chapter files (the
common cmapr input) pass through untouched:

* **Back-matter** starts at a recognized heading (BIBLIOGRAPHY, INDEX,
  REFERENCES, …) in the last half of the document, or at an index-shaped
  run (consecutive short lines ending in page-number lists) even without a
  heading.
* **Front-matter** ends after the copyright markers (©, ISBN, "all rights
  reserved") and/or the CONTENTS block (heading + lines that end with page
  numbers), all required to sit in the first fifth of the document.

`trim_matter` returns the body plus a report of what was cut so callers
can echo it — silent truncation is never acceptable.
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# Headings that open back-matter sections (whole line, optional numbering).
_BACK_HEADING_RE = re.compile(
    r"^\s*(?:[\divxlc]+\.?\s+)?"
    r"(bibliography|references|works\s+cited|index(?:\s+of\s+\w+)?|"
    r"appendix(?:\s+[a-z\d])?|glossary|endnotes)\s*$",
    re.IGNORECASE,
)

# An index-shaped line: short, comma-separated trailing page numbers
# ("semiosis, 23, 45, 112-13" / "Peirce, C. S., 7n, 215").
_INDEX_LINE_RE = re.compile(
    r"^.{1,60}?,\s*\d{1,4}(?:n|f|ff)?(?:\s*[,–-]\s*\d{1,4}(?:n|f|ff)?)*\s*$"
)

# Copyright-page markers.
_COPYRIGHT_RE = re.compile(
    r"©|\bcopyright\b|\ball\s+rights\s+reserved\b|\bisbn\b|"
    r"\blibrary\s+of\s+congress\b",
    re.IGNORECASE,
)

# A TOC-shaped line: text then a trailing page number, optionally
# dot-leadered or em-dashed ("1.2. The signs of an obstinacy — 15").
_TOC_LINE_RE = re.compile(r"^.{3,80}?[.\s—–-]\s*\d{1,4}\s*$")
_CONTENTS_RE = re.compile(r"^\s*(contents|table\s+of\s+contents)\s*$", re.IGNORECASE)

_INDEX_RUN_MIN = 10  # consecutive index-shaped lines to call it an index


def detect_back_matter(lines: list) -> Tuple[Optional[int], str]:
    """Line index where back-matter starts, or (None, ''). Last-half only."""
    floor = len(lines) // 2
    for i in range(floor, len(lines)):
        if _BACK_HEADING_RE.match(lines[i]):
            return i, f"back-matter heading {lines[i].strip()!r}"
    # No heading: look for an index-shaped run.
    run_start, run = None, 0
    for i in range(floor, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if _INDEX_LINE_RE.match(stripped):
            if run == 0:
                run_start = i
            run += 1
            if run >= _INDEX_RUN_MIN:
                return run_start, "index-shaped line run (no heading)"
        else:
            run = 0
    return None, ""


def detect_front_matter(lines: list) -> Tuple[Optional[int], str]:
    """Line index where the body starts, or (None, ''). First-fifth only."""
    ceiling = max(len(lines) // 5, 1)
    last_marker = None
    reasons = []

    for i in range(ceiling):
        if _COPYRIGHT_RE.search(lines[i]):
            last_marker = i
            if "copyright page" not in reasons:
                reasons.append("copyright page")

    # CONTENTS block: heading plus the run of TOC-shaped lines after it.
    for i in range(ceiling):
        if _CONTENTS_RE.match(lines[i]):
            end = i
            for j in range(i + 1, min(len(lines), i + 200)):
                stripped = lines[j].strip()
                if not stripped or _TOC_LINE_RE.match(stripped):
                    end = j
                else:
                    break
            if end > i:
                last_marker = max(last_marker or 0, end)
                reasons.append("contents block")
            break

    if last_marker is None:
        return None, ""
    return last_marker + 1, " + ".join(reasons)


def trim_matter(text: str) -> Tuple[str, dict]:
    """Strip detected front/back-matter; return (body, report).

    The report carries ``front_lines`` / ``back_lines`` counts and the
    detection reasons (empty report values mean nothing was trimmed).
    """
    lines = text.split("\n")
    report = {"front_lines": 0, "back_lines": 0, "front": "", "back": ""}

    back_start, back_reason = detect_back_matter(lines)
    if back_start is not None:
        report["back_lines"] = len(lines) - back_start
        report["back"] = back_reason
        lines = lines[:back_start]

    front_end, front_reason = detect_front_matter(lines)
    if front_end is not None:
        report["front_lines"] = front_end
        report["front"] = front_reason
        lines = lines[front_end:]

    return "\n".join(lines), report


def detect_matter(text: str) -> dict:
    """Detection-only variant for warnings (no text modification)."""
    lines = text.split("\n")
    back_start, back_reason = detect_back_matter(lines)
    front_end, front_reason = detect_front_matter(lines)
    return {
        "front_end": front_end,
        "front": front_reason,
        "back_start": back_start,
        "back": back_reason,
    }
