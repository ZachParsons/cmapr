"""
File loading utilities for text corpus.

Provides functions to load individual files or entire directories
into Document and Corpus objects.
"""

import re
from pathlib import Path
from typing import Dict, Optional, Union

from .models import Corpus, Document

try:
    import pdfplumber

    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False


def load_text(file_path: Union[str, Path]) -> str:
    """
    Load text content from a file with encoding fallback.

    Attempts UTF-8 first, falls back to Latin-1 if UTF-8 fails.
    This handles most text files including those with special characters.

    Args:
        file_path: Path to text file

    Returns:
        File contents as string

    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(file_path)

    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Fallback to Latin-1 for files with special encoding
        return path.read_text(encoding="latin-1")


def resolve_pdf_backend(backend: str = "auto") -> str:
    """Resolve "auto" to docling when importable, else pdfplumber.

    Adopted per the B.2/B.3 evaluation (docs/plans/structured-ingestion.md):
    98% heading recall against the hand-curated TOC baseline, full book in
    ~34s — docling is preferred whenever the ``ingest`` extra is present.
    """
    if backend != "auto":
        return backend
    try:
        import docling  # noqa: F401, PLC0415

        return "docling"
    except ImportError:
        return "pdfplumber"


def load_pdf(file_path: Union[str, Path], backend: str = "auto") -> str:
    """
    Extract text from a PDF file.

    Backends:
    - ``auto`` (default): docling when installed, else pdfplumber.
    - ``pdfplumber``: raw text layer, page by page. Running
      headers/footers come along (clean with ``--clean-ocr`` downstream).
    - ``docling``: layout-aware extraction (requires the ``ingest`` extra).
      Page headers/footers are dropped as furniture, headings come out as
      standalone normalized lines — much friendlier to the automatic
      structure detector. Slower (~0.1–0.6s/page CPU; layout model
      downloads on first use).

    Args:
        file_path: Path to PDF file
        backend: "auto", "pdfplumber", or "docling"

    Returns:
        Extracted text from all pages

    Raises:
        ImportError: If the chosen backend is not installed
        FileNotFoundError: If file doesn't exist
    """
    backend = resolve_pdf_backend(backend)
    if backend == "docling":
        return _load_pdf_docling(file_path)

    if backend != "pdfplumber":
        raise ValueError(f"Unknown PDF backend: {backend!r}")

    if not PDF_SUPPORT:
        raise ImportError(
            "pdfplumber is required for PDF support. "
            "Install with: pip install pdfplumber"
        )

    path = Path(file_path)
    pages_text = []

    with pdfplumber.open(path) as pdf:
        for page_num, page in enumerate(pdf.pages, 1):
            text = page.extract_text()
            if text:  # Only add non-empty pages
                pages_text.append(text)

    # Join pages with double newline (paragraph separator)
    return "\n\n".join(pages_text)


# Glyph noise observed in typeset-PDF heading numbers (docling B.0 trial):
# Greek capital iota / omicron and Cyrillic О standing in for I/O, "*" for
# ".", and spaced-out dots ("1 . 5").
_HEADING_GLYPHS = str.maketrans({"Ι": "I", "О": "O", "Ο": "O"})


def _normalize_heading(text: str) -> str:
    """Normalize a docling heading line for the structure detector."""
    t = text.translate(_HEADING_GLYPHS)
    t = re.sub(r"\s+", " ", t).strip()
    # "1.5*6" → "1.5.6"; "1 . 5 . 3" → "1.5.3" (leading section numbers only)
    t = re.sub(r"(?<=\d)\s*\*\s*(?=\d)", ".", t)
    m = re.match(r"^((?:[\dIVXL]+\s*\.\s*)+[\dIVXL]*\.?)\s*(.*)$", t)
    if m:
        number = re.sub(r"\s+", "", m.group(1))
        t = f"{number} {m.group(2)}".strip()
    return t


def _load_pdf_docling(
    file_path: Union[str, Path],
    page_range=None,
    collected_headings: Optional[list] = None,
) -> str:
    """Layout-aware PDF extraction via docling (``ingest`` extra).

    When ``collected_headings`` is a list, every emitted heading is also
    appended to it as ``{"title": ..., "level": ...}`` so callers can feed
    them to the structure detector (C.3 — TOC file optional).
    """
    try:
        from docling.datamodel.base_models import InputFormat  # noqa: PLC0415
        from docling.datamodel.pipeline_options import PdfPipelineOptions  # noqa: PLC0415
        from docling.document_converter import (  # noqa: PLC0415
            DocumentConverter,
            PdfFormatOption,
        )
        from docling_core.types.doc import SectionHeaderItem, TitleItem  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            "docling is required for the docling PDF backend. "
            "Install with: uv sync --extra ingest"
        ) from e

    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    converter = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    kwargs = {"page_range": page_range} if page_range else {}
    doc = converter.convert(str(file_path), **kwargs).document

    # Body items only — docling classifies running headers/footers as page
    # furniture, which iterate_items skips. Headings become standalone
    # blank-line-separated lines so structure detection can find them.
    lines = []
    for item, _level in doc.iterate_items():
        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue
        if isinstance(item, (SectionHeaderItem, TitleItem)):
            heading = _normalize_heading(text)
            lines.append(f"\n{heading}\n")
            if collected_headings is not None:
                collected_headings.append(
                    {"title": heading, "level": getattr(item, "level", 1)}
                )
        else:
            lines.append(text)
    return "\n".join(lines)


def load_file(
    file_path: Union[str, Path],
    metadata: Optional[Dict] = None,
    pdf_backend: str = "auto",
) -> Document:
    """
    Load a single file into a Document object.

    Supports both text files (.txt, .md, etc.) and PDF files (.pdf).

    Args:
        file_path: Path to text or PDF file
        metadata: Optional metadata dict (if None, extracts from filename)
        pdf_backend: "pdfplumber" (raw text layer) or "docling"
            (layout-aware; requires the ``ingest`` extra)

    Returns:
        Document object with text and metadata

    Raises:
        FileNotFoundError: If file doesn't exist
        ImportError: If PDF file but the chosen backend is not installed
    """
    path = Path(file_path)

    # Detect file type and load accordingly
    detected_headings: list = []
    if path.suffix.lower() == ".pdf":
        backend = resolve_pdf_backend(pdf_backend)
        if backend == "docling":
            text = _load_pdf_docling(path, collected_headings=detected_headings)
        else:
            text = load_pdf(path, backend=backend)
    else:
        text = load_text(path)

    # Create metadata if not provided
    if metadata is None:
        metadata = {
            "source_path": str(path.resolve()),
            "filename": path.name,
            "title": path.stem,  # Filename without extension
        }
    else:
        # Ensure source_path is set
        if "source_path" not in metadata:
            metadata["source_path"] = str(path.resolve())

    # Backend-supplied headings let structure detection skip heuristics
    # (and make the --toc file optional) — see preprocessing/structure.py.
    if detected_headings:
        metadata["detected_headings"] = detected_headings

    return Document(text=text, metadata=metadata)


def load_directory(
    directory_path: Union[str, Path], pattern: str = "*.txt", recursive: bool = False
) -> Corpus:
    """
    Load all matching files from a directory into a Corpus.

    Args:
        directory_path: Path to directory
        pattern: Glob pattern for matching files (default: "*.txt")
        recursive: If True, search subdirectories recursively

    Returns:
        Corpus containing all loaded documents

    Raises:
        NotADirectoryError: If path is not a directory
        FileNotFoundError: If directory doesn't exist
    """
    dir_path = Path(directory_path)

    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    if not dir_path.is_dir():
        raise NotADirectoryError(f"Not a directory: {dir_path}")

    # Use rglob for recursive, glob for non-recursive
    glob_method = dir_path.rglob if recursive else dir_path.glob
    file_paths = sorted(glob_method(pattern))

    corpus = Corpus()
    for file_path in file_paths:
        if file_path.is_file():
            doc = load_file(file_path)
            corpus.add_document(doc)

    return corpus
