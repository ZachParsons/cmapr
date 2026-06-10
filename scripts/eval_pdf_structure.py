"""
B.2 structure-comparison harness (docs/plans/structured-ingestion.md).

Extracts headings from a PDF via the docling backend and scores them
against a hand-curated TOC file (ground truth), reporting precision /
recall on normalized section titles.

Usage:
    uv run python scripts/eval_pdf_structure.py \
        data/input/Eco_1984_SPL.pdf data/input/eco_spl_toc.txt [MAX_PAGE]
"""

import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path


def normalize(title: str) -> str:
    """Comparable form: number/page stripped, casefolded, alnum+space only."""
    t = title.strip()
    t = re.sub(r"^[\d.IVXLivxl\s*]+\.?\s+", "", t)  # leading section number
    t = re.sub(r"[—–-]?\s*\d+\s*$", "", t)  # trailing page number
    t = re.sub(r"[^a-z0-9 ]", "", t.casefold())
    return re.sub(r"\s+", " ", t).strip()


def parse_ground_truth(toc_path: Path) -> list:
    from concept_mapper.preprocessing.structure import DocumentStructureDetector

    detector = DocumentStructureDetector()
    entries = detector._parse_toc_markdown(toc_path.read_text())
    return [e["title"] for e in entries if e.get("title")]


def extract_headings(pdf_path: Path, max_page=None) -> list:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import SectionHeaderItem, TitleItem

    from concept_mapper.corpus.loader import _normalize_heading

    opts = PdfPipelineOptions(do_ocr=False, do_table_structure=False)
    conv = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )
    kwargs = {"page_range": (1, max_page)} if max_page else {}
    doc = conv.convert(str(pdf_path), **kwargs).document
    return [
        _normalize_heading(item.text)
        for item, _ in doc.iterate_items()
        if isinstance(item, (SectionHeaderItem, TitleItem)) and item.text.strip()
    ]


def fuzzy_match(a: str, b: str) -> bool:
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    return SequenceMatcher(None, a, b).ratio() >= 0.8


def main():
    pdf = Path(sys.argv[1])
    toc = Path(sys.argv[2])
    max_page = int(sys.argv[3]) if len(sys.argv) > 3 else None

    truth = parse_ground_truth(toc)
    truth_norm = [normalize(t) for t in truth]
    print(f"ground truth: {len(truth)} TOC entries")

    t0 = time.time()
    headings = extract_headings(pdf, max_page)
    print(f"docling: {len(headings)} headings in {time.time() - t0:.0f}s")

    extracted_norm = [normalize(h) for h in headings]
    matched_truth = set()
    matched_extracted = set()
    for i, t in enumerate(truth_norm):
        for j, e in enumerate(extracted_norm):
            if j not in matched_extracted and fuzzy_match(t, e):
                matched_truth.add(i)
                matched_extracted.add(j)
                break

    recall = len(matched_truth) / max(len(truth_norm), 1)
    precision = len(matched_extracted) / max(len(extracted_norm), 1)
    print(f"\nrecall    {len(matched_truth)}/{len(truth_norm)} = {recall:.0%}"
          f"  (TOC entries docling found)")
    print(f"precision {len(matched_extracted)}/{len(extracted_norm)} = {precision:.0%}"
          f"  (docling headings that are real TOC entries)")

    misses = [truth[i] for i in range(len(truth)) if i not in matched_truth]
    if misses:
        print(f"\nmissed TOC entries ({len(misses)}):")
        for m in misses[:20]:
            print(f"  - {m}")
    extras = [headings[j] for j in range(len(headings)) if j not in matched_extracted]
    if extras:
        print(f"\nextra docling headings ({len(extras)}, first 15):")
        for e in extras[:15]:
            print(f"  + {e}")


if __name__ == "__main__":
    main()
