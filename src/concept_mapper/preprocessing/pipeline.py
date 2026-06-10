"""
Preprocessing pipeline.

Provides unified entry point for preprocessing documents through
all stages: tokenization → POS tagging → lemmatization.
"""

from pathlib import Path
from typing import List, Optional

from ..corpus.models import Document, ProcessedDocument
from .cleaning import clean_text
from .lemmatize import lemmatize_tagged
from .noun_chunks import extract_noun_chunks
from .segment import get_paragraph_indices
from .structure import DocumentStructureDetector
from .tagging import tag_tokens
from .tokenize import tokenize_sentences, tokenize_words


def preprocess(
    document: Document,
    detect_structure: bool = True,
    clean_ocr: bool = False,
    toc_file: Optional[Path] = None,
    use_spacy: bool = False,
    resolve_coref: bool = False,
    trim_matter: bool = False,
) -> ProcessedDocument:
    """
    Preprocess a single document through full pipeline.

    Pipeline stages:
    0a. Front/back-matter trimming (optional) - copyright/TOC/index removal
    0. Text cleaning (optional) - OCR/PDF artifact removal
    0b. Coreference resolution (optional, requires the coref extra)
    1. Sentence tokenization
    2. Word tokenization
    3. POS tagging
    4. Lemmatization
    5. Structure detection (optional)
    6. Noun chunk extraction (optional, requires spaCy)

    Args:
        document: Input Document object
        detect_structure: Whether to detect document structure (default: True)
        clean_ocr: Whether to clean OCR/PDF artifacts (default: False)
        toc_file: Optional path to table of contents file for guided structure detection
        use_spacy: Extract multi-word noun chunks via spaCy (default: False).
                   Results stored in ``ProcessedDocument.metadata["noun_chunks"]``.
        resolve_coref: Rewrite pronominal mentions with their antecedents
                       before tokenization (default: False; requires the
                       coref extra — see ``preprocessing/coref.py``).
        trim_matter: Strip detected front-matter (copyright page, contents
                     block) and back-matter (bibliography, index) before any
                     other stage (default: False). Trim details land in
                     ``ProcessedDocument.metadata["matter_trimmed"]``.

    Returns:
        ProcessedDocument with all linguistic annotations

    Example:
        >>> doc = Document(text="The cats sat. They ran.", metadata={"title": "Test"})
        >>> processed = preprocess(doc)
        >>> processed.num_sentences
        2
        >>> processed.lemmas[:3]
        ['the', 'cat', 'sit']
    """
    text = document.text

    # 0a. Front/back-matter trimming — before everything, so cleaning and
    # structure detection only ever see body text.
    matter_report = None
    if trim_matter:
        from .matter import trim_matter as _trim_matter  # noqa: PLC0415

        text, matter_report = _trim_matter(text)

    # 0. Clean OCR/PDF artifacts if requested
    if clean_ocr:
        text = clean_text(text)

    # 0b. Coreference resolution (after cleaning, before tokenization, so all
    # downstream stages see the resolved text as the document text)
    if resolve_coref:
        from .coref import resolve_coreferences  # noqa: PLC0415

        text = resolve_coreferences(text)

    # 1. Sentence tokenization
    sentences = tokenize_sentences(text)

    # 2. Word tokenization
    tokens = tokenize_words(text)

    # 3. POS tagging
    pos_tags = tag_tokens(tokens)

    # 4. Lemmatization
    lemmas = lemmatize_tagged(pos_tags)

    # 5. Structure detection — an explicit TOC file wins; otherwise
    # backend-supplied headings (docling PDF path) make `--toc` optional;
    # the heuristics remain the fallback.
    structure_nodes = []
    sentence_locations = []
    if detect_structure:
        try:
            detector = DocumentStructureDetector()
            structure_nodes, sentence_locations = detector.detect(
                text,
                sentences,
                toc_file=toc_file,
                headings=document.metadata.get("detected_headings"),
            )
        except Exception:
            # Fail gracefully - structure detection is optional
            pass

    # 6. Paragraph segmentation
    paragraph_indices = get_paragraph_indices(text, sentences)

    doc_metadata = document.metadata.copy()
    if matter_report and (matter_report["front_lines"] or matter_report["back_lines"]):
        doc_metadata["matter_trimmed"] = matter_report

    # 7. Noun chunk extraction (optional)
    if use_spacy:
        doc_metadata["noun_chunks"] = extract_noun_chunks(text)

    return ProcessedDocument(
        raw_text=text,
        sentences=sentences,
        tokens=tokens,
        pos_tags=pos_tags,
        lemmas=lemmas,
        metadata=doc_metadata,
        structure_nodes=structure_nodes,
        sentence_locations=sentence_locations,
        paragraph_indices=paragraph_indices,
    )


def preprocess_corpus(documents: List[Document]) -> List[ProcessedDocument]:
    """
    Preprocess multiple documents.

    Args:
        documents: List of Document objects

    Returns:
        List of ProcessedDocument objects

    Example:
        >>> docs = [Document(text="Text 1"), Document(text="Text 2")]
        >>> processed = preprocess_corpus(docs)
        >>> len(processed)
        2
    """
    return [preprocess(doc) for doc in documents]
