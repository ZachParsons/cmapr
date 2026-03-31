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
from .segment import get_paragraph_indices
from .structure import DocumentStructureDetector
from .tagging import tag_tokens
from .tokenize import tokenize_sentences, tokenize_words

# ---------------------------------------------------------------------------
# spaCy noun-chunk extraction (optional; loaded lazily)
# ---------------------------------------------------------------------------

_SPACY_NLP = None

_LEADING_DETS = frozenset(
    {
        "the", "a", "an", "this", "that", "these", "those",
        "its", "their", "our", "your", "my", "his", "her",
    }
)


def _get_spacy_nlp():
    """Load and cache the spaCy en_core_web_sm model."""
    global _SPACY_NLP
    if _SPACY_NLP is None:
        import spacy  # noqa: PLC0415

        try:
            _SPACY_NLP = spacy.load("en_core_web_sm")
        except OSError:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' not found. "
                "Run: python -m spacy download en_core_web_sm"
            )
    return _SPACY_NLP


def _extract_noun_chunks(text: str) -> List[str]:
    """
    Extract multi-word noun phrases from *text* using spaCy.

    Returns a deduplicated list of lowercased multi-word phrases with leading
    determiners stripped (e.g. 'the sign vehicle' → 'sign vehicle').
    """
    nlp = _get_spacy_nlp()
    max_len = nlp.max_length - 100
    seen: set = set()
    for start in range(0, len(text), max_len):
        doc = nlp(text[start : start + max_len])
        for chunk in doc.noun_chunks:
            words = chunk.text.lower().split()
            # Strip leading determiners/articles
            while words and words[0] in _LEADING_DETS:
                words = words[1:]
            if len(words) < 2:
                continue
            # Reject if any token is a single character (OCR artifact)
            if any(len(w) < 2 for w in words):
                continue
            seen.add(" ".join(words))
    return list(seen)


def preprocess(
    document: Document,
    detect_structure: bool = True,
    clean_ocr: bool = False,
    toc_file: Optional[Path] = None,
    use_spacy: bool = False,
) -> ProcessedDocument:
    """
    Preprocess a single document through full pipeline.

    Pipeline stages:
    0. Text cleaning (optional) - OCR/PDF artifact removal
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

    # 0. Clean OCR/PDF artifacts if requested
    if clean_ocr:
        text = clean_text(text)

    # 1. Sentence tokenization
    sentences = tokenize_sentences(text)

    # 2. Word tokenization
    tokens = tokenize_words(text)

    # 3. POS tagging
    pos_tags = tag_tokens(tokens)

    # 4. Lemmatization
    lemmas = lemmatize_tagged(pos_tags)

    # 5. Structure detection
    structure_nodes = []
    sentence_locations = []
    if detect_structure:
        try:
            detector = DocumentStructureDetector()
            structure_nodes, sentence_locations = detector.detect(
                text, sentences, toc_file=toc_file
            )
        except Exception:
            # Fail gracefully - structure detection is optional
            pass

    # 6. Paragraph segmentation
    paragraph_indices = get_paragraph_indices(text, sentences)

    doc_metadata = document.metadata.copy()

    # 7. Noun chunk extraction (optional)
    if use_spacy:
        doc_metadata["noun_chunks"] = _extract_noun_chunks(text)

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
