"""
Preprocessing pipeline.

Provides unified entry point for preprocessing documents through
all stages: tokenization → POS tagging → lemmatization.
"""

from typing import List

from ..corpus.models import Document, ProcessedDocument
from .lemmatize import lemmatize_tagged
from .tagging import tag_tokens
from .tokenize import tokenize_sentences, tokenize_words


def preprocess(document: Document) -> ProcessedDocument:
    """
    Preprocess a single document through full pipeline.

    Pipeline stages:
    1. Sentence tokenization
    2. Word tokenization
    3. POS tagging
    4. Lemmatization

    Args:
        document: Input Document object

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

    # 1. Sentence tokenization
    sentences = tokenize_sentences(text)

    # 2. Word tokenization
    tokens = tokenize_words(text)

    # 3. POS tagging
    pos_tags = tag_tokens(tokens)

    # 4. Lemmatization
    lemmas = lemmatize_tagged(pos_tags)

    return ProcessedDocument(
        raw_text=text,
        sentences=sentences,
        tokens=tokens,
        pos_tags=pos_tags,
        lemmas=lemmas,
        metadata=document.metadata.copy(),
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
