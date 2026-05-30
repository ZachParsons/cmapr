"""
Sentence-embedding-based ranker for identifying definitional sentences.

Uses sentence-transformers to find the most "definition-shaped" sentence per
term in a corpus. The ranker computes cosine similarity between candidate
sentences and a fixed prototype built from canonical definitional phrasings.

Optional dependency. Install with: ``uv sync --extra embeddings``.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


# Templates representing canonical definitional phrasings. The model captures
# structural similarity, so the prototype need not mention any specific term.
_DEFINITIONAL_TEMPLATES = [
    "A concept is defined as a particular notion.",
    "The term means a specific kind of thing.",
    "By this concept we mean a particular phenomenon.",
    "This concept refers to a specific kind of thing.",
    "This concept denotes a specific entity.",
    "It is a kind of category in the field.",
    "It is a type of phenomenon in the domain.",
    "This can be understood as a specific notion.",
    "We define this concept as a particular thing.",
    "The term names a particular kind of object.",
]


def _load_sentence_transformer(name: str):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise RuntimeError(
            "sentence-transformers is required for definitional ranking. "
            "Install with: uv sync --extra embeddings"
        ) from e
    return SentenceTransformer(name)


class DefinitionRanker:
    """
    Rank candidate sentences by similarity to a definitional prototype.

    The prototype is the mean (renormalised) embedding of a small set of
    canonical definitional phrasings. Higher cosine similarity → sentence
    structurally resembles a definition.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier. Default: all-MiniLM-L6-v2 (~80 MB,
        fast on CPU).
    cache_dir : Path | None
        If given, encoded sentence embeddings are persisted there keyed by
        a content hash of the sentence list.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        cache_dir: Optional[Path] = None,
    ):
        self.model_name = model_name
        self._model = None
        self._prototype = None
        self._cache_dir = cache_dir
        # Master lookup populated by `precompute`: sentence text → row index
        # into `_all_embeddings`. Lets per-node rank() reuse the same encoded
        # vectors instead of re-encoding sentence subsets.
        self._sentence_to_idx: dict = {}
        self._all_embeddings = None
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def precompute(self, sentences: List[str]) -> None:
        """
        Encode the full sentence list once. Subsequent `rank()` calls whose
        sentences are a subset of this list reuse the cached vectors.
        Idempotent — calling again with the same sentences is a no-op.
        """
        if (
            self._all_embeddings is not None
            and len(self._sentence_to_idx) == len(sentences)
            and all(s in self._sentence_to_idx for s in sentences)
        ):
            return
        embeddings = self._encode_with_cache(sentences)
        self._all_embeddings = embeddings
        self._sentence_to_idx = {s: i for i, s in enumerate(sentences)}

    @property
    def model(self):
        if self._model is None:
            self._model = _load_sentence_transformer(self.model_name)
        return self._model

    @property
    def prototype(self):
        if self._prototype is None:
            import numpy as np

            embeddings = self.model.encode(
                _DEFINITIONAL_TEMPLATES,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            mean = np.mean(embeddings, axis=0)
            norm = np.linalg.norm(mean)
            self._prototype = mean / norm if norm > 0 else mean
        return self._prototype

    def rank(self, sentences: List[str]) -> List[Tuple[str, float]]:
        """
        Return (sentence, cosine_similarity) pairs ranked most → least
        definitional. Input list order is preserved for ties.

        If `precompute` was called with a superset of these sentences, reuses
        those vectors; otherwise encodes (and caches) the input list directly.
        """
        if not sentences:
            return []
        if self._all_embeddings is not None and all(
            s in self._sentence_to_idx for s in sentences
        ):
            indices = [self._sentence_to_idx[s] for s in sentences]
            embeddings = self._all_embeddings[indices]
        else:
            embeddings = self._encode_with_cache(sentences)
        scores = embeddings @ self.prototype
        # Stable sort: higher score first, original index breaks ties
        indexed = list(enumerate(zip(sentences, scores.tolist())))
        indexed.sort(key=lambda x: (-x[1][1], x[0]))
        return [pair for _, pair in indexed]

    def _encode_with_cache(self, sentences: List[str]):
        import numpy as np

        if self._cache_dir is None:
            return self.model.encode(
                sentences,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

        h = hashlib.sha256()
        for s in sentences:
            h.update(s.encode("utf-8"))
            h.update(b"\n")
        key = h.hexdigest()[:16]
        model_slug = self.model_name.rsplit("/", 1)[-1]
        cache_file = self._cache_dir / f"{model_slug}-{key}.npy"
        if cache_file.exists():
            try:
                return np.load(cache_file)
            except Exception as e:  # noqa: BLE001
                logger.warning("Embeddings cache corrupt at %s: %s", cache_file, e)

        embeddings = self.model.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        try:
            np.save(cache_file, embeddings)
        except Exception as e:  # noqa: BLE001
            logger.warning("Could not write embeddings cache %s: %s", cache_file, e)
        return embeddings


def enrich_graph_with_definitions(
    graph,
    docs,
    threshold: float = 0.30,
    cache_dir: Optional[Path] = None,
    model_name: Optional[str] = None,
) -> int:
    """
    Attach a ``definition`` attribute to graph nodes by picking the most
    definitional sentence in the corpus that mentions each node's term.

    Nodes that already carry a ``definition`` attribute are left untouched.
    Nodes whose top candidate scores below ``threshold`` receive nothing —
    the existing tooltip fallback (definition-edge → kind-of → property →
    relation) still applies.

    Returns the number of nodes that received a new definition.
    """
    ranker = DefinitionRanker(
        model_name=model_name or DefinitionRanker.DEFAULT_MODEL,
        cache_dir=cache_dir,
    )

    all_sentences = [s for doc in docs for s in doc.sentences]
    if not all_sentences:
        return 0

    # Pre-encode the full sentence list once; per-node ranking then
    # slices the same matrix instead of re-encoding sentence subsets.
    ranker.precompute(all_sentences)

    n_added = 0
    nx_graph = graph.graph
    for node_id in graph.nodes():
        if nx_graph.nodes[node_id].get("definition"):
            continue
        term_l = node_id.lower()
        candidates = [s for s in all_sentences if term_l in s.lower()]
        if not candidates:
            continue
        ranked = ranker.rank(candidates)
        best_sentence, score = ranked[0]
        if score < threshold:
            continue
        nx_graph.nodes[node_id]["definition"] = best_sentence.strip()
        n_added += 1
    return n_added
