"""
Multi-chapter clustered graph construction.

Splits a corpus by chapter (or section), builds a per-cluster sub-graph,
namespaces nodes as ``f"{term}__{label}"``, and links recurring terms
with ``recurrence`` edges. Drives ``cmapr cluster``.

Complement of ``aggregate_graphs`` (which collapses per-chapter graphs
into a single unified view); this one preserves chapter structure.
"""

from typing import Dict, List, Optional

from concept_mapper.graph.model import ConceptGraph


def cluster_by_structure(
    docs: list,
    seed_terms: list,
    *,
    by: str = "chapter",
    node_filter=None,
    pmi_threshold: float = 1.0,
    term_scores: Optional[Dict[str, float]] = None,
) -> ConceptGraph:
    """
    Build a clustered concept graph: one sub-graph per chapter (or section)
    with nodes namespaced as ``f"{term}__{label}"``.

    For each cluster, ``build_proposition_graph`` is called against a
    sub-corpus restricted to that cluster's sentences. Nodes from the
    sub-graph are copied into the result with the namespaced id and a
    ``chapter`` (or ``section``) attribute carrying the original label;
    ``term`` carries the un-namespaced term so the viz can group by it.

    ``recurrence`` edges are added between consecutive same-term nodes in
    the document's natural cluster order (the order in which clusters
    are first encountered while walking ``sentence_locations``). Each
    recurrence edge carries ``weight = span``, where ``span`` is the
    total number of clusters the term appears in.

    Parameters
    ----------
    docs : list[ProcessedDocument]
        Same shape as ``cmapr graph`` consumes.
    seed_terms : list[str]
        Terms from the rarities list.
    by : {"chapter", "section"}
        Which structure level to cluster on. Defaults to ``"chapter"``.
    node_filter, pmi_threshold, term_scores
        Forwarded to ``build_proposition_graph`` for each cluster.

    Returns
    -------
    ConceptGraph
        Directed graph with clustered nodes and recurrence edges. Empty
        if there are no docs or no seed terms.
    """
    from concept_mapper.graph.builders import build_proposition_graph  # noqa: PLC0415

    result = ConceptGraph(directed=True)
    if not docs or not seed_terms:
        return result

    label_attr = "chapter" if by == "chapter" else "section"

    # ------------------------------------------------------------------
    # Phase 1 — sentence → cluster label map, preserving first-seen order
    # ------------------------------------------------------------------
    sentence_label: Dict[tuple, str] = {}
    cluster_order: List[str] = []
    for doc_idx, doc in enumerate(docs):
        # Default fallback when there's no structure metadata
        for sent_idx in range(len(doc.sentences)):
            sentence_label[(doc_idx, sent_idx)] = "Document"

        for loc in getattr(doc, "sentence_locations", []) or []:
            # Resilient: accept either SentenceLocation dataclass or a raw
            # dict (some callers do ProcessedDocument(**doc_data) which
            # leaves nested fields as dicts).
            if isinstance(loc, dict):
                get = loc.get
                sent_index = loc.get("sent_index")
            else:
                get = lambda k, d=None: getattr(loc, k, d)  # noqa: E731
                sent_index = getattr(loc, "sent_index", None)
            if sent_index is None:
                continue
            if by == "chapter":
                label = (
                    get("chapter_title")
                    or get("section_title")
                    or get("subsection_title")
                    or "Document"
                )
            else:
                label = (
                    get("section_title")
                    or get("subsection_title")
                    or get("chapter_title")
                    or "Document"
                )
            sentence_label[(doc_idx, sent_index)] = label

    for (doc_idx, sent_idx), label in sorted(sentence_label.items()):
        if label not in cluster_order:
            cluster_order.append(label)

    # ------------------------------------------------------------------
    # Phase 2 — sub-graph per cluster
    # ------------------------------------------------------------------
    # Track which clusters each term appears in, in cluster_order order
    term_clusters: Dict[str, List[str]] = {}

    for label in cluster_order:
        # Build a sub-corpus: ProcessedDocument-shaped objects whose
        # `sentences` are filtered to this cluster. PropositionExtractor
        # only reads `.sentences` and `.metadata`, so a duck is enough.
        class _SubDoc:
            __slots__ = ("sentences", "metadata")

            def __init__(self, sentences, metadata):
                self.sentences = sentences
                self.metadata = metadata

        sub_docs = []
        for doc_idx, doc in enumerate(docs):
            kept = [
                doc.sentences[s]
                for s in range(len(doc.sentences))
                if sentence_label.get((doc_idx, s)) == label
            ]
            if not kept:
                continue
            sub_docs.append(_SubDoc(kept, dict(getattr(doc, "metadata", {}) or {})))

        if not sub_docs:
            continue

        sub_graph = build_proposition_graph(
            docs=sub_docs,
            seed_terms=seed_terms,
            node_filter=node_filter,
            pmi_threshold=pmi_threshold,
            term_scores=term_scores,
        )

        # Copy nodes with namespaced id
        for orig_id in sub_graph.nodes():
            ns_id = f"{orig_id}__{label}"
            attrs = dict(sub_graph.get_node(orig_id))
            attrs["term"] = orig_id
            attrs[label_attr] = label
            # `label` attribute used by D3 for display — show un-namespaced term
            attrs.setdefault("label", orig_id)
            result.add_node(ns_id, **attrs)
            term_clusters.setdefault(orig_id, []).append(label)

        # Copy edges with namespaced endpoints
        for s, t in sub_graph.edges():
            attrs = dict(sub_graph.get_edge(s, t))
            result.add_edge(f"{s}__{label}", f"{t}__{label}", **attrs)

    # ------------------------------------------------------------------
    # Phase 3 — recurrence edges (same term across consecutive clusters)
    # ------------------------------------------------------------------
    for term, clusters in term_clusters.items():
        if len(clusters) < 2:
            continue
        span = len(clusters)
        for i in range(len(clusters) - 1):
            src = f"{term}__{clusters[i]}"
            tgt = f"{term}__{clusters[i + 1]}"
            result.add_edge(
                src,
                tgt,
                relation_type="recurrence",
                verb="recurs in",
                weight=span,
                evidence=[],
            )

    return result
