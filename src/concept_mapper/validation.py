"""Validation module for ensuring outputs are not empty."""


class EmptyOutputError(ValueError):
    """Raised when attempting to write empty or meaningless output."""
    pass


def validate_corpus(corpus_data: list | dict, min_docs: int = 1) -> None:
    """Validate that corpus data is not empty.

    Args:
        corpus_data: List of processed documents or dict with 'documents' key
        min_docs: Minimum number of documents required

    Raises:
        EmptyOutputError: If corpus is empty or has insufficient documents
    """
    # Handle dict format (e.g., {"documents": [...]})
    if isinstance(corpus_data, dict):
        # If it's a dict but not a standard corpus format, skip detailed validation
        if "documents" in corpus_data:
            docs = corpus_data["documents"]
        else:
            # For other dict formats, just check it's not empty
            if not corpus_data:
                raise EmptyOutputError(
                    "Cannot save empty corpus. No documents were processed."
                )
            return
    else:
        docs = corpus_data

    if not docs:
        raise EmptyOutputError(
            "Cannot save empty corpus. No documents were processed."
        )

    if len(docs) < min_docs:
        raise EmptyOutputError(
            f"Corpus has only {len(docs)} document(s), "
            f"minimum {min_docs} required."
        )

    # Check if documents have actual content (only for list format)
    if isinstance(docs, list) and docs:
        # Only validate tokens if docs have the expected structure
        if isinstance(docs[0], dict) and "tokens" in docs[0]:
            total_tokens = sum(
                len(doc.get("tokens", []))
                for doc in docs
            )
            if total_tokens == 0:
                raise EmptyOutputError(
                    "Corpus documents contain no tokens. Documents may be empty or "
                    "preprocessing failed."
                )


def validate_term_list(terms_data: list | dict, min_terms: int = 1) -> None:
    """Validate that term list is not empty.

    Args:
        terms_data: List of terms or dict with 'terms' key
        min_terms: Minimum number of terms required

    Raises:
        EmptyOutputError: If term list is empty or has insufficient terms
    """
    # Handle both list and dict formats
    if isinstance(terms_data, dict):
        terms = terms_data.get("terms", [])
    else:
        terms = terms_data

    if not terms:
        raise EmptyOutputError(
            "No terms detected. Try:\n"
            "  - Lowering the --threshold value\n"
            "  - Using a different --method (hybrid, ratio, tfidf, neologism)\n"
            "  - Checking that the input text is substantial (>500 words)\n"
            "  - Verifying the text is in English"
        )

    if len(terms) < min_terms:
        raise EmptyOutputError(
            f"Only {len(terms)} term(s) detected, minimum {min_terms} required. "
            f"Try lowering the --threshold or increasing --top-n."
        )


def validate_graph(graph_data: dict, require_edges: bool = True) -> None:
    """Validate that graph data is not empty.

    Args:
        graph_data: Graph data dict with 'nodes' and 'links'/'edges' keys
        require_edges: Whether to require at least one edge

    Raises:
        EmptyOutputError: If graph has no nodes or edges (when required)
    """
    # Handle different key names
    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("links", graph_data.get("edges", []))

    if not nodes:
        raise EmptyOutputError(
            "Cannot save empty graph. No nodes were created. "
            "Check that the term list contains valid terms."
        )

    if require_edges and not edges:
        raise EmptyOutputError(
            "Graph has no edges. Try:\n"
            "  - Lowering the --threshold value\n"
            "  - Using --method relations instead of cooccurrence\n"
            "  - Checking that multiple terms appear in the corpus\n"
            f"  - Current graph has {len(nodes)} node(s) but no connections"
        )


def validate_networkx_graph(graph, require_edges: bool = True) -> None:
    """Validate that NetworkX graph is not empty.

    Args:
        graph: NetworkX graph object
        require_edges: Whether to require at least one edge

    Raises:
        EmptyOutputError: If graph has no nodes or edges (when required)
    """
    if graph.number_of_nodes() == 0:
        raise EmptyOutputError(
            "Cannot export empty graph. No nodes were created."
        )

    if require_edges and graph.number_of_edges() == 0:
        raise EmptyOutputError(
            f"Graph has {graph.number_of_nodes()} node(s) but no edges. "
            f"Try lowering the threshold or using a different method."
        )


def validate_concept_graph(graph, require_edges: bool = True) -> None:
    """Validate that ConceptGraph is not empty.

    Args:
        graph: ConceptGraph object with node_count() and edge_count() methods
        require_edges: Whether to require at least one edge

    Raises:
        EmptyOutputError: If graph has no nodes or edges (when required)
    """
    if graph.node_count() == 0:
        raise EmptyOutputError(
            "Cannot save empty graph. No nodes were created. "
            "Check that the term list contains valid terms."
        )

    if require_edges and graph.edge_count() == 0:
        raise EmptyOutputError(
            f"Graph has {graph.node_count()} node(s) but no edges. Try:\n"
            f"  - Lowering the --threshold value\n"
            f"  - Using --method relations instead of cooccurrence\n"
            f"  - Checking that multiple terms appear in the corpus"
        )


def validate_search_results(matches: list, query: str) -> None:
    """Validate that search results are not empty.

    Args:
        matches: List of search matches
        query: The search query

    Raises:
        EmptyOutputError: If no matches found
    """
    if not matches:
        raise EmptyOutputError(
            f"No matches found for query: '{query}'"
        )


def validate_csv_data(rows: list, file_type: str = "CSV") -> None:
    """Validate that CSV data is not empty.

    Args:
        rows: List of data rows (not including header)
        file_type: Type of file for error message

    Raises:
        EmptyOutputError: If no data rows to write
    """
    if not rows:
        raise EmptyOutputError(
            f"Cannot write {file_type} with no data rows. "
            f"Only headers would be written."
        )
