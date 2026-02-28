"""
Command-line interface for Concept Mapper.

Commands:
  ingest      Parse raw text files into a processed corpus JSON.
  rarities    Score and rank terms by rarity/significance across a corpus.
  search      Find sentences containing a term, with optional context.
  concordance Show a term in its surrounding text context (KWIC view).
  graph       Build a co-occurrence or relation graph from a corpus.
  export      Convert a graph file to D3, GraphML, CSV, GEXF, or HTML.
  diagram     Render a dependency parse tree for a sentence.
  analyze     Analyse a term's contextual terms across a windowed neighborhood.
  replace     Replace one term with another throughout a corpus.
"""

import click
import json
import sys
from pathlib import Path

# Import core functionality
from concept_mapper.corpus.loader import load_file, load_directory
from concept_mapper.corpus.models import ProcessedDocument
from concept_mapper.preprocessing.pipeline import preprocess
from concept_mapper.analysis.reference import load_reference_corpus
from concept_mapper.analysis.rarity import PhilosophicalTermScorer
from concept_mapper.analysis.cooccurrence import build_cooccurrence_matrix
from concept_mapper.analysis.relations import get_relations
from concept_mapper.search.find import find_sentences
from concept_mapper.search.concordance import concordance
from concept_mapper.terms.models import TermList
from concept_mapper.terms.manager import TermManager
from concept_mapper.graph import graph_from_cooccurrence, graph_from_relations
from concept_mapper.export import (
    export_d3_json,
    export_graphml,
    export_csv,
    export_gexf,
    generate_html,
)
from concept_mapper.validation import (
    validate_corpus,
    validate_term_list,
    validate_concept_graph,
)
from concept_mapper.storage.utils import derive_identifier, infer_output_path


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option(
    "--output-dir", "-o", type=click.Path(), default="output", help="Output directory"
)
@click.pass_context
def cli(ctx, verbose, output_dir):
    """
    Concept Mapper: Extract and visualize conceptual vocabularies from texts.

    A tool for analyzing philosophical texts to identify author-specific
    terminology and map relationships between concepts.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    ctx.obj["output_dir"] = Path(output_dir)
    ctx.obj["output_dir"].mkdir(parents=True, exist_ok=True)


# ============================================================================
# Ingest Command
# ============================================================================


@cli.command()
@click.argument("path", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), help="Output corpus file")
@click.option("--recursive", "-r", is_flag=True, help="Process directory recursively")
@click.option(
    "--pattern", "-p", default="*.txt", help="File pattern for recursive mode"
)
@click.option(
    "--clean-ocr",
    is_flag=True,
    help="Clean OCR/PDF artifacts (spacing, split words, page numbers)",
)
@click.option(
    "--toc",
    type=click.Path(exists=True),
    help="Table of contents file for guided structure detection",
)
@click.pass_context
def ingest(ctx, path, output, recursive, pattern, clean_ocr, toc):
    """
    Load and preprocess documents.

    PATH can be a file or directory. Processes documents through
    tokenization, POS tagging, and lemmatization.

    Use --clean-ocr for texts from PDFs with OCR errors (spacing issues,
    split words, page numbers, etc.).

    Use --toc to provide a table of contents file for accurate structure
    detection when automatic detection fails or is unreliable.

    Examples:
        cmapr ingest document.txt -o corpus.json
        cmapr ingest corpus/ -r -p "*.txt" -o corpus.json
        cmapr ingest scanned.txt --clean-ocr -o corpus.json
        cmapr ingest eco_spl.txt --toc eco_spl_toc.txt -o corpus.json
    """
    verbose = ctx.obj["verbose"]
    output_dir = ctx.obj["output_dir"]

    if verbose:
        click.echo(f"Loading documents from {path}...")

    # Load documents
    path = Path(path)
    if path.is_file():
        doc = load_file(path)
        docs = [doc]
    else:
        if not recursive:
            click.echo("Error: Directory requires --recursive flag", err=True)
            sys.exit(1)
        docs = load_directory(path, pattern=pattern)

    if verbose:
        click.echo(f"Loaded {len(docs)} document(s)")

    # Convert TOC path to Path object if provided
    toc_file = Path(toc) if toc else None

    # Preprocess
    if verbose:
        if clean_ocr and toc_file:
            click.echo(
                "Preprocessing documents (with OCR cleaning and TOC-guided structure detection)..."
            )
        elif clean_ocr:
            click.echo("Preprocessing documents (with OCR cleaning)...")
        elif toc_file:
            click.echo(
                "Preprocessing documents (with TOC-guided structure detection)..."
            )
        else:
            click.echo("Preprocessing documents...")

    processed = []
    with click.progressbar(docs, label="Processing") as bar:
        for doc in bar:
            processed.append(preprocess(doc, clean_ocr=clean_ocr, toc_file=toc_file))

    # Save
    if output:
        output_path = Path(output)
    else:
        # NEW: Derive output filename from input source
        output_path = infer_output_path(path, output_dir, "corpus")

    if verbose:
        identifier = derive_identifier(path)
        click.echo(f"Derived identifier: '{identifier}'")
        click.echo(f"Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize processed documents
    serialized = []
    for doc in processed:
        serialized.append(doc.to_dict())

    # Validate before saving
    validate_corpus(serialized)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serialized, f, indent=2, ensure_ascii=False)

    click.echo(f"✓ Saved {len(processed)} processed document(s) to {output_path}")


# ============================================================================
# Rarities Command
# ============================================================================


@cli.command()
@click.argument("corpus", type=click.Path(exists=True))
@click.option(
    "--method",
    "-m",
    type=click.Choice(["ratio", "tfidf", "neologism", "hybrid"]),
    default="hybrid",
    help="Detection method",
)
@click.option(
    "--threshold", "-t", type=float, default=2.0, help="Minimum score threshold"
)
@click.option(
    "--top-n", "-n", type=int, default=50, help="Number of top terms to output"
)
@click.option("--output", "-o", type=click.Path(), help="Output terms file (JSON)")
@click.pass_context
def rarities(ctx, corpus, method, threshold, top_n, output):
    """
    Detect rare/philosophical terms in corpus.

    Analyzes corpus to identify author-specific terminology using
    statistical rarity analysis.

    Examples:
        cmapr rarities corpus.json --method hybrid --top-n 30
        cmapr rarities corpus.json -o terms.json
    """
    verbose = ctx.obj["verbose"]
    output_dir = ctx.obj["output_dir"]

    # Load corpus
    if verbose:
        click.echo(f"Loading corpus from {corpus}...")

    from concept_mapper.corpus.models import ProcessedDocument

    with open(corpus, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [ProcessedDocument(**doc_data) for doc_data in data]

    if verbose:
        click.echo(f"Loaded {len(docs)} document(s)")

    # Load reference corpus
    if verbose:
        click.echo("Loading reference corpus...")

    reference = load_reference_corpus()

    # Detect terms
    if verbose:
        click.echo(f"Detecting rare terms (method={method})...")

    scorer = PhilosophicalTermScorer(docs, reference, use_lemmas=True)
    candidates = scorer.score_all(min_score=threshold, top_n=top_n)

    # Check if any terms were found
    if not candidates:
        click.echo("\nNo rare terms detected. Try:")
        click.echo("  - Lowering the --threshold value")
        click.echo("  - Using a different --method (hybrid, ratio, tfidf, neologism)")
        click.echo("  - Checking that the input text is substantial (>500 words)")
        click.echo("  - Verifying the text is in English")
        sys.exit(1)

    # Display results
    click.echo(f"\nTop {len(candidates)} rare terms:")
    click.echo("-" * 60)

    for term, score, components in candidates:
        click.echo(f"{term:30} {score:6.2f}")

    # Save if requested
    if output:
        output_path = Path(output)
    else:
        # NEW: Derive output filename from corpus filename
        output_path = infer_output_path(Path(corpus), output_dir, "terms")

    if verbose:
        identifier = derive_identifier(Path(corpus))
        click.echo(f"Derived identifier: '{identifier}'")
        click.echo(f"Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create term list
    term_data = [
        {"term": term, "metadata": {"score": score}} for term, score, _ in candidates
    ]

    # Validate before saving (should never fail here since we checked above)
    validate_term_list(term_data)

    term_list = TermList.from_dict({"terms": term_data})
    manager = TermManager(term_list)
    manager.export_to_json(output_path)

    click.echo(f"\n✓ Saved {len(candidates)} terms to {output_path}")


# ============================================================================
# Search Command
# ============================================================================


@cli.command()
@click.argument("corpus", type=click.Path(exists=True))
@click.argument("term")
@click.option(
    "--context", "-c", type=int, default=0, help="Number of context sentences"
)
@click.option(
    "--lemma",
    "-l",
    is_flag=True,
    help="Match lemmatized forms (e.g., 'run' matches 'running', 'ran')",
)
@click.option(
    "--diagram",
    "-d",
    is_flag=True,
    help="Generate sentence diagrams for all matches",
)
@click.option(
    "--diagram-format",
    type=click.Choice(["ascii", "table", "tree"]),
    default="ascii",
    help="Diagram output format (requires --diagram)",
)
@click.option(
    "--extract-significant",
    "-e",
    is_flag=True,
    help="Extract significant terms from matching sentences",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.1,
    help="Minimum significance score (requires --extract-significant, default: 0.1)",
)
@click.option(
    "--pos",
    "-p",
    type=click.Choice(["nouns", "verbs", "adjectives", "adverbs"]),
    multiple=True,
    help="POS types to extract (can specify multiple). Defaults to nouns and verbs.",
)
@click.option(
    "--top-n",
    type=int,
    help="Maximum terms per sentence (requires --extract-significant)",
)
@click.option(
    "--aggregate",
    "-a",
    is_flag=True,
    help="Show aggregated results across all sentences (requires --extract-significant)",
)
@click.option(
    "--detailed",
    is_flag=True,
    help="Show detailed output with scores (requires --extract-significant)",
)
@click.option(
    "--scoring-mode",
    type=click.Choice(["corpus_frequency", "hybrid"]),
    default="corpus_frequency",
    help="Scoring method: corpus_frequency (default, main content words) or hybrid (rare/distinctive terms)",
)
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.option(
    "--start-from-section",
    type=str,
    default=None,
    help=(
        "Skip content before this section number. "
        "E.g. '1' excludes front-matter before chapter 1."
    ),
)
@click.option(
    "--exclude-sections",
    type=str,
    default=None,
    help=(
        "Exclude sections whose title matches this regex (case-insensitive). "
        "E.g. 'index|bibliography' to skip back-matter."
    ),
)
@click.pass_context
def search(
    ctx,
    corpus,
    term,
    context,
    lemma,
    diagram,
    diagram_format,
    extract_significant,
    threshold,
    pos,
    top_n,
    aggregate,
    detailed,
    scoring_mode,
    output,
    start_from_section,
    exclude_sections,
):
    """
    Search for term occurrences in corpus.

    Examples:
        cmapr search corpus.json "consciousness"
        cmapr search corpus.json "being" --context 2
        cmapr search corpus.json "run" --lemma
        cmapr search corpus.json "intentionality" --diagram
        cmapr search corpus.json "dialectic" --diagram --diagram-format tree
        cmapr search corpus.json "intentionality" --extract-significant --threshold 1.5
        cmapr search corpus.json "capitalism" -e -p nouns -p verbs --top-n 5
        cmapr search corpus.json "consciousness" -e --aggregate --detailed
    """
    verbose = ctx.obj["verbose"]

    # Load corpus
    from concept_mapper.corpus.models import ProcessedDocument

    with open(corpus, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [ProcessedDocument.from_dict(doc_data) for doc_data in data]

    if verbose:
        search_type = "lemma-based" if lemma else "exact"
        click.echo(
            f"Searching ({search_type}) for '{term}' in {len(docs)} document(s)..."
        )

    # Handle extract-significant mode
    if extract_significant:
        from concept_mapper.search.extract import (
            extract_significant_terms,
            format_results_by_sentence,
            format_results_detailed,
            aggregate_across_sentences,
        )

        if verbose:
            click.echo(f"Extracting significant terms (threshold: {threshold})...")

        # Convert pos tuple to list (None if empty)
        pos_types = list(pos) if pos else None

        # Extract significant terms
        results = extract_significant_terms(
            term,
            docs,
            threshold=threshold,
            pos_types=pos_types,
            top_n=top_n,
            match_lemma=lemma,
            scoring_mode=scoring_mode,
        )
        results = _filter_significant_results(
            results, start_from_section, exclude_sections
        )

        if not results:
            click.echo(f"No significant terms found in sentences containing '{term}'")
            return

        click.echo(
            f"\nFound significant terms in {len(results)} sentence(s) containing '{term}':"
        )
        click.echo("=" * 70)

        # Display results based on flags
        if aggregate:
            # Show aggregated results across all sentences
            aggregated = aggregate_across_sentences(results, top_n=top_n)
            click.echo("\nAggregated significant terms (avg score, count):")
            for term_name, avg_score, count in aggregated:
                click.echo(
                    f"  • {term_name}: {avg_score:.2f} (appears in {count} sentence(s))"
                )

        if detailed:
            # Show detailed output with full context
            click.echo(format_results_detailed(results))
        elif not aggregate:
            # Show simple output grouped by sentence
            click.echo("\n" + format_results_by_sentence(results, show_scores=detailed))

        # Save if requested
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                if aggregate:
                    f.write(f"Aggregated significant terms for '{term}':\n")
                    f.write("=" * 70 + "\n\n")
                    aggregated = aggregate_across_sentences(results, top_n=top_n)
                    for term_name, avg_score, count in aggregated:
                        f.write(
                            f"{term_name}: {avg_score:.2f} (appears in {count} sentence(s))\n"
                        )
                elif detailed:
                    f.write(format_results_detailed(results))
                else:
                    f.write(format_results_by_sentence(results, show_scores=True))

            click.echo(f"\n✓ Saved results to {output_path}")

        return

    # Search
    matches = find_sentences(term, docs, match_lemma=lemma)
    matches = _filter_sentence_matches(matches, start_from_section, exclude_sections)

    if not matches:
        click.echo(f"No matches found for '{term}'")
        return

    # Display results
    click.echo(f"\nFound {len(matches)} occurrence(s) of '{term}':")
    click.echo("=" * 70)

    # Generate diagrams if requested
    if diagram:
        from concept_mapper.syntax.diagram import diagram_sentence

        if verbose:
            click.echo("Generating sentence diagrams...")

        for i, match in enumerate(matches, 1):
            click.echo(
                f"\n[{i}] {match.doc_id or 'document'} (sentence {match.sent_index}):"
            )
            click.echo(f"    {match.sentence}")
            click.echo("\n    Diagram:")

            try:
                result = diagram_sentence(match.sentence, output_format=diagram_format)
                # Indent diagram output
                for line in result.split("\n"):
                    click.echo(f"    {line}")
            except Exception as e:
                click.echo(f"    ⚠ Could not diagram sentence: {e}")

            click.echo()  # Blank line between entries
    else:
        for i, match in enumerate(matches, 1):
            click.echo(
                f"\n[{i}] {match.doc_id or 'document'} (sentence {match.sent_index}):"
            )

            if context > 0:
                # Get context from document
                doc = next(
                    d for d in docs if d.metadata.get("source_path") == match.doc_id
                )

                start = max(0, match.sent_index - context)
                end = min(len(doc.sentences), match.sent_index + context + 1)

                for j in range(start, end):
                    prefix = ">>> " if j == match.sent_index else "    "
                    click.echo(f"{prefix}{doc.sentences[j]}")
            else:
                click.echo(f"    {match.sentence}")

    # Save if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            if diagram:
                from concept_mapper.syntax.diagram import diagram_sentence

                for i, match in enumerate(matches, 1):
                    f.write(
                        f"[{i}] {match.doc_id or 'document'} (sentence {match.sent_index}):\n"
                    )
                    f.write(f"{match.sentence}\n\n")
                    f.write("Diagram:\n")
                    try:
                        result = diagram_sentence(
                            match.sentence, output_format=diagram_format
                        )
                        f.write(result)
                        f.write("\n")
                    except Exception as e:
                        f.write(f"⚠ Could not diagram sentence: {e}\n")
                    f.write("\n" + "=" * 70 + "\n\n")
            else:
                for match in matches:
                    f.write(f"{match.sentence}\n")

        click.echo(f"\n✓ Saved {len(matches)} matches to {output_path}")


# ============================================================================
# Concordance Command
# ============================================================================


@cli.command()
@click.argument("corpus", type=click.Path(exists=True))
@click.argument("term")
@click.option("--width", "-w", type=int, default=50, help="Context width in characters")
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def concordance_cmd(ctx, corpus, term, width, output):
    """
    Display KWIC (Key Word In Context) concordance.

    Examples:
        cmapr concordance corpus.json "consciousness"
        cmapr concordance corpus.json "being" --width 80
    """
    verbose = ctx.obj["verbose"]

    # Load corpus
    from concept_mapper.corpus.models import ProcessedDocument

    with open(corpus, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [ProcessedDocument(**doc_data) for doc_data in data]

    if verbose:
        click.echo(f"Building concordance for '{term}'...")

    # Generate concordance
    lines = concordance(term, docs, width=width)

    if not lines:
        click.echo(f"No matches found for '{term}'")
        return

    # Display
    click.echo(f"\nKWIC Concordance for '{term}' ({len(lines)} occurrences):")
    click.echo("=" * (width * 2 + 20))

    for line in lines:
        click.echo(
            f"{line.left_context:>{width}} | {line.keyword} | {line.right_context:<{width}}"
        )

    # Save if requested
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for line in lines:
                f.write(
                    f"{line.left_context:>{width}} | {line.keyword} | {line.right_context:<{width}}\n"
                )

        click.echo(f"\n✓ Saved concordance to {output_path}")


# ============================================================================
# Graph Command
# ============================================================================


@cli.command()
@click.argument("corpus", type=click.Path(exists=True))
@click.option(
    "--terms",
    "-t",
    type=click.Path(exists=True),
    required=True,
    help="Terms file (JSON)",
)
@click.option(
    "--method",
    "-m",
    type=click.Choice(["cooccurrence", "relations"]),
    default="cooccurrence",
    help="Graph construction method",
)
@click.option("--threshold", type=float, default=0.3, help="Edge weight threshold")
@click.option("--output", "-o", type=click.Path(), help="Output graph file")
@click.pass_context
def graph(ctx, corpus, terms, method, threshold, output):
    """
    Build concept graph from corpus.

    Examples:
        cmapr graph corpus.json -t terms.json -m cooccurrence
        cmapr graph corpus.json -t terms.json -m relations -o graph.json
    """
    verbose = ctx.obj["verbose"]
    output_dir = ctx.obj["output_dir"]

    # Load corpus
    from concept_mapper.corpus.models import ProcessedDocument

    with open(corpus, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [ProcessedDocument(**doc_data) for doc_data in data]

    if verbose:
        click.echo(f"Loaded {len(docs)} document(s)")

    # Load terms
    manager = TermManager()
    manager.import_from_json(Path(terms))
    term_list = manager.term_list

    if verbose:
        click.echo(f"Loaded {len(term_list)} terms")

    # Build graph
    if method == "cooccurrence":
        if verbose:
            click.echo("Building co-occurrence matrix...")

        matrix = build_cooccurrence_matrix(
            term_list, docs, method="pmi", window="sentence"
        )

        if verbose:
            click.echo(f"Building graph (threshold={threshold})...")

        concept_graph = graph_from_cooccurrence(matrix, threshold=threshold)

    else:  # relations
        if verbose:
            click.echo("Extracting relations...")

        all_relations = []
        with click.progressbar(term_list, label="Extracting") as bar:
            for term_entry in bar:
                relations = get_relations(term_entry.term, docs)
                all_relations.extend(relations)

        if verbose:
            click.echo(f"Found {len(all_relations)} relations")
            click.echo("Building graph...")

        concept_graph = graph_from_relations(all_relations)

        # Connect any isolated nodes via co-occurrence fallback
        from concept_mapper.graph.operations import find_isolated_nodes, connect_isolated_nodes
        isolated = find_isolated_nodes(concept_graph)
        if isolated:
            if verbose:
                click.echo(f"Found {len(isolated)} isolated node(s), building co-occurrence fallback...")
            fallback_matrix = build_cooccurrence_matrix(
                term_list, docs, method="pmi", window="sentence"
            )
            connected = connect_isolated_nodes(concept_graph, fallback_matrix)
            if verbose:
                click.echo(f"Connected {connected}/{len(isolated)} isolated node(s)")

    click.echo(
        f"\n✓ Graph: {concept_graph.node_count()} nodes, {concept_graph.edge_count()} edges"
    )

    # Validate before saving
    validate_concept_graph(concept_graph)

    # Save
    if output:
        output_path = Path(output)
    else:
        # NEW: Derive output filename from corpus filename with optional method suffix
        suffix = f"_{method}" if method != "cooccurrence" else ""
        output_path = infer_output_path(
            Path(corpus), output_dir, "graphs", suffix=suffix
        )

    if verbose:
        identifier = derive_identifier(Path(corpus))
        click.echo(f"Derived identifier: '{identifier}'")
        click.echo(f"Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Export to D3 JSON
    export_d3_json(concept_graph, output_path)

    click.echo(f"✓ Saved graph to {output_path}")


# ============================================================================
# Export Command
# ============================================================================


@cli.command()
@click.argument("graph_file", type=click.Path(exists=True))
@click.option(
    "--format",
    "-f",
    type=click.Choice(["d3", "html", "graphml", "csv", "gexf"]),
    default="html",
    help="Export format",
)
@click.option("--output", "-o", type=click.Path(), help="Output path")
@click.option(
    "--title",
    type=str,
    default="Concept Network",
    help="Visualization title (for HTML)",
)
@click.pass_context
def export(ctx, graph_file, format, output, title):
    """
    Export graph to various formats.

    Examples:
        cmapr export graph.json --format html -o viz/
        cmapr export graph.json --format graphml -o graph.graphml
        cmapr export graph.json --format csv -o output/
    """
    verbose = ctx.obj["verbose"]
    output_dir = ctx.obj["output_dir"]

    # Load graph from D3 JSON
    from concept_mapper.export import load_d3_json

    if verbose:
        click.echo(f"Loading graph from {graph_file}...")

    # Since we save graphs as D3 JSON, we need to reconstruct the graph
    # This is a simplified approach - in practice you might want to save the graph object itself
    data = load_d3_json(Path(graph_file))

    # Reconstruct graph
    from concept_mapper.graph import ConceptGraph

    # Check if it's directed by looking at the data
    concept_graph = ConceptGraph(directed=False)

    # Add nodes
    for node_data in data["nodes"]:
        concept_graph.add_node(
            node_data["id"], **{k: v for k, v in node_data.items() if k != "id"}
        )

    # Add edges
    for link_data in data["links"]:
        concept_graph.add_edge(
            link_data["source"],
            link_data["target"],
            **{k: v for k, v in link_data.items() if k not in ["source", "target"]},
        )

    if verbose:
        click.echo(
            f"Loaded graph: {concept_graph.node_count()} nodes, {concept_graph.edge_count()} edges"
        )

    # Export
    if output:
        output_path = Path(output)
    else:
        # NEW: Derive output path from graph filename
        identifier = derive_identifier(Path(graph_file))
        exports_dir = output_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        if format == "html":
            output_path = exports_dir / identifier
        elif format == "csv":
            output_path = exports_dir / identifier / "csv"
        else:
            output_path = exports_dir / f"{identifier}.{format}"

    if verbose:
        identifier = derive_identifier(Path(graph_file))
        click.echo(f"Derived identifier: '{identifier}'")
        click.echo(f"Output: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if verbose:
        click.echo(f"Exporting to {format}...")

    if format == "d3":
        export_d3_json(concept_graph, output_path)
        click.echo(f"✓ Exported to {output_path}")

    elif format == "html":
        html_path = generate_html(concept_graph, output_path, title=title)
        click.echo(f"✓ Generated HTML visualization at {html_path}")
        click.echo(f"  Open in browser: file://{html_path.absolute()}")

    elif format == "graphml":
        export_graphml(concept_graph, output_path)
        click.echo(f"✓ Exported to {output_path}")

    elif format == "csv":
        export_csv(concept_graph, output_path)
        click.echo(f"✓ Exported to {output_path}/nodes.csv and {output_path}/edges.csv")

    elif format == "gexf":
        export_gexf(concept_graph, output_path)
        click.echo(f"✓ Exported to {output_path}")


# ============================================================================
# Diagram Command
# ============================================================================


@cli.command()
@click.argument("sentence")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["ascii", "table", "tree"]),
    default="ascii",
    help="Output format",
)
@click.option("--output", "-o", type=click.Path(), help="Save to file")
@click.pass_context
def diagram(ctx, sentence, format, output):
    """
    Create a dependency parse diagram of a sentence.

    Uses Stanza to perform deep syntactic analysis and visualizes
    the grammatical structure.

    Examples:
        cmapr diagram "The cat sat on the mat."
        cmapr diagram "Intentionality obscures social processes." --format tree
        cmapr diagram "The question is complex." -o diagram.txt
    """
    from concept_mapper.syntax.diagram import diagram_sentence, save_diagram

    verbose = ctx.obj.get("verbose", False)

    if verbose:
        click.echo("Parsing sentence...")

    if output:
        save_diagram(sentence, Path(output), output_format=format)
        click.echo(f"✓ Diagram saved to {output}")
    else:
        result = diagram_sentence(sentence, output_format=format)
        click.echo(result)


# ============================================================================
# Analyze Context Command
# ============================================================================


def _get_pos_label(pos_tag):
    """
    Convert Penn Treebank POS tag to readable label.

    Args:
        pos_tag: Penn Treebank POS tag (e.g., NN, VBD, JJ)

    Returns:
        Readable POS label (e.g., noun, verb, adj)
    """
    if pos_tag.startswith("NN"):
        return "noun"
    elif pos_tag.startswith("VB"):
        return "verb"
    elif pos_tag.startswith("JJ"):
        return "adj"
    elif pos_tag.startswith("RB"):
        return "adv"
    elif pos_tag.startswith("IN"):
        return "prep"
    elif pos_tag.startswith("DT"):
        return "det"
    elif pos_tag.startswith("PR"):
        return "pron"
    elif pos_tag.startswith("CD"):
        return "num"
    elif pos_tag == "CC":
        return "conj"
    else:
        return "other"


def _build_hierarchical_path(loc, level, structure_nodes=None):
    """
    Build a hierarchical path string for a location.

    Reconstructs the full hierarchy based on section numbering (e.g., "1.5.1"
    belongs to section "1.5" which belongs to chapter "1"), looking up titles
    from structure nodes when available.

    Args:
        loc: SentenceLocation object
        level: Display level (chapter, section, subsection)
        structure_nodes: Optional list of StructureNode objects for title lookup

    Returns:
        Hierarchical path string like "1. Signs / 1.5. Title / 1.5.1. Subtitle"
    """
    # Build a lookup map from structure nodes if provided
    node_map = {}
    if structure_nodes:
        for node in structure_nodes:
            node_map[node.number] = node

    parts = []

    if level == "level_0":
        # Just top level
        if loc.chapter:
            title = loc.chapter_title or f"Level 0 {loc.chapter}"
            parts.append(f"{loc.chapter}. {title}" if loc.chapter_title else title)

    elif level == "level_1":
        # Level 0 / Level 1
        if loc.section:
            # Extract parent number from section (e.g., "1.5" -> "1")
            parent_num = (
                loc.section.split(".")[0] if "." in loc.section else loc.section
            )

            # Add parent level
            if parent_num in node_map:
                parent_title = node_map[parent_num].title
                parts.append(f"{parent_num}. {parent_title}")
            elif loc.chapter == parent_num and loc.chapter_title:
                parts.append(f"{parent_num}. {loc.chapter_title}")
            else:
                parts.append(f"Level 0 {parent_num}")

            # Add current level
            section_title = loc.section_title or f"Level 1 {loc.section}"
            parts.append(
                f"{loc.section}. {section_title}"
                if loc.section_title
                else section_title
            )

    elif level == "level_2":
        # Level 0 / Level 1 / Level 2
        if loc.subsection:
            # Extract parent numbers from subsection (e.g., "1.5.1" -> "1", "1.5")
            parts_nums = loc.subsection.split(".")
            if len(parts_nums) >= 3:
                level0_num = parts_nums[0]
                level1_num = f"{parts_nums[0]}.{parts_nums[1]}"

                # Add level 0
                if level0_num in node_map:
                    level0_title = node_map[level0_num].title
                    parts.append(f"{level0_num}. {level0_title}")
                else:
                    parts.append(f"Level 0 {level0_num}")

                # Add level 1
                if level1_num in node_map:
                    level1_title = node_map[level1_num].title
                    parts.append(f"{level1_num}. {level1_title}")
                else:
                    parts.append(f"Level 1 {level1_num}")

                # Add level 2
                level2_title = loc.subsection_title or f"Level 2 {loc.subsection}"
                parts.append(
                    f"{loc.subsection}. {level2_title}"
                    if loc.subsection_title
                    else level2_title
                )

    return " / ".join(parts) if parts else "Unstructured Content"


def _get_structure_depth_mapping(structure_nodes):
    """
    Build a mapping from tree depth to structure level names.

    Args:
        structure_nodes: List of StructureNode objects

    Returns:
        Dict mapping depth (0, 1, 2...) to level names (e.g., "chapter", "section")
        Returns None if no structure
    """
    if not structure_nodes:
        return None

    # Determine hierarchy by analyzing number patterns
    # e.g., "1" is depth 0, "1.2" is depth 1, "1.2.3" is depth 2
    levels_by_depth = {}

    for node in structure_nodes:
        depth = node.number.count(".")  # "1" = 0, "1.2" = 1, "1.2.3" = 2
        if depth not in levels_by_depth:
            levels_by_depth[depth] = node.level

    return levels_by_depth


def _get_location_field(loc, level_name):
    """
    Get the appropriate number and title from a location based on level name.

    Args:
        loc: SentenceLocation object
        level_name: Generic level name (e.g., "level_0", "level_1", "level_2")

    Returns:
        Tuple of (number, title) or (None, None) if not found
    """
    # Map generic level names to SentenceLocation fields
    # Note: SentenceLocation fields are named chapter/section/subsection for legacy reasons,
    # but they represent arbitrary depth levels (0, 1, 2, 3)
    level_mapping = {
        "level_0": (loc.chapter, loc.chapter_title),
        "level_1": (loc.section, loc.section_title),
        "level_2": (loc.subsection, loc.subsection_title),
        "level_3": (loc.paragraph, None),
    }

    return level_mapping.get(level_name, (None, None))


def _group_relations_by_structure(relations, level=1, structure_nodes=None):
    """
    Group relations by document structure using tree depth levels.

    Args:
        relations: List of ContextualRelation objects
        level: Integer tree depth (0=no grouping, 1=top level, 2=second level, etc.)
        structure_nodes: Optional list of StructureNode objects for hierarchical path lookup

    Returns:
        Dictionary mapping structure keys to lists of relations
    """
    from collections import defaultdict

    grouped = defaultdict(
        lambda: {"relations": [], "title": "", "path": "", "parent": None}
    )

    # Level 0 means no grouping - put everything together
    if level == 0:
        for rel in relations:
            grouped["all"]["relations"].append(rel)
            grouped["all"]["title"] = "All Content"
            grouped["all"]["path"] = "All Content"
        return grouped

    # Build depth mapping from structure nodes
    depth_mapping = _get_structure_depth_mapping(structure_nodes)

    if not depth_mapping:
        # No structure available - put everything in unstructured
        for rel in relations:
            grouped["unstructured"]["relations"].append(rel)
            grouped["unstructured"]["title"] = "Unstructured Content"
            grouped["unstructured"]["path"] = "Unstructured Content"
        return grouped

    # Map requested level to actual structure level name
    # Level 1 = depth 0 (top level), Level 2 = depth 1, etc.
    target_depth = level - 1
    target_level_name = depth_mapping.get(target_depth)

    if not target_level_name:
        # Requested level doesn't exist - use deepest available level
        max_depth = max(depth_mapping.keys())
        target_depth = max_depth
        target_level_name = depth_mapping[max_depth]

    for rel in relations:
        if not rel.evidence_locations:
            # No structure info - put in "unstructured" group
            key = "unstructured"
            grouped[key]["relations"].append(rel)
            grouped[key]["title"] = "Unstructured Content"
            continue

        # Use first location for grouping
        loc = rel.evidence_locations[0]

        # Get the appropriate field from location based on target level
        number, title = _get_location_field(loc, target_level_name)

        if number:
            key = f"{number}"
            grouped[key]["relations"].append(rel)
            grouped[key]["title"] = title or f"{target_level_name.title()} {number}"
            grouped[key]["path"] = _build_hierarchical_path(
                loc, target_level_name, structure_nodes
            )
        else:
            # Fallback to unstructured
            key = "unstructured"
            grouped[key]["relations"].append(rel)
            grouped[key]["title"] = "Unstructured Content"
            grouped[key]["path"] = "Unstructured Content"

    return grouped


def _display_structured_text(term, grouped_relations, docs, verbose=False):
    """
    Display relations grouped by structure in text format.

    Args:
        term: Search term
        grouped_relations: Dictionary from _group_relations_by_structure
        docs: List of ProcessedDocument objects (for structure info)
        verbose: If True, show detailed output with scores and evidence
    """
    from collections import defaultdict

    click.echo(f"\nFound contextual relations for '{term}'")
    click.echo("=" * 80)

    # Sort groups by key
    sorted_groups = sorted(grouped_relations.items())

    for group_key, group_data in sorted_groups:
        path = group_data.get("path", group_data["title"])
        relations = group_data["relations"]

        if not relations:
            continue

        # Display group header with hierarchical path
        click.echo(f"\n{'=' * 80}")
        click.echo(f"{path}")
        click.echo(f"{'=' * 80}")

        # Group relations by type
        by_type = defaultdict(list)
        for rel in relations:
            by_type[rel.relation_type].append(rel)

        # Display each type
        if verbose:
            # Verbose mode: show detailed output with scores and evidence
            for rel_type in ["svo", "copular", "prep", "cooccurrence"]:
                if rel_type not in by_type:
                    continue

                rels = sorted(by_type[rel_type], key=lambda r: r.score, reverse=True)[
                    :10
                ]

                click.echo(f"\n  {rel_type.upper()} Relations ({len(rels)} shown):")
                click.echo("  " + "-" * 76)

                for i, rel in enumerate(rels, 1):
                    evidence_info = f"{len(rel.evidence)} occurrence(s)"

                    # Show first location if available
                    loc_str = ""
                    if rel.evidence_locations:
                        loc = rel.evidence_locations[0]
                        if loc.section:
                            loc_str = f" [{loc.section}]"
                        elif loc.chapter:
                            loc_str = f" [{loc.chapter}]"

                    click.echo(
                        f"    {i}. {rel.source} → {rel.target} "
                        f"(score: {rel.score:.2f}, {evidence_info}){loc_str}"
                    )

                    # Show first evidence sentence (full text)
                    if rel.evidence:
                        click.echo(f'       "{rel.evidence[0]}"')
        else:
            # Minimal mode: list significant terms with POS tags
            # Build term -> POS mapping from documents
            term_pos = {}
            for doc in docs:
                for i, (token, pos) in enumerate(doc.pos_tags):
                    term = doc.lemmas[i] if i < len(doc.lemmas) else token
                    if term not in term_pos:
                        term_pos[term] = pos

            # Collect terms with their POS tags
            term_pos_list = []
            for rel in relations:
                pos = term_pos.get(rel.target, "UNK")
                # Convert Penn Treebank tags to readable labels
                pos_label = _get_pos_label(pos)
                term_pos_list.append((pos_label, rel.target))

            if term_pos_list:
                # Group by POS for better organization
                from collections import defaultdict

                by_pos = defaultdict(list)
                for pos_label, term in term_pos_list:
                    by_pos[pos_label].append(term)

                click.echo("\n  Significant terms:")
                for pos_label in sorted(by_pos.keys()):
                    terms = sorted(set(by_pos[pos_label]))
                    for term in terms:
                        click.echo(f"    {pos_label}: {term}")


def _get_structure_summary(docs):
    """
    Get summary of document structure.

    Args:
        docs: List of ProcessedDocument objects

    Returns:
        Dictionary with structure information
    """
    has_structure = any(doc.structure_nodes for doc in docs)

    if not has_structure:
        return {"has_structure": False}

    # Collect all structure nodes
    all_chapters = []
    all_sections = []

    for doc in docs:
        for node in doc.structure_nodes:
            if node.level == "chapter":
                all_chapters.append(
                    {
                        "number": node.number,
                        "title": node.title,
                    }
                )
            elif node.level == "section":
                all_sections.append(
                    {
                        "number": node.number,
                        "title": node.title,
                    }
                )

    return {
        "has_structure": True,
        "num_chapters": len(all_chapters),
        "num_sections": len(all_sections),
        "chapters": all_chapters,
    }


def _location_passes_filters(location, start_section=None, exclude_pattern=None):
    """
    Return True if a sentence location passes the section filters.

    Args:
        location: SentenceLocation or None
        start_section: Chapter number string; exclude sentence if chapter < this value
        exclude_pattern: Regex string; exclude if any title field matches (case-insensitive)
    """
    import re

    if location is None:
        return True

    if start_section is not None:
        chapter = location.chapter
        if chapter is None:
            # No chapter label means pre-chapter content (front matter); exclude it
            return False
        try:
            if float(chapter) < float(start_section):
                return False
        except ValueError:
            pass  # Non-numeric chapter numbers pass through

    if exclude_pattern is not None:
        titles = [
            location.chapter_title,
            location.section_title,
            location.subsection_title,
        ]
        for title in titles:
            if title and re.search(exclude_pattern, title, re.IGNORECASE):
                return False

    return True


def _filter_sentence_matches(matches, start_section=None, exclude_pattern=None):
    """Filter a list of SentenceMatch objects by section filters."""
    if start_section is None and exclude_pattern is None:
        return matches
    return [
        m
        for m in matches
        if _location_passes_filters(m.location, start_section, exclude_pattern)
    ]


def _filter_relations(relations, start_section=None, exclude_pattern=None):
    """Filter a list of ContextualRelation objects by section filters."""
    if start_section is None and exclude_pattern is None:
        return relations
    filtered = []
    for rel in relations:
        loc = rel.evidence_locations[0] if rel.evidence_locations else None
        if _location_passes_filters(loc, start_section, exclude_pattern):
            filtered.append(rel)
    return filtered


def _filter_significant_results(results, start_section=None, exclude_pattern=None):
    """Filter a list of SignificantTermsResult objects by section filters."""
    if start_section is None and exclude_pattern is None:
        return results
    return [
        r
        for r in results
        if _location_passes_filters(
            r.sentence_match.location, start_section, exclude_pattern
        )
    ]


def _parse_window(window_str):
    """Parse window string like 's0', 's1', 'p1' into (entity_type, radius)."""
    if not window_str or len(window_str) < 2:
        raise click.BadParameter(
            "Window format: entity type + radius number (e.g. 's0', 's1', 'p0')"
        )
    entity = window_str[0].lower()
    if entity not in ("s", "p"):
        raise click.BadParameter(
            f"Entity type must be 's' (sentence) or 'p' (paragraph), got '{entity}'"
        )
    try:
        radius = int(window_str[1:])
    except ValueError:
        raise click.BadParameter(
            f"Radius must be an integer, e.g. 's0' or 's1'. Got: '{window_str[1:]}'"
        )
    if radius < 0:
        raise click.BadParameter("Radius must be >= 0")
    return entity, radius


def _build_sentence_path(location, structure_nodes=None):
    """Build full hierarchical path for a sentence location."""
    if location is None:
        return "unstructured"
    if location.subsection:
        return _build_hierarchical_path(location, "level_2", structure_nodes)
    elif location.section:
        return _build_hierarchical_path(location, "level_1", structure_nodes)
    elif location.chapter:
        return _build_hierarchical_path(location, "level_0", structure_nodes)
    return "unstructured"


def _offset_label(offset, radius):
    """Generate slot label for a window offset."""
    if offset == 0:
        return "current:"
    elif offset < 0:
        return (
            "previous:" if radius <= 1 or abs(offset) == 1 else f"prev {abs(offset)}:"
        )
    else:
        return "next:" if radius <= 1 or offset == 1 else f"next {offset}:"


def _compute_window_slots(match, doc, entity_type, radius):
    """
    Compute window slots as list of (offset, [sentences]) pairs.

    For entity_type='s': each slot is a single sentence.
    For entity_type='p': each slot is an entire paragraph.
    """
    total = len(doc.sentences)

    if entity_type == "s":
        slots = []
        for offset in range(-radius, radius + 1):
            idx = match.sent_index + offset
            sentences = [doc.sentences[idx]] if 0 <= idx < total else []
            slots.append((offset, sentences))
        return slots

    # Paragraph mode
    if not doc.paragraph_indices or match.sent_index >= len(doc.paragraph_indices):
        # Fall back to sentence mode
        return _compute_window_slots(match, doc, "s", radius)

    match_para = doc.paragraph_indices[match.sent_index]
    unique_paras = sorted(set(doc.paragraph_indices[:total]))

    try:
        match_pos = unique_paras.index(match_para)
    except ValueError:
        return [(0, [doc.sentences[match.sent_index]])]

    slots = []
    for offset in range(-radius, radius + 1):
        target_pos = match_pos + offset
        if 0 <= target_pos < len(unique_paras):
            target_para = unique_paras[target_pos]
            sentences = [
                doc.sentences[i]
                for i in range(total)
                if i < len(doc.paragraph_indices)
                and doc.paragraph_indices[i] == target_para
            ]
        else:
            sentences = []
        slots.append((offset, sentences))
    return slots


def _display_window_analysis(
    term,
    docs,
    entity_type,
    radius,
    pos_types,
    threshold,
    top_n,
    lemma,
    start_section=None,
    exclude_sections=None,
):
    """Display per-occurrence window analysis for a search term."""
    from concept_mapper.search.find import find_sentences
    from concept_mapper.search.extract import (
        extract_terms_from_sentence_set,
        build_corpus_term_freqs,
    )

    matches = find_sentences(term, docs, match_lemma=lemma)
    matches = _filter_sentence_matches(matches, start_section, exclude_sections)
    if not matches:
        click.echo(f"No occurrences of '{term}' found")
        return

    pos_types_list = list(pos_types) if pos_types else None

    # Build corpus frequency map once — reused for every window call
    term_freqs, max_freq = build_corpus_term_freqs(docs, pos_types_list)
    structure_nodes = (
        docs[0].structure_nodes if docs and docs[0].structure_nodes else None
    )

    entity_name = "sentence" if entity_type == "s" else "paragraph"
    click.echo(
        f"\nWindow analysis for '{term}' "
        f"({entity_name} ±{radius}, {len(matches)} occurrence(s)):"
    )

    # Build a doc_id → doc map for fast lookup
    doc_map = {d.metadata.get("source_path", f"doc_{i}"): d for i, d in enumerate(docs)}

    last_path = None
    for match in matches:
        path = _build_sentence_path(match.location, structure_nodes)
        if path != last_path:
            click.echo("\n" + "=" * 80)
            click.echo(f"path: {path}")
            last_path = path
        else:
            click.echo("\n" + "-" * 80)
        click.echo(f'sentence: "{match.sentence.strip()}"')
        from concept_mapper.syntax.diagram import parse_sentence, format_as_tree

        doc = parse_sentence(match.sentence.strip())
        for sent in doc.sentences:
            click.echo("\nsentence diagram: " + format_as_tree(sent))
        click.echo("\nsignificant terms:")

        doc = doc_map.get(match.doc_id, docs[0] if docs else None)
        if not doc:
            continue

        slots = _compute_window_slots(match, doc, entity_type, radius)

        for offset, sentences in slots:
            label = _offset_label(offset, radius)
            click.echo(f"\n{label}")
            if not sentences:
                continue

            terms = extract_terms_from_sentence_set(
                sentences,
                pos_types=pos_types_list,
                threshold=threshold,
                top_n=top_n,
                exclude_term=term,
                term_freqs=term_freqs,
                max_freq=max_freq,
            )

            if not terms:
                click.echo("  (no significant terms)")
            else:
                for t, pos_label in terms:
                    click.echo(f'"{t}" {pos_label}')


@cli.command(name="analyze")
@click.argument("corpus", type=click.Path(exists=True))
@click.argument("term")
@click.option(
    "--top-n",
    "-n",
    type=int,
    help="Limit to top N most significant terms per context",
)
@click.option(
    "--threshold",
    "-t",
    type=float,
    default=0.1,
    help="Minimum significance score (default: 0.1)",
)
@click.option(
    "--pos",
    "-p",
    type=click.Choice(["nouns", "verbs", "adjectives", "adverbs"]),
    multiple=True,
    help="POS types to extract (can specify multiple). Defaults to nouns and verbs.",
)
@click.option(
    "--lemma",
    "-l",
    is_flag=True,
    help="Match lemmatized forms (e.g., 'run' matches 'running', 'ran')",
)
@click.option(
    "--no-relations",
    is_flag=True,
    help="Skip grammatical relation extraction (faster, only co-occurrence)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["text", "json", "csv", "graph"]),
    default="text",
    help="Output format",
)
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.option(
    "--group-by",
    "-g",
    type=click.IntRange(min=0),
    default=1,
    help="Group by structure tree level: 0=none, 1=top level, 2=second level, etc. (default: 1)",
)
@click.option(
    "--show-structure",
    is_flag=True,
    help="Display full document structure outline",
)
@click.option(
    "--verbose",
    "-v",
    "show_details",
    is_flag=True,
    help="Show detailed output with scores, evidence sentences, and occurrence counts",
)
@click.option(
    "--window",
    "-w",
    type=str,
    default=None,
    help=(
        "Show significant terms in a window around each occurrence. "
        "Format: entity type (s=sentence, p=paragraph) + radius. "
        "E.g. 's0' (same sentence only), 's1' (±1 sentence), 'p0' (same paragraph), 'p1' (±1 paragraph)."
    ),
)
@click.option(
    "--start-from-section",
    type=str,
    default=None,
    help=(
        "Skip content before this section number. "
        "E.g. '1' excludes front-matter before chapter 1; '0' includes the introduction."
    ),
)
@click.option(
    "--exclude-sections",
    type=str,
    default=None,
    help=(
        "Exclude sections whose title matches this regex (case-insensitive). "
        "E.g. 'index|bibliography' to skip back-matter."
    ),
)
@click.pass_context
def analyze(
    ctx,
    corpus,
    term,
    top_n,
    threshold,
    pos,
    lemma,
    no_relations,
    format,
    output,
    group_by,
    show_structure,
    show_details,
    window,
    start_from_section,
    exclude_sections,
):
    """
    Analyze contextual relations for a search term.

    Extracts significant terms co-occurring with the search term and
    identifies grammatical relations between them.

    Examples:
        cmapr analyze corpus.json "consciousness"
        cmapr analyze corpus.json "being" --top-n 10
        cmapr analyze corpus.json "intentionality" --lemma -p nouns -p verbs
        cmapr analyze corpus.json "dialectic" --format json -o relations.json
        cmapr analyze corpus.json "semiotic" -w s0
        cmapr analyze corpus.json "semiotic" -w s1
        cmapr analyze corpus.json "semiotic" -w p0 --top-n 5
    """
    from concept_mapper.analysis.contextual_relations import analyze_context
    import json

    verbose = ctx.obj["verbose"]

    # Load corpus
    from concept_mapper.corpus.models import ProcessedDocument

    with open(corpus, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [ProcessedDocument.from_dict(doc_data) for doc_data in data]

    # Handle window mode
    if window is not None:
        entity_type, radius = _parse_window(window)
        _display_window_analysis(
            term,
            docs,
            entity_type,
            radius,
            pos,
            threshold,
            top_n,
            lemma,
            start_section=start_from_section,
            exclude_sections=exclude_sections,
        )
        return

    if verbose:
        click.echo(f"Analyzing contextual relations for '{term}'...")

    # Convert pos tuple to list (None if empty)
    pos_types = list(pos) if pos else None

    # Extract contextual relations
    relations = analyze_context(
        search_term=term,
        docs=docs,
        significance_threshold=threshold,
        pos_types=pos_types,
        match_lemma=lemma,
        extract_relations=not no_relations,
        top_n=top_n,
    )

    relations = _filter_relations(relations, start_from_section, exclude_sections)

    if not relations:
        click.echo(f"No significant contextual relations found for '{term}'")
        return

    # Show structure summary if requested
    if show_structure:
        structure_info = _get_structure_summary(docs)
        if structure_info["has_structure"]:
            click.echo("\nDocument Structure:")
            click.echo(f"  Chapters: {structure_info['num_chapters']}")
            click.echo(f"  Sections: {structure_info['num_sections']}")
            click.echo()

    # Display results based on format
    if format == "text":
        # Check if we have structure and should group
        has_structure = any(rel.evidence_locations for rel in relations)

        if has_structure and group_by > 0:
            # Use structured display
            # Extract structure nodes from first doc (assuming single doc corpus)
            structure_nodes = (
                docs[0].structure_nodes if docs and docs[0].structure_nodes else None
            )
            grouped = _group_relations_by_structure(
                relations, level=group_by, structure_nodes=structure_nodes
            )
            _display_structured_text(term, grouped, docs, verbose=show_details)
        else:
            # Use flat display
            from collections import defaultdict

            if show_details:
                # Verbose mode: show detailed output
                click.echo(
                    f"\nFound {len(relations)} contextual relations for '{term}':"
                )
                click.echo("=" * 70)

                by_type = defaultdict(list)
                for rel in relations:
                    by_type[rel.relation_type].append(rel)

                for rel_type in ["svo", "copular", "prep", "cooccurrence"]:
                    if rel_type in by_type:
                        rels = sorted(
                            by_type[rel_type], key=lambda r: r.score, reverse=True
                        )[:20]
                        click.echo(f"\n{rel_type.upper()} Relations (top 20):")
                        for i, rel in enumerate(rels, 1):
                            evidence_info = f"{len(rel.evidence)} sentence(s)"
                            click.echo(
                                f"  {i}. {rel.source} --{rel.relation_type}--> {rel.target} "
                                f"(score: {rel.score:.2f}, {evidence_info})"
                            )
            else:
                # Minimal mode: list significant terms with POS tags
                click.echo(f"\nSignificant terms co-occurring with '{term}':")

                # Build term -> POS mapping from documents
                term_pos = {}
                for doc in docs:
                    for i, (token, pos) in enumerate(doc.pos_tags):
                        term = doc.lemmas[i] if i < len(doc.lemmas) else token
                        if term not in term_pos:
                            term_pos[term] = pos

                # Collect terms with their POS tags
                term_pos_list = []
                for rel in relations:
                    pos = term_pos.get(rel.target, "UNK")
                    pos_label = _get_pos_label(pos)
                    term_pos_list.append((pos_label, rel.target))

                # Group by POS for better organization
                from collections import defaultdict

                by_pos = defaultdict(list)
                for pos_label, term in term_pos_list:
                    by_pos[pos_label].append(term)

                for pos_label in sorted(by_pos.keys()):
                    terms = sorted(set(by_pos[pos_label]))
                    for term in terms:
                        click.echo(f"  {pos_label}: {term}")

    elif format == "json":
        from concept_mapper.analysis.contextual_relations import (
            ContextualRelationExtractor,
        )

        extractor = ContextualRelationExtractor(docs)
        relations_data = extractor.to_dict(relations)

        # Add structure information
        structure_info = _get_structure_summary(docs)

        json_data = {
            "search_term": term,
            "num_relations": len(relations),
            "relations": relations_data,
            "structure": structure_info,
        }

        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            click.echo(f"✓ Saved {len(relations)} relations to {output_path}")
        else:
            click.echo(json.dumps(json_data, indent=2, ensure_ascii=False))

    elif format == "csv":
        import csv

        if output:
            output_path = Path(output)
        else:
            output_path = Path("relations.csv")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "source",
                    "relation_type",
                    "target",
                    "score",
                    "evidence_count",
                    "chapter",
                    "section",
                    "subsection",
                ]
            )
            for rel in relations:
                # Get first location if available
                chapter = ""
                section = ""
                subsection = ""
                if rel.evidence_locations:
                    loc = rel.evidence_locations[0]
                    chapter = loc.chapter or ""
                    section = loc.section or ""
                    subsection = loc.subsection or ""

                writer.writerow(
                    [
                        rel.source,
                        rel.relation_type,
                        rel.target,
                        rel.score,
                        len(rel.evidence),
                        chapter,
                        section,
                        subsection,
                    ]
                )

        click.echo(f"✓ Saved {len(relations)} relations to {output_path}")

    elif format == "graph":
        from concept_mapper.analysis.contextual_relations import (
            ContextualRelationExtractor,
        )

        extractor = ContextualRelationExtractor(docs)
        graph_data = extractor.to_graph_data(relations)

        if output:
            output_path = Path(output)
        else:
            output_path = Path("relations_graph.json")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(graph_data, f, indent=2, ensure_ascii=False)

        click.echo(
            f"✓ Saved graph with {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges to {output_path}"
        )


@cli.command()
@click.argument("corpus")
@click.argument("source")
@click.argument("target")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: print to stdout)",
)
@click.option(
    "--preview",
    is_flag=True,
    help="Show preview of changes without saving",
)
@click.pass_context
def replace(ctx, corpus, source, target, output, preview):
    """
    Replace term with synonym while preserving inflections.

    Replaces SOURCE term with TARGET synonym throughout the corpus,
    automatically preserving grammatical inflections (tense, number, degree).

    SOURCE and TARGET can be:
    - Single words: "run" "sprint"
    - Multi-word phrases (comma-separated): "body,without,organs" "medium"

    Examples:
        # Single word replacement
        cmapr replace corpus.json "run" "sprint" -o output.txt

        # Phrase to single word
        cmapr replace corpus.json "body,without,organs" "medium" -o output.txt

        # Phrase to phrase
        cmapr replace corpus.json "body,without,organs" "blank,resistant,field" -o output.txt

        # Preview changes
        cmapr replace corpus.json "run" "sprint" --preview
    """
    from .transformations.replacement import ReplacementSpec, SynonymReplacer

    verbose = ctx.obj["verbose"]

    # Load corpus
    if verbose:
        click.echo(f"Loading corpus from {corpus}...")

    with open(corpus, "r", encoding="utf-8") as f:
        corpus_data = json.load(f)

    if not corpus_data:
        click.echo("Error: Empty corpus", err=True)
        ctx.exit(1)

    docs = [ProcessedDocument.from_dict(d) for d in corpus_data]

    # Parse source and target (detect multi-word phrases)
    def parse_term(term_str):
        """Parse term string into single word or phrase list."""
        if "," in term_str:
            # Multi-word phrase: split by comma
            return [lemma.strip() for lemma in term_str.split(",")]
        else:
            # Single word
            return term_str.strip()

    source_lemma = parse_term(source)
    target_lemma = parse_term(target)

    # Create replacement spec
    spec = ReplacementSpec(source_lemma, target_lemma)

    if verbose:
        source_display = (
            " ".join(source_lemma) if isinstance(source_lemma, list) else source_lemma
        )
        target_display = (
            " ".join(target_lemma) if isinstance(target_lemma, list) else target_lemma
        )
        click.echo(f'Replacing "{source_display}" with "{target_display}"...')

    # Perform replacements
    replacer = SynonymReplacer()
    results = []

    for doc in docs:
        replaced_text = replacer.replace_in_document(spec, doc, case_sensitive=False)
        results.append(replaced_text)

    # Combine all documents
    combined_text = "\n\n".join(results)

    # Preview or save
    if preview:
        click.echo("\nPreview of changes:")
        click.echo("=" * 70)
        # Show first 1000 characters
        preview_text = combined_text[:1000]
        if len(combined_text) > 1000:
            preview_text += "\n... (truncated)"
        click.echo(preview_text)
        click.echo("=" * 70)
        click.echo(
            f"\nTotal length: {len(combined_text)} characters across {len(results)} document(s)"
        )
    elif output:
        # Save to file
        with open(output, "w", encoding="utf-8") as f:
            f.write(combined_text)
        click.echo(f"✓ Saved replaced text to {output}")
    else:
        # Print to stdout
        click.echo(combined_text)


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
