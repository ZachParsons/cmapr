"""
Command-line interface for Concept Mapper.

Provides unified command-line access to all functionality.
"""

import click
import json
import sys
from pathlib import Path

# Import core functionality
from concept_mapper.corpus.loader import load_file, load_directory
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
@click.pass_context
def ingest(ctx, path, output, recursive, pattern, clean_ocr):
    """
    Load and preprocess documents.

    PATH can be a file or directory. Processes documents through
    tokenization, POS tagging, and lemmatization.

    Use --clean-ocr for texts from PDFs with OCR errors (spacing issues,
    split words, page numbers, etc.).

    Examples:
        cmapr ingest document.txt -o corpus.json
        cmapr ingest corpus/ -r -p "*.txt" -o corpus.json
        cmapr ingest scanned.txt --clean-ocr -o corpus.json
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

    # Preprocess
    if verbose:
        if clean_ocr:
            click.echo("Preprocessing documents (with OCR cleaning)...")
        else:
            click.echo("Preprocessing documents...")

    processed = []
    with click.progressbar(docs, label="Processing") as bar:
        for doc in bar:
            processed.append(preprocess(doc, clean_ocr=clean_ocr))

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

    docs = [ProcessedDocument(**doc_data) for doc_data in data]

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


def _group_relations_by_structure(relations, level="chapter"):
    """
    Group relations by document structure.

    Args:
        relations: List of ContextualRelation objects
        level: Grouping level (chapter, section, subsection)

    Returns:
        Dictionary mapping structure keys to lists of relations
    """
    from collections import defaultdict

    grouped = defaultdict(lambda: {"relations": [], "title": "", "parent": None})

    for rel in relations:
        if not rel.evidence_locations:
            # No structure info - put in "unstructured" group
            key = "unstructured"
            grouped[key]["relations"].append(rel)
            grouped[key]["title"] = "Unstructured Content"
            continue

        # Use first location for grouping
        loc = rel.evidence_locations[0]

        if level == "chapter" and loc.chapter:
            key = f"{loc.chapter}"
            grouped[key]["relations"].append(rel)
            grouped[key]["title"] = loc.chapter_title or f"Chapter {loc.chapter}"
        elif level == "section" and loc.section:
            key = f"{loc.section}"
            grouped[key]["relations"].append(rel)
            grouped[key]["title"] = loc.section_title or f"Section {loc.section}"
            grouped[key]["parent"] = loc.chapter_title if loc.chapter else None
        elif level == "subsection" and loc.subsection:
            key = f"{loc.subsection}"
            grouped[key]["relations"].append(rel)
            grouped[key]["title"] = (
                loc.subsection_title or f"Subsection {loc.subsection}"
            )
            grouped[key]["parent"] = loc.section_title if loc.section else None
        else:
            # Fallback to unstructured
            key = "unstructured"
            grouped[key]["relations"].append(rel)
            grouped[key]["title"] = "Unstructured Content"

    return grouped


def _display_structured_text(term, grouped_relations, docs):
    """
    Display relations grouped by structure in text format.

    Args:
        term: Search term
        grouped_relations: Dictionary from _group_relations_by_structure
        docs: List of ProcessedDocument objects (for structure info)
    """
    from collections import defaultdict

    click.echo(f"\nFound contextual relations for '{term}'")
    click.echo("=" * 80)

    # Sort groups by key
    sorted_groups = sorted(grouped_relations.items())

    for group_key, group_data in sorted_groups:
        title = group_data["title"]
        relations = group_data["relations"]

        if not relations:
            continue

        # Display group header
        click.echo(f"\n{'=' * 80}")
        click.echo(f"{title}")
        click.echo(f"{'=' * 80}")

        # Group relations by type
        by_type = defaultdict(list)
        for rel in relations:
            by_type[rel.relation_type].append(rel)

        # Display each type
        for rel_type in ["svo", "copular", "prep", "cooccurrence"]:
            if rel_type not in by_type:
                continue

            rels = sorted(by_type[rel_type], key=lambda r: r.score, reverse=True)[:10]

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

                # Show first evidence sentence
                if rel.evidence:
                    first_evidence = (
                        rel.evidence[0][:100] + "..."
                        if len(rel.evidence[0]) > 100
                        else rel.evidence[0]
                    )
                    click.echo(f'       "{first_evidence}"')


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
    type=click.Choice(["none", "chapter", "section", "subsection"]),
    default="chapter",
    help="Group results by document structure (default: chapter)",
)
@click.option(
    "--show-structure",
    is_flag=True,
    help="Display full document structure outline",
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
    """
    from concept_mapper.analysis.contextual_relations import analyze_context
    import json

    verbose = ctx.obj["verbose"]

    # Load corpus
    from concept_mapper.corpus.models import ProcessedDocument

    with open(corpus, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [ProcessedDocument.from_dict(doc_data) for doc_data in data]

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

        if has_structure and group_by != "none":
            # Use structured display
            grouped = _group_relations_by_structure(relations, level=group_by)
            _display_structured_text(term, grouped, docs)
        else:
            # Use flat display
            click.echo(f"\nFound {len(relations)} contextual relations for '{term}':")
            click.echo("=" * 70)

            # Group by relation type
            from collections import defaultdict

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


# ============================================================================
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
