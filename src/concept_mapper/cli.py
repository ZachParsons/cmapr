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
@click.pass_context
def ingest(ctx, path, output, recursive, pattern):
    """
    Load and preprocess documents.

    PATH can be a file or directory. Processes documents through
    tokenization, POS tagging, and lemmatization.

    Examples:
        concept-mapper ingest document.txt -o corpus.json
        concept-mapper ingest corpus/ -r -p "*.txt" -o corpus.json
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
        click.echo("Preprocessing documents...")

    processed = []
    with click.progressbar(docs, label="Processing") as bar:
        for doc in bar:
            processed.append(preprocess(doc))

    # Save
    if output:
        output_path = Path(output)
    else:
        # Default: output/corpus/corpus.json
        corpus_dir = output_dir / "corpus"
        corpus_dir.mkdir(parents=True, exist_ok=True)
        output_path = corpus_dir / "corpus.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Serialize processed documents
    serialized = []
    for doc in processed:
        serialized.append(
            {
                "raw_text": doc.raw_text,
                "sentences": doc.sentences,
                "tokens": doc.tokens,
                "lemmas": doc.lemmas,
                "pos_tags": doc.pos_tags,
                "metadata": doc.metadata,
            }
        )

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
        concept-mapper rarities corpus.json --method hybrid --top-n 30
        concept-mapper rarities corpus.json -o terms.json
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
        # Default: output/terms/terms.json
        terms_dir = output_dir / "terms"
        terms_dir.mkdir(parents=True, exist_ok=True)
        output_path = terms_dir / "terms.json"

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
    "--lemma", "-l", is_flag=True, help="Match lemmatized forms (e.g., 'run' matches 'running', 'ran')"
)
@click.option("--output", "-o", type=click.Path(), help="Output file")
@click.pass_context
def search(ctx, corpus, term, context, lemma, output):
    """
    Search for term occurrences in corpus.

    Examples:
        concept-mapper search corpus.json "consciousness"
        concept-mapper search corpus.json "being" --context 2
        concept-mapper search corpus.json "run" --lemma
    """
    verbose = ctx.obj["verbose"]

    # Load corpus
    from concept_mapper.corpus.models import ProcessedDocument

    with open(corpus, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = [ProcessedDocument(**doc_data) for doc_data in data]

    if verbose:
        search_type = "lemma-based" if lemma else "exact"
        click.echo(f"Searching ({search_type}) for '{term}' in {len(docs)} document(s)...")

    # Search
    matches = find_sentences(term, docs, match_lemma=lemma)

    if not matches:
        click.echo(f"No matches found for '{term}'")
        return

    # Display results
    click.echo(f"\nFound {len(matches)} occurrence(s) of '{term}':")
    click.echo("=" * 70)

    for i, match in enumerate(matches, 1):
        click.echo(
            f"\n[{i}] {match.doc_id or 'document'} (sentence {match.sent_index}):"
        )

        if context > 0:
            # Get context from document
            doc = next(d for d in docs if d.metadata.get("source_path") == match.doc_id)

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
        concept-mapper concordance corpus.json "consciousness"
        concept-mapper concordance corpus.json "being" --width 80
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
        concept-mapper graph corpus.json -t terms.json -m cooccurrence
        concept-mapper graph corpus.json -t terms.json -m relations -o graph.json
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
        # Default: output/graphs/graph.json
        graphs_dir = output_dir / "graphs"
        graphs_dir.mkdir(parents=True, exist_ok=True)
        output_path = graphs_dir / "graph.json"

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
        concept-mapper export graph.json --format html -o viz/
        concept-mapper export graph.json --format graphml -o graph.graphml
        concept-mapper export graph.json --format csv -o output/
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
        # Default: output/exports/
        exports_dir = output_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)

        if format == "html":
            output_path = exports_dir / "visualization"
        elif format == "csv":
            output_path = exports_dir / "csv"
        else:
            output_path = exports_dir / f"graph.{format}"

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
        concept-mapper diagram "The cat sat on the mat."
        concept-mapper diagram "Abstraction obscures social processes." --format tree
        concept-mapper diagram "The question is complex." -o diagram.txt
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
# Main Entry Point
# ============================================================================


def main():
    """Main entry point for the CLI."""
    cli(obj={})


if __name__ == "__main__":
    main()
