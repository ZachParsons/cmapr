#!/usr/bin/env python3
"""
Complete Concept Mapper Workflow (Python API)

Demonstrates end-to-end pipeline using the Python API directly.
"""

from pathlib import Path
from concept_mapper.corpus.loader import load_file
from concept_mapper.preprocessing.pipeline import preprocess
from concept_mapper.analysis.reference import load_reference_corpus
from concept_mapper.analysis.rarity import PhilosophicalTermScorer
from concept_mapper.analysis.cooccurrence import build_cooccurrence_matrix
from concept_mapper.graph import graph_from_cooccurrence, graph_from_relations
from concept_mapper.analysis.relations import get_relations
from concept_mapper.terms.models import TermList
from concept_mapper.terms.manager import TermManager
from concept_mapper.export import (
    export_d3_json,
    export_graphml,
    export_csv,
    export_gexf,
    generate_html,
)


def main():
    """Run complete workflow."""
    print("=" * 50)
    print("Concept Mapper: Complete Workflow (Python API)")
    print("=" * 50)
    print()

    # Configuration
    input_file = Path("examples/sample_text.txt")
    output_dir = Path("examples")
    output_dir.mkdir(exist_ok=True)

    # Step 1: Load and preprocess
    print("[1/6] Loading and preprocessing text...")
    doc = load_file(input_file)
    processed = preprocess(doc)
    print(f"  Loaded {len(processed.sentences)} sentences")
    print(f"  Tokenized {len(processed.tokens)} words")
    print()

    # Step 2: Load reference corpus
    print("[2/6] Loading reference corpus...")
    reference = load_reference_corpus()
    print(f"  Reference corpus: {len(reference):,} unique words")
    print()

    # Step 3: Detect philosophical terms
    print("[3/6] Detecting philosophical terms...")
    scorer = PhilosophicalTermScorer([processed], reference, use_lemmas=True)
    candidates = scorer.score_all(min_score=1.5, top_n=20)

    print(f"\n  Top {len(candidates)} rare terms:")
    print("  " + "-" * 40)
    for term, score, components in candidates[:10]:
        print(f"  {term:25} {score:6.2f}")
    if len(candidates) > 10:
        print(f"  ... and {len(candidates) - 10} more")
    print()

    # Create term list
    term_list = TermList(
        [{"term": term, "score": score} for term, score, _ in candidates]
    )

    # Save terms
    terms_file = output_dir / "terms.json"
    manager = TermManager(term_list)
    manager.export_to_json(terms_file)
    print(f"  Saved terms to {terms_file}")
    print()

    # Step 4: Build co-occurrence graph
    print("[4/6] Building co-occurrence graph...")
    matrix = build_cooccurrence_matrix(
        term_list, [processed], method="pmi", window="sentence"
    )
    graph_cooccur = graph_from_cooccurrence(matrix, threshold=0.3)
    print(
        f"  Graph: {graph_cooccur.node_count()} nodes, {graph_cooccur.edge_count()} edges"
    )

    # Save graph
    graph_cooccur_file = output_dir / "graph_cooccur.json"
    export_d3_json(graph_cooccur, graph_cooccur_file)
    print(f"  Saved to {graph_cooccur_file}")
    print()

    # Step 5: Build relations graph
    print("[5/6] Building relations graph...")
    all_relations = []
    for term_data in term_list:
        relations = get_relations(term_data["term"], [processed])
        all_relations.extend(relations)

    graph_relations = graph_from_relations(all_relations)
    print(
        f"  Graph: {graph_relations.node_count()} nodes, {graph_relations.edge_count()} edges"
    )
    print(f"  Relations extracted: {len(all_relations)}")

    # Save graph
    graph_relations_file = output_dir / "graph_relations.json"
    export_d3_json(graph_relations, graph_relations_file)
    print(f"  Saved to {graph_relations_file}")
    print()

    # Step 6: Export and visualize
    print("[6/6] Generating visualizations and exports...")

    # HTML visualization
    viz_dir = output_dir / "visualization"
    html_path = generate_html(graph_cooccur, viz_dir, title="Heidegger Concept Network")
    print(f"  HTML visualization: {html_path}")

    # Other formats
    export_graphml(graph_cooccur, output_dir / "graph.graphml")
    print(f"  GraphML: {output_dir / 'graph.graphml'}")

    export_csv(graph_cooccur, output_dir / "csv")
    print(
        f"  CSV: {output_dir / 'csv' / 'nodes.csv'}, {output_dir / 'csv' / 'edges.csv'}"
    )

    export_gexf(graph_cooccur, output_dir / "graph.gexf")
    print(f"  GEXF: {output_dir / 'graph.gexf'}")
    print()

    # Summary
    print("=" * 50)
    print("Workflow Complete!")
    print("=" * 50)
    print()
    print("To view the visualization:")
    print(f"  open {html_path.absolute()}")
    print()


if __name__ == "__main__":
    main()
