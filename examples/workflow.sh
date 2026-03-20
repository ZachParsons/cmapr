#!/bin/bash
# Concept Mapper Workflow
# Full pipeline from raw text to interactive HTML visualization.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

INPUT_TEXT="$PROJECT_ROOT/samples/eco_spl.txt"
TOC_FILE="$PROJECT_ROOT/samples/eco_spl_toc.txt"
OUTPUT_DIR="$PROJECT_ROOT/output"

echo "=================================="
echo "Concept Mapper: Full Workflow"
echo "=================================="
echo ""

# One-shot: ingest → rarities → graph → HTML export
cmapr --output-dir "$OUTPUT_DIR" run "$INPUT_TEXT" \
  --toc "$TOC_FILE" \
  --top-n 50 \
  --start-from-section 1 \
  --exclude-sections 'index|bibliography|references' \
  --format html \
  --title "Eco - Semiotics & Philosophy of Language"

echo ""
echo "=================================="
echo "Done."
echo "=================================="
echo ""
echo "Outputs:"
echo "  Corpus:    $OUTPUT_DIR/corpus/eco_spl.json"
echo "  Rarities:  $OUTPUT_DIR/rarities/eco_spl.json"
echo "  Graph:     $OUTPUT_DIR/graphs/eco_spl.json"
echo "  HTML:      $OUTPUT_DIR/exports/eco_spl/index.html"
echo ""
echo "To view:"
echo "  open $OUTPUT_DIR/exports/eco_spl/index.html"
echo ""

# Optional: also export other formats from the saved graph
# cmapr export "$OUTPUT_DIR/graphs/eco_spl.json" --format graphml -o "$OUTPUT_DIR/exports/eco_spl.graphml"
# cmapr export "$OUTPUT_DIR/graphs/eco_spl.json" --format gexf    -o "$OUTPUT_DIR/exports/eco_spl.gexf"
# cmapr export "$OUTPUT_DIR/graphs/eco_spl.json" --format csv     -o "$OUTPUT_DIR/exports/eco_spl/csv/"
