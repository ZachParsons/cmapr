#!/bin/bash
# Complete Concept Mapper Workflow
# Demonstrates end-to-end pipeline from text to visualization

set -e  # Exit on error

echo "=================================="
echo "Concept Mapper: Complete Workflow"
echo "=================================="
echo ""

# Get project root directory (parent of examples/)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Configuration
INPUT_TEXT="$PROJECT_ROOT/samples/sample1_analytic_pragmatism.txt"
OUTPUT_DIR="$PROJECT_ROOT/output"

# Output files now derive from source filename automatically
CORPUS_FILE="$OUTPUT_DIR/corpus/sample1_analytic_pragmatism.json"
TERMS_FILE="$OUTPUT_DIR/terms/sample1_analytic_pragmatism.json"
GRAPH_COOCCUR="$OUTPUT_DIR/graphs/sample1_analytic_pragmatism.json"
GRAPH_RELATIONS="$OUTPUT_DIR/graphs/sample1_analytic_pragmatism_relations.json"
VIZ_DIR="$OUTPUT_DIR/exports/sample1_analytic_pragmatism"

# Step 1: Ingest
echo "[1/5] Ingesting and preprocessing text..."
concept-mapper --output-dir "$OUTPUT_DIR" ingest "$INPUT_TEXT"
echo ""

# Step 2: Detect philosophical terms
echo "[2/5] Detecting philosophical terms..."
concept-mapper --output-dir "$OUTPUT_DIR" rarities "$CORPUS_FILE" \
  --method hybrid \
  --threshold 1.5 \
  --top-n 20
echo ""

# Step 3: Build co-occurrence graph
echo "[3/5] Building co-occurrence graph..."
concept-mapper --output-dir "$OUTPUT_DIR" graph "$CORPUS_FILE" \
  --terms "$TERMS_FILE" \
  --method cooccurrence \
  --threshold 0.3
echo ""

# Step 4: Build relations graph
echo "[4/5] Building relations graph..."
concept-mapper --output-dir "$OUTPUT_DIR" graph "$CORPUS_FILE" \
  --terms "$TERMS_FILE" \
  --method relations
echo ""

# Step 5: Generate visualization
echo "[5/5] Generating HTML visualization..."
concept-mapper --output-dir "$OUTPUT_DIR" export "$GRAPH_COOCCUR" \
  --format html \
  --title "Analytic Philosophy Concept Network"
echo ""

# Also export other formats (using explicit filenames for compatibility)
echo "Exporting additional formats..."
concept-mapper export "$GRAPH_COOCCUR" --format graphml -o "$OUTPUT_DIR/exports/sample1_analytic_pragmatism.graphml"
concept-mapper export "$GRAPH_COOCCUR" --format csv -o "$OUTPUT_DIR/exports/sample1_analytic_pragmatism/csv/"
concept-mapper export "$GRAPH_COOCCUR" --format gexf -o "$OUTPUT_DIR/exports/sample1_analytic_pragmatism.gexf"
echo ""

# Summary
echo "=================================="
echo "Workflow Complete!"
echo "=================================="
echo ""
echo "Output files:"
echo "  - Corpus: $CORPUS_FILE"
echo "  - Terms: $TERMS_FILE"
echo "  - Co-occurrence graph: $GRAPH_COOCCUR"
echo "  - Relations graph: $GRAPH_RELATIONS"
echo "  - Visualization: $VIZ_DIR/index.html"
echo "  - GraphML: $OUTPUT_DIR/exports/graph.graphml"
echo "  - CSV: $OUTPUT_DIR/exports/csv/"
echo "  - GEXF: $OUTPUT_DIR/exports/graph.gexf"
echo ""
echo "To view the visualization:"
echo "  open $VIZ_DIR/index.html"
echo ""
