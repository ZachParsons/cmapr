#!/bin/bash
# Complete Concept Mapper Workflow
# Demonstrates end-to-end pipeline from text to visualization

set -e  # Exit on error

echo "=================================="
echo "Concept Mapper: Complete Workflow"
echo "=================================="
echo ""

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Configuration (paths relative to script directory)
INPUT_TEXT="$SCRIPT_DIR/sample_text.txt"
OUTPUT_DIR="$SCRIPT_DIR"
CORPUS_FILE="$OUTPUT_DIR/corpus.json"
TERMS_FILE="$OUTPUT_DIR/terms.json"
GRAPH_COOCCUR="$OUTPUT_DIR/graph_cooccur.json"
GRAPH_RELATIONS="$OUTPUT_DIR/graph_relations.json"
VIZ_DIR="$OUTPUT_DIR/visualization"

# Step 1: Ingest
echo "[1/5] Ingesting and preprocessing text..."
concept-mapper ingest "$INPUT_TEXT" --output "$CORPUS_FILE"
echo ""

# Step 2: Detect philosophical terms
echo "[2/5] Detecting philosophical terms..."
concept-mapper rarities "$CORPUS_FILE" \
  --method hybrid \
  --threshold 1.5 \
  --top-n 20 \
  --output "$TERMS_FILE"
echo ""

# Step 3: Build co-occurrence graph
echo "[3/5] Building co-occurrence graph..."
concept-mapper graph "$CORPUS_FILE" \
  --terms "$TERMS_FILE" \
  --method cooccurrence \
  --threshold 0.3 \
  --output "$GRAPH_COOCCUR"
echo ""

# Step 4: Build relations graph
echo "[4/5] Building relations graph..."
concept-mapper graph "$CORPUS_FILE" \
  --terms "$TERMS_FILE" \
  --method relations \
  --output "$GRAPH_RELATIONS"
echo ""

# Step 5: Generate visualization
echo "[5/5] Generating HTML visualization..."
concept-mapper export "$GRAPH_COOCCUR" \
  --format html \
  --title "Heidegger Concept Network" \
  --output "$VIZ_DIR"
echo ""

# Also export other formats
echo "Exporting additional formats..."
concept-mapper export "$GRAPH_COOCCUR" --format graphml -o "$OUTPUT_DIR/graph.graphml"
concept-mapper export "$GRAPH_COOCCUR" --format csv -o "$OUTPUT_DIR/csv/"
concept-mapper export "$GRAPH_COOCCUR" --format gexf -o "$OUTPUT_DIR/graph.gexf"
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
echo "  - GraphML: $OUTPUT_DIR/graph.graphml"
echo "  - CSV: $OUTPUT_DIR/csv/"
echo "  - GEXF: $OUTPUT_DIR/graph.gexf"
echo ""
echo "To view the visualization:"
echo "  open $VIZ_DIR/index.html"
echo ""
