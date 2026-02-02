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
INPUT_TEXT="$PROJECT_ROOT/samples/eco_spl.txt"
OUTPUT_DIR="$PROJECT_ROOT/output"

# Output files now derive from source filename automatically
CORPUS_FILE="$OUTPUT_DIR/corpus/eco_spl.json"
TERMS_FILE="$OUTPUT_DIR/terms/eco_spl.json"
GRAPH_COOCCUR="$OUTPUT_DIR/graphs/eco_spl.json"
GRAPH_RELATIONS="$OUTPUT_DIR/graphs/eco_spl_relations.json"
VIZ_DIR="$OUTPUT_DIR/exports/eco_spl"

# Step 1: Ingest
echo "[1/5] Ingesting and preprocessing text..."
cmapr --output-dir "$OUTPUT_DIR" ingest "$INPUT_TEXT"
echo ""

# Step 2: Detect semiotic terms
echo "[2/5] Detecting semiotic and philosophical terms..."
cmapr --output-dir "$OUTPUT_DIR" rarities "$CORPUS_FILE" \
  --method ratio \
  --threshold 0.3 \
  --top-n 50
echo ""

# Step 3: Build co-occurrence graph
echo "[3/5] Building co-occurrence graph..."
cmapr --output-dir "$OUTPUT_DIR" graph "$CORPUS_FILE" \
  --terms "$TERMS_FILE" \
  --method cooccurrence \
  --threshold 0.3
echo ""

# Step 4: Build relations graph
echo "[4/5] Building relations graph..."
cmapr --output-dir "$OUTPUT_DIR" graph "$CORPUS_FILE" \
  --terms "$TERMS_FILE" \
  --method relations
echo ""

# Step 5: Generate visualization
echo "[5/5] Generating HTML visualization..."
cmapr --output-dir "$OUTPUT_DIR" export "$GRAPH_COOCCUR" \
  --format html \
  --title "Eco - Semiotics & Philosophy of Language"
echo ""

# Also export other formats (using explicit filenames for compatibility)
echo "Exporting additional formats..."
cmapr export "$GRAPH_COOCCUR" --format graphml -o "$OUTPUT_DIR/exports/eco_spl.graphml"
cmapr export "$GRAPH_COOCCUR" --format csv -o "$OUTPUT_DIR/exports/eco_spl/csv/"
cmapr export "$GRAPH_COOCCUR" --format gexf -o "$OUTPUT_DIR/exports/eco_spl.gexf"
echo ""

# Summary
echo "=================================="
echo "Workflow Complete!"
echo "=================================="
echo ""
echo "Output files:"
echo "  - Corpus: $CORPUS_FILE (~110K words preprocessed)"
echo "  - Terms: $TERMS_FILE (50 semiotic concepts)"
echo "  - Co-occurrence graph: $GRAPH_COOCCUR (50 nodes, ~469 edges)"
echo "  - Relations graph: $GRAPH_RELATIONS"
echo "  - Visualization: $VIZ_DIR/index.html"
echo "  - GraphML: $OUTPUT_DIR/exports/eco_spl.graphml"
echo "  - CSV: $OUTPUT_DIR/exports/eco_spl/csv/"
echo "  - GEXF: $OUTPUT_DIR/exports/eco_spl.gexf"
echo ""
echo "To view the visualization:"
echo "  open $VIZ_DIR/index.html"
echo ""
