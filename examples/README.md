# Example Workflows

This directory contains complete example workflows demonstrating how to use Concept Mapper from start to finish.

## Quick Start

Run the complete workflow script:

```bash
cd examples
bash workflow.sh
```

This will:
1. Ingest and preprocess the sample text
2. Detect philosophical terms
3. Build concept graphs (co-occurrence and relations)
4. Generate HTML visualization
5. Export to multiple formats (GraphML, CSV, GEXF)

## Files

### `workflow.sh`
Complete bash workflow demonstrating the full pipeline:
- Uses `samples/sample1_analytic_pragmatism.txt` as input
- Outputs to `output/` directory
- Generates corpus, terms, graphs, and visualizations

### `workflow.py` (if exists)
Python script version of the workflow, showing API usage

## Input Data

All example workflows use sample texts from the `samples/` directory:
- `samples/sample1_analytic_pragmatism.txt` - Analytic philosophy (212 words)
- `samples/sample2_poststructural_political.txt` - Poststructural philosophy (203 words)
- `samples/sample3_mind_consciousness.txt` - Philosophy of mind (290 words)

## Output Location

All generated outputs go to the `output/` directory:
```
output/
├── corpus/           # Processed corpora
├── terms/            # Extracted terms
├── graphs/           # Concept graphs
└── exports/          # Visualizations and export formats
```

## Customization

To adapt these workflows for your own texts:

1. Place your text files in `samples/`
2. Edit `workflow.sh` to reference your files
3. Run the workflow
4. Find results in `output/`

## See Also

- [README.md](../README.md) - Full project documentation
- [API Reference](../docs/api-reference.md) - Complete usage guide
- [samples/README.md](../samples/README.md) - Available sample texts
