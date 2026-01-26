# Output Validation

This document describes the validation system that prevents empty or meaningless outputs from being written to files.

## Overview

The validation system ensures that all output operations (saving corpus files, term lists, graphs, exports, etc.) fail with descriptive error messages if the output would be empty or contain only boilerplate content.

## Validation Module

Location: `src/concept_mapper/validation.py`

### Exception

- **`EmptyOutputError`**: Raised when attempting to write empty or meaningless output

### Validation Functions

1. **`validate_corpus(corpus_data, min_docs=1)`**
   - Ensures corpus has at least the minimum number of documents
   - Checks that documents contain actual tokens (not empty)
   - Raises error with helpful suggestions if validation fails

2. **`validate_term_list(terms_data, min_terms=1)`**
   - Ensures term list has at least the minimum number of terms
   - Works with both list and dict formats
   - Provides actionable suggestions (lower threshold, try different method, etc.)

3. **`validate_graph(graph_data, require_edges=True)`**
   - Ensures graph has nodes
   - Optionally requires at least one edge
   - Works with dict format (D3 JSON, etc.)
   - Suggests fixes like lowering threshold or using different methods

4. **`validate_concept_graph(graph, require_edges=True)`**
   - Validates ConceptGraph objects
   - Uses `node_count()` and `edge_count()` methods
   - Provides helpful error messages with suggestions

5. **`validate_networkx_graph(graph, require_edges=True)`**
   - Validates NetworkX graph objects
   - Uses `number_of_nodes()` and `number_of_edges()` methods

6. **`validate_search_results(matches, query)`**
   - Ensures search results are not empty
   - Includes the query in the error message

7. **`validate_csv_data(rows, file_type="CSV")`**
   - Ensures CSV has data rows beyond just headers

## Where Validation Is Applied

### CLI Commands (`src/concept_mapper/cli.py`)

- **`ingest`**: Validates corpus before saving
- **`rarities`**: Validates term list before saving
- **`graph`**: Validates graph before exporting
- **`search`**: Already had validation (kept as-is)

### Export Modules

#### `src/concept_mapper/export/d3.py`
- **`export_d3_json()`**: Validates ConceptGraph before export

#### `src/concept_mapper/export/formats.py`
- **`export_graphml()`**: Validates ConceptGraph before export
- **`export_dot()`**: Validates ConceptGraph before export
- **`export_csv()`**: Validates ConceptGraph before export
- **`export_gexf()`**: Validates ConceptGraph before export
- **`export_json_graph()`**: Validates ConceptGraph before export

#### `src/concept_mapper/export/html.py`
- Validation happens via `export_d3_json()` call (no direct validation needed)

### Storage Backend (`src/concept_mapper/storage/json_backend.py`)

- **`save_corpus()`**: Validates corpus data
- **`save_term_list()`**: Validates term list
- **`save_graph()`**: Validates graph data

### Term Management

#### `src/concept_mapper/terms/manager.py`
- **`export_to_txt()`**: Validates term list before export
- **`export_to_csv()`**: Validates CSV data rows before export
- **`export_to_json()`**: Validates term list before export

#### `src/concept_mapper/terms/models.py`
- **`TermList.save()`**: Validates term list before saving

### Analysis Modules

#### `src/concept_mapper/analysis/cooccurrence.py`
- **`save_cooccurrence_matrix()`**: Validates matrix has terms before export

## Error Messages

All validation errors provide actionable suggestions for fixing the issue:

### Empty Corpus
```
Cannot save empty corpus. No documents were processed.
```

### Empty Terms
```
No terms detected. Try:
  - Lowering the --threshold value
  - Using a different --method (hybrid, ratio, tfidf, neologism)
  - Checking that the input text is substantial (>500 words)
  - Verifying the text is in English
```

### Empty Graph
```
Cannot save empty graph. No nodes were created.
Check that the term list contains valid terms.
```

### Graph Without Edges
```
Graph has 5 node(s) but no edges. Try:
  - Lowering the --threshold value
  - Using --method relations instead of cooccurrence
  - Checking that multiple terms appear in the corpus
```

### Empty CSV
```
Cannot write terms CSV with no data rows.
Only headers would be written.
```

## Testing

Validation can be tested with:

```python
from concept_mapper.validation import validate_corpus, EmptyOutputError

# This will raise EmptyOutputError
try:
    validate_corpus([])
except EmptyOutputError as e:
    print(f"Caught expected error: {e}")
```

All validation functions are tested in the test suite.

## Benefits

1. **Early Detection**: Problems are caught before writing empty files
2. **Clear Errors**: Users get actionable error messages instead of silent failures
3. **Debugging Aid**: Empty results indicate bugs in the pipeline
4. **User Experience**: Users understand what went wrong and how to fix it
5. **Data Quality**: Ensures all saved outputs contain meaningful data
