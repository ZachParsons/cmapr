# Implementation Plan: Concept Graph

Derived from `docs/specs/graph.md`. Each task is independently testable before the next begins.

---

## Dev loop

```bash
# Rebuild corpus (once, or after input text changes):
cmapr ingest data/input/eco_spl1.txt --clean-ocr \
    --output data/output/corpus/eco_spl1/corpus.json

# Iterate on graph (corpus + terms already built):
cmapr graph data/output/corpus/eco_spl1/corpus.json \
    --terms data/output/rarities/eco_spl1/terms.json \
    --output data/output/graphs/eco_spl1/graph.json && \
cmapr export data/output/graphs/eco_spl1/graph.json --format html && \
open data/output/exports/eco_spl1/index.html
```

---

## Phase 1 — Test corpus

**1.1 Extract one chapter of Eco's SPL**
- Copy one chapter into `data/input/eco_ch1.txt`
- Run ingest + rarities once; save outputs to `data/output/corpus/eco_ch1/` and `data/output/rarities/eco_ch1/`
- Verify: rarities produces 30–80 terms, corpus loads cleanly

**Verify:**
```bash
cmapr ingest data/input/eco_spl1.txt --clean-ocr \
    --output data/output/corpus/eco_spl1/corpus.json
cmapr rarities data/output/corpus/eco_spl1/corpus.json --top-n 50
# Expect: 30–80 clean terms, no suffix fragments (tion, ence, lated, con-)
python -c "import json; d=json.load(open('data/output/rarities/eco_spl1/terms.json')); print(len(d), 'terms')"
```

---

## Phase 2 — Node filtering

Spec ref: *Nodes > Inclusion criteria*, *Decision 1 (node roles)*

**2.1 Implement `NodeFilter`**
- File: `src/concept_mapper/graph/node_filter.py`
- Function: `is_valid_node(term: str, corpus_vocab: set[str], term_freqs: Counter, stopwords: set) -> bool`
- Checks in order:
  1. POS: must be noun, verb, adjective, or adverb (reuse `POS_TAG_GROUPS` from `search/extract.py`)
  2. Length ≥ 4
  3. Not all-caps with length ≤ 4 (abbreviation check)
  4. Not a stopword
  5. Frequency ≥ 3 in corpus
  6. Fragment check: NOT (absent from WordNet AND any corpus word starts with it)
- Acceptance: unit tests covering each rejection criterion; `sign`, `interpretant`, `semiosis` pass; `nn`, `structu`, `moke` fail

**2.2 Apply `NodeFilter` at graph construction time**
- Both seed nodes (from rarities) and extracted nodes go through `NodeFilter`
- Acceptance: graph command output contains no terms shorter than 4 chars, no all-caps abbreviations

**Verify:**
```bash
python -m pytest tests/test_node_filter.py -v
# Expect: all 36 tests pass, including suffix fragment tests (tion, ence rejected;
# form, sign pass via WordNet guard)
```

---

## Phase 3 — Typed proposition extractor

Spec ref: *Edges > Edge types*, *Copular disambiguation rules*, *Decision 4 (scanning scope)*

**3.1 Implement `PropositionExtractor`**
- File: `src/concept_mapper/graph/proposition_extractor.py`
- Class: `PropositionExtractor(docs: list[ProcessedDocument])`
- Core method: `extract(term_a: str, term_b: str) -> list[Proposition]`
  - Find sentences containing both `term_a` and `term_b` (v1 scope: both present)
  - For each such sentence, attempt extraction in priority order:
    1. `_extract_definition(sentence, term_a, term_b)` — explicit markers: *by X I mean*, *X is defined as*, *X denotes*
    2. `_extract_kind_of(sentence, term_a, term_b)` — copular + kind marker: *X is a type/kind/sort/species of Y*
    3. `_extract_production(sentence, term_a, term_b)` — SVO where verb matches production verbs: *produces, generates, gives rise to, implies, creates*
    4. `_extract_dependence(sentence, term_a, term_b)` — SVO/prep where verb matches dependence verbs: *presupposes, depends on, requires, needs*
    5. If none found: return `None` (caller handles cooccurrence fallback)
  - Returns `Proposition(source, target, label, type, evidence_sentence)`

**3.2 `Proposition` dataclass**
- Fields: `source: str`, `target: str`, `label: str`, `type: str`, `evidence: str`, `directed: bool`
- `directed=True` for all typed extractions; `directed=False` for cooccurrence fallback

**3.3 Unit tests for `PropositionExtractor`**
- Planted sentences for each edge type; verify correct type and direction returned
- Verify that a sentence with no matching pattern returns `None`
- Test copular disambiguation: definition marker → `definition`; kind marker → `kind-of`; plain NP → `None` (property deferred)

**Verify:**
```bash
python -m pytest tests/test_proposition_extractor.py -v
# Expect: all tests pass, each edge type fires correctly, direction is correct
```

---

## Phase 4 — Composition pattern extractor

Spec ref: *Decision 12*

**4.1 Implement `extract_composition(sentence, term_list) -> list[Proposition]`**
- File: `src/concept_mapper/graph/proposition_extractor.py` (add to same module)
- Pattern: compound subject (A, B, C) + composition verb (form, constitute, compose, make up, consist of) + object X
- Where 2+ members of the compound subject are in `term_list`
- Returns:
  - One `production`/`dependence` edge per component → X (based on verb)
  - One `component` edge per co-component pair (undirected, label: *co-constitutes*)
- Acceptance: sentence *"The sign, the interpretant, and the referent form a triadic relation"* → produces sign↔interpretant, sign↔referent, interpretant↔referent `component` edges plus directed edges to `relation`

**Verify:**
```bash
python -m pytest tests/test_proposition_extractor.py -v -k "Composition"
# Expect: component edges between all co-constituents, production edges to composed term
```

---

## Phase 5 — Graph construction pipeline

Spec ref: *Graph model > How edges should be constructed*

**5.1 Implement `build_proposition_graph`**
- File: `src/concept_mapper/graph/builders.py` (new function alongside existing builders)
- Signature: `build_proposition_graph(docs, term_list, pmi_threshold=1.0, prune_ratio=3.0) -> ConceptGraph`
- Algorithm:
  1. For each pair in `term_list` that co-occur in ≥ 1 sentence, call `PropositionExtractor.extract(a, b)`
  2. If propositions found → add as typed edges (multigraph: allow multiple types per pair)
  3. If no proposition found AND PMI(a,b) ≥ `pmi_threshold` → add `cooccurrence` fallback edge
  4. Run `extract_composition` over all sentences containing 2+ terms from `term_list`
  5. Apply `NodeFilter` to all extracted (non-seed) nodes; discard those that fail
  6. Merge same-type duplicate edges: aggregate evidence, increment weight
  7. Prune to `prune_ratio` (Phase 6)
- Acceptance: graph from eco_ch1 produces typed edge labels (not all "co-occurs with"), ratio ≤ 3:1

**5.2 Wire into `cmapr graph` command**
- Replace the current `analyze_context`-based loop with `build_proposition_graph`
- Keep existing `--terms`, `--threshold`, `--count`, `--with-relations` options
- Acceptance: `cmapr graph eco_ch1/corpus.json --terms eco_ch1/terms.json` completes and outputs edge types in summary

**Verify:**
```bash
python -m pytest tests/test_graph.py -v
# Unit tests for builders

cmapr graph data/output/corpus/eco_spl1/corpus.json \
    --terms data/output/rarities/eco_spl1/terms.json \
    --output data/output/graphs/eco_spl1/graph.json
# Expect: completes without error, prints edge type summary

python -c "
import json
g = json.load(open('data/output/graphs/eco_spl1/graph.json'))
links = g['links']
types = {}
for e in links:
    types[e.get('type','cooccurrence')] = types.get(e.get('type','cooccurrence'), 0) + 1
ratio = len(links) / max(len(g['nodes']), 1)
print('nodes:', len(g['nodes']), '  links:', len(links), f'  ratio: {ratio:.1f}:1')
print('link types:', types)
cooc = types.get('cooccurrence', 0)
print(f'typed (non-cooc): {len(links)-cooc}/{len(links)} = {(len(links)-cooc)/max(len(links),1)*100:.0f}%')
"
# Expect: ratio ≤ 3:1, cooccurrence < 50% of links
```

---

## Phase 6 — Edge pruning

Spec ref: *Implementation priorities #3*

**6.1 Implement `prune_to_ratio`**
- File: `src/concept_mapper/graph/operations.py` (add to existing operations)
- Signature: `prune_to_ratio(graph: ConceptGraph, target_ratio: float = 3.0) -> ConceptGraph`
- Pruning order:
  1. Remove `cooccurrence` edges first (lowest priority regardless of weight), except where they are the sole edge for a node
  2. Then remove lowest-weight grammatical edges until ratio ≤ `target_ratio`
  3. Never remove an edge that would isolate a node
- Acceptance: graph with ratio 4:1 → after pruning ratio ≤ 3:1; no isolated nodes introduced

**Verify:**
```bash
python -m pytest tests/test_graph_operations.py -v
# Unit tests: ratio enforced, no isolated nodes

python -c "
import json
g = json.load(open('data/output/graphs/eco_spl1/graph.json'))
links = g['links']
ratio = len(links) / max(len(g['nodes']), 1)
node_ids = {n['id'] for n in g['nodes']}
connected = {e['source'] for e in links} | {e['target'] for e in links}
isolated = node_ids - connected
print(f'ratio: {ratio:.2f}:1  (target ≤ 3:1)')
print(f'isolated nodes: {len(isolated)}  (target 0)')
"
# Expect: ratio ≤ 3.0, 0 isolated nodes
```

---

## Phase 7 — Visualization tuning

Spec ref: *Visualization*, *v1 vs v2 scope*

**7.1 Update D3 force parameters in HTML template**
- File: `src/concept_mapper/export/html.py`
- Changes:
  - Charge: `-400` or stronger (currently default D3 value)
  - Link distance: proportional to `1 / edge_weight`
  - Add collision detection (`d3.forceCollide`) with radius = node label length × font size estimate
  - If node degree > 10: reduce its link strength by 50%
  - Pin dragged nodes: on `dragend`, set `node.fx = node.x` and `node.fy = node.y` so the node stays where placed; the simulation continues but the pinned node is held fixed

**7.2 Edge label rendering**
- Always show edge label on the link line (not just on hover) for v1
- If labels overlap: fall back to hover-only (threshold: when edge density exceeds 3 edges per 100px²)

**7.3 Edge visual differentiation by type**
- Color per edge type (assign fixed palette: definition=blue, kind-of=green, production=orange, dependence=red, component=purple, cooccurrence=light grey)
- `cooccurrence` edges: dashed stroke, 50% opacity
- `component` edges: undirected (no arrowhead)
- All other v1 types: directed arrowhead

**7.4 Node size by significance score**
- Pass rarity score from terms file through to graph JSON node attributes
- Scale node radius: `r = 4 + score * 3` (adjustable)
- Font size proportional to radius: `font-size = max(10, r * 1.2)`

**Verify:**
```bash
cmapr graph data/output/corpus/eco_spl1/corpus.json \
    --terms data/output/rarities/eco_spl1/terms.json \
    --output data/output/graphs/eco_spl1/graph.json && \
cmapr export data/output/graphs/eco_spl1/graph.json --format html && \
open data/output/exports/eco_spl1/index.html
# Visual check: edge labels visible, arrows on directed edges, cooccurrence edges dashed,
# nodes sized by score, no JS console errors
```

---

## Phase 8 — Integration test

**8.1 End-to-end test on eco_ch1**
- Run full dev loop command
- Verify:
  - Node count 30–150 (chapter scale)
  - Edge:node ratio ≤ 3:1
  - At least 50% of edges have a non-cooccurrence type label
  - HTML loads in browser without JS errors
  - Edge labels are visible and readable at default zoom

**8.2 Regression: existing tests still pass**
- `make test` green after all changes

**Verify:**
```bash
# Full pipeline end-to-end
cmapr graph data/output/corpus/eco_spl1/corpus.json \
    --terms data/output/rarities/eco_spl1/terms.json \
    --output data/output/graphs/eco_spl1/graph.json && \
cmapr export data/output/graphs/eco_spl1/graph.json --format html && \
open data/output/exports/eco_spl1/index.html

# Check all acceptance criteria
python -c "
import json
g = json.load(open('data/output/graphs/eco_spl1/graph.json'))
nodes, links = g['nodes'], g['links']
ratio = len(links) / max(len(nodes), 1)
types = {}
for e in links:
    types[e.get('type','cooccurrence')] = types.get(e.get('type','cooccurrence'), 0) + 1
cooc = types.get('cooccurrence', 0)
typed_pct = (len(links) - cooc) / max(len(links), 1) * 100
short = [n['id'] for n in nodes if len(n['id']) < 4]
print(f'nodes: {len(nodes)}  (target 30–150)')
print(f'links: {len(links)}  ratio: {ratio:.1f}:1  (target ≤ 3:1)')
print(f'typed links: {typed_pct:.0f}%  (target ≥ 50%)')
print(f'short nodes (<4): {short}  (target [])')
print('link types:', types)
"

# Regression
python -m pytest -q
# Expect: all tests pass
```

---

## Deferred (not in this plan)

- `property` and `opposition` edge types
- v2 scanning scope (scan all sentences for either term)
- Interactive special-terms dictionary (per-work unknown word vetting)
- Evidence selection spec (which sentence to surface in tooltip)
- v2 interactivity (node expand/collapse, detail panel, edge type toggle)
- Multi-word term support (requires spaCy)
- Seed-word workflow B options (B1–B5: POS filter, count limit, group-by section, depth limit)
