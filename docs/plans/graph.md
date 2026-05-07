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

## Deferred — now in scope

---

## Phase 9 — Opposition edge type ✅

Spec ref: *Edges > Edge types* (deferred from v1)

**9.1 Implement `_try_opposition` in `PropositionExtractor`**
- Pattern: explicit contrast markers between two terms
  - *X vs Y*, *X versus Y*
  - *X as opposed to Y*, *X rather than Y*
  - *X not Y*, *unlike X, Y*
  - *X is the opposite of Y*, *X contrasts with Y*
- Label: `"opposes"`, type: `"opposition"`, directed: `False` (contrast is symmetric)
- Insert in `_extract_from_sentence` chain after `_try_dependence`, before `_try_property`

**9.2 Add to HTML legend and color palette**
- Color: `#d4a0a0` (muted rose) — distinct from dependence red
- Undirected (no arrowhead), solid stroke

**9.3 Unit tests**
- `"X vs Y"`, `"X as opposed to Y"`, `"X is the opposite of Y"` → `opposition`
- Verify `directed=False`

**Verify:**
```bash
python -m pytest tests/test_proposition_extractor.py -v -k "opposition"
# Expect: all opposition tests pass, directed=False confirmed

cmapr graph data/output/corpus/eco_spl1/corpus.json \
    -t data/output/rarities/eco_spl1/terms.json \
    --output data/output/graphs/eco_spl1/graph.json

python -c "
import json
g = json.load(open('data/output/graphs/eco_spl1/graph.json'))
opp = [e for e in g['links'] if e.get('type') == 'opposition']
print(f'opposition edges: {len(opp)}')
for e in opp[:5]:
    print(f'  {e[\"source\"]} ↔ {e[\"target\"]}  ({e.get(\"evidence\", [\"\"])[0][:60]})')
"
# Pass: at least 1 opposition edge present, none have directed=True
# Fail: 0 opposition edges (corpus may be too short — try full eco_spl1.txt)
```

---

## Phase 10 — Evidence selection ✅

Spec ref: *Evidence selection spec* (deferred from v1)

**10.1 Score and rank evidence sentences per edge**
- File: `src/concept_mapper/graph/proposition_extractor.py`
- When merging duplicate (source, target, type) edges, keep the best evidence sentence, not just the first
- Scoring heuristics (in priority order):
  1. Sentence contains a definition marker → highest score
  2. Sentence contains both terms within 15 words of each other → bonus
  3. Shorter sentences preferred (more precise)
  4. Sentences from early in the document preferred (introductions define terms)
- Store top-3 sentences as `evidence` list on the edge

**10.2 Surface evidence in tooltip**
- `html.py`: tooltip already renders `d.evidence[0]`; show all stored sentences (up to 3), separated by `<hr>`

**Verify:**
```bash
cmapr graph data/output/corpus/eco_spl1/corpus.json \
    -t data/output/rarities/eco_spl1/terms.json \
    --output data/output/graphs/eco_spl1/graph.json

python -c "
import json
g = json.load(open('data/output/graphs/eco_spl1/graph.json'))
links = g['links']
multi = [e for e in links if isinstance(e.get('evidence'), list) and len(e['evidence']) > 1]
single = [e for e in links if isinstance(e.get('evidence'), list) and len(e['evidence']) == 1]
missing = [e for e in links if not e.get('evidence')]
print(f'edges with 2-3 evidence sentences: {len(multi)}')
print(f'edges with 1 evidence sentence:    {len(single)}')
print(f'edges missing evidence:             {len(missing)}')
if multi:
    e = multi[0]
    print(f'sample ({e[\"source\"]} → {e[\"target\"]}):')
    for i, s in enumerate(e[\"evidence\"], 1): print(f'  [{i}] {s[:80]}')
"
# Pass: evidence is a list (not a string); at least some edges have 2+ sentences
# Fail: missing > 0 (evidence not being stored); evidence is a string (old format)

cmapr export data/output/graphs/eco_spl1/graph.json --format html \
    --output data/output/exports/eco_spl1/index.html && \
open data/output/exports/eco_spl1/index.html
# Visual: hover an edge — tooltip should show up to 3 sentences separated by <hr>
```

---

## Phase 11 — v2 interactivity

Spec ref: *v2 interactivity* (deferred from v1)

**11.1 Edge type toggle** ✅
- Add checkboxes to the legend (one per edge type)
- Toggling a type hides/shows all edges of that type and their labels
- Isolated nodes (all edges hidden) are also hidden

**11.2 Node detail panel** ✅
- Clicking a node opens a side panel (right side, 280px wide)
- Panel shows: term, POS, frequency, rarity score, community, all connected edges with their type and verb label
- Clicking elsewhere closes it

**11.3 Node expand/collapse** ✅
- Double-click a node: expand — show only that node and its direct neighbours; also releases pin
- Double-click the same node again: collapse — restore full graph
- Interacts correctly with the type filter (hidden types stay hidden in focused view)

**Verify:**
```bash
cmapr graph data/output/corpus/eco_spl1/corpus.json \
    -t data/output/rarities/eco_spl1/terms.json \
    --output data/output/graphs/eco_spl1/graph.json && \
cmapr export data/output/graphs/eco_spl1/graph.json --format html \
    --output data/output/exports/eco_spl1/index.html && \
open data/output/exports/eco_spl1/index.html

# 11.1 — Uncheck "cooccurrence" in legend
# Pass: dashed edges disappear; nodes with no remaining edges hide
# Fail: edges remain visible after unchecking; no checkboxes in legend

# 11.2 — Click any node
# Pass: detail panel opens on the right showing term name, frequency, score, connected edges
# Fail: panel does not open; panel opens blank; clicking elsewhere does not close it
```

---

---

## Phase 13 — Multi-word term support

Spec ref: *Multi-word term support (requires spaCy)* (deferred from v1)

**13.1 spaCy integration**
- Add `spacy` as optional dependency (`pip install cmapr[spacy]`)
- `cmapr ingest --spacy` flag: use spaCy pipeline instead of NLTK for tokenisation and POS
- Extract noun chunks as candidate multi-word terms (e.g. *sign vehicle*, *unlimited semiosis*, *triadic relation*)

**13.2 Multi-word term handling in graph**
- `NodeFilter`: skip length check for multi-word terms (already ≥2 words)
- `PropositionExtractor`: substring matching already works for multi-word terms; no change needed
- Graph node IDs: use the full phrase (e.g. `"sign vehicle"`)

**13.3 Rarities: multi-word candidate scoring**
- Score by TF-IDF of the full phrase across documents
- Merge single-word and multi-word candidates into one ranked list

**Verify:**
```bash
cmapr ingest data/input/eco_spl1.txt --clean-ocr --spacy \
    --output data/output/corpus/eco_spl1_spacy/corpus.json
cmapr rarities data/output/corpus/eco_spl1_spacy/corpus.json --top-n 60
# Expect: list includes multi-word terms like "sign vehicle", "sign system"
```

---

## Phase 14 — Seed-word workflow B options

Spec ref: *Seed-word workflow B options (B1–B5)* (deferred from v1)

**B1 — POS filter flag**
- `cmapr rarities --pos noun,verb` — restrict candidates to specified POS tags
- Default: all POS (current behaviour)

**B2 — Count limit**
- `cmapr rarities --top-n N` already implemented; ensure it interacts correctly with POS filter

**B3 — Group by section**
- `cmapr rarities --by-section` — output terms grouped by document section (requires section metadata from ingest)
- Section metadata: `ProcessedDocument.metadata["section"]`

**B4 — Depth limit**
- `cmapr graph --depth N` — only include nodes reachable within N hops from the highest-scoring seed node
- Useful for focusing on a term's immediate conceptual neighbourhood

**B5 — Neighbourhood export**
- `cmapr graph --focus <term>` — build graph centred on one term, include all nodes connected to it

**Verify:**
```bash
cmapr rarities data/output/corpus/eco_spl1/corpus.json --pos noun --top-n 40
# Expect: only noun terms

cmapr graph data/output/corpus/eco_spl1/corpus.json \
    --terms data/output/rarities/eco_spl1/terms.json \
    --focus sign --depth 2 \
    --output data/output/graphs/eco_spl1/sign_neighbourhood.json
# Expect: small graph centred on 'sign'
```

---

## Phase 15 — Term vetting intermediate workflow

The core vetting machinery is already implemented (formerly Phase 12). This phase tracks the remaining delta to align with the intended design.

**Implemented:**
- `cmapr rarities --vet` — interactive prompt per term, saves decisions to vetting file
- Vetting file at `data/output/rarities/<work>/vetting.json`, format `{"accept": [...], "reject": [...]}`
- Accepted/rejected decisions applied before top-n filtering and output

**Remaining:**
- **15.1** Drop `s/Enter=skip` from `--vet` prompt — currently the CLI accepts `s` and Enter as skip; simplify to `y/n` only (unvetted terms continue to pass through unchanged regardless)
- **15.2** Phase 16 (web UI) becomes the primary vetting interface; `--vet` stays as CLI fallback

**Verify:**
```bash
# From scratch — ingest must already be complete
cmapr rarities data/output/corpus/eco_spl1/corpus.json --top-n 60 --vet
# Prompts y/n per term; saves to data/output/rarities/eco_spl1/vetting.json
# Pass: prompt shows term, frequency, score; accepts y and n; saves file on exit
# Fail: prompt accepts 's' or blank Enter as a third option (post-15.1 fix: these should be rejected)

# Re-run without --vet to confirm vetting is applied
cmapr rarities data/output/corpus/eco_spl1/corpus.json --top-n 60 \
    --output data/output/rarities/eco_spl1/terms.json

python -c "
import json
terms = json.load(open('data/output/rarities/eco_spl1/terms.json'))
vet   = json.load(open('data/output/rarities/eco_spl1/vetting.json'))
ids   = {t['term'] for t in terms}
print('rejected still present:', [t for t in vet.get('reject', []) if t in ids])
print('accepted missing:',       [t for t in vet.get('accept', []) if t not in ids])
"
# Pass: rejected=[], accepted=[]
# Fail: rejected terms appear in output; accepted terms are missing
```

---

## Phase 16 — Local web UI ✅

**Stack:** FastAPI + Jinja2. No frontend build step. Served locally; `cmapr serve` opens the browser.

**Entry point**
- New CLI command: `cmapr serve [--port 8000]`
- Opens `http://localhost:8000` in the default browser
- Serves all UI pages and exposes pipeline steps as POST endpoints

**State detection**
- On load, the UI scans `data/output/corpus/` for existing corpora
- If a corpus exists for the selected work, ingest is skipped and the UI jumps to term review
- "Re-run ingest" button available to force re-ingest (e.g. after the source text changes)

**Step 1 — Configure run**
- File picker for source text (`.txt` or `.pdf`)
- Optional: TOC file path
- Toggles: `--clean-ocr`, `--spacy`
- If corpus already exists: show "Corpus found — skip to term review" shortcut

**Step 2 — Term review**
- Runs `rarities` against the corpus, displays results as a checkbox list
- Each row: term · frequency · rarity score · checkbox (checked = accept)
- Pre-checked state reflects existing vetting file if present
- Submit saves vetting file and proceeds to Step 3

**Step 3 — Graph options**
- `--top-n`, `--threshold`, `--depth`, `--focus` inputs
- "Re-run graph" is always available from this step without returning to Step 1 or 2
- Submit triggers graph build

**Step 4 — Result**
- Embedded D3 visualization (reuses existing HTML export template)
- "Re-run graph" button returns to Step 3 with current options pre-filled
- "Export" button downloads the graph JSON or standalone HTML

**Verify:**
```bash
pip install 'concept-mapper[serve]'   # one-time, adds fastapi + uvicorn
cmapr serve
# Browser opens at localhost:8000 — run from the project root (the directory with data/)

# New run: enter source file path → ingest → term review → graph options → visualization
# Existing corpus: click work name on home page → skip ingest → term review → graph options
# Re-run graph: "Re-run graph" button on result page → graph options → new visualization
```
