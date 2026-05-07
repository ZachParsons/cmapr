# Graph Pipeline QA

Manual spot-check for the full graph workflow: ingest → rarities → graph → export → visual review.

Run these steps in order. Each section lists the command, then what to verify before continuing.
The reference corpus is `eco_spl1.txt` (Eco's *Semiotics and the Philosophy of Language*, part 1).

---

## Prerequisites

```bash
# Confirm input file exists
ls data/input/eco_spl1.txt

# Confirm cmapr is on PATH
cmapr --version
```

---

## Step 1 — Ingest

```bash
cmapr ingest data/input/eco_spl1.txt --clean-ocr \
    --output data/output/corpus/eco_spl1/corpus.json
```

**Check:**
- Exit code 0, no error output
- `data/output/corpus/eco_spl1/corpus.json` exists and is valid JSON
- Sentence count is plausible:

```bash
python -c "
import json
docs = json.load(open('data/output/corpus/eco_spl1/corpus.json'))
sents = sum(len(d['sentences']) for d in docs)
toks  = sum(len(d['tokens']) for d in docs)
print(f'{len(docs)} doc(s), {sents} sentences, {toks} tokens')
"
# Expect: 1 doc, 400–2000 sentences, 10000–60000 tokens
```

**With TOC** (optional, for section-aware output):

```bash
cmapr ingest data/input/eco_spl1.txt --clean-ocr \
    --toc data/input/eco_spl_toc.txt \
    --output data/output/corpus/eco_spl1_w_toc/corpus.json
```

Check that `structure_nodes` is populated:

```bash
python -c "
import json
docs = json.load(open('data/output/corpus/eco_spl1_w_toc/corpus.json'))
nodes = docs[0].get('structure_nodes', [])
print(f'{len(nodes)} structure nodes')
for n in nodes[:5]: print(' ', n)
"
# Expect: 5+ structure nodes (chapters/sections)
```

---

## Step 2 — Rarities

```bash
cmapr rarities data/output/corpus/eco_spl1/corpus.json \
    --top-n 60 \
    --output data/output/rarities/eco_spl1/terms.json
```

**Check:**
- Exit code 0
- Output prints a term list; count is 30–80

```bash
python -c "
import json
terms = json.load(open('data/output/rarities/eco_spl1/terms.json'))
print(f'{len(terms)} terms')
for t in terms[:15]: print(f'  {t[\"term\"]:30s}  score={t[\"metadata\"][\"score\"]:.3f}')
"
```

**Look for:**
- Domain vocabulary: *sign*, *signification*, *semiosis*, *interpretant*, *referent*, *denotation*, *connotation*, *symbol*, *icon*, *index*, *code*, *meaning*
- No obvious junk: no terms shorter than 4 chars, no all-caps abbreviations, no suffix fragments (`tion`, `ence`, `structu`, `lated`)
- Scores decrease monotonically top to bottom

**Variant — POS filter (nouns only):**

```bash
cmapr rarities data/output/corpus/eco_spl1/corpus.json \
    --pos noun --top-n 40 \
    --output data/output/rarities/eco_spl1/terms_noun.json
```

Check count is ≤ the unfiltered run.

**Variant — multi-word terms (requires spaCy):**

```bash
cmapr ingest data/input/eco_spl1.txt --clean-ocr --spacy \
    --output data/output/corpus/eco_spl1_spacy/corpus.json
cmapr rarities data/output/corpus/eco_spl1_spacy/corpus.json \
    --top-n 60 \
    --output data/output/rarities/eco_spl1_spacy/terms.json
```

Check that multi-word terms appear in the list (e.g. *sign vehicle*, *sign system*, *sign function*).

---

## Step 2b — Term vetting (optional)

Run after Step 2. Confirms that accepted/rejected decisions survive a re-run.

```bash
cmapr rarities data/output/corpus/eco_spl1/corpus.json --top-n 60 --vet
# At the prompt: accept a few domain terms (y), reject a known junk term (n)
# Exit when done — saves to data/output/rarities/eco_spl1/vetting.json
```

**Check:**

```bash
python -c "
import json, pathlib
vf = pathlib.Path('data/output/rarities/eco_spl1/vetting.json')
if not vf.exists():
    print('FAIL: vetting.json not created')
else:
    v = json.loads(vf.read_text())
    print('accepted:', v.get('accept', []))
    print('rejected:', v.get('reject', []))
"
# Expect: accepted and rejected lists contain the terms you chose
```

Re-run rarities without `--vet` and confirm decisions are applied:

```bash
cmapr rarities data/output/corpus/eco_spl1/corpus.json --top-n 60 \
    --output data/output/rarities/eco_spl1/terms.json

python -c "
import json
terms = json.load(open('data/output/rarities/eco_spl1/terms.json'))
vet   = json.load(open('data/output/rarities/eco_spl1/vetting.json'))
ids   = {t['term'] for t in terms}
still_in  = [t for t in vet.get('reject', []) if t in ids]
still_out = [t for t in vet.get('accept', []) if t not in ids]
print('rejected terms still in output:', still_in,  '(expect [])')
print('accepted terms missing:',         still_out, '(expect [])')
"
```

---

## Step 3 — Graph construction

```bash
cmapr graph data/output/corpus/eco_spl1/corpus.json \
    -t data/output/rarities/eco_spl1/terms.json \
    --output data/output/graphs/eco_spl1/graph.json
```

**Check structure:**

```bash
python -c "
import json
g = json.load(open('data/output/graphs/eco_spl1/graph.json'))
nodes, links = g['nodes'], g['links']
ratio = len(links) / max(len(nodes), 1)
types = {}
for e in links:
    t = e.get('type', 'cooccurrence')
    types[t] = types.get(t, 0) + 1
cooc = types.get('cooccurrence', 0)
typed_pct = (len(links) - cooc) / max(len(links), 1) * 100
short = [n['id'] for n in nodes if len(n['id']) < 4]

print(f'nodes: {len(nodes)}   (target: 20–150)')
print(f'links: {len(links)}   ratio: {ratio:.1f}:1   (target: ≤ 3:1)')
print(f'typed (non-cooc): {typed_pct:.0f}%   (target: ≥ 30%)')
print(f'short node IDs (<4 chars): {short}   (target: [])')
print()
print('edge types:')
for t, n in sorted(types.items(), key=lambda x: -x[1]):
    print(f'  {t:20s} {n}')
"
```

**Look for:**
- Ratio ≤ 3:1 (the prune step should have enforced this)
- At least some typed edges (`definition`, `kind-of`, `production`, `dependence`) — not all `cooccurrence`
- At least one `opposition` edge (if corpus contains "vs", "as opposed to", etc.)
- No node IDs shorter than 4 characters (NodeFilter should have excluded them)
- No isolated nodes (every node appears in at least one link)
- `evidence` on each edge is a list (not a string), with 1–3 sentences

```bash
# Isolated nodes are stripped before serialisation, so the JSON is always clean.
# Check stderr from the graph command instead — any dropped nodes are reported there:
cmapr graph data/output/corpus/eco_spl1/corpus.json \
    -t data/output/rarities/eco_spl1/terms.json \
    --output data/output/graphs/eco_spl1/graph.json 2>&1 | grep -i "isolated" || echo "no isolated nodes"
# Expect: "no isolated nodes"

# Check opposition edges and evidence format:
python -c "
import json
g = json.load(open('data/output/graphs/eco_spl1/graph.json'))
links = g['links']
opp = [e for e in links if e.get('type') == 'opposition']
bad_ev = [e for e in links if not isinstance(e.get('evidence'), list)]
print(f'opposition edges: {len(opp)}')
if opp: print(f'  sample: {opp[0][\"source\"]} ↔ {opp[0][\"target\"]}')
print(f'edges with non-list evidence: {len(bad_ev)}  (expect 0)')
"
```

**Variant — focus on one term:**

```bash
cmapr graph data/output/corpus/eco_spl1/corpus.json \
    -t data/output/rarities/eco_spl1/terms.json \
    --focus sign --depth 2 \
    --output data/output/graphs/eco_spl1/sign_neighbourhood.json

python -c "
import json
g = json.load(open('data/output/graphs/eco_spl1/sign_neighbourhood.json'))
print(f'neighbourhood: {len(g[\"nodes\"])} nodes, {len(g[\"links\"])} links')
ids = [n[\"id\"] for n in g[\"nodes\"]]
print(\"nodes:\", ids)
"
# Expect: small graph; 'sign' present; all nodes connected to 'sign' within 2 hops
```

---

## Step 4 — Export to HTML

```bash
cmapr export data/output/graphs/eco_spl1/graph.json \
    --format html \
    --title "Eco SPL1 — Concept Graph" \
    --output data/output/exports/eco_spl1/index.html

open data/output/exports/eco_spl1/index.html
```

**Visual checks in browser:**

- [x] Page loads without blank screen or JS console errors (`⌘⌥J` to open DevTools)
- [x] Nodes are visible and labelled
- [x] Node size varies (larger nodes = higher rarity score)
- [x] Edge colors differ by type (blue=definition, green=kind-of, orange=production, red=dependence, grey=cooccurrence)
- [x] `cooccurrence` edges are dashed, other edges are solid with arrowheads
- [x] Hovering an edge shows a tooltip with 1–3 ranked evidence sentences (separated by `<hr>` when multiple)
- [x] Legend is visible (bottom-left); checkboxes are present for each edge type
- [x] Unchecking "cooccurrence" hides dashed edges; nodes with no remaining edges also hide
- [x] Clicking a node opens the detail panel (right side) showing term, frequency, connected edges
- [x] Clicking empty canvas closes the detail panel
- [x] Dragging a node pins it in place; simulation continues around it
- [x] Double-clicking a node shows only that node and its direct neighbours (expand); all other nodes/edges hide
- [x] Double-clicking the same node again restores the full graph (collapse)
- [x] Type filter (legend checkboxes) still works correctly while in expanded view

**Sanity check — expected terms visible as nodes:**

Look for at least a handful of these in the graph: *sign*, *signification*, *semiosis*, *interpretant*, *code*, *symbol*, *icon*, *index*, *denotation*, *connotation*, *meaning*, *referent*

---

## Step 5 — End-to-end via `run` command

The `run` command chains ingest → rarities → graph → export in one call.
`--output-dir` is a global flag on `cmapr`, not on `run`:

```bash
cmapr --output-dir data/output/runs/eco_spl1 \
    run data/input/eco_spl1.txt \
    --clean-ocr \
    --top-n 50

open data/output/runs/eco_spl1/exports/eco_spl1/index.html
```

**Check:**
- All four stages complete without error
- Output directory contains `corpus/`, `rarities/`, `graphs/`, `exports/` subdirectories
- HTML opens and passes the same visual checks as Step 4

---

## Step 6 — Web UI (`cmapr serve`)

Run from the project root (the directory containing `data/`).

```bash
pip install 'concept-mapper[serve]'   # one-time
cmapr serve
# Expect: browser opens at http://localhost:8000
```

**New run flow:**

1. Enter `data/input/eco_spl1.txt` in the source text field; submit
2. Wait for ingest + rarities to complete; review page loads
3. Uncheck 2–3 junk terms; click "Continue to graph options"
4. Leave defaults; click "Build graph"
5. Result page loads with embedded D3 visualization

**Check at each step:**

- [x] Home page lists existing works (if any) as quick-resume links
- [x] Submitting an invalid file path redirects home with an error message
- [x] Review page shows all candidate terms as checked checkboxes with score and frequency
- [x] Unchecked terms are absent from the graph (confirm by checking node list vs. submitted form)
- [x] Graph options page shows top-n, threshold, depth, focus fields
- [x] Result page embeds the D3 visualization in an iframe
- [x] "Open full screen" link opens the standalone HTML in a new tab
- [x] "Re-run graph" button goes to graph options with the same work pre-selected, without re-ingesting

**Existing corpus shortcut:**

```bash
# After the first run, corpus already exists
# Go to http://localhost:8000
# Click the work name under "Resume existing corpus"
# Expect: jumps directly to the review page (no ingest)
```

**Re-run graph without re-ingesting:**

```bash
# From the result page, click "Re-run graph"
# Change top-n or focus; submit
# Expect: new graph builds without running ingest or rarities again
```

---

## Common failure modes

| Symptom | Likely cause |
|---------|--------------|
| Rarities returns 0 terms | Threshold too high; try `--threshold 0.0` |
| Graph has ratio > 3:1 | `prune_to_ratio` not running; check `--output` path is writable |
| All edges are `cooccurrence` | `PropositionExtractor` not finding patterns; corpus may be too short |
| Short/junk nodes in graph | `NodeFilter` not applied to extracted nodes |
| Tooltip is empty | Evidence not stored on edge; check `proposition_extractor._score_sentence` |
| JS console error: `d.evidence is not iterable` | Edge `evidence` field is a string not a list |
| Node detail panel blank | Node missing `frequency` or `score` attribute in graph JSON |
| Double-click expands but graph stays blank | `applyVisibility` not finding neighbour IDs — check `l.source.id` vs `l.source` after D3 force simulation binds objects |
| `cmapr serve` fails to start | Run `pip install 'concept-mapper[serve]'`; confirm you are in the project root |
| Web UI shows blank result iframe | Check browser console for mixed-content errors; ensure `/exports/{work}/index.html` route returns 200 |
| Re-run graph re-ingests unexpectedly | `/build` calls rarities but not ingest; if ingest runs, check server logs for subprocess error in the rarities step that falls through to ingest |
