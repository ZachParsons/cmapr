# Graph Specification

A specification for what the concept graph should represent, what data it should contain, and how it should be displayed.

---

## Goal

Produce a readable map of an author's idiosyncratic conceptual vocabulary — the terms they use in specialized ways, and the propositions that express how those terms relate to each other. The graph should be derived from and faithful to the source text, not a statistical artifact of word proximity.

---

## Nodes

### What a node represents

A node is a **significant content word or phrase** from the author's vocabulary. It should be something a reader of the text would recognize as a meaningful term — not a fragment, abbreviation, function word, or statistical noise.

### Inclusion criteria

A node must pass **all** of the following:

- **POS**: noun, verb, adjective, or adverb (NN, NNS, VB, VBZ, VBG, JJ, RB, etc.) — no determiners, pronouns, prepositions, punctuation
- **Length**: at least 4 characters
- **Not a fragment**: not a prefix of a longer word that appears in the corpus (e.g. *structu* is a fragment of *structure*)
- **Not an abbreviation**: not all-caps with length ≤ 4 (e.g. *nn*, *dt*, *s-code* needs evaluation)
- **Not a stopword**: not in the extended stopword list
- **Minimum frequency**: appears at least 3 times in the corpus
- **Not garbled**: fails WordNet lookup AND falls below corpus minimum frequency (3). A word absent from WordNet is not automatically rejected — philosophical neologisms (*semiosis*, *interpretant*, *Dasein*) are valid terms that won't appear in WordNet. Rejection requires both signals. Future: interactive per-work "special terms" dictionary where unknown words are explicitly accepted or rejected by the user (see future spec).

### What makes a node significant (ranking)

Nodes should be ranked by the existing `PhilosophicalTermScorer` hybrid score (rarity + TF-IDF + neologism + definitional context + capitalization). Only the top-scoring terms should appear in the graph — the rarities term list is the right input, not the full vocabulary.

### Target node count

100–400 nodes for a book-length corpus. Below 100 is too sparse to be interesting; above 400 becomes unreadable.

---

## Edges

### What an edge represents

An edge is a **proposition** — a claim the author makes about how two terms relate. It should be readable as a sentence fragment: *sign produces meaning*, *interpretant is a mental entity*, *semiosis depends on a code*.

Each edge has:
- **source**: the subject term (node)
- **target**: the object term (node)
- **label**: the relational verb or preposition (the proposition)
- **type**: one of the semantic categories below
- **evidence**: the sentence(s) from the text that support this edge

### Edge types (semantic categories)

| Type | Pattern | Example label | Status |
|------|---------|---------------|--------|
| `definition` | Explicit authorial definition: *by X I mean Y*, *X is defined as*, *X denotes Y* | *is defined as*, *means*, *denotes* | v1 |
| `kind-of` | X is a type/kind/species of Y | *is a kind of*, *is a type of* | v1 |
| `production` | X produces / generates / gives rise to Y | *produces*, *generates*, *implies* | v1 |
| `dependence` | X presupposes / depends on Y | *presupposes*, *depends on*, *requires* | v1 |
| `component` | A, B, C together form/constitute/compose X — A and B are co-components | *co-constitutes*, *together with* | v1 |
| `cooccurrence` | fallback: X appears near Y (no proposition found) | *(co-occurs with)* | v1 |
| `property` | X is characterized by Y; X has Y | *has*, *involves*, *requires* | deferred |
| `opposition` | X contrasts with / differs from Y | *contrasts with*, *differs from*, *opposes* | deferred |

`cooccurrence` edges are a **last resort fallback** — not the primary edge type. A graph composed mostly of cooccurrence edges is a failure state.

**Note on `definition`:** The `definition` edge type is reserved for *explicit* metalinguistic definitions only — sentences where the author explicitly defines a term. The full conceptual definition of a term in the graph is emergent: it is the cluster of all typed edges pointing to and from that node (kind-of + production + dependence + property etc. together constitute what the term *means*). The node detail panel should surface this cluster as the term's working definition.

**Copular disambiguation rules (v1):**
1. Explicit marker (*by X I mean*, *X is defined as*, *X denotes*) → `definition`
2. Kind marker (*X is a type/kind/sort/species of Y*) → `kind-of`
3. Plain NP complement (*X is a vehicle*, *X is a process*) → `property` *(deferred; no edge extracted in v1)*
4. AP complement (*X is important*, *X is abstract*) → `property` *(deferred; no edge extracted in v1)*

Rule 3 is provisional — plain NP copular sentences may warrant reclassification as `definition` in some contexts. Revisit once real extraction results are available.

### Edge quality hierarchy

1. **Definitional** — extracted from explicit authorial definitions (*X is*, *X means*, *by X I mean*)
2. **Grammatical** — extracted from SVO/copular/prepositional patterns in sentences containing both terms
3. **Cooccurrence** — statistical fallback when no grammatical relation is found

### Target edge count

Edges:nodes ratio should be close to **1:1** (sparse, tree-like). A ratio above 3:1 produces an unreadable hairball. A ratio above 6:1 is a failure state (current output is 112 edges / 78 nodes ≈ 1.4:1 by count but the edge labels are all identical, making it useless).

---

## Graph model

### What the graph should NOT be

- A co-occurrence matrix (all edges = "co-occurs with")
- A hub-and-spoke structure (one dominant node connected to everything)
- A dense hairball (high edge:node ratio, overlapping labels)

### What the graph should be

- A **sparse network** where most nodes have 1–3 edges
- **Asymmetric**: some nodes are hubs (the most central concepts), most are leaves or small clusters
- **Labeled**: edge labels are readable propositions, not "co-occurs with"
- **Community-structured**: related concepts cluster together, with labeled edges bridging clusters
- **Grounded**: every edge traceable to a specific sentence in the text

### How edges should be constructed (pipeline)

The current `analyze`-based approach (hub-and-spoke per search term) is wrong for graph construction. The correct approach:

1. Start from the rarities term list (pre-curated significant terms)
2. For each pair of terms that co-occur in at least one sentence, attempt to extract a grammatical proposition
3. If a proposition is found, add a labeled edge
4. If no proposition is found but co-occurrence is strong (PMI or LLR above threshold), add a cooccurrence edge as fallback
5. Prune: remove cooccurrence-only edges unless they're the sole connection for a node
6. Result: a graph where most edges carry meaningful labels

---

## Visualization

### Layout

- **Force-directed** (D3) with tuned parameters:
  - High repulsion charge (push nodes apart): `charge: -400` or stronger
  - Short link distance for strong edges, longer for weak: link distance proportional to `1 / edge_weight`
  - Collision detection to prevent label overlap
- **No dominant center**: if one node has degree > 10, reduce its attraction force

### Node display

- **Size**: proportional to the node's significance score (hybrid rarity score), not degree
- **Label**: always visible, font-size proportional to significance
- **Color**: community/cluster membership
- **Tooltip**: show definition, frequency, top evidence sentence on hover

### Edge display

- **Label**: always show the proposition verb/preposition on the edge (not "co-occurs with")
- **Width**: proportional to edge weight (number of supporting sentences)
- **Color**: by edge type (definition = one color, kind-of = another, etc.)
- **Arrow**: directed edges for asymmetric relations (X produces Y ≠ Y produces X)
- **Cooccurrence edges**: visually distinct (dashed, lighter color) to distinguish from proposition edges

### Readability thresholds

- If nodes > 300: hide low-degree nodes by default, add a slider to reveal them
- If edge labels overlap: show labels only on hover
- Minimum font size: 10px

### v1 vs v2 scope

**v1 (current priority):** static layout that is correct and readable. Force parameters tuned, edge labels visible, directed arrows, co-occurrence edges visually distinct. No new interactivity.

**v2 (future):** node expand/collapse, node detail panel (definition, frequency, source location), edge type show/hide toggle, section subgraph navigation.

---

## Commands

Two commands cover the graph workflow:

- **`cmapr run`** — full pipeline from scratch: ingest → rarities → graph → export. One command when starting fresh.
- **`cmapr graph`** — isolated graph step only, takes a pre-built corpus and terms file. Use this during development or when ingest/rarities are already done and only the graph needs to be regenerated.

```
# Full pipeline (run once or from scratch):
cmapr run data/input/eco_ch1.txt --top-n 30

# Isolated graph step (iterative development):
cmapr graph data/output/corpus/eco_ch1/corpus.json \
    --terms data/output/rarities/eco_ch1/terms.json \
    --output data/output/graphs/eco_ch1/graph.json
```

---

## Workflows

### A) Threshold-driven (no seed words)

The user passes a significance threshold (and optionally a count) and the app selects the top-scoring rarities terms automatically, builds the graph from all of them.

```
cmapr graph corpus.json --terms terms.json --threshold 2.0 --count 50
```

The `--terms` file is always required for `graph`. `--threshold` and `--count` filter it further. `cmapr run` generates the terms file automatically as part of the pipeline.

### B) Seed-word-driven (one or more starting words)

The user names one or more terms explicitly. The graph shows those terms as the seed nodes and expands outward to terms that are significantly linked to them.

```
cmapr graph corpus.json --seed "sign" --seed "interpretant"
```

This is closer to what `analyze` does for a single term, but generalized to multiple seeds and merged into a single graph.

#### B1) POS filter — nouns or verbs

Limit the linked (non-seed) nodes to a specific part of speech. Useful for a graph of "what does sign *do*" (verbs) vs. "what is sign *related to*" (nouns).

```
cmapr graph corpus.json --seed "sign" --pos nouns
cmapr graph corpus.json --seed "sign" --pos verbs
```

#### B2) Count limit

Cap the number of linked nodes per seed term.

```
cmapr graph corpus.json --seed "sign" --count 20
```

#### B3) Group by section

Split and visually group the linked nodes by where they appear in the source text — by part, chapter, or paragraph. Produces a graph where community membership reflects document structure rather than just co-occurrence clustering.

```
cmapr graph corpus.json --seed "sign" --group-by chapter
cmapr graph corpus.json --seed "sign" --group-by paragraph
```

#### B4) Depth limit

Control how many hops from the seed words the graph expands. Depth 0 = only the seed term and its direct neighbors. Depth 1 = seed + neighbors + neighbors-of-neighbors. Uses the same window logic as `analyze --window`.

```
cmapr graph corpus.json --seed "sign" --depth 0   # seed + immediate neighbors only
cmapr graph corpus.json --seed "sign" --depth 1   # two hops out
```

#### B5) graph = batch analyze + merge

The `graph` command is structurally a batch-and-merge run of `analyze`: run `analyze` for each seed term, collect all the `ContextualRelation` objects, merge them into a single graph. This is already the implementation direction — the spec formalizes it.

For workflow A (threshold-driven), the seed set is auto-populated from the rarities list filtered by threshold. For workflow B, the seed set is user-supplied. The graph construction logic is the same either way.

---

## Current state vs. target state

| Dimension | Current | Target |
|-----------|---------|--------|
| Node quality | Mixed: valid terms + fragments + abbreviations + OCR artifacts | Only significant, clean content words |
| Edge semantics | All "co-occurs with" | Typed propositions with readable labels |
| Edge:node ratio | ~6:1 (12171/1866) or ~1.4:1 (112/78) | ~1:1 |
| Graph shape | Hub-and-spoke or dense hairball | Sparse asymmetric network |
| Evidence | None surfaced | Every edge links to source sentence |
| Layout | Default D3 force params | Tuned for readability: strong repulsion, collision detection |

---

## Implementation priorities

1. **Node filtering** — enforce inclusion criteria at graph construction time; reject fragments, abbreviations, OCR artifacts
2. **Edge extraction** — replace co-occurrence-only edges with grammatical proposition extraction; make co-occurrence a fallback
3. **Edge pruning** — enforce ~1:1 ratio; prune in order: (1) co-occurrence edges first regardless of weight, (2) then lowest-weight grammatical edges. Pruning threshold (target ratio) is a configurable parameter, default 2:1.
4. **Visualization tuning** — D3 force parameters, edge labels, directed arrows, cooccurrence edge styling
5. **Evidence surfacing** — make every edge traceable to source text in the tooltip. *(Future spec needed: how many sentences to surface, how to rank/select the best evidence sentence when multiple exist. Defer to v2.)*

---

## Decisions

**1. Non-rarities terms as nodes**
Yes. A term that is not in the rarities list may still appear as a node if it is reliably linked to an existing node — i.e. it appears as the target of a grammatical proposition (SVO object, copular complement, prepositional object) in a sentence containing a rarities term. The rarities list defines the seed vocabulary; the graph may expand beyond it through extraction.

Two node roles:
- **Seed node** — sourced from the rarities term list. Admission is by rarity score threshold.
- **Extracted node** — sourced from grammatical extraction (object/complement of a seed node). Must pass the same inclusion criteria as seed nodes (length ≥ 4, frequency ≥ 3, not garbled, not a stopword, valid POS) before being admitted as a node. Future: consider a lower admission bar for extracted nodes that appear as the target of multiple independent extractions.

**2. Multi-word terms**
Multi-word terms (*body without organs*, *intentional stance*, *sign-function*) should be treated as single nodes. This requires noun-chunk or phrase detection at extraction time rather than single-token extraction. Currently unimplemented — requires spaCy or a chunking pass.

**3. Edge direction**
Edges should be directed when the data supports it. If node1 is the grammatical subject, the edge verb is the predicate, and node2 is the object in the same sentence, the edge is directed node1 → node2. Undirected edges are used only when direction cannot be determined (e.g. pure co-occurrence fallback).

**4. Co-occurrence as edge source**
Co-occurrence (same sentence or same paragraph) is used to *discover* candidate term pairs — terms that may be related. Once a candidate pair is found via co-occurrence, attempt grammatical extraction to find a proposition. If a proposition is found, use it as the edge. If not, a co-occurrence edge is the fallback. Co-occurrence is a discovery mechanism, not a final edge type.

**v1 scanning scope:** Only scan sentences where *both* terms appear together. **v2:** Also scan all sentences containing either term individually, then match extracted relations across terms. Document as future enhancement.

**5. Seed words always appear as nodes**
Yes. User-supplied seed words (workflow B) are always nodes regardless of rarities score. They are the fixed points the graph is built around.

**6. Depth 0 definition**
Depth 0 = terms that appear in the same entity as the seed term, where the entity is determined by the window argument: same sentence if a sentence window is given, same paragraph if a paragraph window is given. Depth 1 = terms linked to the depth-0 neighbors, and so on.

---

**7. Visualization grouping by section**
Separate subgraphs per section — one D3 force layout per section (part, chapter, etc.), displayed as distinct panels or selectable views. Nodes that appear in multiple sections may appear in multiple subgraphs.

**9. Multigraph edges**
Multiple edges between the same source/target pair are allowed if they have different types. E.g. if "sign" both *is a kind of* and *produces* "meaning" (in different sentences), both edges are kept. Same-type duplicate edges are merged (aggregated into one with combined evidence and incremented weight).

**10. Co-occurrence edges scope**
Co-occurrence fallback edges are only added between two terms when those terms have a *direct* grammatical relation to each other (subject↔object, copular subject↔complement, etc.) and no typed proposition was found. Terms that merely appear in the same sentence but are not grammatically connected to each other (e.g. two subjects of the same verb, or two terms in a list) do NOT get a co-occurrence edge in v1.

Exception: the **composition pattern** (see Decision 12). Future: consider adding non-composition indirect co-occurrence edges as a separate weak edge category with visual differentiation (grammatical = solid, co-occurrence = dashed/lighter) and toggle-ability.

**12. Composition/constitution pattern**
When a sentence matches the pattern *"A, B, and C form/constitute/compose/make up X"*, two things are extracted:
1. Directed edges from each component to X: A → *constitutes* → X, B → *constitutes* → X, C → *constitutes* → X (edge type: `production` or `dependence` depending on context)
2. Undirected `component` edges between all co-components: A ↔ B, A ↔ C, B ↔ C (edge label: *co-constitutes*)

Rationale: co-components in a composition pattern are not merely co-occurring — they mutually define each other through their shared role in constituting a single entity (e.g. sign + interpretant + referent constituting semiosis in Peircean semiotics). This is semantically distinct from two terms that happen to share a sentence.

**11. Visualization versions**
v1: static force-directed layout, readable labels, correct data. No interactivity beyond what already exists (drag, zoom). v2: node expand/collapse, node detail panel, edge type toggle, section subgraph navigation.

**8. Graph as end artifact**
The graph is the primary end artifact. It is a standalone interactive HTML visualization. Future interactivity requirements:
- **Node expand/collapse**: expand a node to show its directly linked neighbors; collapse to hide them
- **Node detail panel**: click a node to reveal its definition, frequency, and source text page number or location
- **Edge show/hide**: toggle edge types (e.g. hide co-occurrence edges, show only definitional edges)
- **Section subgraph navigation**: switch between section subgraphs
- These are future requirements — the current priority is correct data and readable layout

# dev workflow
cmapr graph data/output/corpus/eco_spl1/corpus.json \
    --terms data/output/rarities/eco_spl1/terms.json \
    --output data/output/graphs/eco_spl1/graph.json && \
cmapr export data/output/graphs/eco_spl1/graph.json --format html && \
open data/output/viz/eco_spl1/index.html