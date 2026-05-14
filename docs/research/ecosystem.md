# Ecosystem Survey

A targeted assessment of concept-mapping, NLP, and knowledge-graph Python libraries — what each does, how it fits with cmapr's current stack, and whether to **replace**, **augment**, or **skip**.

The goal is not a generic library catalog. It's: for each component of cmapr, is there an off-the-shelf piece worth importing? And which libraries open *new* capabilities (link prediction, neural relation extraction, semantic similarity) that the current pipeline doesn't have?

> Source: maintainer assessment as of 2026-05. Reassess yearly — ML library landscape shifts faster than this list does.

---

## Verdict at a glance

| Library | Role | Fit with cmapr | Verdict |
|---|---|---|---|
| **REBEL** | Neural relation extraction (seq2seq) | Augment `graph/proposition_extractor.py` with an optional `--neural` mode | **Augment** (high-leverage) |
| **sentence-transformers** | Sentence/term embeddings | Power evidence ranking, term clustering, semantic similarity queries | **Augment** (high-leverage) |
| **PyKEEN** | KG embeddings + link prediction | Embed `ConceptGraph`, suggest missing edges, similarity queries | **Augment** (new capability) |
| **ConceptNet** | Public commonsense KG | Cross-reference / validate extracted relations against an external baseline | **Augment** (v2 idea) |
| **gensim** | Topic modeling (LDA, NMF), word2vec | Topic-based distinctiveness signal alongside rarity score | **Augment** (modest) |
| **pyvis** | vis.js wrapper for network viz | Could replace `export/html.py` but loses cluster force, type filters, panel | **Skip** |
| **rdflib** | RDF/OWL/Turtle serialization | Alt export format; orthogonal to current JSON/GraphML/GEXF | **Skip** unless RDF needed |
| **spaCy** | NLP pipeline (already used) | Already wired as `[spacy]` extra for noun-chunk extraction | **Already integrated** |
| **NetworkX** | Graph algorithms (already used) | Core dependency; backs `ConceptGraph` | **Already integrated** |

---

## Detail: high-leverage augmentations

### REBEL (Babelscape) — neural relation extraction

**What it does.** Seq2seq transformer (BART-based) fine-tuned to read a sentence and emit `(subject, relation, object)` triples directly. Trained on Wikipedia/Wikidata; outputs ~250 Wikidata relation types. Hugging Face model: `Babelscape/rebel-large` (~500M params).

**Fit with cmapr.** Replaces the work of `PropositionExtractor`'s regex extractors (`_try_definition`, `_try_kind_of`, etc.) for the catch-all case. REBEL's output relation labels don't map 1-to-1 to cmapr's typology (`definition`, `kind-of`, `production`, `dependence`, `component`, `opposition`, `property`, `relation`), but the high-confidence triples could be classified into cmapr's types via a lookup table (e.g., REBEL's `subclass_of` → cmapr's `kind-of`).

**Pros.**
- Massively better recall on complex sentences (passives, relative clauses, nested clauses) than the regex extractors.
- Catches relations the regex patterns don't even attempt (e.g., causal, spatial, temporal).
- Same `Proposition` output shape — no schema change.

**Cons.**
- ~500M-param model: ~2 GB on disk, ~2s/sentence on CPU. Practical only as opt-in (`--neural`) or via batch GPU.
- Relation label mapping is lossy: many REBEL relations don't fit cmapr's seven types and would be lumped into the generic `relation`.
- Heavy optional dependency (PyTorch + transformers).

**Recommendation.** Add as an opt-in `[neural]` extra: `pip install 'concept-mapper[neural]'`. New `cmapr graph --neural` flag fans out to REBEL per sentence containing 2+ seed terms, classifies relations into cmapr types via lookup, merges with regex output (deduplicate on source/target/type, prefer the higher-confidence proposition). Reasonable size-of-prize: 2–3× more typed edges on the same corpus.

---

### sentence-transformers — semantic embeddings

**What it does.** Bi-encoder library producing dense sentence/term embeddings via fine-tuned BERT variants. Common models: `all-MiniLM-L6-v2` (fast, 22M params), `all-mpnet-base-v2` (better, 110M params). Apache-2.0.

**Fit with cmapr.** Three immediate uses:

1. **Evidence ranking** (`_score_sentence` in `proposition_extractor.py`). Current heuristic favors definition markers + proximity + brevity. A neural relevance score (cosine similarity between the candidate sentence and a constructed "X relates to Y" probe) would catch evidence that doesn't match the existing markers.

2. **Term clustering**. Current community detection (`detect_communities` in `graph/metrics.py`) uses Louvain on edge structure. A semantic clustering pass (k-means or HDBSCAN on term embeddings) could complement it: graph-structural communities vs. distributional-semantic communities are not the same thing.

3. **Concept similarity queries**. New CLI verb: `cmapr similar TERM --top-n 10` returns the k nearest terms by embedding. Useful for vetting (find aliases) and for suggesting seed terms.

**Pros.**
- Small models run fine on CPU; one-time embedding of the term list (50–100 terms) is sub-second.
- Apache license, well-maintained, vast model zoo.
- Drop-in for many uses; no architectural lock-in.

**Cons.**
- One more dependency (PyTorch transitive).
- Embedding quality is domain-sensitive — generic sentence-transformer models may underperform on philosophical/specialized text vs. domain-tuned ones.

**Recommendation.** Add as `[neural]` (shared with REBEL) or separate `[similarity]` extra. First high-value addition: similarity-based evidence ranking inside `proposition_extractor.py`, gated by `--neural` flag. Second: `cmapr similar` command.

---

### PyKEEN — knowledge graph embeddings

**What it does.** Library for training KG embedding models (TransE, RotatE, ComplEx, ConvE, ~40 variants) over `(head, relation, tail)` triples. Produces vectors per entity and per relation; supports link prediction (score every possible `(h, r, t)` and surface the highest-scoring ones not in the training set). MIT license.

**Fit with cmapr.** A `cmapr graph` output is exactly a KG: typed directed triples over a fixed entity set. Two concrete uses:

1. **Link prediction**. Train a TransE model on the extracted edges, then surface high-confidence triples that the extractor missed. Output as suggested edges with confidence scores — manual vet, then optionally add to the graph.

2. **Entity similarity**. Cosine over entity embeddings → "which nodes behave similarly in the graph?" — different from semantic similarity (sentence-transformers) because it's purely structural.

**Pros.**
- Closes the loop on extraction recall: catches relations the regex/REBEL pipelines miss because the supporting sentence is too implicit.
- No new annotation needed; trains on cmapr's own output.

**Cons.**
- Needs a reasonable-sized graph to train on (~100s of edges minimum; the existing 50-term graphs are borderline).
- Embedding model selection / hyperparameter tuning is a real chunk of work.
- Conceptually more involved than the other items here; pays off most after the graph extractor is at peak quality.

**Recommendation.** Defer to v3. Real value once REBEL + sentence-transformers have lifted extraction quality and the graphs are large enough to learn from. When it does land, the natural surface is `cmapr predict GRAPH --top-n 20 --model transe` writing a `suggestions.json` of high-confidence missing edges for manual vetting.

---

### ConceptNet — external commonsense KG

**What it does.** Large multilingual commonsense knowledge graph (8M+ nodes, 21M+ edges) with relations like `IsA`, `PartOf`, `RelatedTo`, `Causes`, `HasA`, `UsedFor`. Available as REST API and as an offline dump. Apache-2.0.

**Fit with cmapr.** Two uses:

1. **Validation**. After extracting a graph from the text, look up each extracted edge in ConceptNet: does ConceptNet agree? Mismatch could mean (a) the text uses a term idiosyncratically (the entire point of cmapr — flag and surface) or (b) the extractor is wrong.

2. **Augmentation**. For each term, fetch its top ConceptNet neighbors as suggested "context" nodes the text may also use. Could feed the seed-term suggestion pipeline.

**Pros.** Free, large, well-typed; provides an external ground-truth baseline that cmapr currently lacks.

**Cons.** ConceptNet's relations don't perfectly align with cmapr's (their `IsA` ≈ cmapr's `kind-of`; their `RelatedTo` is too generic). Coverage of *philosophical* concepts is patchy — works great for "dog IsA mammal", less so for "sublation IsA dialectical movement".

**Recommendation.** v2 idea. Concrete first step: `cmapr conceptnet TERM` shows ConceptNet's neighborhood for a term — purely informational, no integration into the extractor or graph. If that proves useful in practice, plumb it into the evidence panel of the HTML viz as a "related concepts" sidebar.

---

## Detail: modest gains

### gensim — topic modeling

**What it does.** Classic NLP library: LDA / NMF topic models, word2vec, doc2vec, FastText. BSD license.

**Fit with cmapr.** Topic distribution per chapter could complement the rarity score: a term that's a topic-defining word in one chapter is structurally important even if its corpus-wide rarity is modest. The cluster feature already separates per-chapter sub-graphs; topic distinctiveness would add a "what is this chapter *about*?" label.

**Pros.** Lightweight, mature, pure-Python (no PyTorch).

**Cons.** Topic modeling needs ~20+ documents to be meaningful. cmapr's typical input is one corpus per author, not a corpus of corpora. Limited applicability for the v1 use case.

**Recommendation.** Skip for now. Revisit once you have multiple authors / multiple works in one analysis session — at that point, topic modeling earns its keep.

---

## Detail: skip

### pyvis — interactive network viz

**What it does.** Python wrapper around vis.js. Generates standalone HTML with a force-directed network and built-in physics controls. Trivial to set up: `Network().add_nodes(...).add_edges(...).show("out.html")`. MIT license.

**Fit with cmapr.** Could replace `export/html.py`. But cmapr's current HTML viz is bespoke — type-filtered legend, expand/collapse, node detail panel, cluster force, per-type edge styling. Reimplementing those on top of pyvis means fighting vis.js's API for every customization.

**Recommendation.** Skip. The investment to migrate would be larger than the maintenance cost of the current D3 code. Reconsider only if the D3 maintenance burden becomes a problem.

### rdflib — RDF triples

**What it does.** Standard library for RDF/OWL/SPARQL in Python. Reads/writes Turtle, N-Triples, JSON-LD. BSD license.

**Fit with cmapr.** Adds RDF/Turtle as an export format. The data shape already maps cleanly: each cmapr edge is essentially a triple. But cmapr already exports GraphML, GEXF, CSV, D3 JSON, HTML — adding RDF is incremental, only worth it if a user genuinely wants to load the graph into a triplestore.

**Recommendation.** Skip until a concrete request. ~50 lines if it ever lands.

---

## Top picks for next initiatives

Ordered by leverage per unit of work, given cmapr's current state:

1. **`[neural]` extra + `cmapr graph --neural` (REBEL + sentence-transformers)** — single biggest improvement to extraction quality. Real shift from "regex covers what regex covers" to "neural fills the gaps." Plan-worthy initiative.
2. **`cmapr similar TERM` (sentence-transformers solo)** — small surface (~150 lines), genuinely useful for vetting and seed-term discovery. Could land in one session.
3. **PyKEEN link prediction** — defer until extraction quality is at peak; the value comes from learning from a high-quality graph.
4. **ConceptNet lookup helper** — exploratory; build only if a concrete use case surfaces.

Out of scope (not on the list above): GUI improvements, alternative export formats, topic modeling. Revisit those when the analytical pipeline is saturated.

---

## What's *not* in this survey

Deliberately omitted because they're outside cmapr's scope or would require restructuring the project:

- **Neo4j / property graph DBs** — replaces JSON storage. Worth considering only if/when cmapr targets corpora >100 documents (current limit per `docs/roadmap.md`).
- **OpenAI / Anthropic / OSS LLMs for extraction** — could supersede REBEL for relation extraction, but introduces cost, latency, and provenance concerns. Possibly worth a future "high-quality batch extraction" mode separate from the interactive pipeline.
- **Streamlit / Gradio frontends** — would replace `cmapr serve` (FastAPI + Jinja2). The current web UI is bespoke for a reason (custom D3 viz, per-stage flow). Reconsider only if the UI roadmap grows.
- **`docling` / `pymupdf4llm` / `unstructured` / `nougat`** — covered by the separate **structured ingestion pipeline** roadmap entry, not this survey.
