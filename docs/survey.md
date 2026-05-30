# Library Survey

A pipeline-keyed catalog of NLP, term-extraction, relation-extraction, knowledge-graph, and visualization libraries — for each one, what it does, which stage of cmapr it could augment, and whether to **adopt**, **augment**, **defer**, or **skip**.

The survey is organized to mirror cmapr's processing pipeline (see [`architecture.md`](architecture.md)):

```
ingest ─► rarities ─► graph ─► export
                                 ▲
                  cross-cutting (merge, cluster, KG-level ops)
```

For each stage, the recap says *what cmapr already does*, then surveys candidates that could replace, augment, or sit alongside the current implementation. The top picks at the end roll the per-stage verdicts up into a recommended sequence.

> Maintainer assessment as of 2026-05. Reassess yearly — ML library landscape shifts faster than this list does.

---

## Verdict at a glance

| Library | Stage | Role | Verdict |
|---|---|---|---|
| **docling / pymupdf4llm / unstructured / nougat** | 1 ingest | PDF → structured text + TOC + layout | **Investigate** (tracked separately as "structured ingestion pipeline") |
| **spaCy** | 1 ingest | Tokenize, POS, NER, noun chunks | **Already integrated** (`[spacy]` extra, noun chunks) |
| **stanza** | 1 ingest / aux | Dependency parsing | **Already integrated** (used by `cmapr diagram`) |
| **KeyBERT** | 2 rarities | Embedding-based keyword extraction | **Augment** (cross-check signal) |
| **YAKE** | 2 rarities | Unsupervised keyphrase extraction, no reference corpus | **Augment** (cheap baseline signal) |
| **PKE** | 2 rarities | Multi-algorithm keyphrase extraction (TopicRank, MultipartiteRank, PositionRank) | **Augment** (research lens) |
| **GLiNER** | 2 rarities / 3 graph | Zero-shot NER with custom labels | **Augment** (high-leverage typing) |
| **BERTopic** | 2 rarities / cross-cutting | Embedding-based topic clustering | **Defer** (needs multi-doc input) |
| **gensim** | 2 rarities | Topic modeling (LDA, NMF), word2vec | **Defer** (needs multi-doc input) |
| **REBEL** | 3 graph | Neural seq2seq relation extraction | **Augment** (high-leverage; roadmap rb.1–rb.7) |
| **sentence-transformers** | 3 graph / 2 rarities | Sentence/term embeddings | **Augment** (high-leverage; roadmap st.1–st.8) |
| **LangExtract** | 3 graph | LLM-based structured extraction with source grounding | **Investigate** (closest match to roadmap goals) |
| **Instructor** | 3 graph | Pydantic-typed LLM outputs over any provider | **Investigate** (lightweight harness if LLM extraction lands) |
| **spaCy-LLM** | 3 graph | LLM-powered NER/relations as spaCy components | **Investigate** (slots into existing spaCy stack) |
| **Outlines / BAML** | 3 graph | Constrained / type-safe LLM generation | **Skip** (Instructor covers the same surface) |
| **Microsoft GraphRAG** | 3 graph | Full LLM knowledge-graph pipeline | **Skip** (overlaps cmapr's whole graph stage) |
| **Stanford OpenIE** | 3 graph | Classical open information extraction | **Skip** (REBEL strictly better) |
| **DyGIE++** | 3 graph | Joint entity + relation extraction | **Skip** (research-grade, low maintenance) |
| **ConceptNet** | 3 graph / cross-cutting | Public commonsense KG | **Augment** (v2 validation/augmentation) |
| **PyKEEN** | cross-cutting | KG embeddings + link prediction | **Defer** (until extraction quality peaks) |
| **pyvis** | 4 export | vis.js HTML viz wrapper | **Skip** (would lose bespoke D3 features) |
| **rdflib** | 4 export | RDF/Turtle/SPARQL serialization | **Skip** (no concrete RDF use case) |
| **Cytoscape.js / Sigma.js / 3d-force-graph** | 4 export | Alternative interactive viz front-ends | **Skip** (current D3 covers it) |
| **NetworkX** | 3 graph / cross-cutting | Graph algorithms | **Already integrated** (backs `ConceptGraph`) |

---

## Stage 1 — Ingest

**What cmapr does now.** `corpus/loader.py` reads `.txt` (UTF-8 → Latin-1 fallback) or `.pdf` (via `pdfplumber`). `preprocessing/cleaning.py` strips OCR artifacts when `--clean-ocr` is set. `preprocessing/tokenize.py`, `tagging.py`, `lemmatize.py` use NLTK; `structure.py` detects chapters/sections (TOC-guided when `--toc` supplied); `noun_chunks.py` optionally calls spaCy.

The known gap is the *ingest seam* itself — cmapr currently expects a clean raw text file plus a manually authored TOC. Anything that produces both directly from a PDF is interesting.

### docling · pymupdf4llm · unstructured · nougat — PDF → structured text

**What they do.** Each takes a PDF and produces structured text plus layout metadata (headings, tables, figures, reading order).

- **docling** (IBM Research, MIT): emits Markdown/JSON, table structure preservation, layout-aware reading order. Newest of the four, fastest-improving.
- **pymupdf4llm** (Artifex, AGPL): thin Markdown wrapper around PyMuPDF. Quick, license-encumbered.
- **unstructured** (Unstructured.io, Apache-2.0): broader ingestion (PDF, DOCX, HTML, EPUB), partition-based API, used heavily in RAG pipelines.
- **nougat** (Meta, MIT): vision-transformer OCR specialized for scientific PDFs — recovers math, tables, multi-column layout.

**Fit with cmapr.** Directly replaces the current `pdfplumber` step in `corpus/loader.py` and could eliminate the two-file workflow (raw text + manually cleaned TOC) by extracting structure directly. nougat is the right pick for OCR-scanned philosophical PDFs; docling/unstructured for born-digital.

**Recommendation.** Tracked separately as the **structured ingestion pipeline** roadmap entry (not duplicated here). Concrete first cut: prototype `cmapr ingest --pdf-backend docling` and compare its TOC against the hand-curated `--toc PATH` baseline on `eco_spl1`.

### spaCy — already integrated

Lazy-loaded under the `[spacy]` extra to extract multi-word noun chunks during ingest (`preprocessing/noun_chunks.py`). Could also supply dependency parses to Stage 3 (currently NLTK POS only); see `docs/roadmap.md` "SpaCy integration" entry for the upsides/downsides of broadening usage.

### stanza — already integrated

Used by `syntax/diagram.py` for the `cmapr diagram` auxiliary command. Dependency parsing only; not on the main ingest path.

---

## Stage 2 — Rarities (term extraction & scoring)

**What cmapr does now.** `analysis/rarity.py:PhilosophicalTermScorer` runs a 5-signal hybrid (ratio vs Brown corpus + TF-IDF vs Brown + WordNet-neologism + definitional-context patterns + mid-sentence capitalization). `terms/scoring.py` post-processes: strip quotes, score multi-word noun chunks, drop proper names, lemma+derivational suffix merge, fragment filter, POS filter, vetting accept/reject. The output is a `TermList` JSON.

Candidates in this stage either (a) provide a *different* distinctiveness signal that could feed the hybrid scorer, or (b) replace the whole rarity step with an embedding-based approach.

### KeyBERT — embedding-based keyword extraction

**What it does.** Computes BERT embeddings for candidate n-grams and the full document, ranks candidates by cosine similarity to the document. MIT.

**Fit with cmapr.** Adds signal 6 to `PhilosophicalTermScorer`: distinctiveness in the *semantic* sense (a term that is centrally about what the document is about), separate from statistical rarity. Catches central concepts that aren't statistically rare in Brown but are densely thematic in the text. Pairs naturally with sentence-transformers if that lands first.

**Recommendation.** **Augment** — once sentence-transformers infrastructure exists (st.1–st.2), KeyBERT is ~50 lines on top. Wire as an opt-in 6th signal with weight ~0.5.

### YAKE — unsupervised keyphrase extraction

**What it does.** Statistical features only — term position, casing, frequency, sentence dispersion. No reference corpus needed. MIT.

**Fit with cmapr.** Drop-in alternative to the Brown-corpus-comparative ratio. Useful when the target text is in a *register* (philosophical, technical) where Brown is a poor reference. Doesn't help when Brown is fine.

**Recommendation.** **Augment** — cheap to add as an alternative `--method yake` next to the existing `ratio | tfidf | neologism | hybrid` modes on `cmapr rarities`.

### PKE — Python Keyphrase Extraction

**What it does.** Unified API across multiple algorithms: TopicRank, MultipartiteRank, PositionRank, KP-Miner, TF-IDF, YAKE. Apache-2.0.

**Fit with cmapr.** Research lens — run several algorithms on the same corpus and compare their top-N to cmapr's hybrid scorer. Useful for *evaluating* the scorer, less so for production extraction.

**Recommendation.** **Augment** — only as an evaluation harness, not in the main pipeline. Could live under `scripts/eval_scorers.py`.

### GLiNER — zero-shot NER

**What it does.** Lightweight transformer that takes a sentence plus a *list of label names* ("philosophical_concept", "thinker", "movement", "work") and tags spans. ~200MB, ~10ms/sentence on CPU. Apache-2.0.

**Fit with cmapr.** Two uses:

1. **Term typing** — after the scorer surfaces a candidate, GLiNER classifies it as concept / person / work / movement. Lets `cmapr rarities` *automatically* drop thinker names (currently a `--no-filter-names` proper-name heuristic) and surface only concepts.
2. **Stage 3 seeds** — once typed, the graph build can apply per-type rules (concepts get full extraction; persons get a lighter "mentioned-by" treatment).

**Pros.** No LLM API. Custom labels. Fast on CPU.
**Cons.** Domain transfer — generic NER training underperforms on philosophical neologisms.

**Recommendation.** **Augment**, after sentence-transformers. Gate behind `--neural`. High leverage: cleaner term lists, less manual vetting.

### BERTopic — topic clustering

**What it does.** Sentence-transformer embeddings → UMAP dimensionality reduction → HDBSCAN clustering → c-TF-IDF per cluster. Produces interpretable topic labels. MIT.

**Fit with cmapr.** Could provide an *alternative* community-detection signal for `cmapr cluster`, where graph-structural communities (Louvain) and semantic communities are not the same thing. Also useful if cmapr ever ingests multiple authors in one session.

**Recommendation.** **Defer** — needs ~20+ documents to be meaningful. Most cmapr inputs are one author × one work.

### gensim — classical topic modeling

**What it does.** LDA / NMF / word2vec / doc2vec. BSD.

**Fit with cmapr.** Same shape as BERTopic but pre-neural. A term that's topic-defining in one chapter is structurally important even at modest corpus-wide rarity.

**Recommendation.** **Defer** — same multi-document constraint as BERTopic. Revisit once multi-work analyses become a use case.

---

## Stage 3 — Graph (relation extraction & build)

**What cmapr does now.** `graph/proposition_extractor.py` runs a regex priority chain per `(term_a, term_b)` pair × per sentence containing both: definition → kind-of → production → dependence → opposition → property → relation → POS-verb fallback. `extract_composition` adds component edges. `_score_sentence` ranks evidence. `graph/builders.py:build_proposition_graph` merges duplicates and adds a PMI co-occurrence fallback for un-typed pairs. `graph/node_filter.py:NodeFilter` filters endpoints.

This is the densest stage of cmapr and the one with the most ecosystem competition. The interesting candidates fall into three groups:

1. **Neural relation extraction** (REBEL) — directly augments the regex chain.
2. **LLM-based structured extraction** (LangExtract, Instructor, spaCy-LLM) — a different architecture entirely.
3. **Embeddings as auxiliary signal** (sentence-transformers, ConceptNet, GLiNER) — feed evidence ranking, validation, and edge typing.

### REBEL (Babelscape) — neural relation extraction

**What it does.** Seq2seq transformer (BART-based) fine-tuned to read a sentence and emit `(subject, relation, object)` triples directly. Trained on Wikipedia/Wikidata; outputs ~250 Wikidata relation types. Hugging Face model: `Babelscape/rebel-large` (~500M params).

**Fit with cmapr.** Replaces the work of `PropositionExtractor`'s regex extractors for the catch-all case. REBEL's output labels don't map 1-to-1 to cmapr's typology (`definition`, `kind-of`, `production`, `dependence`, `component`, `opposition`, `property`, `relation`), but the high-confidence triples can be classified into cmapr types via a lookup (e.g., REBEL's `subclass_of` → cmapr's `kind-of`).

**Pros.**
- Massively better recall on complex sentences (passives, relative clauses, nested clauses).
- Catches relations the regex patterns don't even attempt (causal, spatial, temporal).
- Same `Proposition` output shape — no schema change.

**Cons.**
- ~500M-param model: ~2 GB on disk, ~2s/sentence on CPU. Opt-in via `--neural` or batch GPU.
- Relation label mapping is lossy: many REBEL relations don't fit cmapr's seven types and get lumped into the generic `relation`.
- Heavy optional dependency (PyTorch + transformers).

**Recommendation.** **Augment** — `[neural]` extra + `cmapr graph --neural` flag. Roadmap **rb.1–rb.7**. Reasonable size-of-prize: 2–3× more typed edges on the same corpus.

### sentence-transformers — semantic embeddings

**What it does.** Bi-encoder library producing dense sentence/term embeddings via fine-tuned BERT variants. Common models: `all-MiniLM-L6-v2` (fast, 22M params), `all-mpnet-base-v2` (better, 110M params). Apache-2.0.

**Fit with cmapr.** Three immediate uses:

1. **Evidence ranking** (`_score_sentence` in `proposition_extractor.py`). Current heuristic favors definition markers + proximity + brevity. A neural relevance score (cosine between the candidate sentence and an `"X relates to Y"` probe) catches evidence that doesn't match the existing markers.
2. **Semantic-similarity edges** — augment the PMI co-occurrence fallback in `build_proposition_graph` with similarity edges. Two terms with cosine ≥ threshold link even if they never co-occur. New edge type `similarity`.
3. **Concept similarity queries** — new CLI verb `cmapr similar TERM --top-n 10` returns the k nearest terms by embedding. Useful for vetting (find aliases) and seed-term discovery.

**Pros.** Small models run fine on CPU; embedding 50–100 terms is sub-second. Apache license. Drop-in for many uses; no architectural lock-in.

**Cons.** One more dependency (PyTorch transitive). Embedding quality is domain-sensitive — generic models may underperform on philosophical text vs. domain-tuned ones.

**Recommendation.** **Augment** — `[neural]` extra (shared with REBEL). Roadmap **st.1–st.8**. Land *before* REBEL because it's the smaller integration and unlocks multiple wins from one piece of plumbing.

### LangExtract (Google) — LLM-based structured extraction

**What it does.** LLM-powered structured information extraction with source grounding — the model returns typed entities and relations *plus* the exact spans in the source text they came from. Backed by Gemini by default; provider-pluggable. Apache-2.0.

**Fit with cmapr.** This is the closest off-the-shelf match to cmapr's graph stage goal: read a sentence, emit typed relations between terms-of-interest, keep the supporting text for the evidence panel. Where REBEL is constrained to its training-set relation labels, LangExtract takes a *schema* you define — cmapr's nine edge types could be the schema directly, with no lookup-table lossiness.

**Pros.** Schema-aligned output (no Wikidata → cmapr type mapping). Source grounding gives the evidence panel for free. Handles long-range and implicit relations that REBEL misses.

**Cons.** Requires an LLM (API cost + latency + provenance) unless run against a local model. Provenance concerns for academic use. Cold-start dependency footprint is larger than REBEL.

**Recommendation.** **Investigate.** Land *after* the sentence-transformers + REBEL initiatives so the comparison is apples-to-apples. If results justify it, slot as a third `--neural` backend: `cmapr graph --neural=rebel | --neural=langextract`.

### Instructor — Pydantic-typed LLM outputs

**What it does.** Thin wrapper that lets you call any LLM (OpenAI, Anthropic, local) with a Pydantic schema and get back a validated instance. Retries on validation failure. MIT.

**Fit with cmapr.** If cmapr ever does its own LLM extraction (not via LangExtract), Instructor is the obvious harness — define `Proposition` as a Pydantic model, ask the LLM to extract relations from a sentence, get typed `Proposition` objects back. Minimal code.

**Recommendation.** **Investigate** — only relevant if cmapr writes its own LLM extraction prompt rather than adopting LangExtract. Not a current top pick.

### spaCy-LLM — LLM-powered spaCy components

**What it does.** Drops LLM-powered NER, relation extraction, classification, and span-tagging into a spaCy pipeline as components. Config-driven prompts; supports OpenAI, Anthropic, and local models. MIT.

**Fit with cmapr.** Slots next to the existing spaCy noun-chunk extractor. Could add NER (concept/person/work disambiguation) and relation extraction without bringing in a separate framework.

**Recommendation.** **Investigate** — most useful if cmapr broadens spaCy use beyond noun chunks (per the open "SpaCy integration" roadmap entry). Not a standalone priority.

### Outlines / BAML — constrained generation

**Outlines.** Apache-2.0. Token-level constrained decoding to enforce regex / JSON schemas / context-free grammars on LLM output.
**BAML.** Apache-2.0. DSL for prompts with structured output types; compiles to runtime in multiple languages.

**Fit with cmapr.** Same surface as Instructor for the cmapr use case — schema-constrained LLM extraction.

**Recommendation.** **Skip** — Instructor covers the same need with less ceremony.

### Microsoft GraphRAG — full LLM knowledge-graph pipeline

**What it does.** End-to-end LLM pipeline: documents → entity/relation extraction → community detection → community summaries → RAG-ready KG. MIT.

**Fit with cmapr.** Conceptually overlaps cmapr's entire graph stage. Would replace, not augment.

**Recommendation.** **Skip** — adopting GraphRAG means giving up cmapr's bespoke pipeline (curated rarities, regex extractors with audit trail, evidence panel). Worth a once-over for ideas (their community-summary output is interesting) but not for direct integration.

### Stanford OpenIE — classical open information extraction

**What it does.** Rule-based extraction of `(subject, relation, object)` triples using dependency parses. Java; Python wrappers exist.

**Fit with cmapr.** REBEL is strictly better for the same niche (typed neural extraction).

**Recommendation.** **Skip.**

### DyGIE++ — joint entity + relation extraction

**What it does.** Research-grade neural model for joint named-entity and relation extraction. PyTorch.

**Fit with cmapr.** Similar to REBEL but less maintained and harder to ship.

**Recommendation.** **Skip** — REBEL is the better-supported neural option.

### ConceptNet — external commonsense KG

**What it does.** Large multilingual commonsense knowledge graph (8M+ nodes, 21M+ edges) with relations like `IsA`, `PartOf`, `RelatedTo`, `Causes`, `HasA`, `UsedFor`. REST API + offline dump. Apache-2.0.

**Fit with cmapr.** Two uses:

1. **Validation.** After extracting a graph, look up each extracted edge in ConceptNet: does it agree? Mismatch could mean (a) the text uses a term idiosyncratically (the entire point of cmapr — flag and surface) or (b) the extractor is wrong.
2. **Augmentation.** For each term, fetch ConceptNet's top neighbors as suggested context nodes. Could feed seed-term suggestions.

**Pros.** Free, large, well-typed; external ground-truth baseline cmapr currently lacks.

**Cons.** Relations don't perfectly align (their `IsA` ≈ cmapr's `kind-of`; `RelatedTo` is too generic). Coverage of philosophical concepts is patchy — works for "dog IsA mammal", less so for "sublation IsA dialectical movement".

**Recommendation.** **Augment (v2).** Concrete first step: `cmapr conceptnet TERM` shows ConceptNet's neighborhood for a term — purely informational, no integration. If useful in practice, plumb into the HTML viz's evidence panel as a "related concepts" sidebar.

---

## Stage 4 — Export & visualization

**What cmapr does now.** `export/d3.py` writes D3 force-directed JSON; `export/html.py` inlines D3 v7 + the JSON + simulation JS for a standalone interactive HTML page (drag, zoom, pan, tooltips, community colors, type-filtered legend, expand/collapse, node detail panel, cluster force when chapter data is present). `export/formats.py` adds GraphML, GEXF, CSV, DOT.

The HTML viz is bespoke for a reason — type-filtered legend, expand/collapse, side panel, cluster force, per-type styling. Adopting a viz framework means reimplementing those.

### pyvis — vis.js HTML wrapper

**What it does.** Python wrapper around vis.js. Generates standalone HTML with force-directed network and built-in physics controls. MIT.

**Fit with cmapr.** Could replace `export/html.py`. But every cmapr customization — type-filtered legend, expand/collapse, panel, cluster force — would need reimplementing on top of vis.js's less flexible API.

**Recommendation.** **Skip.** Reconsider only if D3 maintenance becomes a problem.

### rdflib — RDF/Turtle serialization

**What it does.** Standard RDF/OWL/SPARQL library. Reads/writes Turtle, N-Triples, JSON-LD. BSD.

**Fit with cmapr.** Each cmapr edge maps cleanly to a triple. ~50 lines of `export_rdf`.

**Recommendation.** **Skip until a concrete request.**

### Cytoscape.js · Sigma.js · 3d-force-graph

**What they do.** Alternative front-end graph viz libraries — Cytoscape.js for bio-network-style layouts and styling DSL, Sigma.js for WebGL-rendered large graphs, 3d-force-graph for three-dimensional force layouts.

**Fit with cmapr.** Cytoscape.js would be the natural upgrade if cmapr ever needed compound nodes (group nesting), better-styled directed edges, or richer interaction; Sigma.js if graphs grow into the thousands of nodes (current ceiling ~200); 3d-force-graph for exploratory three-dimensional cluster views.

**Recommendation.** **Skip** for now. The current D3 force code is well-fitted to the 50–200 node range; none of these libraries justify the rewrite at present scale. Re-evaluate at the 1000-node threshold.

---

## Cross-cutting / post-graph

These don't fit one stage — they operate on the extracted graph as a whole, or sit alongside cmapr as separate analytical lenses.

### PyKEEN — KG embeddings + link prediction

**What it does.** Library for training KG embedding models (TransE, RotatE, ComplEx, ConvE, ~40 variants) over `(head, relation, tail)` triples. Produces vectors per entity and per relation; supports link prediction (score every possible `(h, r, t)` and surface the highest-scoring ones not in the training set). MIT.

**Fit with cmapr.** A `cmapr graph` output *is* a KG: typed directed triples over a fixed entity set. Two concrete uses:

1. **Link prediction** — train a TransE model on the extracted edges, surface high-confidence triples the extractor missed. Output as suggestions with confidence scores; manual vet, then optionally add to the graph.
2. **Entity similarity** — cosine over entity embeddings → "which nodes behave similarly in the graph?" Different from semantic similarity (sentence-transformers) because it's purely structural.

**Pros.** Closes the loop on extraction recall: catches relations the regex/REBEL pipelines miss because the supporting sentence is too implicit. No new annotation; trains on cmapr's own output.

**Cons.** Needs a reasonable-sized graph to train on (~100s of edges minimum; current 50-term graphs are borderline). Embedding model selection / hyperparameter tuning is real work. Pays off most after the graph extractor is at peak quality.

**Recommendation.** **Defer to v3.** Real value once REBEL + sentence-transformers have lifted extraction quality. Natural surface when it lands: `cmapr predict GRAPH --top-n 20 --model transe` writes a `suggestions.json` of candidate missing edges for manual vetting.

### NetworkX — already integrated

Core dependency; backs `ConceptGraph`. Provides centrality, community detection (Louvain), ego graphs for `--focus`, density, shortest path.

### Neo4j / property-graph databases

Replaces the JSON storage layer. Worth considering only if cmapr ever targets corpora > 100 documents (current scale per `docs/roadmap.md` § Known Limitations).

**Recommendation.** **Defer** indefinitely; revisit at the multi-author / multi-work scale threshold.

### Hosted LLMs (OpenAI, Anthropic, OSS) for extraction

Could supersede REBEL for relation extraction — better quality at higher cost, latency, and provenance burden. Possibly worth a future "high-quality batch extraction" mode separate from the interactive pipeline. LangExtract is the most cmapr-shaped entry point into this space (see Stage 3).

**Recommendation.** **Defer** — track via LangExtract above. Not a separate initiative.

### Streamlit / Gradio frontends

Would replace `cmapr serve` (FastAPI + Jinja2). The current web UI is bespoke for the four-stage flow.

**Recommendation.** **Skip** unless the UI roadmap grows.

---

## Top picks for next initiatives

Ordered by leverage per unit of work, given cmapr's current state:

1. **sentence-transformers integration** (Stage 2 + Stage 3) — `[neural]` extra + `cmapr similar TERM` + semantic evidence ranking + similarity edges. Smaller and lossless next to REBEL; unlocks multiple downstream wins from one piece of plumbing. Roadmap **st.1–st.8**.
2. **REBEL integration** (Stage 3) — `cmapr graph --neural` for neural relation extraction. Land after sentence-transformers so the embedding infrastructure is reusable. Roadmap **rb.1–rb.7**. Reasonable size-of-prize: 2–3× more typed edges.
3. **GLiNER for term typing** (Stage 2) — concept vs thinker vs work classification on rarities output. Cleaner term lists, less manual vetting. Small surface; depends on the `[neural]` extra existing.
4. **KeyBERT as 6th rarity signal** (Stage 2) — embedding-based distinctiveness. ~50 lines on top of sentence-transformers.
5. **LangExtract evaluation** (Stage 3) — once REBEL is in place, run LangExtract on the same corpus and compare yield + provenance. If it wins, add as a third `--neural` backend.
6. **PyKEEN link prediction** (cross-cutting) — defer until the extractor is at peak quality and graphs are large enough to learn from.
7. **ConceptNet lookup helper** (Stage 3) — exploratory; build when a concrete validation use case surfaces.
8. **Structured ingestion pipeline** (Stage 1) — tracked as a separate roadmap entry (docling/pymupdf4llm/unstructured/nougat).

Out of scope (deliberately not on the list): GUI improvements, alternative export formats, topic modeling, RDF export, graph-DB backends. Revisit when the analytical pipeline saturates.

---

## What's *not* in this survey

Deliberately omitted because they're outside cmapr's scope or would require restructuring the project:

- **Neo4j / property-graph DBs** — covered above under cross-cutting; defer until scale demands it.
- **Microsoft GraphRAG / similar end-to-end LLM KG builders** — covered under Stage 3; overlaps cmapr's whole graph stage rather than augmenting it.
- **Streamlit / Gradio** — covered under cross-cutting; not a current priority.
- **`docling` / `pymupdf4llm` / `unstructured` / `nougat`** — listed here for completeness but the integration work is tracked under the separate **structured ingestion pipeline** roadmap entry.

---

## Related docs

- [`roadmap.md`](roadmap.md) — past / present / future + live Status block; sentence-transformers (st.1–st.8) and REBEL (rb.1–rb.7) tasks live there
- [`architecture.md`](architecture.md) — pipeline diagram + module map; the stage structure here mirrors its Stage 1–4 sections
- [`plans/`](plans/) — per-initiative implementation plans (drafted before code lands)
