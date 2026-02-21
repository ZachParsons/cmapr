# Roadmap

**Past:** completed phases, significant additions/pivots.
**Present:** planned work, WIP.
**Future:** unplanned features, ideas.

---

## Concept Mapper

A tool for extracting and visualizing an author's idiosyncratic conceptual vocabulary from primary texts. Identifies neologisms and terms with specialized technical meaning that are statistically distinctive compared to general English corpora. Maps concepts through co-occurrence and grammatical relations, exporting as D3 visualizations.

**Examples of target terms:** Aristotle's *eudaimonia*, Spinoza's *affect*, Hegel's *sublation*, Dennett's *intentional stance*, Deleuze & Guattari's *body without organs*.

---

## Status

**🎉 Feature-complete** — all 11 phases implemented, tested, and documented.

| Phase | Description | Tests |
|-------|-------------|-------|
| 0 | Project scaffolding, storage layer, test corpus | 12 |
| 1 | Corpus loading, preprocessing pipeline (tokenization, POS, lemmas) | 46 |
| 2 | Frequency analysis, Brown corpus reference, TF-IDF | 21 |
| 3 | Philosophical term detection (multi-method rarity analysis) | 103 |
| 4 | Term list management (curation, import/export, auto-population) | 47 |
| 5 | Search & concordance (find, KWIC, context windows, dispersion) | 52 |
| 6 | Co-occurrence analysis (PMI, LLR, matrices) | 45 |
| 7 | Relation extraction (SVO, copular, prepositional) | 35 |
| 8 | Graph construction (networkx, builders, operations, metrics) | 62 |
| 9 | Export & visualization (D3 JSON, GraphML, DOT, CSV, HTML) | 30 |
| 10 | CLI interface (Click, unified command-line access) | 23 |
| 11 | Documentation & polish | — |

**665 tests passing, 2 skipped** (pydot-dependent DOT export tests).

### Post-completion additions (February 2026)

- **OCR text cleaning** — `preprocessing/cleaning.py`, `--clean-ocr` flag. 21 tests.
- **PDF input support** — `load_pdf()` via pdfplumber, auto-detected in `load_file()`. 6 tests.
- **Paragraph segmentation** — `preprocessing/segment.py`, paragraph boundary detection. 21 tests.
- **Synonym replacement** — inflection-preserving term replacement. `transformations/`, `cmapr replace`. 59 tests. See `docs/replacement.md`.
- **Contextual relation extraction** — integrated SVO + co-occurrence workflow. `analysis/contextual_relations.py`, `cmapr analyze`. 38 tests.

---

## Architecture

```
src/concept_mapper/
├── corpus/           # Document loading and models
├── preprocessing/    # Tokenization, POS tagging, lemmatization, cleaning, structure
├── analysis/         # Frequency, rarity, co-occurrence, relations, contextual
├── terms/            # Term list management
├── search/           # Search, concordance, context, dispersion, extraction
├── graph/            # Graph construction, operations, metrics
├── export/           # D3 JSON, GraphML, GEXF, DOT, CSV, HTML
├── transformations/  # Synonym replacement with inflection
├── storage/          # Storage abstraction and utilities
└── cli.py            # Click CLI (cmapr command)
```

**Dependencies:** NLTK (tokenization, POS, lemmas, WordNet), NetworkX (graphs), Click (CLI), pdfplumber (PDF input).

---

## Known Limitations

1. **English only** — NLTK resources are English-centric.
2. **Pattern-based relations** — spaCy dependency parsing deferred (Python 3.14 incompatibility); NLTK pattern-matching works well for philosophical texts.
3. **Scale** — optimized for academic texts (10–100 documents), not massive corpora.
4. **Graph layout** — force-directed only.

---

## Future Work

- SpaCy integration when Python 3.14 compatible (richer dependency parsing)
- Multi-language support
- Usage-based definition generation (aggregate co-occurrences and relations into empirical definitions)
- Temporal analysis across an author's career
- Web interface
- Database backend for large-scale corpora

---

## Acronym Reference

| Acronym | Meaning |
|---------|---------|
| POS | Part of Speech |
| TF-IDF | Term Frequency–Inverse Document Frequency |
| KWIC | Key Word In Context |
| PMI | Pointwise Mutual Information |
| LLR | Log-Likelihood Ratio |
| SVO | Subject-Verb-Object |
| CLI | Command Line Interface |

## References

- Lane 2019, *Natural Language Processing in Action*
- Rockwell & Sinclair 2016, *Hermeneutica*
- Moretti, *Graphs, Maps, Trees*
