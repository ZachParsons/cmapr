# Implementation Plan: Multi-chapter Clustered Visualization

> **Status / next / open issues:** see `docs/roadmap.md` § Status. This file is the per-feature detail view; the roadmap is the at-a-glance view.

Complement to `cmapr merge`: where merge collapses chapters into a unified view, `cmapr cluster` keeps them separate and links recurring concepts across chapters with a new `recurrence` edge type.

---

## Dev loop

```bash
# Build per-chapter graph from a structure-aware corpus:
cmapr cluster data/output/corpus/eco_spl1_w_toc/corpus.json \
    -t data/output/rarities/eco_spl1_w_toc/terms.json \
    -o data/output/graphs/eco_spl1_clustered/graph.json && \
cmapr export data/output/graphs/eco_spl1_clustered/graph.json --format html \
    -o data/output/exports/eco_spl1_clustered/index.html && \
open data/output/exports/eco_spl1_clustered/index.html
```

---

## Tasks

- [x] **1.1 `cluster_by_structure` operation** — `src/concept_mapper/graph/operations.py`. Walks `sentence_locations`, slices docs per chapter, calls `build_proposition_graph` per cluster, namespaces nodes as `<term>__<chapter>`, adds `recurrence` edges between consecutive same-term occurrences with `weight = span` (number of chapters the term appears in). Resilient to dict-shaped or dataclass-shaped `SentenceLocation`.

- [x] **1.2 Re-export** — `graph/__init__.py` exposes `cluster_by_structure` alongside `aggregate_graphs`, `merge_graphs`.

- [x] **2.1 `cmapr cluster` CLI** — `src/concept_mapper/cli.py` after `merge`. Args: `CORPUS`, `-t/--terms`, `-o/--output`. Options: `--by chapter|section` (default `chapter`), `--prune-ratio FLOAT`. Loads docs via `ProcessedDocument(**doc_data)`, builds NodeFilter, calls `cluster_by_structure`, optionally prunes, exports D3 JSON. Echoes chapter count and recurrence-edge count. Module command index updated.

- [x] **3.1 D3 viz — recurrence edge type** — `src/concept_mapper/export/html.py`. Adds `"recurrence": "#7a8aa0"` to `EDGE_COLORS`, an entry in `LEGEND_TYPES` (dashed), and `"recurrence"` to `DIRECTED_TYPES` (chain order). Legend checkbox toggles visibility via the existing `hiddenTypes` mechanism.

- [x] **3.2 D3 viz — cluster force** — `src/concept_mapper/export/html.py`. When ≥ 2 distinct `chapter` values are present in the data, compute centroids on a circle (`(cx + r·cos(2πk/N), cy + r·sin(2πk/N))`, radius `min(width, height)·0.35`), and add a custom `simulation.force("cluster", alpha => …)` that nudges each node's velocity toward its centroid (factor `0.05`). When fewer than 2 chapters, the force is not added — non-clustered graphs render exactly as before.

- [x] **3.3 D3 export — pass cluster attrs** — `export/d3.py:to_d3_dict` propagates `chapter`, `section`, and `term` node attributes into the D3 output dict so the viz can read them.

- [x] **3.4 Consolidation guard** — `graph/operations.py:consolidate_duplicate_labels` now uses `(label, chapter)` as the dedup key when nodes carry a chapter attribute. Without this, the existing duplicate-label rule would silently collapse `sign__Chapter 1` and `sign__Chapter 2` (same `label="sign"`) on every export.

- [x] **4.1 Unit tests** — `tests/test_graph.py::TestClusterByStructure`. 7 cases: two-chapter shared term yields one recurrence edge with weight=2; three-chapter term skips middle and chains directly with weight=2; term in one chapter only → no recurrence; no structure metadata → single fallback "Document" cluster; empty docs → empty graph; empty seeds → empty graph; namespaced nodes carry `term`, `chapter`, and un-namespaced `label` attributes.

- [x] **4.2 CLI tests** — `tests/test_cli.py::TestClusterCommand`. 4 cases: completes on synthetic chaptered corpus; output namespaces all node IDs with `__<chapter>` and surfaces `chapter` attribute; ≥1 recurrence edge present; `--prune-ratio` honored.

- [x] **4.3 Manual HTML verification** — TBD by user. Expected: clusters visually separated on the canvas, recurrence edges thread between them, legend toggle hides/shows them.

**Verify:**
```bash
python -m pytest tests/test_graph.py::TestClusterByStructure tests/test_cli.py::TestClusterCommand -v

cmapr ingest data/input/eco_spl1.txt --clean-ocr \
    --toc data/input/eco_spl_toc.txt \
    --output data/output/corpus/eco_spl1_w_toc/corpus.json
cmapr rarities data/output/corpus/eco_spl1_w_toc/corpus.json --top-n 60 \
    --output data/output/rarities/eco_spl1_w_toc/terms.json
cmapr cluster data/output/corpus/eco_spl1_w_toc/corpus.json \
    -t data/output/rarities/eco_spl1_w_toc/terms.json \
    -o data/output/graphs/eco_spl1_clustered/graph.json
cmapr export data/output/graphs/eco_spl1_clustered/graph.json --format html \
    -o data/output/exports/eco_spl1_clustered/index.html
open data/output/exports/eco_spl1_clustered/index.html

python -c "
import json
g = json.load(open('data/output/graphs/eco_spl1_clustered/graph.json'))
chapters = sorted({n.get('chapter') for n in g['nodes']})
rec = [e for e in g['links'] if e.get('type') == 'recurrence']
print(f'chapters: {chapters}')
print(f'namespaced nodes: {sum(1 for n in g[\"nodes\"] if \"__\" in n[\"id\"])}/{len(g[\"nodes\"])}')
print(f'recurrence edges: {len(rec)}')
"
```

Pass: ≥ 2 chapters present, all nodes namespaced, ≥ 1 recurrence edge, HTML loads with visible cluster separation and a working `recurrence` legend toggle.

---

## Out of scope for v1

- **Display options B and C** (constellation, timeline spine) — explicitly deferred per `docs/roadmap.md`.
- **Auto-generating per-chapter graph files** — single-corpus path covers the v1 use case.
- **Cross-chapter typed propositions** — `recurrence` is the only cross-cluster edge type. No extracting `production` edges from `sign__ch1` to `meaning__ch3` (would require sentence pairs that span chapters, which the spec didn't ask for).
