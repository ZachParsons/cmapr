# Cluster Pipeline QA

Manual spot-check for `cmapr cluster`: corpus with structure → per-chapter sub-graphs + recurrence edges → HTML cluster viz.

Run after `docs/qa/graph.md` if you haven't already — this exercises a structure-aware corpus that the basic graph pipeline doesn't require.

---

## Prerequisites

- [ ] A structure-aware corpus already ingested. The reference is `eco_spl1_w_toc` (Eco's *Semiotics and the Philosophy of Language*, part 1, ingested with `--toc`).
- [ ] A terms file produced by `cmapr rarities` against that corpus.

```bash
ls data/output/corpus/eco_spl1_w_toc/corpus.json
ls data/output/rarities/eco_spl1_w_toc/terms.json
```

If those are missing:

```bash
cmapr ingest data/input/eco_spl1.txt --clean-ocr \
    --toc data/input/eco_spl_toc.txt \
    --output data/output/corpus/eco_spl1_w_toc/corpus.json
cmapr rarities data/output/corpus/eco_spl1_w_toc/corpus.json --top-n 60 \
    --output data/output/rarities/eco_spl1_w_toc/terms.json
```

---

## Step 1 — Cluster

```bash
cmapr cluster data/output/corpus/eco_spl1_w_toc/corpus.json \
    -t data/output/rarities/eco_spl1_w_toc/terms.json \
    -o data/output/graphs/eco_spl1_clustered/graph.json
```

**Check stdout:**

- [ ] Exit code 0, no traceback.
- [ ] Output line shows ≥ 2 chapter(s), some nodes, some edges, and a non-zero recurrence count. Example: `2 chapter(s), 87 nodes, 142 edges (15 recurrence)`.

**Check JSON structure:**

```bash
python -c "
import json
g = json.load(open('data/output/graphs/eco_spl1_clustered/graph.json'))
chapters = sorted({n.get('chapter') for n in g['nodes']})
namespaced = sum(1 for n in g['nodes'] if '__' in n['id'])
rec = [e for e in g['links'] if e.get('type') == 'recurrence']
print(f'chapters: {chapters}')
print(f'namespaced nodes: {namespaced}/{len(g[\"nodes\"])}')
print(f'recurrence edges: {len(rec)}')
if rec:
    sample = rec[0]
    print(f'sample recurrence: {sample[\"source\"]} -> {sample[\"target\"]} weight={sample[\"weight\"]}')
"
```

- [ ] `chapters` is a list of ≥ 2 distinct strings (e.g. chapter titles from the TOC).
- [ ] All nodes are namespaced (`__<chapter>` suffix).
- [ ] At least 1 recurrence edge present.
- [ ] Sample recurrence edge: source and target share the same un-namespaced term (`sign__Chapter 1 → sign__Chapter 3`).
- [ ] Recurrence edge `weight` equals the number of chapters the term appears in (`span`).

**Variant — cluster by section instead of chapter:**

```bash
cmapr cluster data/output/corpus/eco_spl1_w_toc/corpus.json \
    -t data/output/rarities/eco_spl1_w_toc/terms.json \
    --by section \
    -o data/output/graphs/eco_spl1_by_section/graph.json
```

- [ ] Section-level clustering produces ≥ chapter-level cluster count (sections are usually finer-grained).
- [ ] Nodes carry a `section` attribute instead of `chapter`.

**Variant — prune to a tighter ratio:**

```bash
cmapr cluster data/output/corpus/eco_spl1_w_toc/corpus.json \
    -t data/output/rarities/eco_spl1_w_toc/terms.json \
    --prune-ratio 2.0 \
    -o data/output/graphs/eco_spl1_clustered_pruned/graph.json
```

- [ ] Pruning summary printed (`Pruned to ratio 2.0: N → M edges`).
- [ ] Resulting `len(links) / len(nodes) ≤ 2.0`.

---

## Step 2 — Export to HTML

```bash
cmapr export data/output/graphs/eco_spl1_clustered/graph.json --format html \
    --title "Eco SPL1 — Clustered by Chapter" \
    -o data/output/exports/eco_spl1_clustered/index.html
open data/output/exports/eco_spl1_clustered/index.html
```

**Visual checks in browser:**

- [ ] Page loads without blank screen or JS console errors (`⌘⌥J` to open DevTools).
- [ ] Nodes are visibly grouped into spatial clusters, one per chapter. Clusters arranged around a rough circle centered on the canvas.
- [ ] Each cluster contains the terms that appeared in that chapter (hover a node — tooltip shows the un-namespaced term).
- [ ] Same term appearing in multiple chapters → multiple distinct nodes, one per cluster.
- [ ] Recurrence edges thread *between* clusters (slate / muted-blue dashed lines with arrowheads in chapter order).
- [ ] Legend shows a `recurrence` row with a dashed swatch and checkbox.
- [ ] Unchecking `recurrence` in the legend hides those cross-cluster threads; checking re-shows them.
- [ ] All other legend toggles (definition, kind-of, production, etc.) still work within each cluster.
- [ ] Node detail panel (click any node) shows term, frequency, score, and connected edges. Recurrence edges to other chapters appear in the connections list.
- [ ] Dragging a node still pins it; expand/collapse (double-click) still works inside a cluster.

**Sanity check — recurring terms visible across clusters:**

Look for terms you expect to recur across chapters (e.g. *sign*, *semiosis*, *interpretant*, *code*). Hover them in different clusters and confirm the term name matches.

---

## Common failure modes

| Symptom | Likely cause |
|---------|--------------|
| `cmapr cluster` exits with `AttributeError: 'dict' object has no attribute 'chapter_title'` | Corpus loaded via `ProcessedDocument(**doc_data)` left `sentence_locations` as raw dicts. `cluster_by_structure` should handle this — file a bug if it doesn't. |
| Only one chapter in output | Corpus has no structure metadata. Re-ingest with `--toc` pointing to a valid TOC file. |
| 0 recurrence edges | No term appears in 2+ chapters. Either the term list is too narrow, or the corpus chapters are too thematically disjoint. |
| All clusters overlap visually | < 2 distinct chapters in the data — the cluster force only activates when ≥ 2 chapters are present. |
| Node IDs not namespaced in HTML | `to_d3_dict` lost the `chapter` attribute. Check `export/d3.py` is passing `chapter`/`section`/`term` through. |
| Legend missing the `recurrence` row | No recurrence edges in the data, or `LEGEND_TYPES` doesn't include it. The legend only shows rows for types present in `data.links`. |
| Same-term nodes collapse into one across chapters | `consolidate_duplicate_labels` not using the chapter-aware key. Verify `(label, chapter)` dedup logic in `graph/operations.py`. |
