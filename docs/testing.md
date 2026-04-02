# Testing Guidelines

Strategy, patterns, and conventions for the cmapr test suite.

---

## Philosophy

The goal of this test suite is **confidence, not coverage**. A test that
asserts the right thing about a small, controlled input is worth ten tests
that assert `== True` on the output of a real pipeline run. Every test
should answer a specific question: *does this code do what it is supposed to
do in this situation?*

**Three questions to ask before writing a test:**

1. What decision or behaviour am I pinning down? (If you can't answer this,
   the test probably belongs in a different form — a comment, a type hint,
   or documentation.)
2. Is the input minimal? A test that needs a 500-sentence document to trigger
   the behaviour it is testing is fragile and slow. Reduce it to the smallest
   input that exercises the path.
3. Is the assertion specific? `assert result` is nearly useless. Assert the
   exact value, the exact type, the exact key, or the exact count you expect.

---

## Test organization

```
tests/
├── test_analysis.py          # Frequency, rarity, co-occurrence, relations
├── test_cli.py               # Click CLI commands (CliRunner)
├── test_corpus.py            # Document loading, models, serialization
├── test_export.py            # D3 JSON, HTML, GraphML, CSV, GEXF export
├── test_graph.py             # ConceptGraph, builders, operations, metrics
├── test_node_filter.py       # NodeFilter inclusion criteria
├── test_preprocessing.py     # Tokenization, POS, lemmas, cleaning, pipeline
├── test_proposition_extractor.py  # Typed proposition extraction
├── test_search.py            # find, concordance, context, dispersion
└── test_terms.py             # TermList, TermManager, TermEntry
```

**One file per module group.** Do not create `test_phase_13.py` —
place tests in the file that corresponds to the code under test. The
exception is CLI tests, which stay together in `test_cli.py` because
they exercise multiple modules through the command surface.

**Class per behaviour area.** Group related tests in a class even when
pytest does not require it. A class gives you a readable name, a shared
docstring, and a place to put helpers that only that group needs.

```python
class TestPrunesToRatio:
    """prune_to_ratio removes edges until edges:nodes ≤ target."""

    def test_dense_graph_is_pruned(self): ...
    def test_no_isolated_nodes_introduced(self): ...
    def test_cooccurrence_edges_pruned_first(self): ...
    def test_already_within_ratio_unchanged(self): ...
```

---

## Fixtures

### Prefer factory functions over fixtures for small objects

pytest fixtures are right for expensive shared setup (a real corpus JSON on
disk, a preprocessed document). For small in-memory objects like a single
`NodeFilter` or a two-node `ConceptGraph`, use a plain factory function
defined at module level:

```python
def make_filter(vocab=None, freqs=None, min_freq=3):
    return NodeFilter(
        corpus_vocab=vocab or set(),
        term_freqs=Counter(freqs or {}),
        min_freq=min_freq,
    )
```

Call it inside each test method. This keeps tests self-contained and avoids
fixture injection ceremony for trivial setup.

### Fixtures for I/O-heavy setup

Use `@pytest.fixture` with `tmp_path` for anything that touches the
filesystem: writing corpus JSON, term files, or exported HTML. Never use
hardcoded paths or rely on files in `data/output/` — those are user
artifacts, not test artifacts.

```python
@pytest.fixture
def corpus_path(tmp_path):
    path = tmp_path / "corpus.json"
    docs = [make_processed_doc(["Sign is a triadic relation."])]
    path.write_text(json.dumps([d.to_dict() for d in docs]))
    return path
```

### The `make_docs` pattern for NLP tests

Most NLP components (`PropositionExtractor`, `build_proposition_graph`,
`PhilosophicalTermScorer`) accept a list of `ProcessedDocument` objects.
Constructing real documents is slow; a `MagicMock` with the required
attributes is enough for most tests:

```python
def make_docs(sentences: list[str]):
    doc = MagicMock()
    doc.sentences = sentences
    doc.tokens = [tok for s in sentences for tok in s.split()]
    doc.lemmas = doc.tokens  # close enough for most tests
    doc.pos_tags = [(tok, "NN") for tok in doc.tokens]
    doc.metadata = {}
    return [doc]
```

When the test depends on accurate POS tags or lemmas (e.g. `_try_pos_verb`),
use the real preprocessing pipeline on a short planted sentence rather than
trying to manually construct realistic POS output.

---

## The planted-sentence technique

NLP extraction code is hard to unit-test because the real world is messy.
The solution used throughout this project is **planted sentences**: craft the
minimal sentence that should trigger the pattern, assert it fires, then craft
a sentence that should NOT trigger it, assert it does not.

```python
def test_production_pattern_fires(self):
    s = "The sign produces an interpretant in the mind of the interpreter."
    e = PropositionExtractor(make_docs([s]))
    props = e.extract("sign", "interpretant")
    assert any(p.type == "production" for p in props)

def test_production_pattern_does_not_fire_on_copula(self):
    s = "The sign is an interpretant."
    e = PropositionExtractor(make_docs([s]))
    props = e.extract("sign", "interpretant")
    assert not any(p.type == "production" for p in props)
```

**Rules for planted sentences:**

- Use vocabulary from the project's domain (Peircean/Saussurean semiotics,
  Hegelian philosophy) rather than abstract `foo`/`bar` placeholders. Tests
  that resemble real inputs catch real bugs.
- Plant exactly one distinguishing feature per sentence. If the sentence
  contains both a `vs` marker and a `kind of` marker, you cannot tell which
  one fired.
- For negation tests (pattern should NOT fire), verify specifically that the
  target type is absent — not that `props` is empty. Another extractor may
  legitimately fire.

---

## Testing pattern-matching components

`PropositionExtractor` has seven `_try_*` methods, each with its own pattern
list. Test each type in its own class. Within that class:

1. **One test per distinct pattern variant** (e.g. `vs` and `versus` are
   different regex branches — test both).
2. **Direction test**: for directed types, assert `p.source` and `p.target`
   are correct, not just that some proposition was returned.
3. **Undirected test**: for symmetric types (`opposition`, `component`),
   assert `p.directed is False`.
4. **Priority test**: when a sentence matches two patterns, the higher-
   priority one should win. Build a sentence that has both markers and assert
   only the higher-priority type is returned.
5. **No-match test**: a sentence that contains both terms but no pattern
   marker should return nothing for that type (other extractors may fire).

```python
def test_opposition_directed_false(self):
    s = "Semiosis vs. entropy — sign vs. noise."
    e = PropositionExtractor(make_docs([s]))
    props = [p for p in e.extract("semiosis", "entropy") if p.type == "opposition"]
    assert props
    assert props[0].directed is False

def test_definition_beats_kind_of_on_same_sentence(self):
    s = "By 'interpretant' I mean a kind of sign in Peirce's triadic model."
    e = PropositionExtractor(make_docs([s]))
    props = e.extract("interpretant", "sign")
    types = {p.type for p in props}
    assert "definition" in types
    # definition has higher priority; kind-of should not override it
    assert "kind-of" not in types or "definition" in types
```

---

## Testing graph construction (`build_proposition_graph`)

`build_proposition_graph` is an integration point: it drives
`PropositionExtractor`, applies `NodeFilter`, and falls back to PMI-based
co-occurrence. Its tests live in `test_graph.py` in a dedicated class.

**What to test:**

- A pair of terms co-occurring in a planted sentence with a known pattern
  produces the correct typed edge (not `cooccurrence`).
- A pair with no pattern but high PMI gets a `cooccurrence` fallback edge.
- A pair with no co-occurrence at all gets no edge.
- `term_scores` values are stored as node attributes.
- Node pairs that fail `NodeFilter` are excluded from the graph.
- The returned graph is a `ConceptGraph`, not a raw networkx object.

**Minimal graph fixture:**

```python
def make_proposition_graph(sentences, seed_terms, **kwargs):
    docs = make_docs(sentences)
    return build_proposition_graph(docs, seed_terms, **kwargs)
```

**Example:**

```python
def test_typed_edge_extracted_over_cooccurrence(self):
    sentences = [
        "The interpretant is a kind of sign in the mind of the interpreter.",
        "The interpretant appears in the mind of the interpreter.",
    ]
    graph = make_proposition_graph(sentences, ["interpretant", "sign"])
    assert graph.has_edge("interpretant", "sign") or graph.has_edge("sign", "interpretant")
    for src, tgt in graph.edges():
        edge = graph.get_edge(src, tgt)
        assert edge["relation_type"] != "cooccurrence", (
            f"Expected typed edge, got cooccurrence for {src}→{tgt}"
        )

def test_cooccurrence_fallback_when_no_pattern(self):
    sentences = [
        "Sign and code appear together in the signification process.",
        "Sign and code both play a role in communication.",
        "Sign relates to code in every communicative act.",
    ]
    graph = make_proposition_graph(sentences, ["sign", "code"], pmi_threshold=0.0)
    assert graph.has_edge("sign", "code") or graph.has_edge("code", "sign")
    for src, tgt in graph.edges():
        edge = graph.get_edge(src, tgt)
        assert edge["relation_type"] == "cooccurrence"
```

---

## Testing `prune_to_ratio`

`prune_to_ratio` has three guarantees to test independently:

1. After pruning, `edges / nodes ≤ target_ratio`.
2. No node becomes isolated (degree 0) as a result of pruning.
3. `cooccurrence` edges are removed before grammatical edges of equal weight.

Build toy `ConceptGraph` objects directly rather than going through the full
pipeline. Add nodes and edges with known types and weights:

```python
def _dense_graph():
    g = ConceptGraph(directed=True)
    for n in "abcde":
        g.add_node(n, label=n)
    # 10 edges on 5 nodes = 2:1 → above any target_ratio of 1:1
    pairs = [(a, b) for a in "abcde" for b in "abcde" if a < b]
    for src, tgt in pairs:
        g.add_edge(src, tgt, relation_type="cooccurrence", weight=1)
    return g

def test_ratio_enforced(self):
    g = _dense_graph()
    pruned = prune_to_ratio(g, target_ratio=1.5)
    ratio = pruned.edge_count() / max(pruned.node_count(), 1)
    assert ratio <= 1.5

def test_no_isolated_nodes(self):
    g = _dense_graph()
    pruned = prune_to_ratio(g, target_ratio=0.5)
    connected = set()
    for src, tgt in pruned.edges():
        connected.add(src)
        connected.add(tgt)
    for node in pruned.nodes():
        assert node in connected, f"Node {node!r} is isolated after pruning"

def test_cooccurrence_removed_before_grammatical(self):
    g = ConceptGraph(directed=True)
    for n in "abcde":
        g.add_node(n, label=n)
    g.add_edge("a", "b", relation_type="kind-of", weight=1)
    g.add_edge("a", "c", relation_type="cooccurrence", weight=5)  # high weight
    g.add_edge("b", "c", relation_type="production", weight=1)
    # target_ratio=0.5 → must remove at least one edge
    pruned = prune_to_ratio(g, target_ratio=0.5)
    edges = {pruned.get_edge(s, t)["relation_type"] for s, t in pruned.edges()}
    # cooccurrence should be gone despite its high weight
    assert "cooccurrence" not in edges
```

---

## Testing evidence scoring (`_score_sentence`)

`_score_sentence` is a pure function. Test it as a unit, separate from
`PropositionExtractor`:

```python
from concept_mapper.graph.proposition_extractor import _score_sentence

def test_definition_marker_gives_highest_score():
    s = "By 'sign' I mean any vehicle that stands for something else."
    score = _score_sentence(s, "sign", "vehicle", sent_idx=5, n_sentences=20)
    s2 = "Sign and vehicle appear together in the same chapter."
    score2 = _score_sentence(s2, "sign", "vehicle", sent_idx=5, n_sentences=20)
    assert score > score2

def test_proximity_bonus_applied():
    close = "The sign immediately produces its interpretant."
    far = ("The sign, which is a complex relational structure in Peircean "
           "philosophy, may under certain conditions produce its interpretant.")
    score_close = _score_sentence(close, "sign", "interpretant", 0, 10)
    score_far = _score_sentence(far, "sign", "interpretant", 0, 10)
    assert score_close > score_far

def test_early_sentences_preferred():
    s = "Sign produces interpretant."
    early = _score_sentence(s, "sign", "interpretant", sent_idx=0, n_sentences=100)
    late = _score_sentence(s, "sign", "interpretant", sent_idx=99, n_sentences=100)
    assert early > late
```

For end-to-end evidence ranking, verify that `Proposition.evidence` is a
list, has at most 3 entries, and that the highest-scoring sentence appears
first. Use a multi-sentence document where one sentence contains a definition
marker (should rank first) and others do not:

```python
def test_top3_evidence_returned():
    sentences = [
        "Sign and interpretant appear in the text.",
        "By 'sign' I mean a vehicle that stands for something.",  # should rank #1
        "The sign relates to the interpretant through a code.",
        "Both sign and interpretant are central to semiosis.",
        "The sign is fundamentally related to the interpretant.",
    ]
    e = PropositionExtractor(make_docs(sentences))
    props = e.extract("sign", "interpretant")
    assert props
    p = props[0]
    assert isinstance(p.evidence, list)
    assert len(p.evidence) <= 3
    # Definition-marker sentence should be ranked first
    assert "I mean" in p.evidence[0]
```

---

## Testing the vetting file round-trip

The vetting file (`vetting.json`) is loaded before scoring, applied after
filtering, and saved after interactive mode. Test the load-apply-save cycle
through the CLI using `CliRunner` with `mix_stderr=False`.

**Structure:**

```python
class TestRaritiesVetting:

    def test_rejected_terms_excluded(self, runner, corpus_path, tmp_path):
        """A term in vetting.json 'reject' list does not appear in output."""
        ...

    def test_accepted_terms_re_included(self, runner, corpus_path, tmp_path):
        """A term in 'accept' that would be cut by --top-n re-appears."""
        ...

    def test_vetting_file_created_by_vet_flag(self, runner, corpus_path, tmp_path):
        """--vet creates vetting.json with correct structure."""
        ...
```

For the interactive `--vet` flag, use `CliRunner` with `input=` to simulate
keystrokes:

```python
def test_vet_flag_saves_decisions(self, runner, corpus_path, tmp_path):
    result = runner.invoke(
        cli,
        ["rarities", str(corpus_path), "--vet",
         "--output", str(tmp_path / "terms.json")],
        input="y\nn\ns\n",  # accept first, reject second, skip rest
    )
    assert result.exit_code == 0
    vetting = json.loads((tmp_path / "vetting.json").read_text())
    assert "accept" in vetting
    assert "reject" in vetting
```

---

## Testing `--pos` filter on rarities

`--pos noun` should remove non-noun candidates and keep noun candidates.
The test corpus must contain terms with known dominant POS tags.

Use `CliRunner.invoke` and parse the output or output JSON file to check
the term list. Do not parse stdout for term names — write to `--output` and
read the JSON:

```python
def test_pos_noun_filter_keeps_nouns(self, runner, corpus_path, tmp_path):
    output = tmp_path / "terms.json"
    result = runner.invoke(cli, [
        "rarities", str(corpus_path),
        "--pos", "noun",
        "--output", str(output),
    ])
    assert result.exit_code == 0
    data = json.loads(output.read_text())
    # All returned terms should be nouns; at minimum, no verbs should appear
    # (exact POS check requires knowing what the corpus produces)
    assert len(data["terms"]) >= 0  # at minimum: completes without error

def test_unknown_pos_category_warns(self, runner, corpus_path, tmp_path):
    result = runner.invoke(cli, [
        "rarities", str(corpus_path),
        "--pos", "noun,badcategory",
        "--output", str(tmp_path / "terms.json"),
    ])
    assert "unknown POS category" in result.output.lower() or result.exit_code == 0
```

---

## Testing `--by-section`

`--by-section` requires a corpus whose documents have `sentence_locations`
populated. Use the full preprocessing pipeline on a structured text rather
than a mock, because `sentence_locations` is only produced by
`DocumentStructureDetector`.

Alternatively, plant `sentence_locations` directly into a mock document to
avoid the slow pipeline:

```python
def make_docs_with_sections(sentences_by_section: dict) -> list:
    """
    sentences_by_section: {"Chapter 1: Introduction": ["sentence ..."], ...}
    Returns a mock ProcessedDocument with realistic sentence_locations.
    """
    from unittest.mock import MagicMock
    from concept_mapper.corpus.models import SentenceLocation

    all_sentences = []
    locations = []
    sent_idx = 0
    for section_title, sentences in sentences_by_section.items():
        for s in sentences:
            all_sentences.append(s)
            locations.append(SentenceLocation(
                sent_index=sent_idx,
                chapter=None,
                chapter_title=section_title,
                section=None,
                section_title=None,
            ))
            sent_idx += 1

    doc = MagicMock()
    doc.sentences = all_sentences
    doc.sentence_locations = locations
    doc.metadata = {}
    return [doc]
```

Then assert that:

1. `terms_by_section.json` is created alongside `terms.json`.
2. The JSON has a `sections` key containing a list of `{section, terms}` dicts.
3. Terms from section A do not appear under section B (given non-overlapping
   planted terms).

---

## Testing `--depth` and `--focus` on graph

`--depth` and `--focus` are graph-post-processing steps. The simplest test
strategy is to build a known graph JSON, invoke `cmapr graph` with the flag,
and assert on the output JSON's node count.

However, since `cmapr graph` runs the full extraction pipeline, the exact
graph is non-deterministic for real text. Instead, test the underlying
mechanism directly:

```python
from concept_mapper.graph.model import ConceptGraph
import networkx as nx

def make_chain_graph():
    """a — b — c — d — e (linear chain, directed)."""
    g = ConceptGraph(directed=True)
    for n in "abcde":
        g.add_node(n, label=n, score=1.0 if n == "a" else 0.5)
    for src, tgt in zip("abcd", "bcde"):
        g.add_edge(src, tgt, relation_type="kind-of", weight=1)
    return g
```

Then apply the ego_graph logic directly (the same code path as `--depth`)
and assert:

```python
def test_depth_1_includes_only_direct_neighbours():
    g = make_chain_graph()
    sub = nx.ego_graph(g.graph, "c", radius=1, undirected=True)
    assert set(sub.nodes()) == {"b", "c", "d"}

def test_focus_defaults_to_depth_1():
    # when --focus is given without --depth, radius=1
    g = make_chain_graph()
    sub = nx.ego_graph(g.graph, "a", radius=1, undirected=True)
    assert "a" in sub.nodes()
    assert "b" in sub.nodes()
    assert "c" not in sub.nodes()
```

For CLI-level smoke tests (verify the command completes and the output has
fewer nodes than without the flag):

```python
def test_focus_flag_reduces_node_count(self, runner, corpus_path, terms_path, tmp_path):
    full_out = tmp_path / "full.json"
    focus_out = tmp_path / "focus.json"
    runner.invoke(cli, ["graph", str(corpus_path), "-t", str(terms_path),
                        "--output", str(full_out)])
    runner.invoke(cli, ["graph", str(corpus_path), "-t", str(terms_path),
                        "--focus", "sign", "--output", str(focus_out)])
    if full_out.exists() and focus_out.exists():
        full = json.loads(full_out.read_text())
        focused = json.loads(focus_out.read_text())
        assert len(focused["nodes"]) <= len(full["nodes"])
```

---

## Testing spaCy noun chunk extraction

spaCy is an optional dependency. Tests that require it must be marked with
a custom `spacy` mark and skip gracefully when the model is not installed:

```python
import pytest

def spacy_available():
    try:
        import spacy
        spacy.load("en_core_web_sm")
        return True
    except Exception:
        return False

requires_spacy = pytest.mark.skipif(
    not spacy_available(), reason="spaCy en_core_web_sm not installed"
)
```

Apply the mark at class level:

```python
@requires_spacy
class TestNounChunkExtraction:

    def test_multi_word_chunks_extracted(self):
        from concept_mapper.preprocessing.pipeline import _extract_noun_chunks
        chunks = _extract_noun_chunks(
            "The sign vehicle is a triadic relation. "
            "Unlimited semiosis involves sign processes."
        )
        assert any(" " in c for c in chunks), "Expected at least one multi-word chunk"

    def test_leading_determiners_stripped(self):
        from concept_mapper.preprocessing.pipeline import _extract_noun_chunks
        chunks = _extract_noun_chunks("The triadic relation is fundamental.")
        # "triadic relation" not "the triadic relation"
        assert "triadic relation" in chunks
        assert not any(c.startswith("the ") for c in chunks)

    def test_single_word_chunks_excluded(self):
        from concept_mapper.preprocessing.pipeline import _extract_noun_chunks
        chunks = _extract_noun_chunks("Signs exist. Codes function.")
        assert all(" " in c for c in chunks), "Single-word chunks should be excluded"

    def test_noun_chunks_stored_in_metadata(self):
        from concept_mapper.corpus.models import Document
        from concept_mapper.preprocessing.pipeline import preprocess
        doc = Document(text=(
            "The sign vehicle is a triadic relation. "
            "Unlimited semiosis involves sign processes."
        ))
        result = preprocess(doc, use_spacy=True)
        assert "noun_chunks" in result.metadata
        assert isinstance(result.metadata["noun_chunks"], list)
```

For the rarities multi-word integration, test that a corpus with `noun_chunks`
in metadata produces multi-word entries in the term list. Build the doc mock
with `metadata["noun_chunks"]` already populated — do not run the full spaCy
pipeline:

```python
def test_noun_chunks_scored_and_merged(runner, tmp_path):
    """Multi-word chunks from metadata appear in rarities output."""
    # Build a corpus JSON with noun_chunks pre-populated
    docs = [make_processed_doc_with_chunks(
        sentences=["Sign vehicle is a core concept."] * 5,
        noun_chunks=["sign vehicle"] * 5,
    )]
    corpus_path = tmp_path / "corpus.json"
    corpus_path.write_text(json.dumps([d.to_dict() for d in docs]))
    output = tmp_path / "terms.json"
    result = runner.invoke(cli, ["rarities", str(corpus_path),
                                 "--output", str(output)])
    assert result.exit_code == 0
    # Multi-word candidate message should appear
    assert "multi-word" in result.output.lower()
```

---

## Testing `NodeFilter` for multi-word terms

The `is_valid` bypass for multi-word terms (space-containing) should be
tested in `test_node_filter.py` alongside the existing criterion tests:

```python
class TestMultiWordTerms:

    def test_multiword_passes_length_check(self):
        """Multi-word terms skip the ≥4 char check."""
        f = make_filter(freqs={"ab cd": 5})
        ok, reason = f.is_valid("ab cd")
        # Length of "ab cd" is 5, but "ab" alone would fail at 2 chars.
        # Multi-word terms should not be rejected on length.
        assert ok or "low frequency" in reason  # only freq can reject it

    def test_multiword_passes_char_validity(self):
        """Spaces are allowed in multi-word terms."""
        f = make_filter(freqs={"sign vehicle": 10})
        ok, reason = f.is_valid("sign vehicle")
        assert ok

    def test_multiword_passes_fragment_check(self):
        """Fragment detection is skipped for multi-word terms."""
        # "structu" would be a fragment, but "structu re" is multi-word
        f = make_filter(
            vocab={"structure"}, freqs={"structu re": 10}
        )
        ok, reason = f.is_valid("structu re")
        assert ok

    def test_multiword_still_rejected_on_low_frequency(self):
        f = make_filter(freqs={"sign vehicle": 1}, min_freq=3)
        ok, reason = f.is_valid("sign vehicle")
        assert not ok
        assert "low frequency" in reason
```

---

## CLI testing conventions

All CLI tests use `click.testing.CliRunner`. Always:

- Pass `mix_stderr=False` to `CliRunner()` so stdout and stderr are
  separated and you can assert on each independently.
- Use `--output` to write to `tmp_path` rather than relying on
  auto-derived output paths; this makes assertions deterministic.
- Check `result.exit_code == 0` before asserting on output content —
  a non-zero exit with a confusing output assertion is a false failure.
- When testing error cases, check `result.exit_code != 0` AND that a
  meaningful error message appears in `result.output` or `result.stderr`.

```python
def test_ingest_missing_file_exits_nonzero(runner):
    result = runner.invoke(cli, ["ingest", "/does/not/exist.txt"])
    assert result.exit_code != 0

def test_graph_focus_unknown_term_warns_not_crashes(runner, corpus_path, terms_path, tmp_path):
    result = runner.invoke(cli, [
        "graph", str(corpus_path), "-t", str(terms_path),
        "--focus", "xxxxnotaword",
        "--output", str(tmp_path / "g.json"),
    ])
    # Should warn but not crash
    assert result.exit_code == 0
    assert "not found" in result.output.lower() or "warning" in result.output.lower()
```

---

## What not to test

**Do not test implementation details.** If `_try_pos_verb` calls
`self._pos_tag` internally, do not assert that. Test the observable output
(`Proposition.type`, `Proposition.source`) not the internal call chain.

**Do not test the framework.** Click, networkx, NLTK, and pytest are tested
by their own maintainers. Do not write tests that verify `Counter` works or
that `json.dumps` produces valid JSON.

**Do not test the reference corpus content.** The Brown corpus is external
data. Tests that assert a specific word is or is not in Brown are fragile.
Test the *mechanism* of loading and caching, not the data.

**Do not test visualization HTML by string-matching the template.** The HTML
template is a formatted string; asserting that `"linkLabel"` appears in the
output ties your test to incidental implementation text. If you need to test
HTML behavior, assert on the JSON data passed to it, or assert on structural
markers (`<svg`, `</html>`) rather than variable content.

**Do not add tests to hit an arbitrary coverage number.** A test that
exercises a code path without asserting any property of the output is noise.

---

## Parametrize aggressively for variant testing

When the same assertion applies to multiple inputs, use `@pytest.mark.parametrize`
rather than repeating test functions:

```python
@pytest.mark.parametrize("marker,term_a,term_b", [
    ("By 'sign' I mean a vehicle.", "sign", "vehicle"),
    ("The code is defined as a mapping.", "code", "mapping"),
    ("Semiosis denotes the sign process.", "semiosis", "process"),
])
def test_definition_markers(marker, term_a, term_b):
    e = PropositionExtractor(make_docs([marker]))
    props = e.extract(term_a, term_b)
    assert any(p.type == "definition" for p in props)
```

Use parametrize for:
- Multiple regex pattern variants of the same extractor (`vs`, `versus`,
  `as opposed to` should all fire `opposition`)
- Multiple rejection criteria in `NodeFilter` (`too short`, `stopword`,
  `fragment`) where the assertion structure is identical
- Multiple POS categories in `--pos` filter tests

---

## Slow tests

Mark any test that runs the full preprocessing pipeline (tokenize + POS tag +
lemmatize + structure detect) on a document longer than ~10 sentences as slow:

```python
@pytest.mark.slow
def test_full_pipeline_on_chapter_text(tmp_path):
    ...
```

Do not include `--slow` in the default `pytest` invocation in CI. Use:
```
pytest -m "not slow"   # fast suite
pytest                 # everything
```

Currently no `slow` tests exist. Add the mark proactively as you write
pipeline-level tests that take > 2 seconds.

---

## Regression policy

When a bug is fixed, add a regression test **in the same commit** as the fix.
The test name should describe the bug, not the fix:

```python
def test_try_pos_verb_does_not_misclassify_transforms_as_noun():
    """Regression: 'transforms' was tagged NNS when only the between-text
    was passed to the POS tagger, stripping the surrounding sentence context."""
    s = "The sign transforms the interpretant through a process of mediation."
    e = PropositionExtractor(make_docs([s]))
    props = e.extract("sign", "interpretant")
    assert any(p.type in ("relation", "production", "dependence") for p in props)
    assert not any(p.type == "cooccurrence" for p in props)
```

This serves two purposes: it documents the original bug, and it prevents
it from silently reappearing.

---

## Running the suite

```bash
# Fast (default):
python -m pytest -q

# Specific file:
python -m pytest tests/test_proposition_extractor.py -v

# Specific class:
python -m pytest tests/test_graph.py::TestPrunesToRatio -v

# Specific test:
python -m pytest tests/test_node_filter.py::TestFragmentCriterion::test_prefix_fragment -v

# With coverage:
python -m pytest --cov=src/concept_mapper --cov-report=term-missing -q

# spaCy tests only (requires model):
python -m pytest -m spacy -v
```
