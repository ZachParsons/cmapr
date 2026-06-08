"""
Tests for the per-node sentence concordance (search/concordance.py).

Backs the graph viz's node-click sidebar: every sentence a term's lemma appears
in, in document order, with structural location and highlight-ready surface
forms. See docs/plans/node-concordance.md.
"""

from concept_mapper.corpus.models import ProcessedDocument, SentenceLocation
from concept_mapper.search.concordance import build_concordance, _format_location


def _doc(sentences, locations=None, **kw):
    return ProcessedDocument(
        raw_text=" ".join(sentences),
        sentences=sentences,
        tokens=kw.get("tokens", []),
        lemmas=kw.get("lemmas", []),
        pos_tags=kw.get("pos_tags", []),
        metadata=kw.get("metadata", {"source_path": "d.txt"}),
        sentence_locations=locations or [],
    )


class TestFormatLocation:
    def test_chapter_and_section(self):
        loc = SentenceLocation(
            sent_index=0,
            chapter="3",
            chapter_title="The Sign",
            section="2.1",
            section_title="Definitions",
        )
        assert _format_location(loc) == "Ch. 3 — The Sign › §2.1 Definitions"

    def test_paragraph_fallback(self):
        loc = SentenceLocation(sent_index=0, paragraph=12)
        assert _format_location(loc) == "¶ 12"

    def test_empty_when_nothing(self):
        assert _format_location(SentenceLocation(sent_index=0)) == ""
        assert _format_location(None) == ""

    def test_accepts_raw_dict(self):
        # corpus loaded via raw splat leaves locations as dicts.
        assert _format_location({"chapter_title": "Codes"}) == "Codes"


class TestBuildConcordance:
    def test_single_token_lemma_match(self):
        # "sign" should match the plural "signs" via lemmatization.
        docs = [_doc(["A sign matters.", "The signs vary.", "Nothing here."])]
        conc = build_concordance(docs, ["sign"])
        recs = conc["sign"]
        assert len(recs) == 2
        assert recs[0]["text"] == "A sign matters."
        assert recs[1]["text"] == "The signs vary."
        # surface forms captured for highlighting
        assert recs[0]["marks"] == ["sign"]
        assert recs[1]["marks"] == ["signs"]

    def test_matches_all_inflected_surface_forms(self):
        # A node term must find every inflected form in the text, not just the
        # exact lemma — including the inflect-singularized noun (semiotics ->
        # semiotic) that POS-aware lemmatization can miss mid-sentence.
        docs = [
            _doc(
                [
                    "Semiotics studies signs.",
                    "Modern semiotics analyses the sign.",
                    "He signed and signs documents.",
                ]
            )
        ]
        semiotic = build_concordance(docs, ["semiotic"])["semiotic"]
        assert len(semiotic) == 2  # both "Semiotics" and "semiotics"

        sign = build_concordance(docs, ["sign"])["sign"]
        # sign / signs / signed all map back to the node term.
        assert len(sign) == 3
        marks = {m.lower() for r in sign for m in r["marks"]}
        assert {"signs", "signed"} <= marks

    def test_document_order_across_docs(self):
        docs = [_doc(["First sign."]), _doc(["Second sign."])]
        recs = build_concordance(docs, ["sign"])["sign"]
        assert [r["text"] for r in recs] == ["First sign.", "Second sign."]

    def test_variants_extend_matching_to_merged_forms(self):
        # A pipeline-merged variant (taxonomy ⇐ taxonomic) is matched and marked.
        docs = [
            _doc(["Taxonomic rank matters.", "A taxonomy is built.", "Unrelated here."])
        ]
        recs = build_concordance(
            docs, ["taxonomy"], variants={"taxonomy": ["taxonomic"]}
        )["taxonomy"]
        texts = [r["text"] for r in recs if "text" in r]
        assert "Taxonomic rank matters." in texts
        assert "A taxonomy is built." in texts
        marks = {m.lower() for r in recs if "text" in r for m in r["marks"]}
        assert {"taxonomic", "taxonomy"} <= marks

    def test_without_variants_excludes_derivational_form(self):
        # No variants map → derivational form is NOT included (only inflections).
        docs = [_doc(["Taxonomic rank matters.", "A taxonomy is built."])]
        recs = build_concordance(docs, ["taxonomy"])["taxonomy"]
        texts = [r["text"] for r in recs if "text" in r]
        assert texts == ["A taxonomy is built."]

    def test_phrase_substring_match(self):
        docs = [_doc(["The content plane differs.", "Only content here."])]
        recs = build_concordance(docs, ["content plane"])["content plane"]
        assert len(recs) == 1
        assert recs[0]["marks"] == ["content plane"]

    def test_location_attached(self):
        loc = SentenceLocation(sent_index=0, chapter="1", chapter_title="Intro")
        docs = [_doc(["A sign here."], locations=[loc])]
        recs = build_concordance(docs, ["sign"])["sign"]
        assert recs[0]["loc"] == "Ch. 1 — Intro"

    def test_no_match_returns_empty(self):
        docs = [_doc(["Nothing relevant."])]
        assert build_concordance(docs, ["sign"])["sign"] == []

    def test_per_term_cap_truncates(self):
        docs = [_doc([f"sign number {i}." for i in range(10)])]
        recs = build_concordance(docs, ["sign"], per_term_cap=3)["sign"]
        sentences = [r for r in recs if "text" in r]
        sentinel = [r for r in recs if "truncated" in r]
        assert len(sentences) == 3
        assert sentinel == [{"truncated": 3, "total": 10}]
