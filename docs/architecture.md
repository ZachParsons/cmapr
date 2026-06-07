# Architecture

Developer navigation map. Use this to find the area of the app to tweak, extend, or trace. Every module has a one-liner; every stage has its internal sub-steps and the functions responsible.

For at-a-glance project state (latest / next / open issues), see `docs/roadmap.md` § Status.

---

## Pipeline diagram

Designed for a wall-sized monitor. Stages are decomposed proportional to algorithmic substance: **rarities** and **graph** drill down to individual functions (every scorer signal, every filter, every extractor); **ingest** lists each preprocessing module and its key functions; **export** is collapsed to formats; every CLI command and every web UI step is connected to the code that runs it.

Boxes show *file path · function/class · one-line behaviour*. Solid arrows (`==>`) are main data flow; dotted (`-.->`) are optional / read-only.

```mermaid
%%{init: {
  "theme": "default",
  "themeVariables": {
    "fontSize": "16px",
    "fontFamily": "ui-monospace, SFMono-Regular, Menlo, monospace",
    "primaryColor": "#fafafa",
    "primaryBorderColor": "#444",
    "lineColor": "#666",
    "clusterBkg": "#f8f8f8",
    "clusterBorder": "#888"
  },
  "flowchart": {
    "nodeSpacing": 70,
    "rankSpacing": 90,
    "padding": 16,
    "diagramPadding": 20,
    "useMaxWidth": false,
    "htmlLabels": true,
    "curve": "basis"
  }
}}%%
flowchart TB
    %% =====================================================================
    %% INPUTS
    %% =====================================================================
    src(["<b>source</b><br/>.txt / .pdf"]):::io

    %% =====================================================================
    %% STAGE 1 — INGEST (medium)
    %% =====================================================================
    subgraph S1["<b>stage 1 · ingest</b>  &nbsp;·&nbsp;  <code>cmapr ingest [--clean-ocr] [--toc PATH] [--spacy]</code>"]
        direction TB
        i1["<b>corpus/loader.py</b><br/><i>load_file</i> · <i>load_directory</i><br/>UTF-8 → Latin-1 fallback<br/>PDF auto-detect via pdfplumber"]

        subgraph S1clean["<b>preprocessing/cleaning.py</b>  ·  <i>clean_text</i>  ·  (<code>--clean-ocr</code>)"]
            direction LR
            c1["<i>fix mid-word splits</i><br/>'pro position' → 'proposition'"]
            c2["<i>repair hyphenated line breaks</i><br/>'sub-<br/>stance' → 'substance'"]
            c3["<i>strip page numbers</i><br/>+ running headers"]
            c4["<i>normalize unicode</i><br/>curly quotes, ligatures"]
            c5["<i>substitution table</i><br/>common OCR confusables"]
        end

        i3a["<b>preprocessing/tokenize.py</b><br/><i>tokenize_sentences</i><br/>NLTK sent_tokenize"]
        i3b["<b>preprocessing/tokenize.py</b><br/><i>tokenize_words</i><br/>NLTK word_tokenize<br/>(plus _preserve_case variant)"]
        i4["<b>preprocessing/tagging.py</b><br/><i>tag_tokens</i> · <i>tag_sentences</i><br/>NLTK Penn Treebank POS<br/><i>filter_by_pos</i> helper"]
        i5["<b>preprocessing/lemmatize.py</b><br/><i>lemmatize_tagged</i><br/>WordNet lemmatizer<br/>+ inflect fallback for OOV"]

        subgraph S1struct["<b>preprocessing/structure.py</b>  ·  <i>DocumentStructureDetector</i>"]
            direction LR
            st1["<i>TOC parser</i><br/>when <code>--toc</code> supplied"]
            st2["<i>numbered headings</i><br/>'1.', '1.1.', '1.1.1.'"]
            st3["<i>named chapters</i><br/>'Chapter 1', 'Part I'"]
            st4["<i>Markdown headings</i><br/>'# ', '## ', '### '"]
            st5["<i>all-caps + blank line</i><br/>section markers"]
            st6["<i>emits StructureNode tree</i><br/>+ SentenceLocation per sentence"]
            st1 --> st6
            st2 --> st6
            st3 --> st6
            st4 --> st6
            st5 --> st6
        end

        i7["<b>preprocessing/segment.py</b><br/><i>segment_paragraphs</i><br/><i>get_paragraph_indices</i><br/>double-newline + indent detection"]

        subgraph S1chunks["<b>preprocessing/noun_chunks.py</b>  ·  (<code>--spacy</code>)"]
            direction TB
            nc1["<i>_get_spacy_nlp</i><br/>lazy-load en_core_web_sm<br/>(module-level cache)"]
            nc2["<i>extract_noun_chunks</i><br/>doc.noun_chunks iterator"]
            nc3["<i>strip leading determiners</i><br/>the/a/an/this/that/...<br/>"]
            nc4["<i>filter</i><br/>len ≥ 2 tokens<br/>drop 1-char tokens (OCR)"]
            nc5["<i>dedupe + store</i><br/>metadata['noun_chunks']"]
            nc1 --> nc2 --> nc3 --> nc4 --> nc5
        end

        i9["<b>preprocessing/pipeline.py</b><br/><i>preprocess</i><br/>orchestrator — calls each stage<br/>builds ProcessedDocument"]
        i10["<b>validation.py</b><br/><i>validate_corpus</i>"]

        i1 --> S1clean --> i3a --> i3b --> i4 --> i5
        i5 --> S1struct
        i5 --> i7
        i5 --> S1chunks
        S1struct --> i9
        i7 --> i9
        S1chunks --> i9
        i9 --> i10
    end

    corpus(("<b>corpus.json</b><br/>ProcessedDocument[]<br/>sentences · tokens · POS · lemmas<br/>+ structure_nodes · sentence_locations<br/>+ paragraph_indices · metadata")):::artifact

    %% =====================================================================
    %% STAGE 2 — RARITIES (very high — every signal + every filter)
    %% =====================================================================
    subgraph S2["<b>stage 2 · rarities</b>  &nbsp;·&nbsp;  <code>cmapr rarities [--top-n N] [--pos CAT] [--by-section] [--vet]</code>"]
        direction TB

        subgraph S2ref["<b>analysis/reference.py</b>"]
            direction LR
            r1["<i>load_reference_corpus</i><br/>Brown corpus<br/>(disk-cached at data/output/cache/)"]
            r2["<i>get_reference_vocabulary</i>"]
            r3["<i>get_reference_size</i>"]
        end

        subgraph S2score["<b>analysis/rarity.py</b>  ·  <code>PhilosophicalTermScorer.score_all</code>  ·  5-signal hybrid"]
            direction TB

            subgraph S2sig1["<b>signal 1 · ratio</b>  (weight 1.0)"]
                direction TB
                s1a["<i>analysis/frequency.py</i><br/><i>word_frequencies</i> · <i>corpus_frequencies</i>"]
                s1b["<i>compare_to_reference</i><br/>corpus_count / brown_count<br/>normalised by total tokens"]
                s1c["<i>get_top_corpus_specific_terms</i>"]
                s1a --> s1b --> s1c
            end

            subgraph S2sig2["<b>signal 2 · TF-IDF</b>  (weight 1.0)"]
                direction TB
                s2a["<i>analysis/tfidf.py</i><br/><i>tf</i> · <i>idf</i> · <i>tfidf</i>"]
                s2b["<i>tfidf_vs_reference</i><br/>TF in corpus ÷ IDF computed<br/>against Brown"]
                s2c["<i>get_distinctive_by_tfidf</i>"]
                s2a --> s2b --> s2c
            end

            subgraph S2sig3["<b>signal 3 · neologism</b>  (weight 0.5)"]
                direction TB
                s3a["<i>_load_wordnet_vocabulary</i><br/>117K WordNet words (cached)"]
                s3b["<i>get_wordnet_neologisms</i><br/>not in WordNet"]
                s3c["<i>get_capitalized_technical_terms</i><br/>non-sentence-initial caps"]
                s3d["<i>get_potential_neologisms</i><br/>combined signal"]
                s3a --> s3b --> s3d
                s3c --> s3d
            end

            subgraph S2sig4["<b>signal 4 · definitional context</b>  (weight 0.3)"]
                direction TB
                s4a["<i>get_definitional_contexts</i><br/>8 patterns:"]
                s4b["copular · explicit_mean ·<br/>metalinguistic · conceptual"]
                s4c["appositive · explicit_define ·<br/>referential · interpretive"]
                s4d["<i>score_by_definitional_context</i><br/>count matches per term"]
                s4a --> s4b --> s4d
                s4a --> s4c --> s4d
            end

            subgraph S2sig5["<b>signal 5 · capitalization</b>  (weight 0.2)"]
                direction TB
                s5a["mid-sentence Capital ratio<br/>(reified abstraction signal)"]
            end

            sigsum["<b>weighted sum</b><br/>total = 1.0·ratio + 1.0·tfidf<br/>+ 0.5·neologism + 0.3·definitional<br/>+ 0.2·capitalization<br/><br/>→ (term, total, components)"]

            S2sig1 --> sigsum
            S2sig2 --> sigsum
            S2sig3 --> sigsum
            S2sig4 --> sigsum
            S2sig5 --> sigsum
        end

        subgraph S2filter["<b>terms/scoring.py</b>  ·  post-scoring filter chain  ·  shared by <code>cmapr rarities</code> + <code>cmapr run</code>"]
            direction TB
            note_chain["<i>single source of truth — both CLI handlers call into here</i>"]:::note
            f1["<i>strip_stray_quotes</i><br/>strip ' ‘ ’ ‚ ‛ from term ends<br/>drop terms that empty out"]
            f2a["<i>score_multi_word_chunks</i><br/>TF-IDF over noun_chunks<br/>scoring formula:<br/>log(1+freq) · (1 + log((N+1)/(df+1)))<br/>min_freq = min(2, n_docs)"]
            f2b["<i>merge_extra_candidates</i><br/>re-sort + raw_candidates update"]
            f3["<i>filter_proper_names</i><br/>pn_ratios from POS tags<br/>pn_ratio ≥ 0.3 AND<br/>Brown freq &lt; 25 ppm"]
            f3b["<i>filter_stopwords</i><br/>shared STOPWORDS (search/extract.py)<br/>drops since/whether/could/…<br/>phrases bypass · runs before top-N"]
            f4["<i>lemma_and_derivational_merge</i><br/>Pass 1: WordNet noun lemma<br/>(semiotics → semiotic)<br/>Pass 2: 13 derivational suffixes<br/>(co-textual → co-text)<br/>Pass 3: merge_derivational_variants<br/>WordNet derivationally_related_forms<br/>(taxonomic → taxonomy, prefer noun)<br/><b>top-N applied here</b>"]
            f5["<i>filter_fragments</i><br/>&lt; 4 chars OR<br/>(not in WordNet AND<br/>+s/+y/+ed/+er/+al/+ic/+is/+sis<br/>completes a WordNet word)"]
            f5b["<i>filter_ocr_artifacts</i><br/>non-WordNet non-stopword AND<br/>fn-word merge (thesecase→these+case)<br/>OR leading-char drop (ictionary→dictionary)"]
            f6["<i>filter_by_pos_categories</i><br/>POS_CATEGORY_MAP:<br/>noun {NN,NNS,NNP,NNPS}<br/>verb {VB,VBD,VBG,VBN,VBP,VBZ}<br/>adj {JJ,JJR,JJS} · adv {RB,RBR,RBS}<br/>multi-word phrases bypass"]
            f7a["<i>load_vetting</i><br/>read vetting.json<br/>(accept + reject sets)"]
            f7b["<i>apply_vetting</i><br/>drop rejected ·<br/>re-include accepted from raw"]
            f7c["<i>save_vetting</i><br/>(when <code>--vet</code>)"]
            note_chain -.-> f1
            f1 --> f2a --> f2b --> f3 --> f3b --> f4 --> f5 --> f5b --> f6
            f7a --> f7b --> f7c
            f6 --> f7b
        end

        subgraph S2bysec["<b>by-section grouping</b>  ·  (<code>--by-section</code>)"]
            direction TB
            bs1["walk doc.sentence_locations<br/>(doc_idx, sent_idx) → label"]
            bs2["for each term:<br/>label = chapter most occurrences"]
            bs3["writes &lt;work&gt;_by_section.json"]
            bs1 --> bs2 --> bs3
        end

        subgraph S2terms["<b>terms/</b>"]
            direction LR
            tm1["<i>terms/models.py</i><br/>TermList · TermEntry"]
            tm2["<i>terms/manager.py</i><br/>TermManager.export_to_json<br/>CSV / TXT import-export"]
            tm3["<i>terms/suggester.py</i><br/>suggest_terms_from_analysis<br/>(auto-populate examples)"]
        end

        i_validate2["<b>validation.py</b><br/><i>validate_term_list</i>"]

        S2ref --> S2score
        S2score --> S2filter
        S2filter --> S2bysec
        S2filter --> S2terms
        S2bysec -.-> S2terms
        S2terms --> i_validate2
    end

    terms_json(("<b>terms.json</b><br/>TermList<br/>[{term, metadata: {score}, ...}]")):::artifact
    bysec_out(["<b>&lt;work&gt;_by_section.json</b><br/>(optional)"]):::io
    vetting_out(["<b>vetting.json</b><br/>{accept: [...], reject: [...]}"]):::io

    %% =====================================================================
    %% STAGE 3 — GRAPH (high — every extractor + build + post-build)
    %% =====================================================================
    subgraph S3["<b>stage 3 · graph</b>  &nbsp;·&nbsp;  <code>cmapr graph [--focus TERM] [--depth N] [--threshold F]</code>"]
        direction TB

        subgraph S3prep["<b>graph/node_filter.py</b>  ·  <code>NodeFilter</code>"]
            direction TB
            nf1["<i>corpus_vocab</i> = set(all_tokens)<br/><i>term_freqs</i> = Counter(all_lemmas)"]
            nf2["<i>is_valid(term, pos)</i> checks:"]
            nf3["1. POS in {NN*, VB*, JJ*, RB*}"]
            nf4["2. length ≥ 4 (single-token only)"]
            nf5["3. not all-caps abbreviation"]
            nf6["4. not in STOPWORDS"]
            nf7["5. not fragment (prefix of corpus word AND not in WordNet)"]
            nf8["6. freq ≥ 3 (single-token only)"]
            nf9["multi-word phrases bypass 2,3,5,6"]
            nf1 --> nf2
            nf2 --> nf3
            nf3 --> nf4
            nf4 --> nf5
            nf5 --> nf6
            nf6 --> nf7
            nf7 --> nf8
            nf8 --> nf9
        end

        subgraph S3extract["<b>graph/proposition_extractor.py</b>  ·  <code>PropositionExtractor</code>  ·  priority chain (first match wins)"]
            direction TB
            note_pri["<i>per (term_a, term_b) pair × per sentence containing both</i>"]:::note
            e1["<b>_try_definition</b><br/>regex on _DEFINITION_PATTERNS<br/>'by X I mean Y' · 'X is defined as Y'<br/>'X denotes Y' · 'X stands for Y'<br/>→ <code>definition</code> edge (directed)"]
            e2["<b>_try_kind_of</b><br/>'X is a {type|kind|sort|species} of Y'<br/>_KIND_MARKERS regex<br/>→ <code>kind-of</code> edge (directed)"]
            e3["<b>_try_production</b><br/>SVO with _PRODUCTION_VERBS:<br/>produce · generate · give rise to ·<br/>imply · create<br/>→ <code>production</code> edge (directed)"]
            e4["<b>_try_dependence</b><br/>SVO/prep with _DEPENDENCE_PHRASES:<br/>presuppose · depend on · require · need<br/>→ <code>dependence</code> edge (directed)"]
            e5["<b>_try_opposition</b><br/>_OPPOSITION_PATTERNS:<br/>X vs Y · X as opposed to Y ·<br/>X rather than Y · X not Y<br/>→ <code>opposition</code> edge (undirected)"]
            e6["<b>_try_property</b><br/>plain copular X is Y<br/>(after definition/kind-of negative lookahead)<br/>→ <code>property</code> edge (directed)"]
            e7["<b>_try_relation</b><br/>SVO with _RELATION_VERBS<br/>broad catch-all verb list<br/>→ <code>relation</code> edge (directed)"]
            e8["<b>_try_pos_verb</b><br/>NLTK POS tag scan between terms<br/>first non-copular non-light verb<br/>passive voice → reverse direction<br/>(no-op on multi-word phrases — token-level lookup)<br/>→ <code>relation</code> edge"]
            note_pri --> e1 --> e2 --> e3 --> e4 --> e5 --> e6 --> e7 --> e8
        end

        subgraph S3compo["<b>extract_composition</b>  ·  parallel pattern"]
            direction TB
            cp1["regex match _COMPOSITION_VERBS:<br/>form · constitute · compose · make up · consist of"]
            cp2["before-verb: list of seed terms (≥ 2)<br/>after-verb: composed entity"]
            cp3["emit <code>component</code> edges<br/>between all constituent pairs"]
            cp4["emit <code>production</code> edges<br/>from each constituent → composed"]
            cp1 --> cp2 --> cp3
            cp2 --> cp4
        end

        subgraph S3evidence["<b>_score_sentence</b>  ·  evidence ranking"]
            direction LR
            ev1["+10 definition marker<br/>(defined as / by X I mean / denotes)"]
            ev2["+5 proximity ≤ 15 words<br/>between term_a and term_b"]
            ev3["− length / 400<br/>shorter = better"]
            ev4["− sentence_index / n_total<br/>earlier = better"]
        end

        subgraph S3build["<b>graph/builders.py</b>  ·  <code>build_proposition_graph</code>"]
            direction TB
            b1["<i>iterate pairs (term_a, term_b)</i><br/>O(n²) over seed_terms"]
            b2["<i>same-type duplicate merge</i><br/>group by (source, target, type)<br/>weight = count of co-occurrence sentences<br/>evidence = top-3 by _score_sentence"]
            b3["<i>PMI cooccurrence fallback</i><br/>only when no typed edge AND<br/>PMI = log2(n_ab · N / (n_a · n_b)) ≥ pmi_threshold<br/>uses analysis/cooccurrence.py"]
            b4["<i>cross-type priority collapse</i><br/>_TYPE_PRIORITY ladder:<br/>definition &gt; kind-of &gt; production &gt;<br/>dependence &gt; component &gt; opposition &gt;<br/>property &gt; relation &gt; cooccurrence<br/>→ one edge per (source, target)"]
            b5["<i>NodeFilter applied to extracted endpoints</i><br/>(same rules as seed filtering)"]
            b6["term_scores → node.score attribute<br/>(used by HTML for sizing)"]
            b1 --> b2 --> b3 --> b4 --> b5 --> b6
        end

        subgraph S3post["<b>post-build operations</b>"]
            direction TB
            po1["<b>operations.py</b><br/><i>consolidate_duplicate_labels</i><br/>dedup key = (label, chapter)<br/>preserves cluster namespacing"]
            po2["<b>operations.py</b><br/><i>find_isolated_nodes</i><br/><i>connect_isolated_nodes</i><br/>last-resort cooccurrence fallback"]
            po3["<b>pruning.py</b><br/><i>prune_to_ratio</i><br/>target ratio = 3.0<br/>pass 1: drop cooccurrence (weakest first)<br/>pass 2: drop typed edges (weakest first)<br/>never isolate a node"]
            po4["<b>operations.py</b> + <i>nx.ego_graph</i><br/>(<code>--focus TERM</code> / <code>--depth N</code>)<br/>centre = focus or highest-scoring seed"]
            po5["<b>metrics.py</b><br/><i>detect_communities</i> · Louvain<br/><i>assign_communities</i> → node.community<br/>(runs at export time)"]
            po6["<b>validation.py</b><br/><i>validate_concept_graph</i>"]
            po1 --> po2 --> po3 --> po4 --> po5 --> po6
        end

        S3prep --> S3extract
        S3extract --> S3build
        S3compo --> S3build
        S3evidence -. used by .-> S3extract
        S3build --> S3post
    end

    graph_json(("<b>graph.json</b><br/>ConceptGraph (D3 format)<br/>nodes: id · label · size · group · score<br/>+ chapter? · section? · term?<br/>links: source · target · weight · type · verb<br/>+ evidence · relation_types? · weight_by_type?")):::artifact

    %% =====================================================================
    %% STAGE 4 — EXPORT (low)
    %% =====================================================================
    subgraph S4["<b>stage 4 · export</b>  &nbsp;·&nbsp;  <code>cmapr export --format {html|d3|graphml|csv|gexf} [--corpus PATH]</code>"]
        direction LR
        x1["<b>export/d3.py</b><br/><i>to_d3_dict</i><br/><i>export_d3_json</i><br/><i>load_d3_json</i>"]
        x2["<b>export/html.py</b><br/><i>generate_html(…, docs=)</i><br/>D3 force sim · legend ·<br/>node detail panel · cluster force ·<br/>concordance sidebar (--corpus) ·<br/>expand/collapse · per-type colours"]
        x2b["<b>search/concordance.py</b><br/><i>build_concordance</i><br/>per-node sentence list +<br/>location + highlight marks<br/>(inlined when --corpus given)"]
        x3["<b>export/formats.py</b><br/><i>export_graphml</i> · <i>export_gexf</i><br/><i>export_csv</i> · <i>export_dot</i>"]
        x2b -.-> x2
    end

    out_html(["<b>index.html</b>"]):::io
    out_other(["<b>d3 · graphml · csv · gexf</b>"]):::io

    %% =====================================================================
    %% GRAPH-LAYER SIDE COMMANDS
    %% =====================================================================
    subgraph Smerge["<code>cmapr merge graph1.json graph2.json ... -o OUT</code>  ·  graph/aggregation.py"]
        direction TB
        m1["<i>aggregate_graphs</i><br/>sum frequencies"]
        m2["frequency-weighted score mean"]
        m3["per-pair multi-type schema:<br/>relation_types · weight_by_type ·<br/>evidence_by_type · verb_by_type"]
        m4["legacy <i>merge_graphs</i> kept<br/>(last-write-wins primitive)"]
        m1 --> m2 --> m3
    end

    subgraph Scluster["<code>cmapr cluster CORPUS -t TERMS</code>  ·  graph/cluster.py"]
        direction TB
        cl1["<i>cluster_by_structure</i><br/>group sentences by chapter"]
        cl2["per-cluster sub-corpus<br/>+ build_proposition_graph"]
        cl3["namespaced nodes:<br/>&lt;term&gt;__&lt;chapter&gt;"]
        cl4["<code>recurrence</code> edges chaining<br/>same-term nodes across chapters<br/>weight = span"]
        cl1 --> cl2 --> cl3 --> cl4
    end

    %% =====================================================================
    %% WORKFLOW WRAPPERS
    %% =====================================================================
    subgraph Srun["<code>cmapr run TEXT [--top-n N] [--format F]</code>"]
        runc["chains stages 1→4 inline<br/>uses terms.scoring.apply_run_pipeline<br/>(same filter chain as rarities)"]
    end

    subgraph Sserve["<code>cmapr serve [--port P]</code>  ·  web UI  ·  server/app.py + templates/"]
        direction TB
        ui_home["<b>home</b> · GET /<br/>scan data/output/corpus/<br/>list works as resume-shortcuts"]
        ui_step1["<b>step 1 · configure</b> · POST /ingest<br/>source path · TOC · checkboxes:<br/><code>--clean-ocr</code> · <code>--spacy</code>"]
        ui_step2["<b>step 2 · term review</b> · POST /review<br/>checkbox-per-term<br/>writes vetting.json"]
        ui_step3["<b>step 3 · graph options</b> · POST /build<br/>top-n · threshold ·<br/>depth · focus"]
        ui_step4["<b>step 4 · result</b> · GET /result<br/>embedded D3 (iframe)<br/>'open full screen' · 'export' ·<br/>'re-run graph' shortcut"]
        ui_home --> ui_step1 --> ui_step2 --> ui_step3 --> ui_step4
        ui_step4 -.->|"re-run graph"| ui_step3
        ui_home -.->|"resume"| ui_step2
    end

    %% =====================================================================
    %% AUXILIARY COMMANDS (read existing artifacts)
    %% =====================================================================
    subgraph Saux["<b>auxiliary commands</b>  ·  alt lenses on existing artifacts"]
        direction TB
        a_search["<code>cmapr search TERM [-c N]</code><br/>search/find.py · context.py ·<br/>dispersion.py · extract.py<br/>KWIC · context · dispersion"]
        a_analyze["<code>cmapr analyze TERM</code><br/>analysis/contextual_relations.py<br/>windowed SVO + co-occurrence<br/>uses analysis/relations.py:<br/>extract_svo · extract_copular · extract_prepositional"]
        a_replace["<code>cmapr replace SRC DST</code><br/>transformations/inflection.py<br/>+ replacement.py<br/>+ phrase_matcher.py<br/>+ text_reconstruction.py<br/>inflection-preserving"]
        a_diagram["<code>cmapr diagram SENTENCE</code><br/>syntax/diagram.py<br/>Stanza dependency parse"]
    end

    %% =====================================================================
    %% CROSS-CUTTING LAYER
    %% =====================================================================
    subgraph Xcross["<b>cross-cutting layer</b>"]
        direction LR
        xv["<b>validation.py</b><br/>validate_corpus · validate_term_list ·<br/>validate_concept_graph"]
        xs["<b>storage/</b><br/>StorageBackend ABC · JSONBackend ·<br/>derive_identifier · infer_output_path"]
        xc["<b>cli.py</b><br/>Click entrypoint · 12 commands<br/>thin shells over the modules below"]
    end

    %% =====================================================================
    %% FLOWS
    %% =====================================================================
    src ==> S1 ==> corpus
    corpus ==> S2 ==> terms_json
    S2 -.-> bysec_out
    S2 -.-> vetting_out
    vetting_out -.->|"loaded on next run"| S2
    terms_json ==> S3
    corpus ==>|"sentences for extraction"| S3
    S3 ==> graph_json ==> S4
    S4 ==> out_html
    S4 ==> out_other

    %% Graph-layer side commands
    graph_json -.-> Smerge
    Smerge ==> merged_out(("<b>merged graph.json</b>")):::artifact ==> S4
    corpus -.-> Scluster
    terms_json -.-> Scluster
    Scluster ==> clustered_out(("<b>clustered graph.json</b>")):::artifact ==> S4

    %% Workflow wrappers
    src -.- Srun -.- out_html
    src -.- Sserve -.- out_html

    %% Auxiliary read-only
    corpus -.->|"read-only"| Saux

    %% =====================================================================
    %% Styling
    %% =====================================================================
    classDef io fill:#fff3e0,stroke:#fb8c00,stroke-width:3px,color:#000
    classDef artifact fill:#e3f2fd,stroke:#1976d2,stroke-width:3px,color:#000
    classDef note fill:#fff9c4,stroke:#fbc02d,stroke-width:2px,color:#000
```

**Rendering size.** The `init` block at the top sets `useMaxWidth: false` and bumps node/rank spacing so Mermaid renders the diagram at its natural width (no fit-to-container shrinking). At default zoom in a Mermaid-aware renderer this produces a canvas wider than 4K — meant for full-screen review on a large monitor, not for fitting in a sidebar.

**Reading the diagram.** Each box is *file path · function/class · one-line behaviour*. Solid arrows (`==>`) are the main data flow; dotted (`-.->`) are optional or read-only. Sub-graphs collect related steps; the order within a sub-graph is execution order.

---

## Module tree

Every Python module in `src/concept_mapper/` with a one-liner. Use `Cmd-F` to jump.

```
src/concept_mapper/
├── cli.py                              # Click entrypoint; one @cli.command per user-facing verb
├── validation.py                       # validate_corpus / validate_term_list / validate_concept_graph
│
├── corpus/
│   ├── loader.py                       # load_file (txt or PDF), load_directory, encoding fallback
│   └── models.py                       # Document, ProcessedDocument, SentenceLocation, StructureNode, Corpus
│
├── preprocessing/
│   ├── pipeline.py                     # preprocess() — main entry; chains the steps below
│   ├── cleaning.py                     # clean_text — OCR/PDF artifact removal (--clean-ocr)
│   ├── tokenize.py                     # tokenize_words, tokenize_sentences (NLTK)
│   ├── tagging.py                      # tag_tokens, filter_by_pos (Penn Treebank tags)
│   ├── lemmatize.py                    # lemmatize, lemmatize_tagged, lemmatize_words (WordNet)
│   ├── segment.py                      # segment_paragraphs, get_paragraph_indices
│   ├── structure.py                    # DocumentStructureDetector — chapter/section detection, TOC-guided
│   └── noun_chunks.py                  # extract_noun_chunks — spaCy multi-word phrase extraction (--spacy)
│
├── analysis/
│   ├── rarity.py                       # PhilosophicalTermScorer (5-signal hybrid); definitional/neologism helpers
│   ├── reference.py                    # load_reference_corpus (Brown), disk cache
│   ├── tfidf.py                        # tf, idf, tfidf, corpus_tfidf_scores, document_tfidf_scores
│   ├── frequency.py                    # word_frequencies, corpus_frequencies, document_frequencies, pos_filtered_frequencies
│   ├── cooccurrence.py                 # cooccurs_in_sentence, pmi, log_likelihood_ratio, build_cooccurrence_matrix
│   ├── relations.py                    # SVOTriple, CopularRelation, PrepRelation, Relation; extract_* + get_relations
│   └── contextual_relations.py         # analyze_context — windowed SVO + co-occurrence (used by `cmapr analyze`)
│
├── terms/
│   ├── models.py                       # TermList, TermEntry (term, lemma, pos, definition, examples, metadata)
│   ├── manager.py                      # TermManager — JSON / CSV / TXT I/O, CRUD, merging
│   ├── suggester.py                    # suggest_terms_from_analysis (wraps PhilosophicalTermScorer + examples)
│   └── scoring.py                      # post-scoring filter chain: quote/slash-strip, multi-word chunks, proper names, stopwords, lemma+suffix+derivational merge, fragments, OCR artifacts, POS, vetting
│
├── search/
│   ├── find.py                         # SentenceMatch, find_sentences, find_sentences_any/all, count_term_occurrences
│   ├── context.py                      # ContextWindow, get_context, get_context_with_highlights
│   ├── dispersion.py                   # dispersion, dispersion_plot_data, get_concentrated_regions
│   ├── extract.py                      # extract_terms_from_sentence_set; POS_TAG_GROUPS, STOPWORDS shared constants
│   └── concordance.py                  # build_concordance — per-node sentence list (location + highlight marks) for the HTML viz sidebar
│
├── syntax/
│   └── diagram.py                      # diagram_sentence — Stanza dependency parse → ASCII / GraphML
│
├── graph/
│   ├── model.py                        # ConceptGraph — wraps nx.DiGraph or nx.Graph; uniform node/edge attr API
│   ├── proposition_extractor.py        # Proposition; PropositionExtractor (regex pattern extractors); _score_sentence
│   ├── node_filter.py                  # NodeFilter — POS / length / abbreviation / stopword / fragment / freq rules
│   ├── builders.py                     # build_proposition_graph (main); graph_from_{cooccurrence, relations, contextual_relations, terms}
│   ├── operations.py                   # find/connect_isolated_nodes, consolidate_duplicate_labels, get_subgraph, filter_by_relation_type
│   ├── pruning.py                      # prune_edges, prune_nodes, prune_to_ratio
│   ├── aggregation.py                  # merge_graphs (last-write-wins), aggregate_graphs (attribute-aware, drives cmapr merge)
│   ├── cluster.py                      # cluster_by_structure (drives cmapr cluster)
│   └── metrics.py                      # centrality, detect_communities, assign_communities, get_connected_components, graph_density, get_shortest_path
│
├── export/
│   ├── d3.py                           # to_d3_dict, export_d3_json, load_d3_json
│   ├── html.py                         # generate_html — standalone interactive D3 page (force sim, legend, panel, expand/collapse, cluster force)
│   └── formats.py                      # export_graphml, export_csv, export_gexf, export_dot
│
├── transformations/
│   ├── inflection.py                   # English conjugation/declension (inflect + pattern3)
│   ├── replacement.py                  # replace_term — drives `cmapr replace`
│   ├── phrase_matcher.py               # multi-token boundary matching for replacement
│   └── text_reconstruction.py          # rebuilds text from token-level edits preserving whitespace
│
├── storage/
│   ├── backend.py                      # StorageBackend ABC
│   ├── json_backend.py                 # JSONBackend (default)
│   └── utils.py                        # derive_identifier (filename → work id), infer_output_path
│
└── server/
    ├── app.py                          # FastAPI routes — wraps the four pipeline stages as web flow
    └── templates/                      # Jinja2 templates: base, index, review, options, result
```

---

## Stage 1 — `cmapr ingest`

**Entry:** `cli.py:ingest` → `preprocessing.pipeline.preprocess(document, clean_ocr, toc_file, use_spacy)`.

**Pipeline (in order):**

1. **Load file** — `corpus.loader.load_file(path)` returns `Document`. Auto-detects `.pdf` (via `pdfplumber`), else reads `.txt` with UTF-8 → Latin-1 fallback.
2. **Clean** (optional, `--clean-ocr`) — `preprocessing.cleaning.clean_text(text)`. Strips OCR artifacts (mid-word splits, page numbers, hyphenated line breaks, weird unicode).
3. **Tokenize sentences** — `tokenize_sentences(text)` → NLTK `sent_tokenize`. List of sentence strings.
4. **Tokenize words** — `tokenize_words(text)` → NLTK `word_tokenize`. Flat token list across the whole doc.
5. **POS tag** — `tag_tokens(tokens)` → NLTK `pos_tag` (Penn Treebank). List of `(word, tag)` tuples.
6. **Lemmatize** — `lemmatize_tagged(pos_tags)` → WordNet lemmatizer, Penn→WordNet POS mapping in `get_wordnet_pos`; inflect-based fallback for unknown forms.
7. **Detect structure** (default on, disable via API) — `preprocessing.structure.DocumentStructureDetector(...).detect(text, sentences, toc_file)`. Returns `(structure_nodes, sentence_locations)`. Hierarchical chapter/section/subsection detection; TOC-guided when `--toc` supplied.
8. **Segment paragraphs** — `preprocessing.segment.get_paragraph_indices(text, sentences)`.
9. **Noun chunks** (optional, `--spacy`) — lazy-loads `en_core_web_sm`, extracts multi-word phrases (strip leading determiners, len ≥ 2), stores in `metadata["noun_chunks"]`.

**Output:** `ProcessedDocument` JSON via `to_dict()`. Schema: `raw_text`, `sentences`, `tokens`, `pos_tags`, `lemmas`, `metadata`, `structure_nodes`, `sentence_locations`, `paragraph_indices`. Written to `data/output/corpus/<work>/corpus.json`.

**Where it's serialized vs round-tripped:** the canonical loader is `ProcessedDocument.from_dict()` (deserializes nested `StructureNode` / `SentenceLocation`); some CLI commands splat the raw dict via `ProcessedDocument(**doc_data)` which leaves nested fields as dicts. Functions reading `sentence_locations` must accept both shapes (see `cluster_by_structure`).

---

## Stage 2 — `cmapr rarities`

**Entry:** `cli.py:rarities` (line ~275).

**Pipeline (in order):**

1. **Resolve output path + vetting file** — `output_dir/rarities/<work>/terms.json`, vetting at `vetting.json` alongside.
2. **Load corpus** — `[ProcessedDocument(**d) for d in json.load(f)]` (raw splat, OK because rarities doesn't touch `sentence_locations` directly).
3. **Load reference corpus** — `analysis.reference.load_reference_corpus()` returns Brown frequencies (disk-cached after first run).
4. **Score all terms** — `PhilosophicalTermScorer(docs, reference, use_lemmas=True).score_all(min_score=threshold)`. Five-signal hybrid:
   - **Ratio**: corpus freq / reference freq (weight 1.0) — `compare_to_reference`
   - **TF-IDF**: in-corpus vs reference (weight 1.0) — `tfidf_vs_reference`
   - **Neologism**: absent from WordNet (weight 0.5) — `get_wordnet_neologisms`
   - **Definitional context**: appears in "X is Y", "by X I mean", etc. (weight 0.3) — `get_definitional_contexts`
   - **Capitalization**: mid-sentence capitals = reified abstraction (weight 0.2)
   Weights/signals configurable per-scorer. Returns `[(term, total_score, components_dict), ...]`.
5. **Strip stray quotes/slashes** — `strip_stray_quotes` handles `'sign'` → `sign` and `/man/` → `man` edge artifacts; internal punctuation (`co-text`) preserved.
6. **Multi-word noun chunks** — if any `metadata["noun_chunks"]` present (spaCy run upstream), TF-IDF score them across docs and merge into the ranked list.
7. **Proper-name filter** (default on, `--no-filter-names`) — `proper_noun_ratios` × Brown frequency check; rejects rare-in-Brown frequently-capitalized terms.
8. **Stopword filter** — `filter_stopwords` drops function words (`since`, `whether`, `could`, …) using the shared `STOPWORDS` set from `search/extract.py`. Phrases bypass. Runs *before* the top-N trim so junk doesn't consume slots.
9. **Lemma + derivational merge** (default on, `--no-lemmatize`) — Pass 1/2 collapse `semiotics` → `semiotic`, `co-textual` → `co-text`; Pass 3 (`merge_derivational_variants`) collapses WordNet `derivationally_related_forms` pairs like `taxonomic` ↔ `taxonomy`, keeping the noun form. Keeps highest cluster score. Applies `[:top_n]` cut here.
10. **Fragment filter** (default on, `--no-filter-fragments`) — drops terms < 4 chars and prefix fragments of WordNet words.
11. **OCR-artifact filter** (default on, same `--no-filter-fragments` gate) — `filter_ocr_artifacts` drops non-WordNet non-stopword terms that are function-word merges (`thesecase` → `these`+`case`) or leading-char drops (`ictionary` → `dictionary`). Genuine neologisms (absent from WordNet) are untouched.
12. **POS filter** (`--pos noun,verb,adj,adv`) — uses `filter_by_pos_tags`. Multi-word phrases always pass.
13. **Vetting** — read existing `vetting.json` (`accept`/`reject` lists); rejected terms removed, accepted terms re-included even past top-n. If `--vet`, interactive y/n prompt per unvetted term; only `y`/`n` accepted (anything else loops).
14. **By-section grouping** (`--by-section`) — assigns each term to the section it appears in most often (resolved from `sentence_locations`), writes `<work>_by_section.json` alongside.
15. **Export** — `TermManager(term_list).export_to_json(output_path)`.

**Output:** `TermList` JSON — list of `{"term": str, "metadata": {"score": float}}`. Optional companion `vetting.json` and `_by_section.json`.

---

## Stage 3 — `cmapr graph`

**Entry:** `cli.py:graph` (line ~1085) → `graph.builders.build_proposition_graph(docs, seed_terms, node_filter, pmi_threshold, term_scores)`.

**Pipeline (in order):**

1. **Load corpus + term list.**
2. **Build `NodeFilter`** — `NodeFilter(corpus_vocab=set(all_tokens), term_freqs=Counter(all_lemmas))`. Filters seed terms before extraction; later re-applied to extracted (non-seed) endpoint terms inside the builder.
3. **`PropositionExtractor(docs)`** — caches sentences and lazy POS-tags per sentence.
4. **Per-pair extraction** — for every unordered pair `(term_a, term_b)` in the filtered seed list:
   - Scan sentences containing both (substring match — phrase-safe).
   - For each candidate sentence, run extractor chain in **priority order** (returns first match):
     1. `_try_definition` — *by X I mean Y*, *X is defined as Y*, *X denotes Y*
     2. `_try_kind_of` — *X is a type/kind/species/sort of Y*
     3. `_try_production` — *X produces/generates/gives rise to Y*
     4. `_try_dependence` — *X presupposes/depends on/requires Y*
     5. `_try_opposition` — *X vs Y*, *X as opposed to Y*, *X is the opposite of Y* (symmetric, undirected)
     6. `_try_property` — plain copular *X is Y* fallback (after definition/kind-of)
     7. `_try_relation` — catch-all SVO with a broad verb list (`_RELATION_VERBS`)
     8. `_try_pos_verb` — POS-based fallback; finds first non-copular non-light verb between the two terms (skips phrase terms because of token-level lookup)
   - Plus `extract_composition` — *A, B, C form/constitute/compose X* — runs over the full seed list, produces `component` edges among constituents + `production` to the composed entity.
5. **Merge same-type duplicates** — `(source, target, relation_type)` aggregated; weight = occurrence count, evidence = top-3 by `_score_sentence` (definition marker > proximity > brevity > position).
6. **Cooccurrence fallback** — for pairs with no typed extraction and PMI ≥ `pmi_threshold` (default 1.0), add `cooccurrence` edge weighted by joint sentence count.
7. **NodeFilter on extracted endpoints** — same criteria as seed filtering, applied to any node that wasn't a seed.
8. **Multigraph collapse** — `ConceptGraph` is a single-edge-per-pair `DiGraph`; when multiple types fire on one pair, the highest-priority type wins (priority ladder in `builders.py:_TYPE_PRIORITY`). To preserve multi-type info, see `cmapr merge` / `aggregate_graphs`.
9. **Score-aware node attrs** — `term_scores` dict copied onto node `score` for HTML sizing.
10. **Prune** — `prune_to_ratio(graph, target_ratio=3.0)` — drops cooccurrence edges first (never the sole edge for a node), then lowest-weight grammatical edges. Never isolates nodes.
11. **Focus / depth** (`--focus TERM`, `--depth N`) — `nx.ego_graph(g, centre, radius=N, undirected=True)`. Centre = focus term or highest-scoring seed.
12. **Validate** — `validation.validate_concept_graph`.
13. **Export** — `export_d3_json(graph, path, include_evidence=True)`.

**Output:** D3 JSON. Nodes: `{id, label, size, group, frequency?, pos?, score?, chapter?, section?, term?}`. Links: `{source, target, weight, type, label, verb, evidence?, relation_types?, weight_by_type?, evidence_by_type?, verb_by_type?}`.

**Edge type vocabulary:**

| Type | Source | Directed | Default color |
|---|---|---|---|
| `definition` | `_try_definition` | yes | `#4e79a7` blue |
| `kind-of` | `_try_kind_of` | yes | `#59a14f` green |
| `production` | `_try_production` | yes | `#f28e2b` orange |
| `dependence` | `_try_dependence` | yes | `#e15759` red |
| `opposition` | `_try_opposition` | no | `#d4a0a0` rose |
| `property` | `_try_property` | yes | `#edc948` yellow |
| `relation` | `_try_relation`, `_try_pos_verb` | yes | `#76b7b2` teal |
| `component` | `extract_composition` (constituent pairs) | no | `#b07aa1` purple |
| `recurrence` | `cluster_by_structure` | yes | `#7a8aa0` slate (dashed) |
| `cooccurrence` | PMI fallback in `build_proposition_graph` | no | `#bbbbbb` grey (dashed) |

---

## Stage 4 — `cmapr export`

**Entry:** `cli.py:export` (line ~1282).

**Pipeline:**

1. **Load D3 JSON** — `export.load_d3_json(path)`.
2. **Reconstruct `ConceptGraph`** — flat `add_node` / `add_edge` from nodes/links.
3. **Convert / write** based on `--format`:
   - `d3` → `export_d3_json` (round-trip).
   - `html` → `generate_html(graph, output_dir, title, docs=None)` — standalone interactive page. Internal flow:
     - `to_d3_dict` runs first: validates, consolidates duplicate labels (with chapter-aware dedup key for cluster graphs), strips isolated nodes (errors logged), computes communities (Louvain via NetworkX), sizes nodes, builds the JSON inlined into the page.
     - When `docs` is supplied (CLI `--corpus`, or in-process from `cmapr run`/`serve`), `search/concordance.build_concordance` precomputes, per node term, every sentence its lemma appears in (document order, structural location, highlight surface-forms) and inlines it as a `CONCORDANCE` const. Omitted → inlines `{}`.
     - Template inlines D3 v7 from CDN, the JSON, and JS for the simulation.
     - Forces: `link` (distance ∝ 1/√weight), `charge` (-400, or -600 for high-degree hubs), `center`, `collide` (label-length aware), optional `cluster` (auto-engages when ≥ 2 distinct `chapter` values are present, pulls nodes toward centroids on a circle around the canvas centre).
     - Edge styling per type from `EDGE_COLORS`; `DIRECTED_TYPES` get arrowheads; `cooccurrence` and `recurrence` rendered dashed.
     - Legend (`LEGEND_TYPES`): checkbox per type that toggles `hiddenTypes` set; `applyVisibility` filters nodes and edges.
     - Tooltip on edge hover: term pair + verb + weight; multi-type edges (from `cmapr merge`) show `also: type2 (×W), ...` plus per-type evidence sections.
     - Click a node → two right sidebars open together: the detail panel (far right, 280px) with frequency, score, and connected edges/types; and, just to its left, the **concordance panel** (380px) listing every sentence the node's lemmatized term appears in — document order, independently scrollable, each with a chapter › section breadcrumb and the term `<mark>`-highlighted (`showConcordance`, fed by the inlined `CONCORDANCE`). Empty-canvas click closes both. The concordance panel is silent when no corpus was supplied.
     - Double-click a node → expand/collapse (show only that node + its direct neighbours; respect type filters).
     - Drag pins a node (`fx`/`fy`); simulation keeps running around it.
   - `graphml` → `export_graphml` (Gephi / yEd / Cytoscape).
   - `csv` → `export_csv` (writes `nodes.csv` + `edges.csv`).
   - `gexf` → `export_gexf` (Gephi).

**Output:** files under `data/output/exports/<work>/`.

---

## Workflow commands (chain or wrap the four stages)

- **`cmapr run` (`cli.py:run`, ~2635)** — chains ingest → rarities → graph → export non-interactively. Single text file in, visualization out. Same code paths as the four commands; not a separate pipeline.
- **`cmapr merge` (`cli.py:merge`, ~2857)** — combines multiple graph JSON files via `aggregate_graphs()` in `graph/operations.py`. Frequencies sum; scores frequency-weighted mean; edges with same-pair-different-type collapse to a single edge that carries additive multi-type fields (`relation_types`, `weight_by_type`, `evidence_by_type`, `verb_by_type`). Used for collapsing per-chapter graphs into one unified view.
- **`cmapr cluster` (`cli.py:cluster`, ~2932)** — builds one sub-graph per chapter (or section) from a single corpus via `cluster_by_structure()`. Nodes namespaced as `<term>__<chapter>`; `recurrence` edges chain consecutive same-term occurrences (weight = span). Used for *preserving* per-chapter structure. Complement of `merge`.
- **`cmapr serve` (`cli.py:serve`, ~2940)** — FastAPI + Jinja2 web UI; routes in `server/app.py`. Same code paths as the four commands; just a different user surface. Requires `pip install 'concept-mapper[serve]'`.

---

## Auxiliary commands (read existing artifacts)

These don't feed back into the pipeline; they're alternative lenses on the same data.

| Command | What it does | Where |
|---|---|---|
| `cmapr search` | Find sentences containing a term; KWIC / context window / dispersion options | `cli.py:search` (~798) → `search/find.py`, `context.py`, `dispersion.py`, `extract.py` |
| `cmapr analyze` | Windowed term-neighbourhood analysis (SVO + co-occurrence + significant terms) | `cli.py:analyze` (~2185) → `analysis/contextual_relations.py` |
| `cmapr replace` | Inflection-preserving synonym replacement throughout a corpus | `cli.py:replace` (~2494) → `transformations/` |
| `cmapr diagram` | Render Stanza dependency parse tree for a single sentence | `cli.py:diagram` (~1406) → `syntax/diagram.py` |

---

## Cross-cutting layer

- **`validation.py`** — `validate_corpus`, `validate_term_list`, `validate_concept_graph`. Called by `ingest`, `rarities`, `graph`, `export`. Fail-fast schema checks.
- **`storage/`** — `StorageBackend` ABC, `JSONBackend` default. `utils.py`: `derive_identifier(path)` (filename → work id) and `infer_output_path(input, output_dir, subdir)` produce the `data/output/<subdir>/<work>/` conventions.
- **`cli.py`** — Click entrypoint; each `@cli.command` is a thin shell that loads inputs, calls into `*/`, writes outputs. Logic does **not** live here; if you're tempted to add it, extract a function to the matching `logic` module. See `.claude/rules.md` § Architecture for the data/logic/presentation layering rule.

---

## Where to extend — recipes

Concrete "if you want to do X, edit Y" map. Each row points at the files that need touching plus the relevant tests.

| Goal | Edit | Tests |
|---|---|---|
| **Add a new edge type** | `graph/proposition_extractor.py`: add `_try_<type>` to the extractor chain; update `_TYPE_PRIORITY` in `graph/builders.py`. `export/html.py`: append to `EDGE_COLORS`, `LEGEND_TYPES`, `DIRECTED_TYPES`. | `tests/test_proposition_extractor.py`; HTML viz checks added to `docs/qa/graph.md`. |
| **Tweak the rarity scorer** | `analysis/rarity.py:PhilosophicalTermScorer` — adjust signal weights, add a new signal as a method, register in `score_term`. | `tests/test_rarity.py`; verify top-N stability on `eco_spl1`. |
| **Add a new node-inclusion rule** | `graph/node_filter.py:NodeFilter.is_valid` — single-token branch (or phrase branch if it should apply to multi-word terms). Update docstring criteria list. | `tests/test_node_filter.py` — add a `Test<RuleName>Criterion` class. |
| **Add a new export format** | `export/formats.py` — new `export_<fmt>(graph, path)`. Register in `cli.py:export` (Choice + branch). | `tests/test_export.py`. |
| **Add a new CLI verb** | `cli.py` — new `@cli.command()` block. Place it next to its logical neighbour; update the module docstring command index at the top. | `tests/test_cli.py` — new `TestXCommand` class. Plus `docs/plans/<feature>.md`. |
| **Tune the proposition extractor's heuristics** | `graph/proposition_extractor.py:_score_sentence` (evidence ranking), or per-extractor regex constants (`_DEFINITION_PATTERNS`, `_KIND_MARKERS`, `_PRODUCTION_VERBS`, `_DEPENDENCE_PHRASES`, `_OPPOSITION_PATTERNS`, `_RELATION_VERBS`, `_COMPOSITION_VERBS`). | `tests/test_proposition_extractor.py` — `TestEvidenceScoring`, per-type test classes. |
| **Change graph pruning heuristics** | `graph/operations.py:prune_to_ratio`. The order is: cooccurrence first (never if sole edge), then lowest-weight grammatical edges. | `tests/test_graph.py:TestPruneToRatio`. |
| **Tweak the HTML viz layout / interactions** | `export/html.py` — the file is a single template string with inlined JS. Edit force params (link distance, charge strength), node size formula, edge dasharray, legend layout, detail-panel HTML, expand/collapse logic. | Browser-side; covered by `docs/qa/graph.md` and `docs/qa/cluster.md` visual checklists. |
| **Add a new pipeline stage to ingest** | `preprocessing/pipeline.py:preprocess()` — call the new module between existing stages. Add the module under `preprocessing/`. Persist new outputs via `ProcessedDocument` (add a field to `corpus/models.py:ProcessedDocument` and update `to_dict`/`from_dict`). | `tests/test_preprocessing.py`. |
| **Change rarities filter ordering** | `cli.py:rarities` (~275–705) — the `candidates = [...]` reductions happen in sequence. Reorder with care; vetting must run after top-n cut so accepted terms can be re-included. | `tests/test_cli.py:TestRaritiesPOSFilter`, `TestRaritiesBySection`. |
| **Add a new neural component (REBEL, sentence-transformers, etc.)** | Add as optional dep in `pyproject.toml` `[project.optional-dependencies]`. New module under `analysis/` (for extraction) or `graph/` (for graph-level ops). Wire as `--neural` flag on the relevant CLI command, lazy-imported. See `docs/survey.md` for the survey-level rationale. | Mirror the spaCy-extra pattern: gate tests with `@requires_spacy`-style skipif. |
| **Add a clustered-viz display option (constellation, timeline spine)** | `export/html.py` — extend the cluster-force block. Read a `clusterLayout` attr from JSON or a new CLI flag; pick centroid placement function accordingly. Centroids on circle (current), grid, or x-axis spine. | `docs/qa/cluster.md` visual checks. |

---

## Data structure reference

Quick lookup. Each entry points at the module that owns the type.

### `Document` (`corpus/models.py`)
Raw input: `text`, `metadata` (`title`, `author`, `date`, `source_path`).

### `ProcessedDocument` (`corpus/models.py`)
The corpus-stage artifact. Fields: `raw_text`, `sentences` (list[str]), `tokens` (list[str]), `pos_tags` (list[(str, str)]), `lemmas` (list[str]), `metadata` (dict — holds `noun_chunks` when spaCy used), `structure_nodes` (list[`StructureNode`]), `sentence_locations` (list[`SentenceLocation`]), `paragraph_indices` (list[int] of sentence→paragraph). `to_dict()` / `from_dict()` round-trip.

### `SentenceLocation` (`corpus/models.py`)
Flattened structural lookup per sentence: `sent_index` (int), `chapter` / `chapter_title` / `section` / `section_title` / `subsection` / `subsection_title` / `paragraph`. All optional. Built by `DocumentStructureDetector`.

### `StructureNode` (`corpus/models.py`)
Hierarchical structure tree: `title`, `level`, `sentence_range`, `children`.

### `TermList`, `TermEntry` (`terms/models.py`)
Curated terms. `TermEntry`: `term`, `lemma?`, `pos?`, `definition?`, `notes?`, `examples` (list), `metadata` (dict — holds `score`). `TermList`: collection with lookup; JSON I/O via `TermManager`.

### `ConceptGraph` (`graph/model.py`)
NetworkX wrapper. Backed by `nx.DiGraph` (directed) or `nx.Graph` (undirected). One edge per `(source, target)` pair (multi-type info goes on the edge attrs, not as parallel edges). Uniform API: `add_node`, `add_edge`, `has_node`, `has_edge`, `get_node`, `get_edge`, `remove_node`, `remove_edge`, `nodes()`, `edges()`, `neighbors`, `degree`, `node_count`, `edge_count`, `copy`. Direct access via `graph._graph` when needed.

### `Proposition` (`graph/proposition_extractor.py`)
Single extracted edge: `source`, `target`, `label` (verb / surface form), `type` (one of the edge type vocabulary), `evidence` (list[str], top-3 sentences), `directed` (bool), `weight` (int).

### `SVOTriple`, `CopularRelation`, `PrepRelation`, `Relation` (`analysis/relations.py`)
Grammatical relation primitives used by `cmapr analyze` and `graph_from_relations`. `Relation` is the aggregator (source, relation_type, target, evidence, metadata).

### `ContextualRelation` (`analysis/contextual_relations.py`)
Windowed analysis output (significance score + relation type + evidence). Consumed by `graph_from_contextual_relations`.

### `SentenceMatch`, `ContextWindow`, `KWICLine` (`search/`)
Search results. `find.py:SentenceMatch` (sentence + positions); `context.py:ContextWindow` (before/match/after); KWIC is produced by `concordance` helpers.

---

## Conventions and constraints

- **Layering rule** (`.claude/rules.md` § Architecture): data → logic → presentation. Logic never imports presentation; data never imports logic. The CLI is presentation; analysis/transformations/graph internals are logic; `corpus/models.py` and `storage/` are data.
- **Graph diagram constraints** (`.claude/rules.md` § Graph Diagram Constraints): no duplicate nodes (consolidated by `consolidate_duplicate_labels`), no unconnected nodes (errors logged, dropped from export), edge labels are derived text not numbers, all edges directed in the viz.
- **Output convention**: every command writes to `data/output/<subdir>/<work>/` where `<work>` is the identifier derived from the source filename via `derive_identifier`.
- **Test mirroring**: `tests/test_<module>.py` mirrors `src/concept_mapper/<module>.py`. CLI surface tested in `tests/test_cli.py`.
- **Doc discipline**: every feature gets a plan file under `docs/plans/`. Every task list uses `[ ]`/`[x]` checkboxes. Commits gate on flipping the relevant boxes. See `.claude/rules.md`.

---

## Related docs

- `docs/roadmap.md` — past / present / future + live Status block
- `docs/plans/graph.md` — graph-spec implementation plan (closed)
- `docs/plans/multi-chapter.md` — clustered viz implementation plan
- `docs/qa/graph.md` — manual QA checklist for ingest→rarities→graph→export
- `docs/qa/cluster.md` — manual QA checklist for `cmapr cluster`
- `docs/survey.md` — library landscape survey, organized by pipeline stage + next-initiative picks
- `docs/api-reference.md` — Python API reference (separate from this navigation map)
- `.claude/rules.md` — code conventions, layering, doc discipline
- `.claude/context.md` — pointer-only (this file is the authoritative architecture map)
