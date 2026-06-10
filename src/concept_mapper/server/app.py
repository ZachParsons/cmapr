"""Local web UI server for cmapr — FastAPI + Jinja2."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

app = FastAPI(title="cmapr")

_TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(_TEMPLATES_DIR))

# ---------------------------------------------------------------------------
# Path helpers — all relative to cwd (project root) so they match the CLI
# ---------------------------------------------------------------------------


def _corpus_path(work: str) -> Path:
    return Path("data/output/corpus") / work / "corpus.json"


def _rarities_path(work: str) -> Path:
    return Path("data/output/rarities") / work / "terms.json"


def _vetting_path(work: str) -> Path:
    return Path("data/output/rarities") / work / "vetting.json"


def _graph_path(work: str) -> Path:
    return Path("data/output/graphs") / work / "graph.json"


def _export_path(work: str) -> Path:
    return Path("data/output/exports") / work / "index.html"


def _params_path(work: str) -> Path:
    return Path("data/output/rarities") / work / "rarities_params.json"


# Cross-work learned filters, grown from review reasons and read by the rarities
# run (output-dir root, matching `cmapr --output-dir data/output`).
def _stopwords_supplement_path() -> Path:
    return Path("data/output/stopwords_extra.json")


def _aliases_path() -> Path:
    return Path("data/output/aliases.json")


def _grow_learned_filters(reasons: dict, accepted_terms: set) -> None:
    """Feed rejection reasons into the cross-work learned filters.

    "common & atopic" terms join the stopword supplement; "duplicate / lemma"
    terms become aliases onto a canonical inferred from the accepted (kept)
    terms. Both files are unions/updates, so corrections accumulate over time.
    """
    atopic = sorted(t for t, code in reasons.items() if code == "atopic")
    duplicates = [t for t, code in reasons.items() if code == "duplicate"]

    if atopic:
        sp = _stopwords_supplement_path()
        sp.parent.mkdir(parents=True, exist_ok=True)
        existing: set = set()
        if sp.exists():
            try:
                existing = set(json.loads(sp.read_text()).get("stopwords", []))
            except (ValueError, OSError):
                existing = set()
        existing.update(atopic)
        sp.write_text(
            json.dumps({"stopwords": sorted(existing)}, indent=2, ensure_ascii=False)
        )

    if duplicates:
        from concept_mapper.terms.scoring import infer_canonical

        ap = _aliases_path()
        ap.parent.mkdir(parents=True, exist_ok=True)
        amap: dict = {}
        if ap.exists():
            try:
                amap = dict(json.loads(ap.read_text()).get("aliases", {}))
            except (ValueError, OSError):
                amap = {}
        pool = sorted(accepted_terms)
        for dup in duplicates:
            canonical = infer_canonical(dup, pool)
            if canonical and canonical.lower() != dup:
                amap[dup] = canonical.lower()
        if amap:
            ap.write_text(
                json.dumps(
                    {"aliases": dict(sorted(amap.items()))},
                    indent=2,
                    ensure_ascii=False,
                )
            )


# Defaults mirror the `cmapr rarities` CLI defaults.
_DEFAULT_PARAMS: dict = {
    "top_n": 100,
    "threshold": 0.5,
    "pos": "",  # comma-separated subset of noun,verb,adj,adv ("" = all)
    "keep_names": False,  # --no-filter-names
    "keep_fragments": False,  # --no-filter-fragments
    "no_lemmatize": False,  # --no-lemmatize
    "min_author_freq": 3,
    "weight_ratio": 1.0,
    "weight_tfidf": 1.0,
    "weight_neologism": 0.5,
    "weight_definitional": 0.3,
    "weight_capitalized": 0.2,
}


def _load_params(work: str) -> dict:
    """Last-used rarities params for this work, falling back to defaults."""
    p = _params_path(work)
    if p.exists():
        try:
            return {**_DEFAULT_PARAMS, **json.loads(p.read_text())}
        except (ValueError, OSError):
            pass
    return dict(_DEFAULT_PARAMS)


def _run_rarities(work: str, params: dict) -> subprocess.CompletedProcess:
    """Run `cmapr rarities` with the given params; persist them on success.

    `top_n` only *caps* the candidate pool — to surface MORE terms, lower
    `threshold` (the minimum score). Both are exposed on the review page.
    """
    terms_p = _rarities_path(work)
    terms_p.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cmapr",
        "rarities",
        str(_corpus_path(work)),
        "--top-n",
        str(params["top_n"]),
        "--threshold",
        str(params["threshold"]),
        "--output",
        str(terms_p),
    ]
    if params.get("pos"):
        cmd += ["--pos", params["pos"]]
    if params.get("keep_names"):
        cmd.append("--no-filter-names")
    if params.get("keep_fragments"):
        cmd.append("--no-filter-fragments")
    if params.get("no_lemmatize"):
        cmd.append("--no-lemmatize")
    cmd += ["--min-freq", str(params.get("min_author_freq", 3))]
    cmd += ["--weight-ratio", str(params.get("weight_ratio", 1.0))]
    cmd += ["--weight-tfidf", str(params.get("weight_tfidf", 1.0))]
    cmd += ["--weight-neologism", str(params.get("weight_neologism", 0.5))]
    cmd += ["--weight-definitional", str(params.get("weight_definitional", 0.3))]
    cmd += ["--weight-capitalized", str(params.get("weight_capitalized", 0.2))]
    result = _run(cmd)
    if result.returncode == 0:
        _params_path(work).write_text(json.dumps(params, indent=2), encoding="utf-8")
    return result


def _list_works() -> list[str]:
    corpus_dir = Path("data/output/corpus")
    if not corpus_dir.exists():
        return []
    return sorted(
        d.name
        for d in corpus_dir.iterdir()
        if d.is_dir() and (d / "corpus.json").exists()
    )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=Path.cwd())


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
def index(request: Request, error: str = ""):
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "works": _list_works(),
            "error": error,
        },
    )


@app.post("/start")
def start(
    file_path: Annotated[str, Form()],
    clean_ocr: Annotated[bool, Form()] = False,
    use_spacy: Annotated[bool, Form()] = False,
    trim_matter: Annotated[bool, Form()] = False,
    toc_path: Annotated[str, Form()] = "",
    force: Annotated[bool, Form()] = False,
):
    work = Path(file_path).stem

    # `force` comes from the review page's "re-run ingest" form — without
    # it, changed ingest options were silently ignored for existing corpora.
    if force or not _corpus_path(work).exists():
        cmd = ["cmapr", "ingest", file_path]
        if clean_ocr:
            cmd.append("--clean-ocr")
        if use_spacy:
            cmd.append("--spacy")
        if trim_matter:
            cmd.append("--trim-matter")
        if toc_path.strip():
            cmd += ["--toc", toc_path.strip()]
        result = _run(cmd)
        if result.returncode != 0:
            err = (result.stderr or result.stdout)[:300].replace('"', "'")
            return RedirectResponse(f"/?error={err}", status_code=303)

    # Run rarities with the default params (generous top-n) so the review
    # page shows a wide candidate pool. The build step does NOT re-run
    # rarities; it filters this pool by the user's vetting + graph top-n. The
    # user can adjust the params (top-n, threshold, POS, filters) from the
    # review page via POST /rarities.
    result = _run_rarities(work, dict(_DEFAULT_PARAMS))
    if result.returncode != 0:
        err = (result.stderr or result.stdout)[:300].replace('"', "'")
        return RedirectResponse(f"/?error={err}", status_code=303)

    return RedirectResponse(f"/review?work={work}", status_code=303)


@app.get("/review", response_class=HTMLResponse)
def review(request: Request, work: str, error: str = ""):
    terms_p = _rarities_path(work)
    if not terms_p.exists():
        return RedirectResponse("/", status_code=303)

    terms = json.loads(terms_p.read_text())
    vet: dict = {}
    vp = _vetting_path(work)
    if vp.exists():
        vet = json.loads(vp.read_text())
    rejected = {t.lower() for t in vet.get("reject", [])}
    # Optional per-term rejection reason (parallel to reject — see POST /vet).
    reasons = {k.lower(): v for k, v in (vet.get("reasons") or {}).items()}

    rows = []
    for t in terms:
        name = t["term"]
        meta = t.get("metadata", {})
        rows.append(
            {
                "term": name,
                "score": round(meta.get("score", 0), 2),
                "freq": meta.get("corpus_count") or meta.get("count", 0),
                "checked": name.lower() not in rejected,
                "reason": reasons.get(name.lower(), ""),
            }
        )

    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "work": work,
            "terms": rows,
            "pool_size": len(rows),
            "params": _load_params(work),
            "error": error,
            "has_vetting": _vetting_path(work).exists(),
            "has_export": _export_path(work).exists(),
        },
    )


@app.post("/rarities")
async def rerun_rarities(request: Request):
    """Re-run only the rarities step with user-adjusted detection params.

    Exposes top-n, minimum score threshold, POS restriction, and the filter
    toggles. `top_n` caps the pool; lower `threshold` to surface more terms.
    """
    form = await request.form()
    work = form.get("work", "")

    try:
        pool_size = int(form.get("pool_size") or 100)
    except ValueError:
        pool_size = 100
    try:
        threshold = float(form.get("threshold") or 0.5)
    except ValueError:
        threshold = 0.5
    try:
        min_author_freq = max(1, int(form.get("min_author_freq") or 3))
    except ValueError:
        min_author_freq = 3

    def _float_param(
        key: str, default: float, lo: float = 0.0, hi: float = 10.0
    ) -> float:
        try:
            return max(lo, min(float(form.get(key) or default), hi))
        except ValueError:
            return default

    params = {
        "top_n": max(10, min(pool_size, 2000)),
        "threshold": max(0.0, min(threshold, 100.0)),
        # POS arrives as 0+ checkbox values; join to the CLI's comma form.
        "pos": ",".join(form.getlist("pos")),
        "keep_names": bool(form.get("keep_names")),
        "keep_fragments": bool(form.get("keep_fragments")),
        "no_lemmatize": bool(form.get("no_lemmatize")),
        "min_author_freq": min_author_freq,
        "weight_ratio": _float_param("weight_ratio", 1.0),
        "weight_tfidf": _float_param("weight_tfidf", 1.0),
        "weight_neologism": _float_param("weight_neologism", 0.5),
        "weight_definitional": _float_param("weight_definitional", 0.3),
        "weight_capitalized": _float_param("weight_capitalized", 0.2),
    }

    result = _run_rarities(work, params)
    if result.returncode != 0:
        err = (result.stderr or result.stdout)[:300].replace('"', "'")
        return RedirectResponse(f"/review?work={work}&error={err}", status_code=303)
    return RedirectResponse(f"/review?work={work}", status_code=303)


@app.post("/vet")
async def vet(request: Request):
    form = await request.form()
    work = form.get("work", "")
    accepted_terms = set(t.lower() for t in form.getlist("terms"))

    # Per-term rejection reason from the single "Reason" column. Field name is
    # "reason::<term>"; only non-empty selections are kept. Stored in a parallel
    # `reasons` map so `reject` stays a plain string list for the CLI loaders.
    reasons: dict = {}

    terms_p = _rarities_path(work)
    if terms_p.exists():
        all_terms = json.loads(terms_p.read_text())
        all_names = [t["term"].lower() for t in all_terms]
        rejected = sorted(n for n in all_names if n not in accepted_terms)
        for t in all_terms:
            name = t["term"]
            r = (form.get(f"reason::{name}") or "").strip()
            if r:
                reasons[name.lower()] = r
    else:
        rejected = []

    vp = _vetting_path(work)
    vp.parent.mkdir(parents=True, exist_ok=True)
    vp.write_text(
        json.dumps(
            {
                "accept": sorted(accepted_terms),
                "reject": rejected,
                "reasons": dict(sorted(reasons.items())),
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    # Feed atopic/duplicate reasons into the cross-work learned filters.
    _grow_learned_filters(reasons, accepted_terms)

    return RedirectResponse(f"/options?work={work}", status_code=303)


@app.get("/options", response_class=HTMLResponse)
def options(request: Request, work: str, error: str = ""):
    return templates.TemplateResponse(
        request,
        "options.html",
        {
            "work": work,
            "error": error,
            "has_export": _export_path(work).exists(),
        },
    )


def _build_seed_terms_file(work: str, top_n: int) -> Path | None:
    """
    Filter the candidate pool (terms.json) by the user's vetting and the
    graph's top-n cap; write the result to seed_terms.json and return its
    path. Does not re-run rarities — the candidate pool is preserved so
    the review page keeps showing the full set.
    """
    terms_p = _rarities_path(work)
    if not terms_p.exists():
        return None
    candidates = json.loads(terms_p.read_text())
    rejected: set[str] = set()
    vp = _vetting_path(work)
    if vp.exists():
        rejected = {t.lower() for t in json.loads(vp.read_text()).get("reject", [])}
    accepted = [c for c in candidates if c["term"].lower() not in rejected]
    accepted.sort(key=lambda c: c.get("metadata", {}).get("score", 0.0), reverse=True)
    seed = accepted[:top_n]
    seed_p = terms_p.parent / "seed_terms.json"
    seed_p.write_text(json.dumps(seed, indent=2, ensure_ascii=False))
    return seed_p


@app.post("/build")
def build(
    work: Annotated[str, Form()],
    top_n: Annotated[int, Form()] = 50,
    threshold: Annotated[float, Form()] = 0.1,
    depth: Annotated[str, Form()] = "",
    focus: Annotated[str, Form()] = "",
    definitions: Annotated[bool, Form()] = False,
    prune_ratio: Annotated[float, Form()] = 3.0,
):
    # Filter candidate pool → seed terms; the pool itself is preserved.
    seed_p = _build_seed_terms_file(work, top_n)
    if seed_p is None:
        return RedirectResponse(
            f"/options?work={work}&error=no+candidate+pool+found", status_code=303
        )

    # Build graph
    graph_p = _graph_path(work)
    graph_p.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cmapr",
        "graph",
        str(_corpus_path(work)),
        "-t",
        str(seed_p),
        "--threshold",
        str(threshold),
        "--output",
        str(graph_p),
    ]
    if depth.strip():
        cmd += ["--depth", depth.strip()]
    if focus.strip():
        cmd += ["--focus", focus.strip()]
    if definitions:
        cmd.append("--definitions")
    cmd += ["--prune-ratio", str(max(0.5, prune_ratio))]
    result = _run(cmd)
    if result.returncode != 0:
        return RedirectResponse(
            f"/options?work={work}&error=graph+failed", status_code=303
        )

    # Export to HTML
    export_p = _export_path(work)
    export_p.parent.mkdir(parents=True, exist_ok=True)
    export_cmd = [
        "cmapr",
        "export",
        str(graph_p),
        "--format",
        "html",
        "--output",
        str(export_p),
        # Enables the node-click concordance sidebar in the embedded graph.
        "--corpus",
        str(_corpus_path(work)),
    ]
    # Extend the concordance to pipeline-merged variants when available.
    variants_p = _rarities_path(work).parent / "variants.json"
    if variants_p.exists():
        export_cmd += ["--variants", str(variants_p)]
    result = _run(export_cmd)
    if result.returncode != 0:
        return RedirectResponse(
            f"/options?work={work}&error=export+failed", status_code=303
        )

    return RedirectResponse(f"/result?work={work}", status_code=303)


@app.get("/result", response_class=HTMLResponse)
def result(request: Request, work: str):
    has_export = _export_path(work).exists()
    return templates.TemplateResponse(
        request,
        "result.html",
        {
            "work": work,
            "has_export": has_export,
        },
    )


@app.get("/exports/{work}/index.html", response_class=HTMLResponse)
def serve_export(work: str):
    ep = _export_path(work)
    if not ep.exists():
        return HTMLResponse(
            "<p>Export not found. Build the graph first.</p>", status_code=404
        )
    return HTMLResponse(ep.read_text())
