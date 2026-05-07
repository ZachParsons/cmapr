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
    toc_path: Annotated[str, Form()] = "",
):
    work = Path(file_path).stem

    if not _corpus_path(work).exists():
        cmd = ["cmapr", "ingest", file_path]
        if clean_ocr:
            cmd.append("--clean-ocr")
        if use_spacy:
            cmd.append("--spacy")
        if toc_path.strip():
            cmd += ["--toc", toc_path.strip()]
        result = _run(cmd)
        if result.returncode != 0:
            err = (result.stderr or result.stdout)[:300].replace('"', "'")
            return RedirectResponse(f"/?error={err}", status_code=303)

    # Run rarities with a generous top-n so the review page shows all candidates
    terms_p = _rarities_path(work)
    terms_p.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            "cmapr",
            "rarities",
            str(_corpus_path(work)),
            "--top-n",
            "100",
            "--output",
            str(terms_p),
        ]
    )
    if result.returncode != 0:
        err = (result.stderr or result.stdout)[:300].replace('"', "'")
        return RedirectResponse(f"/?error={err}", status_code=303)

    return RedirectResponse(f"/review?work={work}", status_code=303)


@app.get("/review", response_class=HTMLResponse)
def review(request: Request, work: str):
    terms_p = _rarities_path(work)
    if not terms_p.exists():
        return RedirectResponse("/", status_code=303)

    terms = json.loads(terms_p.read_text())
    vet: dict = {}
    vp = _vetting_path(work)
    if vp.exists():
        vet = json.loads(vp.read_text())
    rejected = {t.lower() for t in vet.get("reject", [])}

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
            }
        )

    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "work": work,
            "terms": rows,
        },
    )


@app.post("/vet")
async def vet(request: Request):
    form = await request.form()
    work = form.get("work", "")
    accepted_terms = set(t.lower() for t in form.getlist("terms"))

    terms_p = _rarities_path(work)
    if terms_p.exists():
        all_terms = json.loads(terms_p.read_text())
        all_names = [t["term"].lower() for t in all_terms]
        rejected = sorted(n for n in all_names if n not in accepted_terms)
    else:
        rejected = []

    vp = _vetting_path(work)
    vp.parent.mkdir(parents=True, exist_ok=True)
    vp.write_text(
        json.dumps(
            {
                "accept": sorted(accepted_terms),
                "reject": rejected,
            },
            indent=2,
            ensure_ascii=False,
        )
    )

    return RedirectResponse(f"/options?work={work}", status_code=303)


@app.get("/options", response_class=HTMLResponse)
def options(request: Request, work: str, error: str = ""):
    return templates.TemplateResponse(
        request,
        "options.html",
        {
            "work": work,
            "error": error,
        },
    )


@app.post("/build")
def build(
    work: Annotated[str, Form()],
    top_n: Annotated[int, Form()] = 50,
    threshold: Annotated[float, Form()] = 0.1,
    depth: Annotated[str, Form()] = "",
    focus: Annotated[str, Form()] = "",
):
    # Re-run rarities with final top-n (vetting file already applied automatically)
    terms_p = _rarities_path(work)
    terms_p.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            "cmapr",
            "rarities",
            str(_corpus_path(work)),
            "--top-n",
            str(top_n),
            "--output",
            str(terms_p),
        ]
    )
    if result.returncode != 0:
        return RedirectResponse(
            f"/options?work={work}&error=rarities+failed", status_code=303
        )

    # Build graph
    graph_p = _graph_path(work)
    graph_p.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "cmapr",
        "graph",
        str(_corpus_path(work)),
        "-t",
        str(terms_p),
        "--threshold",
        str(threshold),
        "--output",
        str(graph_p),
    ]
    if depth.strip():
        cmd += ["--depth", depth.strip()]
    if focus.strip():
        cmd += ["--focus", focus.strip()]
    result = _run(cmd)
    if result.returncode != 0:
        return RedirectResponse(
            f"/options?work={work}&error=graph+failed", status_code=303
        )

    # Export to HTML
    export_p = _export_path(work)
    export_p.parent.mkdir(parents=True, exist_ok=True)
    result = _run(
        [
            "cmapr",
            "export",
            str(graph_p),
            "--format",
            "html",
            "--output",
            str(export_p),
        ]
    )
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
