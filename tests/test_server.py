"""Tests for the local web UI server (Phase 16)."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

try:
    from fastapi.testclient import TestClient
    from concept_mapper.server.app import app
except ImportError:
    pytest.skip(
        "serve dependencies not installed — run: pip install 'concept-mapper[serve]'",
        allow_module_level=True,
    )

client = TestClient(app, follow_redirects=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def tmp_data(tmp_path, monkeypatch):
    """Redirect all server path helpers to a tmp data directory."""

    monkeypatch.chdir(tmp_path)

    corpus_dir = tmp_path / "data/output/corpus/mywork"
    corpus_dir.mkdir(parents=True)
    rarities_dir = tmp_path / "data/output/rarities/mywork"
    rarities_dir.mkdir(parents=True)

    corpus = [
        {"sentences": [{"text": "Sign is a symbol.", "tokens": []}], "tokens": []}
    ]
    (corpus_dir / "corpus.json").write_text(json.dumps(corpus))

    terms = [
        {"term": "sign", "metadata": {"score": 2.5, "corpus_count": 40}},
        {"term": "interpretant", "metadata": {"score": 1.8, "corpus_count": 25}},
        {"term": "semiosis", "metadata": {"score": 1.2, "corpus_count": 12}},
    ]
    (rarities_dir / "terms.json").write_text(json.dumps(terms))

    return tmp_path


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------


class TestIndex:
    def test_renders(self):
        r = client.get("/")
        assert r.status_code == 200
        assert "cmapr" in r.text

    def test_shows_existing_work(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/")
        assert "mywork" in r.text

    def test_shows_error_param(self):
        r = client.get("/?error=something+went+wrong")
        assert r.status_code == 200
        assert "something went wrong" in r.text


# ---------------------------------------------------------------------------
# POST /start
# ---------------------------------------------------------------------------


class TestStart:
    def test_redirects_to_review_when_corpus_exists(self, tmp_data):
        with (
            patch("concept_mapper.server.app._run") as mock_run,
            TestClient(app, follow_redirects=False) as c,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            r = c.post("/start", data={"file_path": "data/input/mywork.txt"})

        assert r.status_code == 303
        assert r.headers["location"] == "/review?work=mywork"
        # _run should only be called for rarities (ingest skipped — corpus exists)
        calls = [call.args[0] for call in mock_run.call_args_list]
        assert not any("ingest" in cmd for cmd in calls), (
            "ingest should be skipped when corpus already exists"
        )

    def test_runs_ingest_when_corpus_missing(self, tmp_data):
        (tmp_data / "data/output/corpus/newwork").mkdir(parents=True, exist_ok=True)
        with (
            patch("concept_mapper.server.app._run") as mock_run,
            TestClient(app, follow_redirects=False) as c,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            r = c.post("/start", data={"file_path": "data/input/newwork.txt"})

        assert r.status_code == 303
        calls = [call.args[0] for call in mock_run.call_args_list]
        assert any("ingest" in cmd for cmd in calls), (
            "ingest should run when corpus does not exist"
        )

    def test_redirects_to_error_on_ingest_failure(self, tmp_data):
        with (
            patch("concept_mapper.server.app._run") as mock_run,
            TestClient(app, follow_redirects=False) as c,
        ):
            mock_run.return_value = MagicMock(
                returncode=1, stdout="", stderr="ingest error"
            )
            # Force ingest by pointing at a file whose work name has no corpus
            r = c.post("/start", data={"file_path": "data/input/noexist.txt"})

        assert r.status_code == 303
        assert "/?error=" in r.headers["location"]


# ---------------------------------------------------------------------------
# GET /review
# ---------------------------------------------------------------------------


class TestReview:
    def test_renders_terms(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/review?work=mywork")
        assert r.status_code == 200
        assert "sign" in r.text
        assert "interpretant" in r.text

    def test_all_checked_with_no_vetting_file(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/review?work=mywork")
        # All checkboxes should be checked (no vetting.json present)
        assert r.text.count("checked") >= 3

    def test_unchecked_for_rejected_terms(self, tmp_data):
        vp = tmp_data / "data/output/rarities/mywork/vetting.json"
        vp.write_text(json.dumps({"accept": [], "reject": ["sign"]}))
        with (
            patch("concept_mapper.server.app._vetting_path", return_value=vp),
            TestClient(app, follow_redirects=False) as c,
        ):
            r = c.get("/review?work=mywork")
        assert r.status_code == 200
        # sign row must not carry the checked attribute
        import re

        sign_input = re.search(r'<input[^>]*value="sign"[^>]*>', r.text, re.DOTALL)
        assert sign_input, "sign checkbox not found in response"
        assert "checked" not in sign_input.group(0), (
            "sign should not be checked when it is in the reject list"
        )

    def test_redirects_home_when_no_terms_file(self, tmp_data):
        (tmp_data / "data/output/rarities/mywork/terms.json").unlink()
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/review?work=mywork")
        assert r.status_code == 303
        assert r.headers["location"] == "/"


# ---------------------------------------------------------------------------
# POST /vet
# ---------------------------------------------------------------------------


class TestVet:
    def test_saves_vetting_and_redirects(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.post(
                "/vet",
                data={
                    "work": "mywork",
                    "terms": ["sign", "semiosis"],
                },
            )
        assert r.status_code == 303
        assert r.headers["location"] == "/options?work=mywork"

        vp = tmp_data / "data/output/rarities/mywork/vetting.json"
        assert vp.exists()
        vet = json.loads(vp.read_text())
        assert "sign" in vet["accept"]
        assert "semiosis" in vet["accept"]
        assert "interpretant" in vet["reject"]

    def test_all_unchecked_rejects_all(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.post("/vet", data={"work": "mywork"})
        assert r.status_code == 303
        vp = tmp_data / "data/output/rarities/mywork/vetting.json"
        vet = json.loads(vp.read_text())
        assert len(vet["reject"]) == 3
        assert vet["accept"] == []


# ---------------------------------------------------------------------------
# GET /options
# ---------------------------------------------------------------------------


class TestOptions:
    def test_renders(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/options?work=mywork")
        assert r.status_code == 200
        assert "mywork" in r.text
        assert "top_n" in r.text

    def test_shows_error_param(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/options?work=mywork&error=graph+failed")
        assert "graph failed" in r.text


# ---------------------------------------------------------------------------
# POST /build
# ---------------------------------------------------------------------------


class TestBuild:
    def test_redirects_to_result_on_success(self, tmp_data):
        with (
            patch("concept_mapper.server.app._run") as mock_run,
            TestClient(app, follow_redirects=False) as c,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="ok", stderr="")
            r = c.post(
                "/build", data={"work": "mywork", "top_n": "50", "threshold": "0.1"}
            )

        assert r.status_code == 303
        assert r.headers["location"] == "/result?work=mywork"

    def test_passes_focus_and_depth_when_provided(self, tmp_data):
        with (
            patch("concept_mapper.server.app._run") as mock_run,
            TestClient(app, follow_redirects=False) as c,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            c.post(
                "/build",
                data={
                    "work": "mywork",
                    "top_n": "40",
                    "threshold": "0.1",
                    "depth": "2",
                    "focus": "sign",
                },
            )

        graph_cmd = next(
            call.args[0] for call in mock_run.call_args_list if "graph" in call.args[0]
        )
        assert "--depth" in graph_cmd
        assert "2" in graph_cmd
        assert "--focus" in graph_cmd
        assert "sign" in graph_cmd

    def test_redirects_to_options_on_graph_failure(self, tmp_data):
        call_count = 0

        def side_effect(cmd, **_):
            nonlocal call_count
            call_count += 1
            if "graph" in cmd:
                return MagicMock(returncode=1, stdout="", stderr="graph error")
            return MagicMock(returncode=0, stdout="", stderr="")

        with (
            patch("concept_mapper.server.app._run", side_effect=side_effect),
            TestClient(app, follow_redirects=False) as c,
        ):
            r = c.post(
                "/build", data={"work": "mywork", "top_n": "50", "threshold": "0.1"}
            )

        assert r.status_code == 303
        assert "/options?work=mywork" in r.headers["location"]


# ---------------------------------------------------------------------------
# GET /result
# ---------------------------------------------------------------------------


class TestResult:
    def test_renders_without_export(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/result?work=mywork")
        assert r.status_code == 200
        assert "mywork" in r.text

    def test_renders_with_export(self, tmp_data):
        export_dir = tmp_data / "data/output/exports/mywork"
        export_dir.mkdir(parents=True)
        (export_dir / "index.html").write_text("<html><body>graph here</body></html>")
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/result?work=mywork")
        assert r.status_code == 200
        assert 'src="/exports/mywork/index.html"' in r.text


# ---------------------------------------------------------------------------
# GET /exports/{work}/index.html
# ---------------------------------------------------------------------------


class TestServeExport:
    def test_serves_export_html(self, tmp_data):
        export_dir = tmp_data / "data/output/exports/mywork"
        export_dir.mkdir(parents=True)
        (export_dir / "index.html").write_text("<html><body>d3 graph</body></html>")
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/exports/mywork/index.html")
        assert r.status_code == 200
        assert "d3 graph" in r.text

    def test_404_when_missing(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/exports/nowork/index.html")
        assert r.status_code == 404
