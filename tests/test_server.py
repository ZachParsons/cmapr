"""Tests for the local web UI server (Phase 16)."""

from __future__ import annotations

import json
import re
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

    def test_renders_corpus_count_as_freq(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/review?work=mywork")
        # corpus_count from term metadata is shown in the Freq column.
        assert ">40</td>" in r.text and ">25</td>" in r.text

    def test_exposes_detection_params_and_sortable_headers(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/review?work=mywork")
        # Params beyond top-n.
        assert 'name="threshold"' in r.text
        assert 'name="pos" value="noun"' in r.text
        assert all(
            tok in r.text for tok in ("keep_names", "keep_fragments", "no_lemmatize")
        )
        # Sortable column headers.
        assert "sortTerms(1" in r.text and "sortTerms(3" in r.text

    def test_review_prefills_persisted_params(self, tmp_data):
        params = {
            "top_n": 120,
            "threshold": 0.25,
            "pos": "noun",
            "keep_names": True,
            "keep_fragments": False,
            "no_lemmatize": False,
        }
        (tmp_data / "data/output/rarities/mywork/rarities_params.json").write_text(
            json.dumps(params)
        )
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/review?work=mywork")
        assert 'value="120"' in r.text  # top_n
        assert 'value="0.25"' in r.text  # threshold


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

    def test_persists_rejection_reasons_parallel_to_reject(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            c.post(
                "/vet",
                data={
                    "work": "mywork",
                    "terms": ["sign"],  # accept sign; reject the rest
                    "reason::interpretant": "duplicate",
                    "reason::semiosis": "",  # blank reasons are dropped
                },
            )
        vet = json.loads(
            (tmp_data / "data/output/rarities/mywork/vetting.json").read_text()
        )
        # reject stays a plain string list (CLI loaders depend on this).
        assert vet["reject"] == sorted(["interpretant", "semiosis"])
        assert all(isinstance(x, str) for x in vet["reject"])
        # reasons recorded in a parallel map, blanks omitted.
        assert vet["reasons"] == {"interpretant": "duplicate"}

    def test_review_prefills_saved_reason(self, tmp_data):
        vp = tmp_data / "data/output/rarities/mywork/vetting.json"
        vp.write_text(
            json.dumps(
                {"accept": [], "reject": ["sign"], "reasons": {"sign": "atopic"}}
            )
        )
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/review?work=mywork")
        sign_select = re.search(
            r'name="reason::sign".*?</select>', r.text, re.DOTALL
        ).group(0)
        assert re.search(r'value="atopic"\s+selected', sign_select)

    def test_reasons_grow_learned_filters(self, tmp_data):
        # atopic → stopword supplement; duplicate → alias onto an accepted term.
        with TestClient(app, follow_redirects=False) as c:
            c.post(
                "/vet",
                data={
                    "work": "mywork",
                    "terms": ["sign"],  # accept sign (canonical for the alias)
                    "reason::interpretant": "atopic",
                    "reason::semiosis": "atopic",
                },
            )
        sw = json.loads((tmp_data / "data/output/stopwords_extra.json").read_text())
        assert {"interpretant", "semiosis"} <= set(sw["stopwords"])


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

    def test_does_not_rerun_rarities(self, tmp_data):
        """The build step must filter the candidate pool, not overwrite it."""
        with (
            patch("concept_mapper.server.app._run") as mock_run,
            TestClient(app, follow_redirects=False) as c,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            c.post("/build", data={"work": "mywork", "top_n": "50", "threshold": "0.1"})
        invoked = [call.args[0] for call in mock_run.call_args_list]
        assert not any("rarities" in cmd for cmd in invoked), (
            "build must not invoke `cmapr rarities` — that would shrink the "
            "candidate pool that the review page shows"
        )

    def test_writes_seed_terms_filtered_by_vetting(self, tmp_data):
        """Build derives seed_terms.json from candidates - rejected, capped to top_n."""
        # Reject 'sign'; only 'interpretant' and 'semiosis' should remain.
        vetting = {"accept": ["interpretant", "semiosis"], "reject": ["sign"]}
        (tmp_data / "data/output/rarities/mywork/vetting.json").write_text(
            json.dumps(vetting)
        )
        with (
            patch("concept_mapper.server.app._run") as mock_run,
            TestClient(app, follow_redirects=False) as c,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            c.post("/build", data={"work": "mywork", "top_n": "10", "threshold": "0.1"})

        seed_p = tmp_data / "data/output/rarities/mywork/seed_terms.json"
        assert seed_p.exists()
        seed = json.loads(seed_p.read_text())
        seed_names = {s["term"] for s in seed}
        assert "sign" not in seed_names
        assert seed_names == {"interpretant", "semiosis"}
        # Sorted by score descending: interpretant (1.8) before semiosis (1.2)
        assert [s["term"] for s in seed] == ["interpretant", "semiosis"]

    def test_seed_terms_capped_to_top_n(self, tmp_data):
        with (
            patch("concept_mapper.server.app._run") as mock_run,
            TestClient(app, follow_redirects=False) as c,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            c.post("/build", data={"work": "mywork", "top_n": "1", "threshold": "0.1"})
        seed = json.loads(
            (tmp_data / "data/output/rarities/mywork/seed_terms.json").read_text()
        )
        assert len(seed) == 1
        assert seed[0]["term"] == "sign"  # highest score

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
# POST /rarities (adjust pool size from review page)
# ---------------------------------------------------------------------------


class TestRerunRarities:
    def test_runs_rarities_with_pool_size(self, tmp_data):
        with (
            patch("concept_mapper.server.app._run") as mock_run,
            TestClient(app, follow_redirects=False) as c,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            r = c.post("/rarities", data={"work": "mywork", "pool_size": "300"})
        cmd = mock_run.call_args.args[0]
        assert cmd[1] == "rarities"
        assert "--top-n" in cmd
        assert "300" in cmd
        assert r.status_code == 303
        assert r.headers["location"] == "/review?work=mywork"

    def test_clamps_pool_size_to_max(self, tmp_data):
        with (
            patch("concept_mapper.server.app._run") as mock_run,
            TestClient(app, follow_redirects=False) as c,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            c.post("/rarities", data={"work": "mywork", "pool_size": "99999"})
        cmd = mock_run.call_args.args[0]
        idx = cmd.index("--top-n")
        assert cmd[idx + 1] == "2000"  # clamped

    def test_clamps_pool_size_to_min(self, tmp_data):
        with (
            patch("concept_mapper.server.app._run") as mock_run,
            TestClient(app, follow_redirects=False) as c,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            c.post("/rarities", data={"work": "mywork", "pool_size": "1"})
        cmd = mock_run.call_args.args[0]
        idx = cmd.index("--top-n")
        assert cmd[idx + 1] == "10"  # clamped

    def test_redirects_with_error_on_failure(self, tmp_data):
        with (
            patch("concept_mapper.server.app._run") as mock_run,
            TestClient(app, follow_redirects=False) as c,
        ):
            mock_run.return_value = MagicMock(returncode=1, stdout="", stderr="oh no")
            r = c.post("/rarities", data={"work": "mywork", "pool_size": "200"})
        assert r.status_code == 303
        assert "/review?work=mywork" in r.headers["location"]
        assert "error=" in r.headers["location"]

    def test_passes_extra_params_to_cli_and_persists(self, tmp_data):
        with (
            patch("concept_mapper.server.app._run") as mock_run,
            TestClient(app, follow_redirects=False) as c,
        ):
            mock_run.return_value = MagicMock(returncode=0, stdout="", stderr="")
            r = c.post(
                "/rarities",
                data={
                    "work": "mywork",
                    "pool_size": "120",
                    "threshold": "0.2",
                    "pos": ["noun", "verb"],
                    "keep_fragments": "true",
                },
            )
        assert r.status_code == 303
        cmd = mock_run.call_args.args[0]
        assert cmd[cmd.index("--threshold") + 1] == "0.2"
        assert cmd[cmd.index("--top-n") + 1] == "120"
        assert cmd[cmd.index("--pos") + 1] == "noun,verb"
        assert "--no-filter-fragments" in cmd
        assert "--no-filter-names" not in cmd  # not requested

        saved = json.loads(
            (tmp_data / "data/output/rarities/mywork/rarities_params.json").read_text()
        )
        assert saved["top_n"] == 120
        assert saved["threshold"] == 0.2
        assert saved["pos"] == "noun,verb"
        assert saved["keep_fragments"] is True


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


# ---------------------------------------------------------------------------
# Loading overlay (shared across templates)
# ---------------------------------------------------------------------------


class TestLoadingOverlay:
    """The loading overlay must render on every page that submits work,
    so users always get feedback that long-running ingest/build jobs began."""

    def _has_overlay(self, html: str) -> bool:
        return 'id="loading"' in html and "loading-overlay" in html

    def test_index_has_overlay(self):
        r = client.get("/")
        assert self._has_overlay(r.text)

    def test_review_has_overlay(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/review?work=mywork")
        assert self._has_overlay(r.text)

    def test_options_has_overlay(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/options?work=mywork")
        assert self._has_overlay(r.text)

    def test_result_has_overlay(self, tmp_data):
        with TestClient(app, follow_redirects=False) as c:
            r = c.get("/result?work=mywork")
        assert self._has_overlay(r.text)
