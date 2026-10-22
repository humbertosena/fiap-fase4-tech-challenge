"""Smoke tests do gerador das páginas didáticas (HTML autocontido)."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest

from credit_sentinel.cli import main
from credit_sentinel.explainers.build import (
    CHART_JS_INTEGRITY,
    CHART_JS_SRC,
    MAX_PAGE_BYTES,
    build_all,
)
from credit_sentinel.settings import Settings

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("docs")
    build_all(settings=Settings(_env_file=None), etapas=[1, 2], out_dir=out)
    return out


def _pages(root: Path) -> list[Path]:
    releases = root / "releases"
    return [
        root / "index.html",
        releases / "etapa-1-contratos.html",
        releases / "etapa-2-drift.html",
    ]


def test_index_page_and_nojekyll_are_created(built: Path) -> None:
    assert (built / "index.html").exists()
    assert (built / ".nojekyll").exists()


def test_reports_are_copied_for_github_pages(built: Path) -> None:
    assert (built / "reports" / "validation" / "bad_batch_report.json").exists()
    assert (built / "reports" / "evidently_drift_report.html").exists()
    assert (built / "reports" / "evidently_drift_summary.json").exists()


def test_index_lists_the_published_stages(built: Path) -> None:
    html = (built / "index.html").read_text(encoding="utf-8")
    assert 'href="releases/etapa-2-drift.html"' in html
    assert 'href="reports/evidently_drift_report.html"' in html
    assert html.count("Em breve") >= 2


def test_navigation_links_stage_two_from_stage_one(built: Path) -> None:
    html = _pages(built)[1].read_text(encoding="utf-8")
    assert 'href="../releases/etapa-2-drift.html"' in html


def test_etapa2_page_has_its_interactive_components(built: Path) -> None:
    html = _pages(built)[2].read_text(encoding="utf-8")
    for element in (
        'id="window-buttons"',
        'id="psi-slider"',
        'id="heatmap"',
        'id="band-chart"',
        'id="perf-chart"',
    ):
        assert element in html
    assert "reports/evidently_drift_report.html" in html
    assert "v0.2.0-beta" in html


@pytest.mark.parametrize("index", [0, 1, 2])
def test_pages_are_self_contained(built: Path, index: int) -> None:
    html = _pages(built)[index].read_text(encoding="utf-8")
    assert "<style>" in html
    assert 'lang="pt-BR"' in html
    assert not re.search(r'(?:src|href)="http://', html)
    sources = re.findall(r'<script[^>]*\ssrc="([^"]+)"', html)
    assert sources == [CHART_JS_SRC]
    assert f'integrity="{CHART_JS_INTEGRITY}"' in html
    assert '<link rel="stylesheet"' not in html


@pytest.mark.parametrize("index", [0, 1, 2])
def test_payload_is_valid_json_and_page_fits_limit(built: Path, index: int) -> None:
    page = _pages(built)[index]
    html = page.read_text(encoding="utf-8")
    pattern = r'<script type="application/json" id="payload">(.*?)</script>'
    match = re.search(pattern, html, re.DOTALL)
    assert match is not None
    assert isinstance(json.loads(match.group(1)), dict)
    assert page.stat().st_size <= MAX_PAGE_BYTES


def test_build_explainers_cli(tmp_path: Path) -> None:
    assert main(["build-explainers", "--etapa", "1", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "releases" / "etapa-1-contratos.html").exists()
