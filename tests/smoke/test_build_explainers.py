"""Smoke tests do gerador das páginas didáticas (HTML autocontido)."""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

import pytest

from credit_sentinel.cli import main
from credit_sentinel.contracts.privacy_sanitizer import scan_for_pii
from credit_sentinel.explainers.build import (
    CHART_JS_INTEGRITY,
    CHART_JS_SRC,
    MAX_PAGE_BYTES,
    build_all,
)
from credit_sentinel.settings import Settings
from credit_sentinel.utils.io import read_json

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.smoke


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("docs")
    build_all(settings=Settings(_env_file=None), etapas=[1, 2, 3, 4], out_dir=out)
    return out


def _pages(root: Path) -> list[Path]:
    releases = root / "releases"
    return [
        root / "index.html",
        releases / "etapa-1-contratos.html",
        releases / "etapa-2-drift.html",
        releases / "etapa-3-observabilidade.html",
        releases / "etapa-4-governanca.html",
    ]


def test_index_page_and_nojekyll_are_created(built: Path) -> None:
    assert (built / "index.html").exists()
    assert (built / ".nojekyll").exists()


def test_reports_are_copied_for_github_pages(built: Path) -> None:
    assert (built / "reports" / "health_dashboard.html").exists()
    assert (built / "reports" / "slo_audit_report.json").exists()
    assert (built / "reports" / "validation" / "bad_batch_report.json").exists()
    assert (built / "reports" / "evidently_drift_report.html").exists()
    assert (built / "reports" / "evidently_drift_summary.json").exists()


def test_index_lists_the_published_stages(built: Path) -> None:
    html = (built / "index.html").read_text(encoding="utf-8")
    assert 'href="releases/etapa-2-drift.html"' in html
    assert 'href="reports/evidently_drift_report.html"' in html
    assert 'href="releases/etapa-4-governanca.html"' in html
    assert 'href="reports/fairness_audit.md"' in html
    assert "Em breve" not in html
    assert 'href="releases/etapa-3-observabilidade.html"' in html
    assert 'href="reports/health_dashboard.html"' in html


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


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_pages_are_self_contained(built: Path, index: int) -> None:
    html = _pages(built)[index].read_text(encoding="utf-8")
    assert "<style>" in html
    assert 'lang="pt-BR"' in html
    assert not re.search(r'(?:src|href)="http://', html)
    sources = re.findall(r'<script[^>]*\ssrc="([^"]+)"', html)
    assert sources == [CHART_JS_SRC]
    assert f'integrity="{CHART_JS_INTEGRITY}"' in html
    assert '<link rel="stylesheet"' not in html


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
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


def test_etapa3_page_has_its_interactive_components(built: Path) -> None:
    html = _pages(built)[3].read_text(encoding="utf-8")
    for element in (
        'id="replay-run"',
        'id="replay-stages"',
        'id="gauges"',
        'id="alert-filters"',
        'id="alert-feed"',
        'id="alias-toggle"',
        'id="pii-input"',
        'id="gallery"',
    ):
        assert element in html, element
    assert "v0.3.0-rc" in html
    assert "reports/health_dashboard.html" in html
    assert 'href="../releases/etapa-2-drift.html"' in html


def test_etapa3_page_embeds_at_most_three_screenshots(built: Path) -> None:
    html = _pages(built)[3].read_text(encoding="utf-8")
    assert 1 <= html.count('<img src="data:image/jpeg;base64,') <= 3


@pytest.mark.parametrize("index", [0, 1, 2, 3, 4])
def test_pages_never_carry_personal_data(built: Path, index: int) -> None:
    html = _pages(built)[index].read_text(encoding="utf-8")
    assert scan_for_pii(html) == []
    assert "CLI-A1B2" not in html


def test_etapa3_numbers_come_from_the_audit(built: Path) -> None:
    html = _pages(built)[3].read_text(encoding="utf-8")
    audit = read_json(Settings(_env_file=None).reports_dir / "slo_audit_report.json")
    assert audit["run_id"] in html
    crisis = next(w for w in audit["windows"] if w["window"] == "mes_3_crise")
    assert f"{round(100 * crisis['slo']['remaining_budget'])} %" in html


def test_etapa4_page_has_its_interactive_components(built: Path) -> None:
    html = _pages(built)[4].read_text(encoding="utf-8")
    for element in (
        'id="stepper"',
        'id="inventory-filters"',
        'id="fair-list"',
        'id="dag"',
        'id="interv-canvas"',
        'id="ret-slider"',
        'id="star-start"',
    ):
        assert element in html, element
    assert "v1.0.0-ga" in html
    assert "reports/causal_ablation.json" in html
    assert 'href="../releases/etapa-3-observabilidade.html"' in html


def test_etapa4_numbers_come_from_the_reports(built: Path) -> None:
    html = _pages(built)[4].read_text(encoding="utf-8")
    reports = Settings(_env_file=None).reports_dir
    audit = read_json(reports / "fairness_audit.json")
    worst = min(g["disparate_impact"] for r in audit["reports"] for g in r["groups"])
    assert f"Menor razão: {worst:.3f}".replace(".", ",") in html
    ablation = read_json(reports / "causal_ablation.json")
    concept = next(s for s in ablation["scenarios"] if s["key"] == "so_concept")
    assert f"{concept['delta_auc']:.3f}".replace(".", ",") in html
