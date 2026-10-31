"""Painel de saúde HTML e amostra de log: marcadores, autocontenção e privacidade."""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import TYPE_CHECKING

import pytest

from credit_sentinel.cli import main
from credit_sentinel.contracts.privacy_sanitizer import scan_for_pii
from credit_sentinel.observability.health_dashboard import (
    SAMPLE_SIZE,
    render_health_dashboard,
    select_events_sample,
)
from credit_sentinel.observability.logger import PipelineEvent, read_events
from credit_sentinel.observability.pipeline_runner import WINDOW_ORDER, AuditReport
from credit_sentinel.observability.slo_monitor import load_slo_config
from credit_sentinel.utils.io import write_json

if TYPE_CHECKING:
    from pathlib import Path

    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.smoke

MARKERS = (
    'id="status-geral"',
    'id="janelas"',
    'id="slos"',
    'id="alertas"',
    'id="latencia"',
    'id="mapa-psi"',
    'id="evidently"',
)


class _Tags(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.tags: list[tuple[str, dict[str, str | None]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.tags.append((tag, dict(attrs)))


@pytest.fixture
def rendered(project_clean: Settings, tmp_path: Path) -> tuple[str, AuditReport]:
    assert main(["run-pipeline", "--window", "all", "--quick", "--no-mlflow"]) == 0
    audit_path = project_clean.logs_dir / "last_run" / "slo_audit_report.json"
    audit = AuditReport.model_validate_json(audit_path.read_text("utf-8"))
    out = tmp_path / "health_dashboard.html"
    config = load_slo_config(project_clean.config_dir / "slo_config.yaml")
    render_health_dashboard(audit, out, config)
    return out.read_text("utf-8"), audit


def test_page_has_every_section_marker(rendered: tuple[str, AuditReport]) -> None:
    html, _ = rendered
    for marker in MARKERS:
        assert marker in html, marker


def test_page_is_self_contained(rendered: tuple[str, AuditReport]) -> None:
    html, _ = rendered
    parser = _Tags()
    parser.feed(html)
    scripts = [a for t, a in parser.tags if t == "script"]
    assert all(a.get("src") is None for a in scripts)
    assert not [
        a for t, a in parser.tags if t == "link" and a.get("rel") == "stylesheet"
    ]
    assert not re.search(r"""(?:src|href)=["']https?://(?!github\.com)""", html)
    assert len(html.encode("utf-8")) < 250_000


def test_every_window_alert_and_slo_appears(rendered: tuple[str, AuditReport]) -> None:
    html, audit = rendered
    for window in WINDOW_ORDER:
        assert f'data-window="{window}"' in html
    for severity in ("P1", "P2", "P3", "P4"):
        assert f'data-severity="{severity}"' in html
    for result in audit.windows[1].slo.results:
        assert result.slo_id in html
    assert "bloqueado" in html.lower()
    assert "runbook_incidentes.md#p1-circuit-breaker" in html


def test_psi_map_lists_the_features_and_the_score(
    rendered: tuple[str, AuditReport],
) -> None:
    html, audit = rendered
    crisis = next(w for w in audit.windows if w.window == "mes_3_crise")
    assert crisis.drift is not None
    for item in crisis.drift.numeric:
        assert item.feature in html
    assert "score do modelo" in html
    assert 'class="psi critico' in html


def test_page_never_carries_personal_data(rendered: tuple[str, AuditReport]) -> None:
    html, _ = rendered
    assert scan_for_pii(html) == []
    assert "CLI-" not in html


def test_evidently_links_are_relative_to_the_reports_folder(
    project_clean: Settings, tmp_path: Path
) -> None:
    audit_path = project_clean.reports_dir / "slo_audit_report.json"
    assert main(["run-pipeline", "--window", "mes_2", "--output-dir", "reports"]) == 0
    audit = AuditReport.model_validate_json(audit_path.read_text("utf-8"))
    out = tmp_path / "painel.html"
    render_health_dashboard(audit, out)
    assert 'href="evidently/mes_2_drift_report.html"' in out.read_text("utf-8")


def test_sample_has_forty_redacted_events_with_the_key_outcomes(
    project_clean: Settings, tmp_path: Path
) -> None:
    assert main(["run-pipeline", "--window", "all", "--quick"]) == 0
    events = read_events(project_clean.logs_dir / "pipeline_events.jsonl")
    sample = select_events_sample(events)
    assert len(sample) == SAMPLE_SIZE == 40
    assert all(isinstance(e, PipelineEvent) for e in sample)
    pairs = {(e.window, e.stage, e.status) for e in sample}
    assert ("bad_batch", "contract_validation", "BLOCKED") in pairs
    assert ("mes_1", "contract_validation", "SUCCESS") in pairs
    assert ("mes_3_crise", "slo_evaluation", "SUCCESS") in pairs
    assert not [e for e in sample if e.window == "mes_2" and e.status == "STARTED"]
    out = write_json(
        [e.model_dump(mode="json", exclude_none=True) for e in sample],
        tmp_path / "s.json",
    )
    assert scan_for_pii(out.read_text("utf-8")) == []


def test_hidden_alerts_stay_hidden_despite_component_display_rules(
    rendered: tuple[str, AuditReport],
) -> None:
    html, _ = rendered
    assert "[hidden] { display: none !important; }" in html
