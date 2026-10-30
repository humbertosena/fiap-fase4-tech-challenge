"""Painel de saúde do pipeline: uma página HTML autocontida gerada do ``AuditReport``.

O painel reaproveita o CSS e o tema dos explainers, mas não depende de nada externo:
os gauges são CSS e o mapa de PSI é uma tabela colorida (com valor e rótulo, nunca só
cor). Mostra o estado geral, os SLOs por janela, o feed de alertas, a latência por
estágio, o mapa de PSI por feature e janela e os links para os relatórios do Evidently.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from credit_sentinel.explainers.build import REPO_URL, STATIC_DIR, br, pct, thousands
from credit_sentinel.observability.logger import PipelineEvent, event_json
from credit_sentinel.utils.io import write_text

if TYPE_CHECKING:
    from credit_sentinel.observability.pipeline_runner import (
        AuditReport,
        WindowRunResult,
    )
    from credit_sentinel.observability.slo_monitor import SloConfig

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_LOCAL = Path(__file__).resolve().parent / "static"
SAMPLE_SIZE = 40
FULL_SAMPLE_WINDOWS = ("bad_batch", "mes_1", "mes_3_crise")
SCORE_ROW = "score do modelo"
PSI_ATTENTION = 0.10
PSI_CRITICAL = 0.25
GAUGE_SCALE = 2.0
RUNBOOK_URL = f"{REPO_URL}/blob/main/docs/runbook_incidentes.md"
STATUS_LABEL = {"OK": "Dentro do limite", "WARN": "Atenção", "BREACHED": "Estourado"}
SEVERITY_LABEL = {
    "P1": "P1 · lote bloqueado",
    "P2": "P2 · drift crítico",
    "P3": "P3 · atenção",
    "P4": "P4 · informativo",
}
SLO_TITLES = {
    "SLO1_integridade": "Integridade dos dados",
    "SLO2_estabilidade_psi": "Estabilidade (PSI)",
    "SLO3_consistencia_ks": "Consistência (KS)",
    "SLO4_performance": "Performance (ROC-AUC)",
    "SLO5_latencia": "Latência do lote",
}
STAGES = (
    "ingestion",
    "contract_validation",
    "inference",
    "drift_detection",
    "slo_evaluation",
    "tracking",
)
STAGE_LABEL = {
    "ingestion": "Ingestão",
    "contract_validation": "Contrato",
    "inference": "Inferência",
    "drift_detection": "Drift",
    "slo_evaluation": "SLOs",
    "tracking": "Tracking",
}


def _psi_level(value: float) -> str:
    if value >= PSI_CRITICAL:
        return "critico"
    return "atencao" if value >= PSI_ATTENTION else "estavel"


def _worst(window: WindowRunResult) -> str:
    return min(alert.severity for alert in window.slo.alerts)


def _overall(audit: AuditReport) -> dict[str, str]:
    worst = min(w.slo.alerts[0].severity for w in audit.windows)
    if worst in {"P1", "P2"}:
        return {"level": "critico", "label": "Crítico"}
    if worst == "P3":
        return {"level": "atencao", "label": "Atenção"}
    return {"level": "estavel", "label": "Saudável"}


def _slo_rows(audit: AuditReport, config: SloConfig | None) -> list[dict[str, Any]]:
    if config is not None:
        specs = [(s.id, s.title or SLO_TITLES.get(s.id, s.id), s) for s in config.slos]
    else:
        ids = {r.slo_id for w in audit.windows for r in w.slo.results}
        specs = [(i, SLO_TITLES.get(i, i), None) for i in SLO_TITLES if i in ids]
    rows = []
    for slo_id, title, spec in specs:
        cells = []
        for window in audit.windows:
            found = next((r for r in window.slo.results if r.slo_id == slo_id), None)
            cells.append({"window": window.window, "result": found})
        limit = (
            None if spec is None else f"{spec.objective.op} {spec.objective.value:g}"
        )
        rows.append({"id": slo_id, "title": title, "limit": limit, "cells": cells})
    return rows


def _psi_map(audit: AuditReport) -> list[dict[str, Any]]:
    features = next(
        (
            [item.feature for item in w.drift.numeric]
            for w in audit.windows
            if w.drift is not None
        ),
        [],
    )
    rows = []
    for name in [*features, SCORE_ROW]:
        cells = []
        for window in audit.windows:
            value = _psi_value(window, name)
            level = None if value is None else _psi_level(value)
            cells.append({"window": window.window, "value": value, "level": level})
        rows.append({"feature": name, "cells": cells})
    return rows


def _psi_value(window: WindowRunResult, name: str) -> float | None:
    if name == SCORE_ROW:
        predictions = window.predictions
        return None if predictions is None else predictions.psi_score_vs_reference
    if window.drift is None:
        return None
    return next((i.psi for i in window.drift.numeric if i.feature == name), None)


def _latency_rows(audit: AuditReport) -> tuple[list[dict[str, Any]], int]:
    peak = max(
        (ms for w in audit.windows for ms in w.stage_latencies_ms.values()), default=0
    )
    rows = []
    for window in audit.windows:
        stages = [
            {
                "label": STAGE_LABEL[stage],
                "ms": window.stage_latencies_ms.get(stage, 0),
                "status": window.stage_status.get(stage, "SKIPPED"),
                "width": _share(window.stage_latencies_ms.get(stage, 0), peak),
            }
            for stage in STAGES
        ]
        rows.append({"window": window.window, "stages": stages})
    return rows, peak


def _share(value: int, peak: int) -> float:
    return 0.0 if peak <= 0 else round(100 * value / peak, 1)


def _evidently_links(audit: AuditReport) -> list[dict[str, str | None]]:
    links = []
    for window in audit.windows:
        ref = window.drift_ref
        href = None if ref is None else ref.removeprefix("reports/")
        links.append({"window": window.window, "href": href})
    return links


def _alerts(audit: AuditReport) -> list[dict[str, Any]]:
    return [
        {**alert.model_dump(), "label": SEVERITY_LABEL[alert.severity]}
        for alert in audit.alerts
    ]


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "j2"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update({"br": br, "pct": pct, "thousands": thousands})
    return env


def _text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def render_health_dashboard(
    audit: AuditReport, out_path: Path, config: SloConfig | None = None
) -> None:
    """Gera o painel de saúde HTML a partir do relatório de auditoria.

    Args:
        audit: Resultado de ``run_pipeline``.
        out_path: Arquivo HTML de destino.
        config: SLOs (para título e limite de cada linha); opcional.
    """
    latency, peak = _latency_rows(audit)
    context = {
        "audit": audit,
        "overall": _overall(audit),
        "windows": [
            {"run": w, "worst": _worst(w), "severity_label": SEVERITY_LABEL[_worst(w)]}
            for w in audit.windows
        ],
        "slo_rows": _slo_rows(audit, config),
        "psi_rows": _psi_map(audit),
        "latency_rows": latency,
        "latency_peak": peak,
        "evidently": _evidently_links(audit),
        "alerts": _alerts(audit),
        "gauge_scale": GAUGE_SCALE,
        "status_label": STATUS_LABEL,
        "severity_label": SEVERITY_LABEL,
        "runbook_url": RUNBOOK_URL,
        "repo_url": REPO_URL,
        "css": _text(STATIC_DIR / "explainer.css")
        + "\n"
        + _text(STATIC_LOCAL / "health_dashboard.css"),
        "js": _text(STATIC_DIR / "explainer.js")
        + "\n"
        + _text(STATIC_LOCAL / "health_dashboard.js"),
    }
    html = _environment().get_template("health_dashboard.html.j2").render(**context)
    write_text(html, out_path)


def select_events_sample(events: list[PipelineEvent]) -> list[PipelineEvent]:
    """Escolhe a amostra de eventos versionada como evidência da execução.

    Usa a última execução do log. Mantém todos os eventos de ``bad_batch``, ``mes_1`` e
    ``mes_3_crise`` e, de ``mes_2``, apenas os desfechos (sem ``STARTED``).

    Args:
        events: Eventos do arquivo de log (podem misturar execuções).

    Returns:
        Eventos da amostra, na ordem original (40 numa execução completa com MLflow).
    """
    if not events:
        return []
    last = events[-1].run_id
    chosen = []
    for event in events:
        if event.run_id != last:
            continue
        if event.window in FULL_SAMPLE_WINDOWS or event.status != "STARTED":
            chosen.append(event)
    return chosen


def write_events_sample(events: list[PipelineEvent], out_path: Path) -> int:
    """Grava a amostra de eventos em JSON Lines (já mascarados pelo logger).

    Args:
        events: Eventos do arquivo de log.
        out_path: Arquivo de destino.

    Returns:
        Quantidade de eventos gravados.
    """
    sample = select_events_sample(events)
    write_text("".join(f"{event_json(e)}\n" for e in sample), out_path)
    return len(sample)
