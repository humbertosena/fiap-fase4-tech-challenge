"""Payload da página da Etapa 3 (observabilidade, SLOs e MLflow).

Todos os números vêm dos artefatos versionados: ``reports/slo_audit_report.json``,
``reports/gates_report.json``, ``config/slo_config.yaml`` e a amostra de log em
``docs/evidencias/etapa3/``. O estado do Model Registry antes e depois da promoção é
derivado do relatório de guardrails e está comprovado pelos prints do MLflow.
"""

from __future__ import annotations

import base64
from typing import TYPE_CHECKING

from pydantic import BaseModel

from credit_sentinel.contracts.privacy_sanitizer import pii_patterns, scan_for_pii
from credit_sentinel.explainers.payloads import AcceptanceItem, ReleaseInfo, require
from credit_sentinel.ml.evaluate_gates import GatesReport
from credit_sentinel.observability.logger import (
    REQUIRED_FIELDS,
    PipelineEvent,
    read_events,
)
from credit_sentinel.observability.pipeline_runner import WINDOW_ORDER, AuditReport
from credit_sentinel.observability.slo_monitor import SloConfig, load_slo_config
from credit_sentinel.tracking.registry import RETIRED_REASON
from credit_sentinel.utils.io import read_json

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from credit_sentinel.settings import Settings

MAX_SCREENSHOT_BYTES = 400_000
MAX_WINDOW_MS = 120_000
EXPECTED_GATES = 5
MIN_PRINTS = 4
AUC_GAIN = 0.02
EVIDENCE_DIR = ("evidencias", "etapa3")
SAMPLE_FILE = "pipeline_events_sample.jsonl"
STAGES = (
    "ingestion",
    "contract_validation",
    "inference",
    "drift_detection",
    "slo_evaluation",
    "tracking",
)
SCREENSHOTS = [
    (
        "mlflow_runs_monitoring.jpg",
        "Runs do pipeline no MLflow",
        "Um run pai e um run aninhado por janela, com PSI, saldo e AUC lado a lado.",
    ),
    (
        "mlflow_evidently_artefato.jpg",
        "Relatório do Evidently como artefato",
        "O mesmo HTML de reports/evidently/, anexado ao run da janela mes_3_crise.",
    ),
    (
        "mlflow_registry_depois.jpg",
        "Model Registry depois da promoção",
        "A versão 2 virou @champion e a versão 1 ficou com a tag retired_reason.",
    ),
]
_FIELD_TEXT = {
    "timestamp": ("data e hora", "Instante UTC do evento (ISO-8601)."),
    "run_id": (
        "texto",
        "uuid4 da execução; liga todos os eventos de um mesmo pipeline.",
    ),
    "stage": ("enum", "Estágio do pipeline: os seis que aparecem no replay."),
    "status": ("enum", "STARTED, SUCCESS, FAILED, BLOCKED ou SKIPPED."),
    "records_processed": ("inteiro", "Registros tratados pelo estágio."),
    "rejected_records": ("inteiro", "Registros rejeitados pelo contrato de dados."),
    "latency_ms": ("inteiro", "Duração do estágio em milissegundos."),
    "model_version": ("texto", "Versão do modelo avaliado."),
    "window": ("texto", "Janela de dados em processamento."),
    "message": ("texto", "Texto livre, já mascarado."),
    "details": ("objeto", "Dados adicionais do estágio, já mascarados."),
}
_PII_FRAGMENTS = [
    "Falha ao analisar o titular ",
    "CLI-",
    "A1B2C3D4E5",
    " (CPF ",
    "123.456",
    ".789-09",
    ") pelo contato ",
    "maria",
    "@banco.com.br",
]
ETAPA3_COMMANDS = [
    "uv sync --all-groups",
    "uv run credit-sentinel generate-data --force",
    "uv run credit-sentinel train",
    "uv run credit-sentinel simulate --force",
    "uv run credit-sentinel run-pipeline --window all",
    "uv run credit-sentinel retrain-challenger",
    "uv run credit-sentinel promote-challenger",
    "make mlflow",
    "uv run credit-sentinel scan-pii reports logs docs",
    "uv run credit-sentinel build-explainers --etapa 3",
]
PIPELINE_HINT = "uv run credit-sentinel run-pipeline --window all"
GATES_HINT = "uv run credit-sentinel retrain-challenger"


class LogField(BaseModel):
    """Um campo do evento de log."""

    name: str
    kind: str
    required: bool
    description: str


class RegistryVersion(BaseModel):
    """Uma versão do ``credit-scoring-model`` com seus aliases e tag."""

    version: int
    model: str
    aliases: list[str]
    tag: str | None = None


class RegistryState(BaseModel):
    """Estado do Model Registry em um momento da demonstração."""

    label: str
    note: str
    versions: list[RegistryVersion]


class PiiPattern(BaseModel):
    """Expressão regular do mascaramento, espelhada em JavaScript."""

    kind: str
    source: str


class PiiDemo(BaseModel):
    """Texto de demonstração (em pedaços) e as expressões que o mascaram."""

    fragments: list[str]
    patterns: list[PiiPattern]


class ScreenshotCard(BaseModel):
    """Print do MLflow embutido em base64."""

    key: str
    title: str
    caption: str
    data_uri: str
    size_bytes: int


class Etapa3Payload(BaseModel):
    """Dados da página da Etapa 3."""

    release: ReleaseInfo
    audit: AuditReport
    slo: SloConfig
    events: list[PipelineEvent]
    gates: GatesReport
    registry: list[RegistryState]
    screenshots: list[ScreenshotCard]
    log_fields: list[LogField]
    example_event: PipelineEvent
    pii: PiiDemo
    stages: list[str]
    acceptance: list[AcceptanceItem]
    commands: list[str]


def log_fields() -> list[LogField]:
    """Campos do evento de log, com os oito obrigatórios marcados.

    Returns:
        Um item por campo de ``PipelineEvent``, na ordem do modelo.
    """
    return [
        LogField(
            name=name,
            kind=_FIELD_TEXT[name][0],
            required=name in REQUIRED_FIELDS,
            description=_FIELD_TEXT[name][1],
        )
        for name in PipelineEvent.model_fields
    ]


def pii_demo() -> PiiDemo:
    """Texto de demonstração do mascaramento, dividido para não conter PII em claro.

    A página junta os pedaços no navegador; assim o HTML versionado nunca traz
    identificador, CPF ou e-mail por inteiro (o ``scan-pii`` da CI varre ``docs/``).

    Returns:
        Fragmentos do texto e as expressões regulares do logger.
    """
    patterns = [PiiPattern(kind=k, source=src) for k, src in pii_patterns()]
    return PiiDemo(fragments=_PII_FRAGMENTS, patterns=patterns)


def registry_states(gates: GatesReport) -> list[RegistryState]:
    """Estado do registry antes e depois da promoção, derivado do relatório de gates.

    Args:
        gates: Relatório com as versões do champion e do challenger.

    Returns:
        Dois estados: antes e depois (igual ao antes se os gates reprovaram).

    Raises:
        ValueError: Se o relatório não trouxer as versões do registry.
    """
    old, new = gates.champion_version, gates.challenger_version
    if old is None or new is None:
        msg = "o relatório não traz as versões do registry; rode retrain-challenger"
        raise ValueError(msg)
    before = RegistryState(
        label="Antes da promoção",
        note="O baseline é o champion e o challenger aguarda os guardrails.",
        versions=[
            RegistryVersion(
                version=old, model=gates.champion_model, aliases=["champion"]
            ),
            RegistryVersion(
                version=new, model=gates.challenger_model, aliases=["challenger"]
            ),
        ],
    )
    if not gates.passed:
        refused = before.model_copy(
            update={"label": "Depois", "note": "Promoção recusada: nenhum alias mudou."}
        )
        return [before, refused]
    after = RegistryState(
        label="Depois da promoção",
        note="O alias champion passou para a nova versão e a antiga foi aposentada.",
        versions=[
            RegistryVersion(
                version=old,
                model=gates.champion_model,
                aliases=[],
                tag=f"retired_reason={RETIRED_REASON}",
            ),
            RegistryVersion(
                version=new, model=gates.challenger_model, aliases=["champion"]
            ),
        ],
    )
    return [before, after]


def screenshot_cards(
    folder: Path, wanted: Sequence[tuple[str, str, str]]
) -> list[ScreenshotCard]:
    """Embute os prints do MLflow em base64.

    Args:
        folder: Pasta com os arquivos JPEG.
        wanted: ``(arquivo, título, legenda)`` na ordem de exibição.

    Returns:
        Um cartão por print.

    Raises:
        ValueError: Se algum arquivo passar de ``MAX_SCREENSHOT_BYTES``.
    """
    cards = []
    for name, title, caption in wanted:
        data = (folder / name).read_bytes()
        if len(data) > MAX_SCREENSHOT_BYTES:
            msg = f"{name} tem {len(data)} bytes; o limite é {MAX_SCREENSHOT_BYTES}"
            raise ValueError(msg)
        uri = "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")
        cards.append(
            ScreenshotCard(
                key=name,
                title=title,
                caption=caption,
                data_uri=uri,
                size_bytes=len(data),
            )
        )
    return cards


def _item(criterion: str, *, ok: bool, evidence: str) -> AcceptanceItem:
    return AcceptanceItem(criterion=criterion, ok=ok, evidence=evidence)


def _window(audit: AuditReport, name: str) -> str:
    result = next(w for w in audit.windows if w.window == name)
    return ", ".join(a.severity for a in result.slo.alerts)


def _log_items(events: list[PipelineEvent], folder: Path) -> list[AcceptanceItem]:
    pairs = {(e.window, e.stage, e.status) for e in events}
    outcomes = [
        ("mes_1", "contract_validation", "SUCCESS"),
        ("bad_batch", "contract_validation", "BLOCKED"),
        ("mes_3_crise", "slo_evaluation", "SUCCESS"),
    ]
    raw = (folder / SAMPLE_FILE).read_text(encoding="utf-8")
    return [
        _item(
            "O log traz SUCCESS (mes_1), BLOCKED (bad_batch) e os SLOs da crise",
            ok=all(o in pairs for o in outcomes),
            evidence=f"{len(events)} eventos na amostra versionada",
        ),
        _item(
            "Todo evento tem os oito campos obrigatórios",
            ok=all(f in e.model_dump() for e in events for f in REQUIRED_FIELDS),
            evidence="cada linha valida em PipelineEvent",
        ),
        _item(
            "A amostra de log não contém dados pessoais",
            ok=scan_for_pii(raw) == [],
            evidence="scan-pii sem achados em pipeline_events_sample.jsonl",
        ),
    ]


def _seconds(audit: AuditReport) -> float:
    slowest = max(sum(w.stage_latencies_ms.values()) for w in audit.windows)
    return slowest / 1000


def _alert_items(audit: AuditReport) -> list[AcceptanceItem]:
    crisis = next(w for w in audit.windows if w.window == "mes_3_crise")
    p2 = [a for a in crisis.slo.alerts if a.severity == "P2"]
    retrain = bool(p2) and "retreino" in p2[0].recommended_action.lower()
    budget = f"{crisis.slo.remaining_budget:.3f}".replace(".", ",")
    slowest = f"{_seconds(audit):.1f}".replace(".", ",")
    return [
        _item(
            "bad_batch gera o alerta P1",
            ok=_window(audit, "bad_batch") == "P1",
            evidence=f"alertas: {_window(audit, 'bad_batch')}",
        ),
        _item(
            "mes_1 gera só o P4 informativo",
            ok=_window(audit, "mes_1") == "P4",
            evidence=f"alertas: {_window(audit, 'mes_1')}",
        ),
        _item(
            "mes_2 gera ao menos um P3",
            ok="P3" in _window(audit, "mes_2"),
            evidence=f"alertas: {_window(audit, 'mes_2')}",
        ),
        _item(
            "mes_3_crise gera P2 com ação de retreino e orçamento negativo",
            ok=retrain and crisis.slo.remaining_budget < 0,
            evidence=f"saldo do orçamento {budget}",
        ),
        _item(
            "Cada janela leva menos de 120 s",
            ok=_seconds(audit) * 1000 < MAX_WINDOW_MS,
            evidence=f"maior janela: {slowest} s",
        ),
    ]


def _model_items(
    gates: GatesReport, slo: SloConfig, settings: Settings
) -> list[AcceptanceItem]:
    top3 = read_json(settings.models_dir / "baseline" / "feature_importance.json")[
        "top3"
    ]
    delta = gates.challenger.roc_auc - gates.champion.roc_auc
    return [
        _item(
            "O SLO2 observa só features do top-3 do modelo",
            ok=set(slo.slo("SLO2_estabilidade_psi").features) <= set(top3),
            evidence=", ".join(top3),
        ),
        _item(
            "Os cinco guardrails aprovam o challenger",
            ok=len(gates.results) == EXPECTED_GATES and gates.passed,
            evidence=", ".join(g.gate_id.split("_")[0] for g in gates.results),
        ),
        _item(
            "AUC do challenger pelo menos 0,02 acima do champion no holdout",
            ok=delta >= AUC_GAIN - 1e-9,
            evidence=f"{delta:+.4f}".replace(".", ","),
        ),
    ]


def _doc_items(
    slo: SloConfig, settings: Settings, folder: Path
) -> list[AcceptanceItem]:
    runbook = (settings.docs_dir / "runbook_incidentes.md").read_text(encoding="utf-8")
    prints = sorted(folder.glob("mlflow_*.jpg"))
    return [
        _item(
            "Os alertas apontam para seções existentes do runbook",
            ok=all(f'id="{rule.runbook_anchor}"' in runbook for rule in slo.alerts),
            evidence=", ".join(rule.runbook_anchor for rule in slo.alerts),
        ),
        _item(
            "Pelo menos quatro prints do MLflow, cada um com até 400 KB",
            ok=len(prints) >= MIN_PRINTS
            and all(p.stat().st_size <= MAX_SCREENSHOT_BYTES for p in prints),
            evidence=f"{len(prints)} prints em docs/evidencias/etapa3/",
        ),
        _item(
            "O painel de saúde foi gerado",
            ok=(settings.reports_dir / "health_dashboard.html").exists(),
            evidence="reports/health_dashboard.html",
        ),
    ]


def build_etapa3(settings: Settings, release: ReleaseInfo) -> Etapa3Payload:
    """Monta o payload da Etapa 3 a partir dos relatórios e da amostra de log.

    Args:
        settings: Configuração do projeto.
        release: Metadados da release.

    Returns:
        Payload validado.
    """
    folder = settings.docs_dir.joinpath(*EVIDENCE_DIR)
    audit_path = require(settings.reports_dir / "slo_audit_report.json", PIPELINE_HINT)
    audit = AuditReport.model_validate(read_json(audit_path))
    gates_path = require(settings.reports_dir / "gates_report.json", GATES_HINT)
    gates = GatesReport.model_validate(read_json(gates_path))
    slo = load_slo_config(settings.config_dir / "slo_config.yaml")
    events = read_events(require(folder / SAMPLE_FILE, PIPELINE_HINT))
    blocked = next(e for e in events if e.status == "BLOCKED")
    return Etapa3Payload(
        release=release,
        audit=audit,
        slo=slo,
        events=events,
        gates=gates,
        registry=registry_states(gates),
        screenshots=screenshot_cards(folder, SCREENSHOTS),
        log_fields=log_fields(),
        example_event=blocked,
        pii=pii_demo(),
        stages=list(STAGES),
        acceptance=[
            *_log_items(events, folder),
            *_alert_items(audit),
            *_model_items(gates, slo, settings),
            *_doc_items(slo, settings, folder),
        ],
        commands=ETAPA3_COMMANDS,
    )


__all__ = ["WINDOW_ORDER", "Etapa3Payload", "build_etapa3"]
