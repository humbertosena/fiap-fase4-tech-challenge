"""Orquestrador do pipeline: ingestão, contrato, inferência, drift, SLO e tracking.

Cada janela passa por seis estágios instrumentados (``StageTimer``). Um lote bloqueado
pelo contrato termina ``BLOCKED``: inferência e drift viram ``SKIPPED``, mas os SLOs
ainda são avaliados para que o alerta P1 exista. O bloqueio é um **resultado esperado**
do pipeline; só insumo ausente ou erro inesperado derrubam a execução. Os hooks de
tracking (MLflow) são opcionais e uma falha neles vira ``FAILED`` no estágio
``tracking``, sem interromper o resto.
"""

from __future__ import annotations

import contextlib
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd  # noqa: TC002 - anotações de métodos privados
from pydantic import BaseModel

from credit_sentinel.contracts.commands import display_path
from credit_sentinel.contracts.schema import build_inference_schema
from credit_sentinel.contracts.validator import (
    BatchContext,
    CircuitBreakerOpenError,
    ContractViolationError,
    ValidationPolicy,
    ValidationReport,
    validate_batch,
)
from credit_sentinel.drift.detection import (
    CONTROL_WINDOW,
    Baseline,
    MissingInputError,
    control_auc,
    evidently_window,
    load_baseline,
    model_frame,
    window_path,
)
from credit_sentinel.drift.predictions import (
    PerformanceSummary,
    PredictionSummary,
    evaluate_performance,
    score_window,
    summarize_predictions,
)
from credit_sentinel.drift.simulator import WINDOWS
from credit_sentinel.drift.statistical_tests import DriftSummary, run_drift_suite
from credit_sentinel.ml.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
)
from credit_sentinel.observability.logger import (
    PipelineLogger,
    Stage,
    StageTimer,
    get_pipeline_logger,
    read_events,
)
from credit_sentinel.observability.slo_monitor import (
    Alert,
    Decision,
    SloConfig,
    SloEvaluation,
    SloObservation,
    evaluate_slos,
    load_slo_config,
)
from credit_sentinel.utils.clock import utc_now
from credit_sentinel.utils.io import read_parquet, write_json, write_text

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence
    from pathlib import Path

    from credit_sentinel.settings import Settings

BAD_BATCH = "bad_batch"
BAD_BATCH_FILE = "bad_batch.parquet"
WINDOW_ORDER = (BAD_BATCH, *WINDOWS)
SLO_CONFIG_FILE = "slo_config.yaml"
LOG_FILE = "pipeline_events.jsonl"
QUICK_ROWS = 1000
_BLOCKING = (ContractViolationError, CircuitBreakerOpenError)
_SKIP_REASON = "lote bloqueado pelo contrato de dados"


class HookFailureError(Exception):
    """Um ou mais hooks de tracking falharam (o pipeline segue)."""


class WindowRunResult(BaseModel):
    """Tudo o que o pipeline produziu para uma janela.

    Attributes:
        window: Nome da janela (``bad_batch``, ``mes_1``...).
        run_id: Execução do pipeline.
        decision: ``BLOCKED`` se o contrato bloqueou o lote.
        validation: Relatório do contrato de dados.
        drift_ref: Caminho do relatório HTML do Evidently, se gerado.
        share_drifted_evidently: Fração das colunas com drift segundo o Evidently.
        drift: Testes nativos de drift (``None`` em lote bloqueado).
        predictions: Comportamento dos scores (``None`` em lote bloqueado).
        performance: Performance contra a verdade de campo.
        slo: Avaliação dos SLOs e alertas.
        stage_latencies_ms: Duração de cada estágio.
        stage_status: Desfecho de cada estágio.
        model_version: Versão do modelo avaliado.
    """

    window: str
    run_id: str
    decision: Decision
    validation: ValidationReport
    drift_ref: str | None
    share_drifted_evidently: float | None = None
    drift: DriftSummary | None
    predictions: PredictionSummary | None
    performance: PerformanceSummary | None
    slo: SloEvaluation
    stage_latencies_ms: dict[str, int]
    stage_status: dict[str, str]
    model_version: str


class AuditReport(BaseModel):
    """Relatório de auditoria de uma execução (``slo_audit_report.json``).

    Attributes:
        generated_at: Instante da geração (respeita ``SOURCE_DATE_EPOCH``).
        run_id: Execução do pipeline.
        model_version: Versão do modelo avaliado.
        quick: Se a execução foi no modo rápido (amostra e sem Evidently).
        reference_auc: AUC de validação, referência do orçamento do SLO4.
        windows: Resultado por janela, na ordem de execução.
        totals: Contagens de decisões, alertas e janelas estouradas.
        alerts: Todos os alertas, do mais grave ao mais leve.
    """

    generated_at: datetime
    run_id: str
    model_version: str
    quick: bool
    reference_auc: float
    windows: list[WindowRunResult]
    totals: dict[str, Any]
    alerts: list[Alert]


class TrackingHook(Protocol):
    """Destino opcional dos resultados de cada janela (ex.: MLflow)."""

    def on_window_complete(
        self, result: WindowRunResult, artifacts: dict[str, Path]
    ) -> None:
        """Recebe o resultado da janela e os arquivos que o acompanham."""


@dataclass(frozen=True)
class PipelineContext:
    """Insumos compartilhados por todas as janelas de uma execução.

    Attributes:
        settings: Configuração do projeto.
        baseline: Modelo e referências.
        config: SLOs e regras de alerta.
        logger: Logger JSON da execução.
        hooks: Hooks de tracking (vazio = sem tracking).
        quick: Amostra de 1.000 linhas e sem Evidently.
        control_auc: AUC da janela de controle, para a queda contra o controle.
    """

    settings: Settings
    baseline: Baseline
    config: SloConfig
    logger: PipelineLogger
    hooks: Sequence[TrackingHook] = ()
    quick: bool = False
    control_auc: float | None = None


def window_file(settings: Settings, name: str) -> Path:
    """Arquivo parquet de uma janela, inclusive o lote corrompido.

    Args:
        settings: Configuração do projeto.
        name: ``bad_batch`` ou o nome de uma janela de produção.

    Returns:
        Caminho do parquet.
    """
    if name == BAD_BATCH:
        return settings.corrupted_dir / BAD_BATCH_FILE
    return window_path(settings, name)


def _alert_summary(evaluation: SloEvaluation) -> str:
    parts = [f"{a.severity} ({a.slo_id or 'lote'})" for a in evaluation.alerts]
    return f"alertas: {', '.join(parts)}"


class _WindowRun:
    """Executa os estágios de uma janela e acumula o que eles produzem."""

    def __init__(self, name: str, ctx: PipelineContext) -> None:
        self.name = name
        self.ctx = ctx
        self.latencies: dict[str, int] = {}
        self.status: dict[str, str] = {}
        self.scores: pd.DataFrame | None = None
        self.predictions: PredictionSummary | None = None
        self.performance: PerformanceSummary | None = None
        self.drift: DriftSummary | None = None
        self.drift_ref: str | None = None
        self.share_drifted: float | None = None

    @contextlib.contextmanager
    def _stage(self, stage: Stage) -> Iterator[StageTimer]:
        timer = StageTimer(self.ctx.logger, stage, self.name)
        try:
            with timer:
                yield timer
        finally:
            self.latencies[stage] = timer.latency_ms
            self.status[stage] = timer.status or "SKIPPED"

    def _skip(self, stage: Stage, reason: str) -> None:
        StageTimer(self.ctx.logger, stage, self.name).skip(reason)
        self.latencies[stage] = 0
        self.status[stage] = "SKIPPED"

    def _ingest(self) -> pd.DataFrame:
        settings = self.ctx.settings
        path = window_file(settings, self.name)
        with self._stage("ingestion") as timer:
            if not path.exists():
                where = display_path(path, settings.project_root)
                msg = f"janela {self.name} não encontrada em {where}"
                raise MissingInputError(msg)
            frame = read_parquet(path)
            if self.ctx.quick and len(frame) > QUICK_ROWS:
                frame = frame.sample(n=QUICK_ROWS, random_state=settings.seed)
            timer.records_processed = len(frame)
            timer.details = {
                "arquivo": display_path(path, settings.project_root),
                "amostra_rapida": self.ctx.quick,
            }
        return frame

    @staticmethod
    def _describe(timer: StageTimer, report: ValidationReport) -> None:
        timer.rejected_records = report.rejected_records
        timer.details = {
            "regras": sorted({v.rule_id for v in report.violations}),
            "rejected_ratio": round(report.rejected_ratio, 4),
        }
        if report.decision == "SUCCESS" and report.rejected_records:
            timer.message = f"{report.rejected_records} linha(s) em quarentena"

    def _check(
        self, frame: pd.DataFrame, timer: StageTimer
    ) -> tuple[pd.DataFrame, ValidationReport]:
        settings = self.ctx.settings
        context = BatchContext(
            batch_name=self.name,
            run_id=f"val-{self.ctx.logger.run_id[:8]}-{self.name}",
            quarantine_dir=settings.quarantine_dir,
            pii_salt=settings.pii_salt,
        )
        try:
            valid, report = validate_batch(
                frame,
                schema=build_inference_schema(),
                policy=ValidationPolicy(),
                context=context,
            )
        except _BLOCKING as exc:
            self._describe(timer, exc.report)
            raise
        self._describe(timer, report)
        return valid, report

    def _validate(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, ValidationReport]:
        try:
            with self._stage("contract_validation") as timer:
                timer.records_processed = len(frame)
                return self._check(frame, timer)
        except _BLOCKING as exc:
            return frame.iloc[0:0], exc.report

    def _infer(self, valid: pd.DataFrame) -> None:
        ctx = self.ctx
        bundle, baseline = ctx.baseline.bundle, ctx.baseline
        with self._stage("inference") as timer:
            salt = ctx.settings.pii_salt.get_secret_value()
            self.scores = score_window(bundle, valid, salt=salt)
            self.predictions = summarize_predictions(
                baseline.holdout_scores, self.scores, self.name
            )
            control = None if self.name == CONTROL_WINDOW else ctx.control_auc
            self.performance = evaluate_performance(
                valid, self.scores, bundle, self.name, control
            )
            timer.records_processed = len(valid)
            timer.details = {
                "roc_auc": round(self.performance.roc_auc, 4),
                "approval_rate": round(self.predictions.approval_rate, 4),
                "psi_score": round(self.predictions.psi_score_vs_reference, 4),
            }

    def _detect(self, valid: pd.DataFrame) -> None:
        ctx = self.ctx
        with self._stage("drift_detection") as timer:
            self.drift = run_drift_suite(
                ctx.baseline.reference_features,
                build_features(valid),
                numeric=NUMERIC_FEATURES,
                categorical=CATEGORICAL_FEATURES,
                window=self.name,
            )
            timer.records_processed = len(valid)
            timer.details = {
                "drifted_features": self.drift.drifted_features,
                "psi_critical_features": self.drift.psi_critical_features,
            }
            if not ctx.quick and self.scores is not None:
                current = model_frame(valid, self.scores)
                found = evidently_window(ctx.settings, ctx.baseline, self.name, current)
                self.drift_ref = found.html
                self.share_drifted = found.share_drifted
                timer.details["share_drifted_evidently"] = round(found.share_drifted, 4)

    def _evaluate(self, report: ValidationReport) -> SloEvaluation:
        ctx = self.ctx
        with self._stage("slo_evaluation") as timer:
            observation = SloObservation(
                validation=report,
                drift=self.drift,
                performance=self.performance,
                latency_ms=sum(self.latencies.values()),
                reference_auc=ctx.baseline.bundle.metrics.roc_auc,
                run_id=ctx.logger.run_id,
            )
            evaluation = evaluate_slos(ctx.config, observation)
            timer.records_processed = report.total_records
            timer.message = _alert_summary(evaluation)
            timer.details = {
                "remaining_budget": round(evaluation.remaining_budget, 4),
                "slos": {r.slo_id: r.status for r in evaluation.results},
                "alerts": [
                    {"severity": a.severity, "slo_id": a.slo_id}
                    for a in evaluation.alerts
                ],
            }
        return evaluation

    def _result(
        self, report: ValidationReport, evaluation: SloEvaluation
    ) -> WindowRunResult:
        return WindowRunResult(
            window=self.name,
            run_id=self.ctx.logger.run_id,
            decision=report.decision,
            validation=report,
            drift_ref=self.drift_ref,
            share_drifted_evidently=self.share_drifted,
            drift=self.drift,
            predictions=self.predictions,
            performance=self.performance,
            slo=evaluation,
            stage_latencies_ms=dict(self.latencies),
            stage_status=dict(self.status),
            model_version=self.ctx.logger.model_version,
        )

    def _artifacts(self, result: WindowRunResult) -> dict[str, Path]:
        settings, logger = self.ctx.settings, self.ctx.logger
        folder = settings.logs_dir / "runs" / logger.run_id / self.name
        files = {
            "validation_report": write_json(
                result.validation, folder / "validation_report.json"
            ),
            "slo_evaluation": write_json(result.slo, folder / "slo_evaluation.json"),
        }
        lines = [
            event.model_dump_json(exclude_none=True)
            for event in read_events(logger.log_path)
            if event.run_id == logger.run_id and event.window == self.name
        ]
        files["window_events"] = write_text(
            "\n".join(lines) + "\n", folder / "events.jsonl"
        )
        if self.drift_ref is not None:
            html = settings.project_root / self.drift_ref
            files["evidently_html"] = html
            files["evidently_json"] = html.with_suffix(".json")
        return files

    def _run_hooks(
        self, result: WindowRunResult, artifacts: dict[str, Path]
    ) -> list[str]:
        failures = []
        for hook in self.ctx.hooks:
            try:
                hook.on_window_complete(result, artifacts)
            except Exception as exc:  # noqa: BLE001 - hook falho não derruba o pipeline
                failures.append(f"{type(hook).__name__}: {exc}")
        return failures

    def _track(self, result: WindowRunResult) -> WindowRunResult:
        if not self.ctx.hooks:
            self._skip("tracking", "nenhum hook de tracking configurado")
        else:
            artifacts = self._artifacts(result)
            with (
                contextlib.suppress(HookFailureError),
                self._stage("tracking") as timer,
            ):
                timer.records_processed = result.validation.total_records
                failures = self._run_hooks(result, artifacts)
                if failures:
                    raise HookFailureError("; ".join(failures))
        return result.model_copy(
            update={
                "stage_latencies_ms": dict(self.latencies),
                "stage_status": dict(self.status),
            }
        )

    def run(self) -> WindowRunResult:
        frame = self._ingest()
        valid, report = self._validate(frame)
        if report.decision == "BLOCKED":
            self._skip("inference", _SKIP_REASON)
            self._skip("drift_detection", _SKIP_REASON)
        else:
            self._infer(valid)
            self._detect(valid)
        evaluation = self._evaluate(report)
        return self._track(self._result(report, evaluation))


def run_window(window_name: str, context: PipelineContext) -> WindowRunResult:
    """Executa os seis estágios do pipeline para uma janela.

    Args:
        window_name: ``bad_batch``, ``mes_1``, ``mes_2`` ou ``mes_3_crise``.
        context: Insumos compartilhados da execução.

    Returns:
        Resultado completo da janela, inclusive o desfecho de cada estágio.

    Raises:
        MissingInputError: Se o parquet da janela não existir.
    """
    return _WindowRun(window_name, context).run()


def _totals(results: list[WindowRunResult]) -> dict[str, Any]:
    alerts = [alert for r in results for alert in r.slo.alerts]
    return {
        "windows": len(results),
        "success": sum(r.decision == "SUCCESS" for r in results),
        "blocked": sum(r.decision == "BLOCKED" for r in results),
        "alerts_by_severity": {
            severity: sum(a.severity == severity for a in alerts)
            for severity in ("P1", "P2", "P3", "P4")
        },
        "windows_over_budget": [
            r.window for r in results if r.slo.remaining_budget < 0
        ],
    }


def run_pipeline(
    windows: Sequence[str],
    *,
    settings: Settings,
    quick: bool = False,
    hook_factory: Callable[[str, str], Sequence[TrackingHook]] | None = None,
) -> AuditReport:
    """Executa o pipeline nas janelas pedidas e consolida a auditoria.

    Args:
        windows: Janelas, na ordem de execução.
        settings: Configuração do projeto.
        quick: Amostra de 1.000 linhas por janela e sem Evidently.
        hook_factory: Recebe ``(run_id, model_version)`` e devolve os hooks de
            tracking; ``None`` desliga o tracking.

    Returns:
        Relatório de auditoria da execução.

    Raises:
        MissingInputError: Se faltar modelo, referência ou janela.
        ValueError: Se alguma janela não existir no projeto.
    """
    unknown = [w for w in windows if w not in WINDOW_ORDER]
    if unknown:
        msg = f"janela desconhecida: {', '.join(unknown)}"
        raise ValueError(msg)
    baseline = load_baseline(settings)
    config = load_slo_config(settings.config_dir / SLO_CONFIG_FILE)
    run_id, version = str(uuid.uuid4()), baseline.bundle.version
    hooks = () if hook_factory is None else tuple(hook_factory(run_id, version))
    with get_pipeline_logger(run_id, version, settings.logs_dir / LOG_FILE) as logger:
        context = PipelineContext(
            settings=settings,
            baseline=baseline,
            config=config,
            logger=logger,
            hooks=hooks,
            quick=quick,
            control_auc=control_auc(settings, baseline),
        )
        results = [run_window(name, context) for name in windows]
    alerts = sorted(
        (a for r in results for a in r.slo.alerts), key=lambda a: a.severity
    )
    return AuditReport(
        generated_at=utc_now(),
        run_id=run_id,
        model_version=version,
        quick=quick,
        reference_auc=baseline.bundle.metrics.roc_auc,
        windows=results,
        totals=_totals(results),
        alerts=alerts,
    )
