"""Data SLOs, error budget normalizado e classificação de alertas P1 a P4.

``config/slo_config.yaml`` declara cinco SLIs com seus limites; este módulo carrega e
valida a configuração (``SloConfig``) e, nas funções de avaliação, transforma as
medições de um lote em consumo de orçamento, estado por SLO e alertas.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, Field, model_validator

from credit_sentinel.utils.io import read_yaml

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from credit_sentinel.contracts.validator import ValidationReport
    from credit_sentinel.drift.predictions import PerformanceSummary
    from credit_sentinel.drift.statistical_tests import (
        DriftSummary,
        NumericDriftResult,
    )

Operator = Literal["<=", ">="]
SliKind = Literal[
    "rejected_ratio",
    "psi_top_features",
    "ks_failing_share",
    "roc_auc",
    "batch_latency_ms",
]
Severity = Literal["P1", "P2", "P3", "P4"]
Trigger = Literal["decision_blocked", "breached", "budget_at_least", "otherwise"]
Status = Literal["OK", "WARN", "BREACHED"]
Decision = Literal["SUCCESS", "BLOCKED"]
_FEATURE_SLIS = ("psi_top_features", "ks_failing_share")


class Objective(BaseModel):
    """Limite de um SLO.

    Attributes:
        op: ``<=`` (quanto menor, melhor) ou ``>=`` (quanto maior, melhor).
        value: Valor limite do SLI.
    """

    op: Operator
    value: float


class SloSpec(BaseModel):
    """Um SLO: o SLI medido, a meta e as features observadas.

    Attributes:
        id: Identificador estável (``SLO1_integridade``...).
        sli: Indicador medido.
        objective: Meta do SLO.
        features: Features que alimentam o SLI (quando aplicável).
        budget_note: Leitura do orçamento em linguagem de negócio.
    """

    id: str
    sli: SliKind
    objective: Objective
    features: list[str] = Field(default_factory=list)
    budget_note: str | None = None

    @model_validator(mode="after")
    def _check_limit_and_features(self) -> Self:
        if self.objective.op == "<=" and self.objective.value <= 0:
            msg = f"{self.id}: o limite de um objetivo '<=' deve ser maior que zero"
            raise ValueError(msg)
        if self.sli in _FEATURE_SLIS and not self.features:
            msg = f"{self.id}: o SLI {self.sli} exige a lista de features"
            raise ValueError(msg)
        return self


class AlertRule(BaseModel):
    """Regra de alerta.

    Attributes:
        severity: Severidade P1 (mais grave) a P4 (informativo).
        when: Gatilho da regra.
        action: Ação recomendada, repetida em cada alerta gerado.
        runbook_anchor: Âncora da seção do ``docs/runbook_incidentes.md``.
        value: Consumo mínimo (só para ``budget_at_least``).
        slos: SLOs considerados pelo gatilho; vazio = todos.
    """

    severity: Severity
    when: Trigger
    action: str
    runbook_anchor: str
    value: float | None = None
    slos: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _budget_rule_needs_a_value(self) -> Self:
        if self.when == "budget_at_least" and self.value is None:
            msg = f"{self.severity}: o gatilho budget_at_least exige `value`"
            raise ValueError(msg)
        return self


class SloConfig(BaseModel):
    """Conteúdo de ``config/slo_config.yaml``.

    Attributes:
        window_batches: Lotes da janela de avaliação (quatro neste projeto).
        warn_at: Consumo a partir do qual o SLO fica em ``WARN``.
        breach_at: Consumo a partir do qual o SLO fica ``BREACHED``.
        slos: SLOs, na ordem de apresentação.
        alerts: Regras de alerta, da mais grave para a mais leve.
    """

    window_batches: int = 4
    warn_at: float = Field(default=0.70, gt=0)
    breach_at: float = Field(default=1.0, gt=0)
    slos: list[SloSpec]
    alerts: list[AlertRule]

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        ids = [s.id for s in self.slos]
        duplicated = sorted({i for i in ids if ids.count(i) > 1})
        if duplicated:
            msg = f"SLO duplicado: {', '.join(duplicated)}"
            raise ValueError(msg)
        unknown = {s for rule in self.alerts for s in rule.slos} - set(ids)
        if unknown:
            msg = f"alerta referencia SLO inexistente: {', '.join(sorted(unknown))}"
            raise ValueError(msg)
        if self.warn_at >= self.breach_at:
            msg = "warn_at deve ser menor que breach_at"
            raise ValueError(msg)
        return self

    def slo(self, slo_id: str) -> SloSpec:
        """Busca um SLO pelo identificador.

        Args:
            slo_id: Identificador (ex.: ``SLO2_estabilidade_psi``).

        Returns:
            A especificação do SLO.

        Raises:
            KeyError: Se o identificador não existir.
        """
        for spec in self.slos:
            if spec.id == slo_id:
                return spec
        raise KeyError(slo_id)


def load_slo_config(path: Path) -> SloConfig:
    """Carrega e valida o arquivo de SLOs.

    Args:
        path: Caminho de ``slo_config.yaml``.

    Returns:
        Configuração validada.
    """
    return SloConfig.model_validate(read_yaml(path))


class SloResult(BaseModel):
    """Medição de um SLO em um lote.

    Attributes:
        slo_id: Identificador do SLO.
        sli: Indicador medido.
        observed: Valor observado do SLI.
        threshold: Limite do SLO.
        op: Direção do objetivo.
        consumption: Fração do error budget consumida (1,0 = no limite).
        status: ``OK``, ``WARN`` ou ``BREACHED``.
        reference: Valor de referência dos objetivos ``>=`` (AUC de validação).
    """

    slo_id: str
    sli: SliKind
    observed: float
    threshold: float
    op: Operator
    consumption: float
    status: Status
    reference: float | None = None


class Alert(BaseModel):
    """Alerta operacional de um lote.

    Attributes:
        severity: P1 (mais grave) a P4 (informativo).
        slo_id: SLO que originou o alerta (``None`` em alertas do lote).
        window: Janela de dados.
        message: Descrição do que aconteceu.
        recommended_action: O que fazer.
        runbook_anchor: Âncora da seção do runbook.
    """

    severity: Severity
    slo_id: str | None
    window: str
    message: str
    recommended_action: str
    runbook_anchor: str


class SloEvaluation(BaseModel):
    """Resultado da avaliação de SLOs de um lote.

    Attributes:
        window: Janela de dados.
        run_id: Execução do pipeline.
        results: Medições dos SLOs avaliados.
        alerts: Alertas, do mais grave ao mais leve.
        remaining_budget: ``1 - média dos consumos`` (negativo = estourado).
        decision: ``BLOCKED`` quando o contrato bloqueou o lote.
        blocked_reason: Motivo do bloqueio, se houver.
    """

    window: str
    run_id: str
    results: list[SloResult]
    alerts: list[Alert]
    remaining_budget: float
    decision: Decision
    blocked_reason: str | None = None


@dataclass(frozen=True)
class SloObservation:
    """Tudo o que se mediu em um lote, entrada de ``evaluate_slos``.

    Attributes:
        validation: Relatório do contrato (sempre disponível).
        drift: Testes nativos de drift; ``None`` se o lote não chegou lá.
        performance: Performance contra a verdade de campo, se existir.
        latency_ms: Duração do processamento do lote.
        reference_auc: AUC de validação, referência do orçamento do SLO4.
        run_id: Execução do pipeline.
    """

    validation: ValidationReport
    drift: DriftSummary | None
    performance: PerformanceSummary | None
    latency_ms: int
    reference_auc: float
    run_id: str


def consumption(
    op: Operator, observed: float, threshold: float, *, reference: float | None = None
) -> float:
    """Fração do error budget consumida por um valor observado.

    Args:
        op: Direção do objetivo.
        observed: Valor observado do SLI.
        threshold: Limite do SLO.
        reference: Valor ideal, obrigatório nos objetivos ``>=``.

    Returns:
        ``observado / limite`` para ``<=``; ``(referência - observado) /
        (referência - limite)``, nunca negativo, para ``>=``.

    Raises:
        ValueError: Objetivo ``>=`` sem referência acima do limite.
    """
    if op == "<=":
        return observed / threshold
    if reference is None or reference <= threshold:
        msg = (
            f"objetivo '>=' exige uma referência acima do limite (referência"
            f" {reference}, limite {threshold}): o modelo já nasce fora do SLO"
        )
        raise ValueError(msg)
    return max(0.0, (reference - observed) / (reference - threshold))


def slo_status(value: float, config: SloConfig) -> Status:
    """Estado do SLO para um consumo de orçamento.

    Args:
        value: Consumo (1,0 = limite).
        config: Configuração com ``warn_at`` e ``breach_at``.

    Returns:
        ``BREACHED`` a partir de ``breach_at``, ``WARN`` a partir de ``warn_at``.
    """
    if value >= config.breach_at:
        return "BREACHED"
    return "WARN" if value >= config.warn_at else "OK"


def _numeric_for(spec: SloSpec, drift: DriftSummary) -> list[NumericDriftResult]:
    found = {item.feature: item for item in drift.numeric}
    missing = [name for name in spec.features if name not in found]
    if missing:
        msg = f"{spec.id}: feature ausente do resumo de drift: {', '.join(missing)}"
        raise ValueError(msg)
    return [found[name] for name in spec.features]


def _read_psi(spec: SloSpec, obs: SloObservation) -> float | None:
    if obs.drift is None:
        return None
    return max(item.psi for item in _numeric_for(spec, obs.drift))


def _read_ks_share(spec: SloSpec, obs: SloObservation) -> float | None:
    if obs.drift is None:
        return None
    items = _numeric_for(spec, obs.drift)
    return sum(item.ks_drift_bonferroni for item in items) / len(items)


def _read_auc(_spec: SloSpec, obs: SloObservation) -> float | None:
    return None if obs.performance is None else obs.performance.roc_auc


_READERS: dict[SliKind, Callable[[SloSpec, SloObservation], float | None]] = {
    "rejected_ratio": lambda _spec, obs: obs.validation.rejected_ratio,
    "psi_top_features": _read_psi,
    "ks_failing_share": _read_ks_share,
    "roc_auc": _read_auc,
    "batch_latency_ms": lambda _spec, obs: float(obs.latency_ms),
}


def _measure(
    spec: SloSpec, observed: float, obs: SloObservation, config: SloConfig
) -> SloResult:
    objective = spec.objective
    reference = obs.reference_auc if objective.op == ">=" else None
    spent = consumption(
        objective.op, observed, objective.value, reference=obs.reference_auc
    )
    return SloResult(
        slo_id=spec.id,
        sli=spec.sli,
        observed=observed,
        threshold=objective.value,
        op=objective.op,
        consumption=spent,
        status=slo_status(spent, config),
        reference=reference,
    )


def _remaining(results: list[SloResult]) -> float:
    if not results:
        return 1.0
    return 1.0 - sum(r.consumption for r in results) / len(results)


def evaluate_slos(config: SloConfig, observation: SloObservation) -> SloEvaluation:
    """Avalia os SLOs de um lote e classifica os alertas.

    SLOs cujo insumo o lote não produziu (drift e AUC de um lote bloqueado) ficam
    de fora da média do orçamento, em vez de contar como zero.

    Args:
        config: SLOs e regras de alerta.
        observation: Medições do lote.

    Returns:
        Avaliação com resultados, alertas e saldo do orçamento.
    """
    results = []
    for spec in config.slos:
        observed = _READERS[spec.sli](spec, observation)
        if observed is not None:
            results.append(_measure(spec, observed, observation, config))
    validation = observation.validation
    evaluation = SloEvaluation(
        window=validation.batch_name,
        run_id=observation.run_id,
        results=results,
        alerts=[],
        remaining_budget=_remaining(results),
        decision=validation.decision,
        blocked_reason=validation.blocked_reason,
    )
    return evaluation.model_copy(update={"alerts": classify_alerts(evaluation, config)})


def _first_rule(config: SloConfig, trigger: Trigger) -> AlertRule | None:
    return next((rule for rule in config.alerts if rule.when == trigger), None)


def _alert(
    rule: AlertRule, window: str, message: str, slo_id: str | None = None
) -> Alert:
    return Alert(
        severity=rule.severity,
        slo_id=slo_id,
        window=window,
        message=message,
        recommended_action=rule.action,
        runbook_anchor=rule.runbook_anchor,
    )


def _fires(rule: AlertRule, result: SloResult) -> bool:
    if rule.slos and result.slo_id not in rule.slos:
        return False
    if rule.when == "breached":
        return result.status == "BREACHED"
    if rule.when == "budget_at_least":
        return result.consumption >= (rule.value or 0.0)
    return False


def _describe(result: SloResult) -> str:
    return (
        f"{result.slo_id}: observado {result.observed:.4g} (objetivo {result.op}"
        f" {result.threshold:g}), {result.consumption:.0%} do error budget"
    )


def _slo_alerts(evaluation: SloEvaluation, config: SloConfig) -> list[Alert]:
    alerts = []
    for result in evaluation.results:
        rule = next((r for r in config.alerts if _fires(r, result)), None)
        if rule is not None:
            alerts.append(
                _alert(rule, evaluation.window, _describe(result), result.slo_id)
            )
    return alerts


def classify_alerts(evaluation: SloEvaluation, config: SloConfig) -> list[Alert]:
    """Classifica os alertas de um lote (P1 a P4).

    Um lote bloqueado gera só o P1. Fora isso, cada SLO gera no máximo um alerta,
    na maior severidade em que se enquadra; sem nenhum, sai um P4 informativo.

    Args:
        evaluation: Resultados dos SLOs do lote.
        config: Regras de alerta.

    Returns:
        Alertas ordenados da maior para a menor severidade.
    """
    blocked = _first_rule(config, "decision_blocked")
    if evaluation.decision == "BLOCKED" and blocked is not None:
        reason = evaluation.blocked_reason or "contrato violado"
        message = f"Lote {evaluation.window} bloqueado pelo contrato: {reason}"
        return [_alert(blocked, evaluation.window, message)]
    alerts = _slo_alerts(evaluation, config)
    informative = _first_rule(config, "otherwise")
    if not alerts and informative is not None:
        saldo = f"{evaluation.remaining_budget:.0%} de saldo no error budget"
        message = f"Lote {evaluation.window} dentro dos objetivos ({saldo})"
        alerts = [_alert(informative, evaluation.window, message)]
    return sorted(alerts, key=lambda alert: alert.severity)
