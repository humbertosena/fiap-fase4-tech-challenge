"""Data SLOs, error budget normalizado e classificação de alertas P1 a P4.

``config/slo_config.yaml`` declara cinco SLIs com seus limites; este módulo carrega e
valida a configuração (``SloConfig``) e, nas funções de avaliação, transforma as
medições de um lote em consumo de orçamento, estado por SLO e alertas.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, Self

from pydantic import BaseModel, Field, model_validator

from credit_sentinel.utils.io import read_yaml

if TYPE_CHECKING:
    from pathlib import Path

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
    def _limit_must_be_positive(self) -> Self:
        if self.objective.op == "<=" and self.objective.value <= 0:
            msg = f"{self.id}: o limite de um objetivo '<=' deve ser maior que zero"
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
