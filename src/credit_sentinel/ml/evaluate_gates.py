"""Guardrails de promoção do challenger (resultado em ``reports/gates_report.json``).

O challenger só vira champion se passar nos cinco gates, medidos no holdout temporal:

1. G1: ROC-AUC pelo menos 0,02 acima do champion;
2. G2: Brier não pior que o do champion;
3. G3: contrato de entrada (assinatura) idêntico;
4. G4: razão de impacto na taxa de aprovação de cada grupo em [0,80, 1,25];
5. G5: inferência de 5.000 linhas em até 5 s.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from credit_sentinel.data import dictionary as dd
from credit_sentinel.drift.predictions import decision_threshold
from credit_sentinel.governance.fairness_audit import (
    IMPACT_LOWER,
    IMPACT_UPPER,
    GroupImpact,
    approval_impact_ratios,
)
from credit_sentinel.ml.evaluation import classification_metrics
from credit_sentinel.ml.train_baseline import ModelBundle, predict_proba

if TYPE_CHECKING:
    import pandas as pd

    from credit_sentinel.settings import Settings

GATE_IDS = (
    "G1_auc_vs_champion",
    "G2_brier",
    "G3_contrato_de_entrada",
    "G4_fairness_aprovacao",
    "G5_latencia",
)
MIN_AUC_GAIN = 0.02
MAX_LATENCY_S = 5.0
LATENCY_ROWS = 5000
_EPS = 1e-9


class GateResult(BaseModel):
    """Resultado de um guardrail.

    Attributes:
        gate_id: Identificador do gate.
        passed: Se o gate aprovou.
        observed: Valor medido.
        threshold: Limite (inferior, nos gates com faixa).
        threshold_upper: Limite superior, quando o gate é uma faixa.
        detail: Leitura em texto.
    """

    gate_id: str
    passed: bool
    observed: float
    threshold: float
    threshold_upper: float | None = None
    detail: str


class ModelMetrics(BaseModel):
    """Métricas de um modelo no holdout temporal.

    Attributes:
        roc_auc: ROC-AUC.
        brier: Erro de calibração.
        f1: F1 no limiar operacional do próprio modelo.
        approval_rate: Fração aprovada no limiar operacional.
    """

    roc_auc: float
    brier: float
    f1: float
    approval_rate: float


class TrainingProtocol(BaseModel):
    """Como o challenger foi treinado e avaliado.

    Attributes:
        sliding_months: Meses recentes da referência usados no treino.
        reference_start: Primeiro dia da referência recente.
        reference_rows: Linhas da referência recente.
        windows: Linhas por janela de produção usada no treino.
        training_rows: Total de linhas de treino.
        crisis_train_until: Último dia da crise incluído no treino.
        holdout_window: Janela do holdout temporal.
        holdout_rows: Linhas do holdout.
        holdout_start: Primeiro dia do holdout.
        holdout_end: Último dia do holdout.
    """

    sliding_months: int
    reference_start: str
    reference_rows: int
    windows: dict[str, int]
    training_rows: int
    crisis_train_until: str
    holdout_window: str
    holdout_rows: int
    holdout_start: str
    holdout_end: str


class GatesReport(BaseModel):
    """Conteúdo de ``reports/gates_report.json``.

    Attributes:
        passed: ``True`` somente se todos os gates aprovaram.
        results: Um resultado por gate, em ordem.
        champion_model: Versão do modelo champion.
        challenger_model: Versão do modelo challenger.
        champion_version: Versão no Model Registry (preenchida ao registrar).
        challenger_version: Versão do challenger no registry (idem).
        holdout_window: Janela do holdout.
        holdout_rows: Linhas do holdout.
        champion: Métricas do champion no holdout.
        challenger: Métricas do challenger no holdout.
        fairness: Razões de impacto por grupo do challenger.
        protocol: Protocolo de treino e avaliação (preenchido pelo retreino).
    """

    passed: bool
    results: list[GateResult]
    champion_model: str
    challenger_model: str
    champion_version: int | None = None
    challenger_version: int | None = None
    holdout_window: str
    holdout_rows: int
    champion: ModelMetrics
    challenger: ModelMetrics
    fairness: list[GroupImpact]
    protocol: TrainingProtocol | None = None


def gate_auc(champion_auc: float, challenger_auc: float) -> GateResult:
    """G1: o challenger precisa superar o champion em ``MIN_AUC_GAIN`` de AUC.

    Args:
        champion_auc: ROC-AUC do champion no holdout.
        challenger_auc: ROC-AUC do challenger no holdout.

    Returns:
        Resultado do gate.
    """
    needed = champion_auc + MIN_AUC_GAIN
    return GateResult(
        gate_id=GATE_IDS[0],
        passed=challenger_auc - needed >= -_EPS,
        observed=challenger_auc,
        threshold=needed,
        detail=(
            f"challenger {challenger_auc:.4f} contra champion {champion_auc:.4f}"
            f" ({challenger_auc - champion_auc:+.4f}; exigido +{MIN_AUC_GAIN:.2f})"
        ),
    )


def gate_brier(champion_brier: float, challenger_brier: float) -> GateResult:
    """G2: a calibração do challenger não pode piorar.

    Args:
        champion_brier: Brier do champion no holdout.
        challenger_brier: Brier do challenger no holdout.

    Returns:
        Resultado do gate.
    """
    return GateResult(
        gate_id=GATE_IDS[1],
        passed=challenger_brier - champion_brier <= _EPS,
        observed=challenger_brier,
        threshold=champion_brier,
        detail=(
            f"challenger {challenger_brier:.4f} contra champion {champion_brier:.4f}"
        ),
    )


def _input_pairs(signature: dict[str, Any]) -> set[tuple[str, str]]:
    return {(item["name"], item["type"]) for item in signature["inputs"]}


def gate_input_contract(
    champion: dict[str, Any], challenger: dict[str, Any]
) -> GateResult:
    """G3: o challenger precisa aceitar exatamente as mesmas colunas de entrada.

    Args:
        champion: Assinatura do champion.
        challenger: Assinatura do challenger.

    Returns:
        Resultado do gate; ``observed`` é o número de colunas divergentes.
    """
    differing = sorted(
        name for name, _ in _input_pairs(champion) ^ _input_pairs(challenger)
    )
    unique = sorted(set(differing))
    return GateResult(
        gate_id=GATE_IDS[2],
        passed=not unique,
        observed=float(len(unique)),
        threshold=0.0,
        detail=(
            "assinatura idêntica"
            if not unique
            else f"colunas divergentes: {', '.join(unique)}"
        ),
    )


def gate_fairness(impacts: list[GroupImpact]) -> GateResult:
    """G4: a razão de impacto de todo grupo deve ficar em [0,80, 1,25].

    Args:
        impacts: Razões por grupo do challenger.

    Returns:
        Resultado do gate; ``observed`` é a razão mais distante de 1,0.
    """
    worst = max(impacts, key=lambda item: abs(item.ratio - 1.0))
    passed = all(IMPACT_LOWER <= i.ratio <= IMPACT_UPPER for i in impacts)
    return GateResult(
        gate_id=GATE_IDS[3],
        passed=passed,
        observed=worst.ratio,
        threshold=IMPACT_LOWER,
        threshold_upper=IMPACT_UPPER,
        detail=(
            f"pior razão: {worst.attribute}={worst.group} com {worst.ratio:.3f}"
            f" (faixa {IMPACT_LOWER:.2f} a {IMPACT_UPPER:.2f})"
        ),
    )


def gate_latency(seconds: float) -> GateResult:
    """G5: a inferência de ``LATENCY_ROWS`` linhas não pode passar de 5 s.

    Args:
        seconds: Duração medida.

    Returns:
        Resultado do gate.
    """
    return GateResult(
        gate_id=GATE_IDS[4],
        passed=seconds <= MAX_LATENCY_S,
        observed=seconds,
        threshold=MAX_LATENCY_S,
        detail=(
            f"{LATENCY_ROWS} linhas em {seconds:.3f} s (limite {MAX_LATENCY_S:.0f} s)"
        ),
    )


def _metrics(bundle: ModelBundle, holdout: pd.DataFrame) -> tuple[ModelMetrics, Any]:
    scores = predict_proba(bundle, holdout)
    threshold = decision_threshold(bundle)
    stats = classification_metrics(
        holdout[dd.TARGET].to_numpy(), scores, threshold=threshold
    )
    metrics = ModelMetrics(
        roc_auc=stats["roc_auc"],
        brier=stats["brier"],
        f1=stats["f1"],
        approval_rate=float((scores < threshold).mean()),
    )
    return metrics, scores < threshold


def _latency(bundle: ModelBundle, holdout: pd.DataFrame, seed: int) -> float:
    batch = holdout.sample(n=LATENCY_ROWS, replace=True, random_state=seed)
    started = time.perf_counter()
    predict_proba(bundle, batch)
    return time.perf_counter() - started


def evaluate_promotion_gates(
    champion: ModelBundle,
    challenger: ModelBundle,
    holdout: pd.DataFrame,
    *,
    settings: Settings,
) -> GatesReport:
    """Avalia os cinco guardrails de promoção no holdout temporal.

    Args:
        champion: Modelo em produção.
        challenger: Modelo candidato.
        holdout: Janela com verdade de campo, nunca vista no treino do challenger.
        settings: Configuração (semente da amostra de latência).

    Returns:
        Relatório com um resultado por gate e as métricas dos dois modelos.
    """
    champ_metrics, _ = _metrics(champion, holdout)
    chall_metrics, approved = _metrics(challenger, holdout)
    impacts = approval_impact_ratios(holdout, approved)
    results = [
        gate_auc(champ_metrics.roc_auc, chall_metrics.roc_auc),
        gate_brier(champ_metrics.brier, chall_metrics.brier),
        gate_input_contract(champion.signature, challenger.signature),
        gate_fairness(impacts),
        gate_latency(_latency(challenger, holdout, settings.seed)),
    ]
    return GatesReport(
        passed=all(result.passed for result in results),
        results=results,
        champion_model=champion.version,
        challenger_model=challenger.version,
        holdout_window="holdout",
        holdout_rows=len(holdout),
        champion=champ_metrics,
        challenger=chall_metrics,
        fairness=impacts,
    )
