"""Predições do baseline por janela e medição da degradação de performance.

As saídas nunca carregam ``id_cliente`` em claro: o identificador vira
``id_cliente_hash`` (HMAC-SHA-256 com o salt do ambiente) antes de qualquer gravação.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel

from credit_sentinel.contracts.privacy_sanitizer import HASH_COLUMN, hash_identifier
from credit_sentinel.data import dictionary as dd
from credit_sentinel.drift.statistical_tests import population_stability_index
from credit_sentinel.ml.evaluation import classification_metrics
from credit_sentinel.ml.train_baseline import predict_proba, split_reference
from credit_sentinel.settings import get_settings

if TYPE_CHECKING:
    from credit_sentinel.ml.train_baseline import ModelBundle

SCORE_COLUMN = "score"
PRED_COLUMN = "pred"
P50, P90, P99 = 0.50, 0.90, 0.99


class PredictionSummary(BaseModel):
    """Proxy de performance sem verdade de campo: o comportamento dos scores.

    Attributes:
        window: Nome da janela.
        n: Quantidade de clientes pontuados.
        p50: Mediana do score (probabilidade de inadimplência).
        p90: Percentil 90 do score.
        p99: Percentil 99 do score.
        mean_score: Score médio.
        approval_rate: Fração aprovada (decisão ``pred`` igual a 0).
        psi_score_vs_reference: PSI do score contra o holdout da referência.
    """

    window: str
    n: int
    p50: float
    p90: float
    p99: float
    mean_score: float
    approval_rate: float
    psi_score_vs_reference: float


class PerformanceSummary(BaseModel):
    """Performance real do modelo na janela (usa o ``inadimplente`` observado).

    Attributes:
        window: Nome da janela.
        roc_auc: ROC-AUC da janela.
        f1: F1 no limiar operacional.
        precision: Precisão no limiar operacional.
        recall: Recall no limiar operacional.
        brier: Erro de calibração (Brier).
        base_rate: Taxa de inadimplência observada.
        auc_delta_vs_reference: ``roc_auc`` menos o AUC de validação do baseline.
        auc_delta_vs_control: ``roc_auc`` menos o AUC da janela de controle, se houver.
    """

    window: str
    roc_auc: float
    f1: float
    precision: float
    recall: float
    brier: float
    base_rate: float
    auc_delta_vs_reference: float
    auc_delta_vs_control: float | None = None


def decision_threshold(bundle: ModelBundle) -> float:
    """Limiar operacional do modelo (F1 da validação).

    Args:
        bundle: Modelo carregado.

    Returns:
        Limiar de F1 do relatório de limiares ou, na falta dele, o das métricas.
    """
    if bundle.threshold is not None:
        return bundle.threshold.f1_threshold
    return bundle.metrics.threshold


def _salt(salt: str | None) -> str:
    return salt if salt is not None else get_settings().pii_salt.get_secret_value()


def score_window(
    bundle: ModelBundle, df: pd.DataFrame, *, salt: str | None = None
) -> pd.DataFrame:
    """Pontua um lote e pseudonimiza o identificador.

    Args:
        bundle: Modelo carregado.
        df: Lote com as colunas do dicionário de dados.
        salt: Segredo do HMAC; ``None`` usa o das settings.

    Returns:
        DataFrame com ``id_cliente_hash``, ``score`` e ``pred`` (1 = negar).
    """
    scores = predict_proba(bundle, df)
    key = _salt(salt)
    hashes = [hash_identifier(str(value), key) for value in df[dd.ID_COLUMN]]
    return pd.DataFrame(
        {
            HASH_COLUMN: hashes,
            SCORE_COLUMN: scores,
            PRED_COLUMN: (scores >= decision_threshold(bundle)).astype(int),
        },
        index=df.index,
    )


def score_reference_holdout(
    bundle: ModelBundle,
    reference: pd.DataFrame,
    *,
    seed: int,
    salt: str | None = None,
) -> pd.DataFrame:
    """Pontua o holdout de validação da referência.

    O modelo viu as demais linhas no treino, então os scores delas são otimistas e
    fariam o controle parecer em drift. O holdout é a referência justa do score.

    Args:
        bundle: Modelo carregado.
        reference: Dataset de referência completo.
        seed: Semente do split de treino (a mesma do treino do baseline).
        salt: Segredo do HMAC; ``None`` usa o das settings.

    Returns:
        Scores do holdout, no formato de ``score_window``.
    """
    _, valid = split_reference(reference, seed=seed)
    return score_window(bundle, valid, salt=salt)


def summarize_predictions(
    scores_ref: pd.DataFrame, scores_cur: pd.DataFrame, window: str
) -> PredictionSummary:
    """Resume a distribuição dos scores de uma janela.

    Args:
        scores_ref: Scores do holdout da referência.
        scores_cur: Scores da janela.
        window: Nome da janela.

    Returns:
        Percentis, média, taxa de aprovação e PSI do score.
    """
    current = scores_cur[SCORE_COLUMN]
    return PredictionSummary(
        window=window,
        n=len(scores_cur),
        p50=float(current.quantile(P50)),
        p90=float(current.quantile(P90)),
        p99=float(current.quantile(P99)),
        mean_score=float(current.mean()),
        approval_rate=float((scores_cur[PRED_COLUMN] == 0).mean()),
        psi_score_vs_reference=population_stability_index(
            scores_ref[SCORE_COLUMN], current
        ),
    )


def evaluate_performance(
    df_with_truth: pd.DataFrame,
    scores: pd.DataFrame,
    bundle: ModelBundle,
    window: str,
    control_auc: float | None = None,
) -> PerformanceSummary:
    """Mede a performance do modelo na janela contra o ``inadimplente`` observado.

    Args:
        df_with_truth: Lote com a coluna ``inadimplente``.
        scores: Saída de ``score_window`` para o mesmo lote, na mesma ordem.
        bundle: Modelo (fornece o limiar e o AUC de validação).
        window: Nome da janela.
        control_auc: AUC da janela de controle, para a queda contra o controle.

    Returns:
        Métricas da janela e quedas de AUC.

    Raises:
        ValueError: Se o lote tiver só uma classe.
    """
    truth = df_with_truth[dd.TARGET].to_numpy()
    if len(set(truth.tolist())) < 2:  # noqa: PLR2004 - duas classes são o mínimo do AUC
        msg = "a avaliação exige as duas classes (adimplente e inadimplente) no lote"
        raise ValueError(msg)
    metrics = classification_metrics(
        truth, scores[SCORE_COLUMN].to_numpy(), threshold=decision_threshold(bundle)
    )
    auc = metrics["roc_auc"]
    return PerformanceSummary(
        window=window,
        roc_auc=auc,
        f1=metrics["f1"],
        precision=metrics["precision"],
        recall=metrics["recall"],
        brier=metrics["brier"],
        base_rate=float(truth.mean()),
        auc_delta_vs_reference=auc - bundle.metrics.roc_auc,
        auc_delta_vs_control=None if control_auc is None else auc - control_auc,
    )
