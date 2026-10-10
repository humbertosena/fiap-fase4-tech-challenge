"""Métricas offline, limiares de decisão e importância de features."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from pydantic import BaseModel
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

if TYPE_CHECKING:
    import pandas as pd
    from sklearn.pipeline import Pipeline

FN_COST = 5.0
FP_COST = 1.0
HISTOGRAM_BINS = 50
_THRESHOLD_GRID = np.linspace(0.0, 1.0, 1001)


class ScoreHistogram(BaseModel):
    """Histograma do score por classe (base do slider de limiar na página HTML)."""

    edges: list[float]
    positives: list[int]
    negatives: list[int]


class ThresholdReport(BaseModel):
    """Limiares de decisão do modelo.

    Attributes:
        f1_threshold: Limiar que maximiza o F1 na validação (operacional).
        cost_threshold: Limiar que minimiza ``FN_COST·FN + FP_COST·FP``.
        fn_cost: Custo relativo de um falso negativo (inadimplente aprovado).
        fp_cost: Custo relativo de um falso positivo (bom pagador negado).
        score_histogram: Distribuição do score por classe.
    """

    f1_threshold: float
    cost_threshold: float
    fn_cost: float
    fp_cost: float
    score_histogram: ScoreHistogram


class FeatureImportance(BaseModel):
    """Importância por permutação (queda média de ROC-AUC)."""

    feature: str
    importance_mean: float
    importance_std: float


def ks_statistic(y: np.ndarray, scores: np.ndarray) -> float:
    """Estatística KS de separação entre classes (máximo de TPR - FPR).

    Args:
        y: Rótulos binários.
        scores: Probabilidades previstas.

    Returns:
        Valor entre 0 e 1.
    """
    fpr, tpr, _ = roc_curve(y, scores)
    return float(np.max(tpr - fpr))


def best_f1_threshold(y: np.ndarray, scores: np.ndarray) -> float:
    """Limiar que maximiza o F1.

    Args:
        y: Rótulos binários.
        scores: Probabilidades previstas.

    Returns:
        Limiar ótimo.
    """
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def best_cost_threshold(
    y: np.ndarray,
    scores: np.ndarray,
    *,
    fn_cost: float = FN_COST,
    fp_cost: float = FP_COST,
) -> float:
    """Limiar que minimiza o custo esperado de erros.

    Args:
        y: Rótulos binários.
        scores: Probabilidades previstas.
        fn_cost: Custo de um falso negativo.
        fp_cost: Custo de um falso positivo.

    Returns:
        Limiar de menor custo na grade de 0,001.
    """
    predicted = scores[None, :] >= _THRESHOLD_GRID[:, None]
    false_neg = ((~predicted) & (y[None, :] == 1)).sum(axis=1)
    false_pos = (predicted & (y[None, :] == 0)).sum(axis=1)
    cost = fn_cost * false_neg + fp_cost * false_pos
    return float(_THRESHOLD_GRID[int(np.argmin(cost))])


def classification_metrics(
    y: np.ndarray, scores: np.ndarray, *, threshold: float
) -> dict[str, Any]:
    """Métricas de ranking, calibração e classificação em um limiar.

    Args:
        y: Rótulos binários.
        scores: Probabilidades previstas.
        threshold: Limiar de decisão.

    Returns:
        Dicionário com ROC-AUC, AP, F1, precisão, recall, Brier, KS e matriz.
    """
    predicted = (scores >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y, scores)),
        "average_precision": float(average_precision_score(y, scores)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "brier": float(brier_score_loss(y, scores)),
        "ks_statistic": ks_statistic(y, scores),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


def score_histogram(
    y: np.ndarray, scores: np.ndarray, *, bins: int = HISTOGRAM_BINS
) -> ScoreHistogram:
    """Histograma de scores em ``[0, 1]`` separado por classe.

    Args:
        y: Rótulos binários.
        scores: Probabilidades previstas.
        bins: Quantidade de bins uniformes.

    Returns:
        Bordas e contagens de positivos e negativos.
    """
    edges = np.linspace(0.0, 1.0, bins + 1)
    positives = np.histogram(scores[y == 1], bins=edges)[0]
    negatives = np.histogram(scores[y == 0], bins=edges)[0]
    return ScoreHistogram(
        edges=[round(float(e), 4) for e in edges],
        positives=positives.astype(int).tolist(),
        negatives=negatives.astype(int).tolist(),
    )


def threshold_report(y: np.ndarray, scores: np.ndarray) -> ThresholdReport:
    """Consolida limiares de F1 e de custo e o histograma de scores.

    Args:
        y: Rótulos binários.
        scores: Probabilidades previstas.

    Returns:
        Relatório de limiares.
    """
    return ThresholdReport(
        f1_threshold=best_f1_threshold(y, scores),
        cost_threshold=best_cost_threshold(y, scores),
        fn_cost=FN_COST,
        fp_cost=FP_COST,
        score_histogram=score_histogram(y, scores),
    )


def feature_importance(
    pipeline: Pipeline, features: pd.DataFrame, y: np.ndarray, *, seed: int
) -> list[FeatureImportance]:
    """Importância por permutação medida em ROC-AUC (10 repetições).

    Args:
        pipeline: Modelo ajustado.
        features: Features de validação.
        y: Rótulos de validação.
        seed: Semente das permutações.

    Returns:
        Importâncias em ordem decrescente.
    """
    result = permutation_importance(
        pipeline, features, y, scoring="roc_auc", n_repeats=10, random_state=seed
    )
    ranked = sorted(
        zip(
            features.columns,
            result.importances_mean,
            result.importances_std,
            strict=True,
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        FeatureImportance(
            feature=name, importance_mean=float(m), importance_std=float(s)
        )
        for name, m, s in ranked
    ]
