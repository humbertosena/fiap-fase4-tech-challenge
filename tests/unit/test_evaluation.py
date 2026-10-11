"""Testes das métricas, limiares e histogramas de avaliação."""

from __future__ import annotations

import numpy as np
import pytest

from credit_sentinel.ml.evaluation import (
    best_cost_threshold,
    best_f1_threshold,
    classification_metrics,
    ks_statistic,
    score_histogram,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def scored() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    y = (rng.random(4000) < 0.1).astype(int)
    scores = np.clip(rng.normal(0.2 + 0.35 * y, 0.15), 0, 1)
    return y, scores


def test_ks_is_one_for_perfect_separation() -> None:
    y = np.array([0, 0, 1, 1])
    assert ks_statistic(y, np.array([0.1, 0.2, 0.8, 0.9])) == pytest.approx(1.0)


def test_ks_is_zero_for_constant_scores() -> None:
    y = np.array([0, 1, 0, 1])
    assert ks_statistic(y, np.full(4, 0.5)) == pytest.approx(0.0)


def test_f1_threshold_separates_obvious_case() -> None:
    y = np.array([0, 0, 0, 1, 1])
    threshold = best_f1_threshold(y, np.array([0.1, 0.2, 0.3, 0.7, 0.9]))
    assert 0.3 < threshold <= 0.7


def test_cost_threshold_is_lower_when_false_negatives_cost_more(
    scored: tuple[np.ndarray, np.ndarray],
) -> None:
    y, scores = scored
    assert best_cost_threshold(y, scores, fn_cost=5) < best_f1_threshold(y, scores)


def test_classification_metrics_at_threshold(
    scored: tuple[np.ndarray, np.ndarray],
) -> None:
    y, scores = scored
    metrics = classification_metrics(y, scores, threshold=0.4)
    (tn, fp), (fn, tp) = metrics["confusion_matrix"]
    assert tn + fp + fn + tp == len(y)
    assert metrics["recall"] == pytest.approx(tp / (tp + fn))
    assert 0 < metrics["brier"] < 0.25
    assert metrics["roc_auc"] > 0.8


def test_score_histogram_keeps_class_counts(
    scored: tuple[np.ndarray, np.ndarray],
) -> None:
    y, scores = scored
    histogram = score_histogram(y, scores, bins=50)
    assert len(histogram.edges) == 51
    assert sum(histogram.positives) == int(y.sum())
    assert sum(histogram.negatives) == int((1 - y).sum())
