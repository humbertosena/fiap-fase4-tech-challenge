"""Testes das predições por janela e da medição de degradação."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from credit_sentinel.contracts.privacy_sanitizer import HASH_COLUMN, hash_identifier
from credit_sentinel.data import dictionary as dd
from credit_sentinel.drift.predictions import (
    PerformanceSummary,
    PredictionSummary,
    decision_threshold,
    evaluate_performance,
    score_reference_holdout,
    score_window,
    summarize_predictions,
)
from credit_sentinel.ml.evaluation import ScoreHistogram, ThresholdReport
from credit_sentinel.ml.train_baseline import (
    ModelBundle,
    TrainingMetrics,
    split_reference,
)
from credit_sentinel.utils.seeds import make_rng

if TYPE_CHECKING:
    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.unit

SALT = "sal-de-teste"
VALIDATION_AUC = 0.80
F1_THRESHOLD = 0.5


class _RiskFromBureauScore:
    """Pipeline falso: quanto menor o score de bureau, maior a probabilidade."""

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        risk = 1.0 - features["score_bureau"].to_numpy(dtype=float) / 1000.0
        return np.column_stack([1.0 - risk, risk])


def _metrics() -> TrainingMetrics:
    return TrainingMetrics(
        roc_auc=VALIDATION_AUC,
        average_precision=0.3,
        f1=0.4,
        precision=0.3,
        recall=0.5,
        brier=0.05,
        ks_statistic=0.5,
        threshold=0.2,
        confusion_matrix=[[1, 0], [0, 1]],
        n_train=80,
        n_valid=20,
        base_rate=0.06,
        trained_at=datetime(2026, 10, 10, tzinfo=UTC),
        model_version="baseline-teste",
    )


def _threshold_report() -> ThresholdReport:
    histogram = ScoreHistogram(edges=[0.0, 1.0], positives=[0], negatives=[0])
    return ThresholdReport(
        f1_threshold=F1_THRESHOLD,
        cost_threshold=0.1,
        fn_cost=5.0,
        fp_cost=1.0,
        score_histogram=histogram,
    )


@pytest.fixture(scope="module")
def bundle() -> ModelBundle:
    return ModelBundle(
        pipeline=_RiskFromBureauScore(),  # type: ignore[arg-type]
        metrics=_metrics(),
        threshold=_threshold_report(),
    )


def _scores(values: np.ndarray) -> pd.DataFrame:
    decisions = (values >= F1_THRESHOLD).astype(int)
    return pd.DataFrame({HASH_COLUMN: "h", "score": values, "pred": decisions})


def test_decision_threshold_prefers_the_f1_threshold(bundle: ModelBundle) -> None:
    assert decision_threshold(bundle) == F1_THRESHOLD
    without_report = ModelBundle(pipeline=bundle.pipeline, metrics=_metrics())
    assert decision_threshold(without_report) == pytest.approx(0.2)


def test_score_window_returns_hashed_ids_scores_and_decisions(
    bundle: ModelBundle, df_reference_small: pd.DataFrame
) -> None:
    frame = df_reference_small.head(50)
    scored = score_window(bundle, frame, salt=SALT)
    assert list(scored.columns) == [HASH_COLUMN, "score", "pred"]
    assert len(scored) == 50
    assert scored[HASH_COLUMN].iloc[0] == hash_identifier(
        frame[dd.ID_COLUMN].iloc[0], SALT
    )
    expected = 1.0 - frame["score_bureau"].to_numpy() / 1000.0
    np.testing.assert_allclose(scored["score"].to_numpy(), expected)
    np.testing.assert_array_equal(
        scored["pred"].to_numpy(), (expected >= F1_THRESHOLD).astype(int)
    )


def test_score_window_never_exposes_the_client_id(
    bundle: ModelBundle, df_reference_small: pd.DataFrame
) -> None:
    scored = score_window(bundle, df_reference_small, salt=SALT)
    assert dd.ID_COLUMN not in scored.columns
    assert not scored[HASH_COLUMN].str.startswith("CLI-").any()
    assert scored[HASH_COLUMN].str.fullmatch(r"[0-9a-f]{64}").all()


def test_score_window_uses_the_settings_salt_by_default(
    bundle: ModelBundle, df_reference_small: pd.DataFrame, project_tmp: Settings
) -> None:
    frame = df_reference_small.head(3)
    scored = score_window(bundle, frame)
    salt = project_tmp.pii_salt.get_secret_value()
    assert scored[HASH_COLUMN].iloc[0] == hash_identifier(
        frame[dd.ID_COLUMN].iloc[0], salt
    )


def test_reference_holdout_is_the_validation_split(
    bundle: ModelBundle, df_reference_small: pd.DataFrame
) -> None:
    holdout = score_reference_holdout(bundle, df_reference_small, seed=42, salt=SALT)
    valid = split_reference(df_reference_small, seed=42)[1]
    assert len(holdout) == len(valid) == 100
    assert holdout[HASH_COLUMN].iloc[0] == hash_identifier(
        valid[dd.ID_COLUMN].iloc[0], SALT
    )


def test_summary_reports_percentiles_mean_and_approval_rate() -> None:
    scores = _scores(np.linspace(0.0, 1.0, 101))
    summary = summarize_predictions(scores, scores, "mes_x")
    assert summary.window == "mes_x"
    assert summary.n == 101
    assert summary.p50 == pytest.approx(0.5)
    assert summary.p90 == pytest.approx(0.9)
    assert summary.p99 == pytest.approx(0.99)
    assert summary.mean_score == pytest.approx(0.5)
    assert summary.approval_rate == pytest.approx(50 / 101)
    assert summary.psi_score_vs_reference == pytest.approx(0.0)


def test_score_psi_grows_when_the_scores_move_up() -> None:
    reference = _scores(make_rng(1).beta(2, 12, 4000))
    mild = _scores(make_rng(2).beta(2, 10, 4000))
    strong = _scores(make_rng(3).beta(4, 6, 4000))
    low = summarize_predictions(reference, mild, "mes_1").psi_score_vs_reference
    high = summarize_predictions(reference, strong, "mes_3").psi_score_vs_reference
    assert 0 <= low < high
    assert high > 0.25


def test_percentiles_rise_when_the_population_gets_riskier() -> None:
    reference = _scores(make_rng(1).beta(2, 12, 4000))
    riskier = _scores(make_rng(2).beta(3, 8, 4000))
    base = summarize_predictions(reference, reference, "ref")
    shifted = summarize_predictions(reference, riskier, "mes_3")
    assert shifted.p90 > base.p90
    assert shifted.p99 > base.p99
    assert shifted.approval_rate < base.approval_rate


def test_performance_of_a_perfect_ranking(bundle: ModelBundle) -> None:
    truth = pd.DataFrame({dd.TARGET: [0, 0, 0, 0, 1, 1]})
    scores = _scores(np.array([0.1, 0.2, 0.3, 0.4, 0.8, 0.9]))
    result = evaluate_performance(truth, scores, bundle, "mes_x")
    assert result.roc_auc == pytest.approx(1.0)
    assert (result.f1, result.precision, result.recall) == (1.0, 1.0, 1.0)
    assert result.base_rate == pytest.approx(2 / 6)
    assert result.auc_delta_vs_reference == pytest.approx(1.0 - VALIDATION_AUC)
    assert result.auc_delta_vs_control is None


def test_performance_reports_the_drop_against_validation_and_control(
    bundle: ModelBundle,
) -> None:
    truth = pd.DataFrame({dd.TARGET: make_rng(4).integers(0, 2, 4000)})
    scores = _scores(make_rng(5).random(4000))
    result = evaluate_performance(truth, scores, bundle, "mes_3", control_auc=0.78)
    assert result.roc_auc == pytest.approx(0.5, abs=0.03)
    assert result.auc_delta_vs_reference == pytest.approx(
        result.roc_auc - VALIDATION_AUC
    )
    assert result.auc_delta_vs_control == pytest.approx(result.roc_auc - 0.78)
    assert 0 <= result.brier <= 1


def test_performance_needs_both_classes(bundle: ModelBundle) -> None:
    truth = pd.DataFrame({dd.TARGET: [0, 0, 0]})
    with pytest.raises(ValueError, match="duas classes"):
        evaluate_performance(truth, _scores(np.array([0.1, 0.2, 0.3])), bundle, "x")


def test_summaries_survive_a_json_round_trip() -> None:
    scores = _scores(np.linspace(0.0, 1.0, 11))
    summary = summarize_predictions(scores, scores, "mes_x")
    assert PredictionSummary.model_validate_json(summary.model_dump_json()) == summary
    performance = PerformanceSummary(
        window="mes_x",
        roc_auc=0.7,
        f1=0.3,
        precision=0.3,
        recall=0.3,
        brier=0.1,
        base_rate=0.1,
        auc_delta_vs_reference=-0.1,
        auc_delta_vs_control=None,
    )
    restored = PerformanceSummary.model_validate_json(performance.model_dump_json())
    assert restored == performance
