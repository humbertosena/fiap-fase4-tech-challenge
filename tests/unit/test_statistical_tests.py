"""Testes dos detectores estatísticos nativos com distribuições controladas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_sentinel.drift.statistical_tests import (
    DriftSummary,
    chi_square_test,
    classify_psi,
    jensen_shannon_divergence,
    ks_test,
    population_stability_index,
    run_drift_suite,
)
from credit_sentinel.utils.seeds import make_rng

pytestmark = pytest.mark.unit

N = 5000


@pytest.fixture(scope="module")
def normal_ref() -> pd.Series:
    return pd.Series(make_rng(1).normal(0.0, 1.0, N))


def _normal(shift: float, seed: int) -> pd.Series:
    return pd.Series(make_rng(seed).normal(shift, 1.0, N))


def _labels(shares: dict[str, float], n: int, seed: int) -> pd.Series:
    picks = make_rng(seed).choice(list(shares), size=n, p=list(shares.values()))
    return pd.Series(picks)


# --- PSI -------------------------------------------------------------------------


def test_psi_is_zero_for_identical_samples(normal_ref: pd.Series) -> None:
    assert population_stability_index(normal_ref, normal_ref) == pytest.approx(0.0)


def test_psi_is_small_for_two_samples_of_the_same_distribution(
    normal_ref: pd.Series,
) -> None:
    assert population_stability_index(normal_ref, _normal(0.0, 2)) < 0.01


def test_psi_grows_with_the_shift(normal_ref: pd.Series) -> None:
    values = [
        population_stability_index(normal_ref, _normal(shift, 3))
        for shift in (0.1, 0.3, 0.6, 1.0)
    ]
    assert values == sorted(values)
    assert values[0] < 0.10
    assert values[-1] > 0.25


def test_psi_flags_opposite_shifts_with_similar_magnitude(
    normal_ref: pd.Series,
) -> None:
    up = population_stability_index(normal_ref, _normal(0.5, 4))
    down = population_stability_index(normal_ref, _normal(-0.5, 4))
    assert 0.15 < up < 0.40
    assert 0.15 < down < 0.40


def test_psi_matches_the_closed_form_for_a_binary_feature() -> None:
    reference = pd.Series([0] * 50 + [1] * 50)
    current = pd.Series([0] * 30 + [1] * 70)
    expected = (0.3 - 0.5) * np.log(0.3 / 0.5) + (0.7 - 0.5) * np.log(0.7 / 0.5)
    assert population_stability_index(reference, current) == pytest.approx(expected)


def test_psi_uses_one_bucket_per_value_for_discrete_features() -> None:
    terms = pd.Series([6, 12, 24, 36, 48, 60] * 100)
    shifted = pd.Series([6, 12, 24, 36, 48, 60] * 100).replace({60: 6})
    assert population_stability_index(terms, terms) == pytest.approx(0.0)
    assert population_stability_index(terms, shifted) > 0.25


def test_psi_detects_a_move_away_from_a_constant_reference() -> None:
    constant = pd.Series([5.0] * 200)
    assert population_stability_index(constant, constant) == pytest.approx(0.0)
    assert population_stability_index(constant, pd.Series([6.0] * 200)) > 0.25


def test_psi_ignores_missing_values(normal_ref: pd.Series) -> None:
    with_gaps = pd.concat([normal_ref, pd.Series([np.nan] * 100)])
    clean = population_stability_index(normal_ref, _normal(0.3, 5))
    gapped = population_stability_index(with_gaps, _normal(0.3, 5))
    assert gapped == pytest.approx(clean)


def test_psi_rejects_empty_samples(normal_ref: pd.Series) -> None:
    with pytest.raises(ValueError, match="vazia"):
        population_stability_index(normal_ref, pd.Series([np.nan, np.nan]))
    with pytest.raises(ValueError, match="vazia"):
        population_stability_index(pd.Series([], dtype=float), normal_ref)


@pytest.mark.parametrize(
    ("value", "level"),
    [
        (0.0, "estavel"),
        (0.0999, "estavel"),
        (0.10, "atencao"),
        (0.2499, "atencao"),
        (0.25, "critico"),
        (1.7, "critico"),
    ],
)
def test_classify_psi_follows_the_10_and_25_thresholds(
    value: float, level: str
) -> None:
    assert classify_psi(value) == level


# --- Kolmogorov-Smirnov ------------------------------------------------------------


def test_ks_detects_a_half_sigma_shift(normal_ref: pd.Series) -> None:
    statistic, p_value, drift = ks_test(normal_ref, _normal(0.5, 6))
    assert drift is True
    assert p_value < 1e-10
    assert statistic == pytest.approx(0.2, abs=0.04)


def test_ks_keeps_quiet_for_the_same_distribution(normal_ref: pd.Series) -> None:
    _, p_value, drift = ks_test(normal_ref, _normal(0.0, 7))
    assert drift is False
    assert p_value > 0.05


def test_ks_alpha_moves_the_decision_threshold() -> None:
    reference = pd.Series(np.linspace(0.0, 1.0, 1000))
    current = pd.Series(np.linspace(0.065, 1.065, 1000))
    _, p_value, loose = ks_test(reference, current, alpha=0.05)
    _, _, strict = ks_test(reference, current, alpha=0.001)
    assert 0.001 < p_value < 0.05
    assert loose is True
    assert strict is False


# --- Qui-quadrado e Jensen-Shannon -----------------------------------------------


def test_chi_square_keeps_quiet_for_the_same_shares() -> None:
    shares = {"a": 0.5, "b": 0.3, "c": 0.2}
    _, p_value, drift = chi_square_test(
        _labels(shares, 3000, 1), _labels(shares, 3000, 2)
    )
    assert drift is False
    assert p_value > 0.05


def test_chi_square_detects_shifted_shares() -> None:
    before = _labels({"a": 0.5, "b": 0.3, "c": 0.2}, 3000, 1)
    after = _labels({"a": 0.4, "b": 0.3, "c": 0.3}, 3000, 2)
    statistic, p_value, drift = chi_square_test(before, after)
    assert drift is True
    assert statistic > 30
    assert p_value < 0.001


def test_chi_square_handles_a_category_that_only_appears_later() -> None:
    before = pd.Series(["a"] * 500 + ["b"] * 500)
    after = pd.Series(["a"] * 400 + ["b"] * 400 + ["novo"] * 200)
    _, p_value, drift = chi_square_test(before, after)
    assert drift is True
    assert p_value < 0.001


def test_chi_square_with_a_single_category_has_nothing_to_test() -> None:
    same = pd.Series(["a"] * 100)
    assert chi_square_test(same, same) == (0.0, 1.0, False)


def test_jensen_shannon_is_zero_for_identical_shares() -> None:
    labels = pd.Series(["a"] * 60 + ["b"] * 40)
    assert jensen_shannon_divergence(labels, labels) == pytest.approx(0.0)


def test_jensen_shannon_is_one_for_disjoint_categories() -> None:
    before = pd.Series(["a"] * 100)
    after = pd.Series(["b"] * 100)
    assert jensen_shannon_divergence(before, after) == pytest.approx(1.0)


def test_jensen_shannon_is_symmetric_and_grows_with_the_shift() -> None:
    base = pd.Series(["a"] * 50 + ["b"] * 50)
    mild = pd.Series(["a"] * 40 + ["b"] * 60)
    strong = pd.Series(["a"] * 10 + ["b"] * 90)
    assert jensen_shannon_divergence(base, mild) == pytest.approx(
        jensen_shannon_divergence(mild, base)
    )
    assert jensen_shannon_divergence(base, mild) < jensen_shannon_divergence(
        base, strong
    )


# --- Suíte completa -----------------------------------------------------------------


def _frames(shift: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    numeric = {f"x{i}": _normal(0.0, 10 + i) for i in range(4)}
    labels = _labels({"a": 0.6, "b": 0.4}, N, 20)
    reference = pd.DataFrame({**numeric, "grupo": labels})
    moved = {f"x{i}": _normal(shift if i == 0 else 0.0, 30 + i) for i in range(4)}
    shifted = {"a": 0.6 - shift / 4, "b": 0.4 + shift / 4}
    current = pd.DataFrame({**moved, "grupo": _labels(shifted, N, 21)})
    return reference, current


def _suite(reference: pd.DataFrame, current: pd.DataFrame) -> DriftSummary:
    return run_drift_suite(
        reference,
        current,
        numeric=[f"x{i}" for i in range(4)],
        categorical=["grupo"],
        window="mes_x",
    )


def test_suite_flags_only_the_shifted_features() -> None:
    summary = _suite(*_frames(1.0))
    assert summary.window == "mes_x"
    assert (summary.reference_rows, summary.current_rows) == (N, N)
    assert set(summary.drifted_features) == {"x0", "grupo"}
    assert summary.n_drifted_features == 2
    assert summary.psi_critical_features == ["x0"]


def test_suite_reports_nothing_for_a_stable_window() -> None:
    summary = _suite(*_frames(0.0))
    assert summary.n_drifted_features == 0
    assert summary.drifted_features == []
    assert summary.psi_critical_features == []


def test_suite_applies_bonferroni_over_the_numeric_features() -> None:
    summary = _suite(*_frames(0.0))
    assert summary.alpha == 0.05
    assert summary.alpha_bonferroni == pytest.approx(0.05 / 4)


def test_suite_bonferroni_is_stricter_than_the_raw_alpha() -> None:
    steady = {f"x{i}": np.linspace(0.0, 1.0, 1000) for i in range(12)}
    reference = pd.DataFrame(steady)
    current = reference.assign(x0=np.linspace(0.065, 1.065, 1000))
    summary = run_drift_suite(
        reference, current, numeric=list(steady), categorical=[], window="mes_x"
    )
    flagged = next(r for r in summary.numeric if r.feature == "x0")
    assert flagged.ks_drift is True
    assert flagged.ks_drift_bonferroni is False
    assert summary.alpha_bonferroni == pytest.approx(0.05 / 12)
    assert summary.drifted_features == []


def test_suite_numeric_result_carries_magnitude() -> None:
    summary = _suite(*_frames(1.0))
    result = next(r for r in summary.numeric if r.feature == "x0")
    assert result.psi_level == "critico"
    assert result.wasserstein_norm == pytest.approx(1.0, abs=0.1)
    assert result.mean_cur - result.mean_ref == pytest.approx(1.0, abs=0.1)
    quiet = next(r for r in summary.numeric if r.feature == "x1")
    assert quiet.psi_level == "estavel"
    assert quiet.wasserstein_norm < 0.1


def test_suite_categorical_result_carries_shares() -> None:
    summary = _suite(*_frames(1.0))
    result = summary.categorical[0]
    assert result.feature == "grupo"
    assert result.chi2_drift is True
    assert result.shares_ref["a"] == pytest.approx(0.6, abs=0.03)
    assert result.shares_cur["a"] == pytest.approx(0.35, abs=0.03)
    assert sum(result.shares_cur.values()) == pytest.approx(1.0)
    assert result.jensen_shannon > 0.1


def test_suite_summary_survives_a_json_round_trip() -> None:
    summary = _suite(*_frames(1.0))
    assert DriftSummary.model_validate_json(summary.model_dump_json()) == summary
