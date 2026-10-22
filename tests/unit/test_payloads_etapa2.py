"""Testes das funções puras do payload da Etapa 2 (histogramas, curva e aceitação)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from credit_sentinel.data import dictionary as dd
from credit_sentinel.explainers.payloads import (
    SCORE_BANDS,
    BandCurve,
    build_payload,
    category_shares,
    overlay_histogram,
    psi_calculator,
    score_band_curve,
)
from credit_sentinel.utils.seeds import make_rng

if TYPE_CHECKING:
    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.unit

N = 4000


def _frame(shift: float, seed: int, *, rate: float = 0.06) -> pd.DataFrame:
    rng = make_rng(seed)
    return pd.DataFrame(
        {
            "renda_mensal": rng.lognormal(8.2 + shift, 0.5, N),
            "score_bureau": rng.integers(300, 950, N),
            "situacao_emprego": rng.choice(dd.EMPLOYMENT, N),
            dd.TARGET: (rng.random(N) < rate).astype(int),
        }
    )


@pytest.fixture(scope="module")
def frames() -> dict[str, pd.DataFrame]:
    return {
        "reference": _frame(0.0, 1),
        "mes_1": _frame(0.0, 2),
        "mes_3_crise": _frame(-0.3, 3, rate=0.13),
    }


def test_missing_artifacts_raise_a_clear_error(settings_tmp: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="detect-drift"):
        build_payload(2, settings=settings_tmp)


def test_overlay_histogram_shares_the_reference_edges(
    frames: dict[str, pd.DataFrame],
) -> None:
    hist = overlay_histogram(frames, "renda_mensal", bins=20)
    assert hist.feature == "renda_mensal"
    assert len(hist.edges) == 21
    assert set(hist.counts) == set(frames)
    assert all(sum(counts) == N for counts in hist.counts.values())
    assert all(len(counts) == 20 for counts in hist.counts.values())


def test_overlay_shows_the_income_moving_left(frames: dict[str, pd.DataFrame]) -> None:
    hist = overlay_histogram(frames, "renda_mensal", bins=20)
    centre = lambda counts: float(  # noqa: E731
        np.average(np.arange(20), weights=counts)
    )
    assert centre(hist.counts["mes_3_crise"]) < centre(hist.counts["reference"])
    assert centre(hist.counts["mes_1"]) == pytest.approx(
        centre(hist.counts["reference"]), abs=0.6
    )


def test_psi_calculator_embeds_the_reference_distribution(
    frames: dict[str, pd.DataFrame],
) -> None:
    calc = psi_calculator(frames, "renda_mensal", bins=40)
    reference = frames["reference"]["renda_mensal"]
    assert len(calc.edges) == 41
    assert sum(calc.counts) == N
    assert calc.mean == pytest.approx(float(reference.mean()))
    assert calc.std == pytest.approx(float(reference.std(ddof=0)))
    assert calc.observed_shift_sd["mes_1"] == pytest.approx(0.0, abs=0.1)
    assert calc.observed_shift_sd["mes_3_crise"] < -0.3
    assert "reference" not in calc.observed_shift_sd


def test_score_band_curve_has_one_rate_per_band_and_window(
    frames: dict[str, pd.DataFrame],
) -> None:
    curve = score_band_curve(frames)
    assert isinstance(curve, BandCurve)
    assert curve.edges == SCORE_BANDS
    bands = len(SCORE_BANDS) - 1
    for key, frame in frames.items():
        assert sum(curve.counts[key]) == len(frame)
        assert len(curve.default_rate[key]) == bands
    rate = curve.default_rate["mes_3_crise"]
    assert all(r is None or 0.0 <= r <= 1.0 for r in rate)


def test_category_shares_sum_to_one_per_dataset(
    frames: dict[str, pd.DataFrame],
) -> None:
    shares = category_shares(frames, "situacao_emprego")
    assert shares.categories == dd.ENUMS["situacao_emprego"]
    for values in shares.shares.values():
        assert sum(values) == pytest.approx(1.0)
