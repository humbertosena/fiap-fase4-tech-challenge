"""Testes de integração do relatório do Evidently AI (API 0.7)."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pytest

from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.generator import generate_portfolio, load_scenarios
from credit_sentinel.drift.evidently_runner import (
    build_report,
    extract_flags,
    save_report,
    share_drifted,
)
from credit_sentinel.ml.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
)
from credit_sentinel.utils.seeds import make_rng

if TYPE_CHECKING:
    from pathlib import Path

    from evidently.core.report import Snapshot

pytestmark = pytest.mark.integration

TARGET = dd.TARGET
PREDICTION = "score"
UUID7 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}")


def _model_frame(df: pd.DataFrame) -> pd.DataFrame:
    features = build_features(df)
    risk = 0.02 + 0.5 * features["taxa_endividamento"] * features["utilizacao_credito"]
    features[TARGET] = df[TARGET].to_numpy()
    features[PREDICTION] = np.clip(risk, 0.0, 1.0).to_numpy()
    return features


@pytest.fixture(scope="module")
def frames(scenarios_path: Path) -> dict[str, pd.DataFrame]:
    params = load_scenarios(scenarios_path)
    spans = {
        "reference": ("reference", 1500, 42, (2025, 7, 1)),
        "control": ("reference", 800, 43, (2026, 7, 1)),
        "crisis": ("mes_3_crise", 800, 45, (2026, 9, 1)),
    }
    out = {}
    for key, (scenario, rows, seed, start) in spans.items():
        period = (pd.Timestamp(*start).date(), pd.Timestamp(*start).date())
        out[key] = _model_frame(
            generate_portfolio(rows, params[scenario], make_rng(seed), period)
        )
    return out


def _report(reference: pd.DataFrame, current: pd.DataFrame) -> Snapshot:
    return build_report(
        reference,
        current,
        numeric=NUMERIC_FEATURES,
        categorical=CATEGORICAL_FEATURES,
        target=TARGET,
        prediction=PREDICTION,
        threshold=0.1,
    )


@pytest.fixture(scope="module")
def crisis(frames: dict[str, pd.DataFrame]) -> Snapshot:
    return _report(frames["reference"], frames["crisis"])


@pytest.fixture(scope="module")
def control(frames: dict[str, pd.DataFrame]) -> Snapshot:
    return _report(frames["reference"], frames["control"])


def test_report_is_saved_as_html_and_json(crisis: Snapshot, tmp_path: Path) -> None:
    html, data = tmp_path / "janela.html", tmp_path / "janela.json"
    save_report(crisis, html, data)
    assert html.stat().st_size > 100_000
    assert "<html" in html.read_text(encoding="utf-8")[:2000].lower()
    payload = json.loads(data.read_text(encoding="utf-8"))
    assert {"metrics", "tests"} <= set(payload)


def test_flags_cover_every_feature_plus_target_and_score(crisis: Snapshot) -> None:
    flags = extract_flags(json.loads(crisis.json()))
    assert set(flags) == {*MODEL_FEATURES, TARGET, PREDICTION}
    assert all(isinstance(value, bool) for value in flags.values())


def test_crisis_flags_the_shocked_columns(crisis: Snapshot) -> None:
    flags = extract_flags(json.loads(crisis.json()))
    assert flags["taxa_endividamento"] is True
    assert flags["utilizacao_credito"] is True
    assert flags[TARGET] is True
    assert flags["idade"] is False


def test_control_window_has_no_drifted_column(control: Snapshot) -> None:
    payload = json.loads(control.json())
    assert not any(extract_flags(payload)[name] for name in MODEL_FEATURES)
    assert share_drifted(payload) == pytest.approx(0.0)


def test_share_drifted_grows_with_the_shift(
    crisis: Snapshot, control: Snapshot
) -> None:
    crisis_share = share_drifted(json.loads(crisis.json()))
    assert 0.0 < crisis_share <= 1.0
    assert crisis_share > share_drifted(json.loads(control.json()))


def test_identifiers_carry_the_given_clock_instead_of_the_wall_clock(
    crisis: Snapshot, tmp_path: Path
) -> None:
    moment = datetime(2026, 10, 20, 21, 20, tzinfo=UTC)
    html = tmp_path / "janela.html"
    save_report(crisis, html, tmp_path / "janela.json", uuid_time=moment)
    stamps = [
        int(found.replace("-", "")[:12], 16)
        for found in UUID7.findall(html.read_text(encoding="utf-8"))
    ]
    start = int(moment.timestamp() * 1000)
    assert stamps
    assert min(stamps) == start
    assert max(stamps) - start < 120_000
