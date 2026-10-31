"""Retreino do challenger: janela deslizante, protocolo e CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from credit_sentinel.cli import main
from credit_sentinel.ml.evaluate_gates import GatesReport
from credit_sentinel.ml.retrain_challenger import (
    CHALLENGER_VERSION,
    build_sliding_window,
    retrain_challenger,
    split_crisis,
)
from credit_sentinel.ml.train_baseline import load_model_bundle
from credit_sentinel.utils.io import read_parquet

if TYPE_CHECKING:
    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.integration


def _months(first: str, last: str, per_month: int = 3) -> pd.DataFrame:
    days = pd.date_range(first, last, freq="MS")
    rows = [
        {"data_referencia": d.replace(day=i + 1), "x": 1}
        for d in days
        for i in range(per_month)
    ]
    return pd.DataFrame(rows)


def test_sliding_window_keeps_the_last_six_reference_months_and_all_windows() -> None:
    reference = _months("2025-07-01", "2026-06-01")
    windows = {
        "mes_1": _months("2026-07-01", "2026-07-01"),
        "mes_2": _months("2026-08-01", "2026-08-01"),
    }
    window = build_sliding_window(reference, windows)
    kept = window["data_referencia"]
    assert kept.min() == pd.Timestamp("2026-01-01")
    assert len(window) == 6 * 3 + 3 + 3
    assert (kept >= "2026-01-01").all()


def test_sliding_window_months_is_configurable() -> None:
    reference = _months("2025-07-01", "2026-06-01")
    window = build_sliding_window(reference, {}, months=3)
    assert window["data_referencia"].min() == pd.Timestamp("2026-04-01")
    assert len(window) == 3 * 3


def test_split_crisis_trains_until_day_15_and_holds_out_the_rest() -> None:
    days = pd.date_range("2026-09-01", "2026-09-30")
    crisis = pd.DataFrame({"data_referencia": days})
    early, late = split_crisis(crisis)
    assert early["data_referencia"].max() == pd.Timestamp("2026-09-15")
    assert late["data_referencia"].min() == pd.Timestamp("2026-09-16")
    assert len(early) + len(late) == len(crisis)


def test_retrain_uses_the_documented_protocol(pipeline_project: Settings) -> None:
    challenger, report = retrain_challenger(settings=pipeline_project)
    assert challenger.version == CHALLENGER_VERSION
    assert isinstance(report, GatesReport)
    assert len(report.results) == 5
    protocol = report.protocol
    crisis = read_parquet(pipeline_project.production_dir / "mes_3_crise.parquet")
    early, late = split_crisis(crisis)
    assert protocol.holdout_window == "mes_3_crise"
    assert protocol.holdout_rows == len(late) == report.holdout_rows
    assert protocol.windows["mes_3_crise_ate_dia_15"] == len(early)
    assert protocol.windows["mes_1"] == 500
    assert protocol.training_rows == protocol.reference_rows + 500 + 500 + len(early)
    assert protocol.reference_start == "2026-01-01"
    assert report.champion_model == "baseline-1.0.0"
    assert report.challenger_model == CHALLENGER_VERSION


def test_cli_writes_the_challenger_and_the_gates_report(
    project_clean: Settings,
) -> None:
    code = main(["retrain-challenger"])
    assert code in {0, 3}
    bundle = load_model_bundle(project_clean.models_dir / "challenger")
    assert bundle.version == CHALLENGER_VERSION
    text = (project_clean.reports_dir / "gates_report.json").read_text("utf-8")
    report = GatesReport.model_validate_json(text)
    assert (code == 0) == report.passed


@pytest.mark.parametrize("passed", [True, False])
def test_cli_exit_code_follows_the_gates(
    project_clean: Settings, monkeypatch: pytest.MonkeyPatch, *, passed: bool
) -> None:
    from credit_sentinel.ml import commands

    real_challenger, real_report = retrain_challenger(settings=project_clean)
    forced = real_report.model_copy(update={"passed": passed})
    monkeypatch.setattr(
        commands, "retrain_challenger", lambda **_: (real_challenger, forced)
    )
    assert main(["retrain-challenger"]) == (0 if passed else 3)
    written = GatesReport.model_validate_json(
        (project_clean.reports_dir / "gates_report.json").read_text("utf-8")
    )
    assert written.passed is passed


def test_missing_window_points_to_simulate(
    project_clean: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    (project_clean.production_dir / "mes_2.parquet").unlink()
    capsys.readouterr()
    assert main(["retrain-challenger"]) == 1
    assert "simulate" in capsys.readouterr().out
