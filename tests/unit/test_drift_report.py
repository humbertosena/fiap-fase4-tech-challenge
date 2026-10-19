"""Testes do relatório consolidado de drift e da tabela em Markdown."""

from __future__ import annotations

from datetime import UTC, datetime

import numpy as np
import pandas as pd
import pytest

from credit_sentinel.drift.predictions import PerformanceSummary, PredictionSummary
from credit_sentinel.drift.report import (
    DriftReport,
    EvidentlyWindow,
    ReferenceInfo,
    WindowReport,
    render_drift_table,
)
from credit_sentinel.drift.statistical_tests import run_drift_suite
from credit_sentinel.utils.seeds import make_rng

pytestmark = pytest.mark.unit

N = 3000


def _frame(shift: float, seed: int) -> pd.DataFrame:
    rng = make_rng(seed)
    labels = rng.choice(["a", "b"], size=N, p=[0.6 - shift / 8, 0.4 + shift / 8])
    return pd.DataFrame(
        {
            "renda": rng.normal(shift, 1.0, N),
            "idade": rng.normal(0, 1, N),
            "grupo": labels,
        }
    )


def _window(name: str, shift: float, auc: float) -> WindowReport:
    drift = run_drift_suite(
        _frame(0.0, 1),
        _frame(shift, 2),
        numeric=["renda", "idade"],
        categorical=["grupo"],
        window=name,
    )
    predictions = PredictionSummary(
        window=name,
        n=N,
        p50=0.03,
        p90=0.15,
        p99=0.45,
        mean_score=0.06,
        approval_rate=0.88,
        psi_score_vs_reference=0.012 + shift / 10,
    )
    performance = PerformanceSummary(
        window=name,
        roc_auc=auc,
        f1=0.33,
        precision=0.3,
        recall=0.4,
        brier=0.05,
        base_rate=0.06,
        auc_delta_vs_reference=auc - 0.829,
        auc_delta_vs_control=None if shift == 0 else auc - 0.806,
    )
    evidently = EvidentlyWindow(
        share_drifted=0.0 if shift == 0 else 0.5,
        flags={"renda": shift != 0, "idade": False, "grupo": shift != 0},
        html=f"reports/evidently/{name}_drift_report.html",
        json_report=f"reports/evidently/{name}_drift_report.json",
    )
    return WindowReport(
        drift=drift,
        predictions=predictions,
        performance=performance,
        evidently=evidently,
    )


@pytest.fixture(scope="module")
def report() -> DriftReport:
    return DriftReport(
        generated_at=datetime(2026, 10, 20, 21, 20, tzinfo=UTC),
        model_version="baseline-1.0.0",
        reference=ReferenceInfo(
            rows=12000, holdout_rows=2400, roc_auc_validation=0.829
        ),
        windows={
            "mes_1": _window("mes_1", 0.0, 0.806),
            "mes_3_crise": _window("mes_3_crise", 1.0, 0.764),
        },
    )


def test_report_survives_a_json_round_trip(report: DriftReport) -> None:
    restored = DriftReport.model_validate_json(report.model_dump_json())
    assert restored == report
    assert list(restored.windows) == ["mes_1", "mes_3_crise"]


def test_table_lists_every_feature_and_window(report: DriftReport) -> None:
    table = render_drift_table(report)
    for token in ("renda", "idade", "grupo", "mes_1", "mes_3_crise"):
        assert token in table
    assert "KS" in table
    assert "PSI" in table


def test_table_uses_pt_br_numbers_and_psi_levels(report: DriftReport) -> None:
    table = render_drift_table(report)
    assert "estável" in table
    assert "crítico" in table
    assert "0,0" in table
    assert not any(
        part.replace(".", "", 1).isdigit() and "." in part and len(part) < 6
        for line in table.splitlines()
        for part in line.split("|")
        if "e-" not in part
    )


def test_table_reports_scores_and_performance(report: DriftReport) -> None:
    table = render_drift_table(report)
    assert "AUC" in table
    assert "0,764" in table
    assert "P90" in table
    assert "aprovação" in table.lower()


def test_table_compares_native_tests_with_evidently(report: DriftReport) -> None:
    table = render_drift_table(report)
    assert "Evidently" in table
    assert "participação de colunas com drift" in table.lower() or "colunas" in table


def test_table_ends_with_a_single_newline(report: DriftReport) -> None:
    table = render_drift_table(report)
    assert table.endswith("\n")
    assert not table.endswith("\n\n")
    assert np.isfinite(report.windows["mes_1"].drift.numeric[0].psi)
