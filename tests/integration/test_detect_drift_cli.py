"""Teste ponta a ponta do subcomando ``detect-drift`` com janelas pequenas."""

from __future__ import annotations

import re
import shutil
from datetime import UTC, datetime

import pandas as pd
import pytest

from credit_sentinel.cli import main
from credit_sentinel.contracts.privacy_sanitizer import scan_paths
from credit_sentinel.drift.report import DriftReport
from credit_sentinel.drift.simulator import WINDOWS, load_windows, simulate_window
from credit_sentinel.settings import Settings, get_settings
from credit_sentinel.utils.io import write_parquet

pytestmark = pytest.mark.integration

SMALL_WINDOW = 500
UUID7 = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12}")


@pytest.fixture(scope="module")
def trained_project(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    """Projeto temporário com referência pequena e baseline treinado (uma vez)."""
    root = tmp_path_factory.mktemp("detect")
    shutil.copytree(Settings(_env_file=None).config_dir, root / "config")
    mp = pytest.MonkeyPatch()
    mp.setenv("CS_PROJECT_ROOT", str(root))
    get_settings.cache_clear()
    assert main(["generate-data", "--rows", "1500"]) == 0
    assert main(["train", "--no-mlflow"]) == 0
    yield get_settings()
    mp.undo()
    get_settings.cache_clear()


@pytest.fixture
def project(trained_project: Settings) -> Settings:
    shutil.rmtree(trained_project.reports_dir, ignore_errors=True)
    for spec in load_windows(trained_project.config_dir / "simulation_scenarios.yaml"):
        small = simulate_window(spec.model_copy(update={"n_rows": SMALL_WINDOW}))
        write_parquet(small, trained_project.production_dir / f"{spec.name}.parquet")
    return trained_project


def test_detect_drift_writes_every_artifact_of_a_window(project: Settings) -> None:
    assert main(["detect-drift", "--window", "mes_3_crise"]) == 0
    reports = project.reports_dir
    assert (reports / "evidently" / "mes_3_crise_drift_report.html").exists()
    assert (reports / "evidently" / "mes_3_crise_drift_report.json").exists()
    scores = pd.read_parquet(reports / "predictions" / "mes_3_crise_scores.parquet")
    assert list(scores.columns) == ["id_cliente_hash", "score", "pred"]
    assert len(scores) == SMALL_WINDOW
    assert not scores["id_cliente_hash"].str.startswith("CLI-").any()
    summary = DriftReport.model_validate_json(
        (reports / "evidently_drift_summary.json").read_text(encoding="utf-8")
    )
    assert list(summary.windows) == ["mes_3_crise"]
    assert (reports / "drift_table.md").read_text(encoding="utf-8").strip()


def test_crisis_report_is_copied_to_the_named_deliverable(project: Settings) -> None:
    assert main(["detect-drift", "--window", "mes_3_crise"]) == 0
    named = project.reports_dir / "evidently_drift_report.html"
    source = project.reports_dir / "evidently" / "mes_3_crise_drift_report.html"
    assert named.read_bytes() == source.read_bytes()


def test_all_windows_are_consolidated_and_ordered_by_severity(
    project: Settings,
) -> None:
    assert main(["detect-drift", "--window", "all"]) == 0
    text = (project.reports_dir / "evidently_drift_summary.json").read_text("utf-8")
    summary = DriftReport.model_validate_json(text)
    assert list(summary.windows) == list(WINDOWS)
    drifted = [w.drift.n_drifted_features for w in summary.windows.values()]
    assert drifted[0] <= drifted[1] <= drifted[2]
    assert summary.windows["mes_3_crise"].evidently.share_drifted > 0
    control = summary.windows["mes_1"].performance
    assert control.auc_delta_vs_control is None
    crisis = summary.windows["mes_3_crise"].performance
    assert crisis.auc_delta_vs_control is not None
    assert summary.reference.roc_auc_validation == pytest.approx(
        summary.windows["mes_1"].performance.auc_delta_vs_reference * -1
        + control.roc_auc
    )


def test_outputs_never_expose_a_client_id(project: Settings) -> None:
    assert main(["detect-drift", "--window", "mes_2"]) == 0
    assert scan_paths([project.reports_dir]) == []


def test_source_date_epoch_stamps_the_summary_and_the_identifiers(
    project: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    epoch = 1792542000  # 2026-10-20 21:00 -03:00
    monkeypatch.setenv("SOURCE_DATE_EPOCH", str(epoch))
    assert main(["detect-drift", "--window", "mes_1"]) == 0
    text = (project.reports_dir / "evidently_drift_summary.json").read_text("utf-8")
    assert DriftReport.model_validate_json(text).generated_at == datetime.fromtimestamp(
        epoch, UTC
    )
    html = (project.reports_dir / "evidently" / "mes_1_drift_report.html").read_text(
        "utf-8"
    )
    stamps = [int(f.replace("-", "")[:12], 16) for f in UUID7.findall(html)]
    assert min(stamps) == epoch * 1000


def test_invalid_window_blocks_the_run_with_exit_2(project: Settings) -> None:
    path = project.production_dir / "mes_2.parquet"
    broken = pd.read_parquet(path).drop(columns=["renda_mensal"])
    write_parquet(broken, path)
    assert main(["detect-drift", "--window", "mes_2"]) == 2
    assert not (project.reports_dir / "evidently_drift_summary.json").exists()


def test_missing_window_points_to_simulate(
    project: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    (project.production_dir / "mes_1.parquet").unlink()
    capsys.readouterr()
    assert main(["detect-drift", "--window", "mes_1"]) == 1
    assert "simulate" in capsys.readouterr().out


def test_missing_model_points_to_train(
    project: Settings,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CS_PROJECT_ROOT", str(project.project_root / "vazio"))
    get_settings.cache_clear()
    capsys.readouterr()
    assert main(["detect-drift"]) == 1
    assert "train" in capsys.readouterr().out
    monkeypatch.undo()
    get_settings.cache_clear()
