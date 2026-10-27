"""Hook do MLflow: run pai, runs aninhados, métricas de drift e artefatos por janela."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest
from mlflow import MlflowClient

from credit_sentinel.observability.pipeline_runner import (
    WINDOW_ORDER,
    AuditReport,
    run_pipeline,
)
from credit_sentinel.tracking.mlflow_tracking import (
    EXPERIMENT_MONITORING,
    MlflowTrackingHook,
    hook_factory,
)

if TYPE_CHECKING:
    from pathlib import Path

    from mlflow.entities import Run

    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.integration

PARENT_TAG = "mlflow.parentRunId"


@dataclass(frozen=True)
class Tracked:
    """Execução do pipeline com tracking e os runs resultantes no MLflow."""

    report: AuditReport
    client: MlflowClient
    parent: Run
    children: dict[str, Run]


def _runs(client: MlflowClient) -> list[Run]:
    experiment = client.get_experiment_by_name(EXPERIMENT_MONITORING)
    assert experiment is not None
    return client.search_runs([experiment.experiment_id])


@pytest.fixture(scope="module")
def tracked(
    pipeline_project: Settings, tmp_path_factory: pytest.TempPathFactory
) -> Tracked:
    uri = f"sqlite:///{tmp_path_factory.mktemp('mlflow')}/mlflow.db"
    report = run_pipeline(
        list(WINDOW_ORDER),
        settings=pipeline_project,
        quick=True,
        hook_factory=hook_factory(uri, pipeline_project.project_root),
    )
    client = MlflowClient(tracking_uri=uri)
    runs = _runs(client)
    parent = next(r for r in runs if PARENT_TAG not in r.data.tags)
    children = {
        r.data.tags["mlflow.runName"]: r for r in runs if PARENT_TAG in r.data.tags
    }
    return Tracked(report, client, parent, children)


def test_one_parent_run_and_one_nested_run_per_window(tracked: Tracked) -> None:
    assert tracked.parent.info.run_name == f"pipeline-{tracked.report.run_id}"
    assert set(tracked.children) == set(WINDOW_ORDER)
    for child in tracked.children.values():
        assert child.data.tags[PARENT_TAG] == tracked.parent.info.run_id
        assert child.info.status == "FINISHED"
    assert tracked.parent.info.status == "FINISHED"


def test_parent_run_summarizes_the_execution(tracked: Tracked) -> None:
    parent = tracked.client.get_run(tracked.parent.info.run_id)
    assert parent.data.params["model_version"] == "baseline-1.0.0"
    assert parent.data.metrics["windows_done"] == 4
    assert parent.data.metrics["windows_blocked"] == 1
    assert parent.data.tags["stage"] == "monitoring"


def test_drift_metrics_are_logged_per_window(tracked: Tracked) -> None:
    crisis = tracked.children["mes_3_crise"].data.metrics
    assert crisis["psi_utilizacao_credito"] > 0.25
    assert crisis["ks_p_utilizacao_credito"] < 0.05
    assert crisis["psi_score"] > 0.1
    assert crisis["budget_SLO2_estabilidade_psi"] > 1.0
    assert crisis["n_alerts_P2"] == 1
    control = tracked.children["mes_1"].data.metrics
    assert control["psi_utilizacao_credito"] < 0.1
    assert control["n_alerts_P4"] == 1
    assert {"roc_auc", "f1", "brier", "base_rate"} <= set(control)


def test_blocked_window_is_tagged_and_has_no_drift_metrics(tracked: Tracked) -> None:
    bad = tracked.children["bad_batch"]
    assert bad.data.tags["decision"] == "BLOCKED"
    assert bad.data.metrics["rejected_ratio"] > 0.01
    assert bad.data.metrics["n_alerts_P1"] == 1
    assert bad.data.metrics["remaining_budget"] < 0
    assert not [name for name in bad.data.metrics if name.startswith("psi_")]
    assert "roc_auc" not in bad.data.metrics


def test_quality_and_latency_metrics(tracked: Tracked) -> None:
    metrics = tracked.children["mes_2"].data.metrics
    assert metrics["validated_records"] == 500
    assert metrics["rejected_records"] == 0
    assert metrics["rejected_ratio"] == 0
    for stage in ("ingestion", "contract_validation", "inference", "drift_detection"):
        assert metrics[f"latency_{stage}_ms"] >= 0
    assert "remaining_budget" in metrics


def test_run_tags_and_params_identify_the_window(tracked: Tracked) -> None:
    crisis = tracked.children["mes_3_crise"]
    assert crisis.data.tags["stage"] == "monitoring"
    assert crisis.data.tags["decision"] == "SUCCESS"
    assert crisis.data.tags["window"] == "mes_3_crise"
    assert crisis.data.tags["release"] == "v0.3.0-rc"
    assert crisis.data.tags["git_sha"]
    assert crisis.data.params["window"] == "mes_3_crise"
    assert crisis.data.params["schema"] == "credit_scoring_inference"
    assert crisis.data.params["limite_SLO2_estabilidade_psi"] == "<= 0.25"


def _artifacts(tracked: Tracked, window: str, folder: str) -> list[str]:
    run_id = tracked.children[window].info.run_id
    return [a.path for a in tracked.client.list_artifacts(run_id, folder)]


def test_window_artifacts_are_attached(tracked: Tracked) -> None:
    assert _artifacts(tracked, "mes_1", "validation") == [
        "validation/validation_report.json"
    ]
    assert _artifacts(tracked, "mes_1", "slo") == ["slo/slo_evaluation.json"]
    assert _artifacts(tracked, "mes_1", "logs") == ["logs/events.jsonl"]


def test_full_mode_attaches_the_evidently_report(
    pipeline_project: Settings, tmp_path: Path
) -> None:
    uri = f"sqlite:///{tmp_path}/mlflow.db"
    run_pipeline(
        ["mes_2"],
        settings=pipeline_project,
        quick=False,
        hook_factory=hook_factory(uri, pipeline_project.project_root),
    )
    client = MlflowClient(tracking_uri=uri)
    (child,) = [r for r in _runs(client) if PARENT_TAG in r.data.tags]
    files = [a.path for a in client.list_artifacts(child.info.run_id, "evidently")]
    assert files == [
        "evidently/mes_2_drift_report.html",
        "evidently/mes_2_drift_report.json",
    ]
    assert child.data.metrics["share_drifted_evidently"] > 0


def test_unreachable_store_marks_tracking_failed_but_the_pipeline_goes_on(
    project_clean: Settings, tmp_path: Path
) -> None:
    blocker = tmp_path / "arquivo"
    blocker.write_text("não é um diretório", encoding="utf-8")
    uri = f"sqlite:///{blocker}/mlflow.db"
    report = run_pipeline(
        ["bad_batch", "mes_1"],
        settings=project_clean,
        quick=True,
        hook_factory=hook_factory(uri, project_clean.project_root),
    )
    assert [w.stage_status["tracking"] for w in report.windows] == ["FAILED"] * 2
    assert [w.stage_status["slo_evaluation"] for w in report.windows] == ["SUCCESS"] * 2
    assert report.totals["blocked"] == 1


def test_hook_class_is_a_tracking_hook(tmp_path: Path) -> None:
    hook = MlflowTrackingHook(f"sqlite:///{tmp_path}/m.db", "exp", "run-1", "v1")
    assert callable(hook.on_window_complete)
