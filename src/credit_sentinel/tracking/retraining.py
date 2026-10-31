"""Registro do challenger no MLflow: run de retreino, versão no registry e alias."""

from __future__ import annotations

from typing import TYPE_CHECKING

from credit_sentinel.ml.mlflow_baseline import resolve_tracking_uri
from credit_sentinel.ml.retrain_challenger import CHALLENGER_DIR
from credit_sentinel.tracking.mlflow_tracking import (
    CURRENT_RELEASE,
    TrainingRunSpec,
    current_git_sha,
    log_training_run,
)
from credit_sentinel.tracking.registry import ModelRegistry
from credit_sentinel.utils.clock import utc_now
from credit_sentinel.utils.io import write_json

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from credit_sentinel.ml.evaluate_gates import GatesReport
    from credit_sentinel.ml.train_baseline import ModelBundle
    from credit_sentinel.settings import Settings

RETRAINING_EXPERIMENT = "credit-sentinel/retraining"
GATES_FOLDER = "gates"


def _delta_metrics(report: GatesReport) -> dict[str, float]:
    champion, challenger = report.champion, report.challenger
    metrics = {
        f"{role}_{name}": getattr(model, name)
        for role, model in (("champion", champion), ("challenger", challenger))
        for name in ("roc_auc", "brier", "approval_rate")
    }
    metrics["delta_roc_auc"] = challenger.roc_auc - champion.roc_auc
    metrics["delta_brier"] = challenger.brier - champion.brier
    metrics["gates_passed"] = float(report.passed)
    metrics |= {f"gate_{g.gate_id}": float(g.passed) for g in report.results}
    return metrics


def _spec(report: GatesReport, settings: Settings) -> TrainingRunSpec:
    protocol = report.protocol
    params = {"holdout_window": report.holdout_window}
    if protocol is not None:
        params |= {
            "training_rows": str(protocol.training_rows),
            "sliding_months": str(protocol.sliding_months),
            "holdout_rows": str(protocol.holdout_rows),
        }
    return TrainingRunSpec(
        experiment=RETRAINING_EXPERIMENT,
        run_name=f"challenger-{utc_now():%Y-%m-%d}",
        tags={
            "stage": "retraining",
            "gates_passed": str(report.passed),
            "git_sha": current_git_sha(settings.project_root),
            "release": CURRENT_RELEASE,
        },
        params=params,
        metrics=_delta_metrics(report),
        artifacts={"bundle": settings.models_dir / CHALLENGER_DIR},
    )


def register_challenger(
    challenger: ModelBundle,
    report: GatesReport,
    settings: Settings,
    example: pd.DataFrame,
    gates_path: Path,
) -> tuple[GatesReport, ModelRegistry]:
    """Registra o run de retreino e a versão do challenger com o alias ``challenger``.

    O relatório de guardrails é regravado com as versões do registry e anexado ao run.

    Args:
        challenger: Modelo retreinado, já salvo em ``models/challenger``.
        report: Resultado dos guardrails.
        settings: Configuração do projeto.
        example: Algumas linhas de features (``input_example``).
        gates_path: Onde gravar o ``gates_report.json`` final.

    Returns:
        O relatório com as versões preenchidas e o registry usado.
    """
    run_id = log_training_run(challenger, _spec(report, settings), settings, example)
    registry = ModelRegistry(resolve_tracking_uri(settings))
    version = registry.version_for_run(run_id)
    registry.set_alias("challenger", version)
    final = report.model_copy(
        update={
            "champion_version": registry.version_of("champion"),
            "challenger_version": version,
        }
    )
    write_json(final, gates_path)
    registry.client.log_artifact(run_id, str(gates_path), GATES_FOLDER)
    return final, registry
