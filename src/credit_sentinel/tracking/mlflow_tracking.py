"""Rastreio das execuções do pipeline no MLflow.

O experimento é ``credit-sentinel/monitoring``. Cada execução vira um run pai
(``pipeline-<run_id>``) com um run aninhado por janela. O run da janela leva
métricas de qualidade, drift (``psi_*`` e ``ks_p_*``), performance, latência e error
budget, mais os relatórios (contrato, SLO, log e Evidently) como artefatos. Usa o
``MlflowClient`` com a URI explícita, sem estado global, para que os testes escrevam
só em um store temporário.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Metric, Param
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID
from pydantic import BaseModel, Field

from credit_sentinel.ml.mlflow_baseline import (
    bundle_metrics,
    bundle_params,
    ensure_experiment,
    log_registered_model,
    resolve_tracking_uri,
)
from credit_sentinel.utils.clock import utc_now

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import pandas as pd

    from credit_sentinel.ml.train_baseline import ModelBundle
    from credit_sentinel.observability.pipeline_runner import (
        TrackingHook,
        WindowRunResult,
    )
    from credit_sentinel.settings import Settings

EXPERIMENT_MONITORING = "credit-sentinel/monitoring"
CURRENT_RELEASE = "v1.0.0-ga"
STAGE_TAG = "monitoring"
_SQLITE_PREFIX = "sqlite:///"
_SEVERITIES = ("P1", "P2", "P3", "P4")
_ARTIFACT_FOLDERS = {
    "validation_report": "validation",
    "slo_evaluation": "slo",
    "window_events": "logs",
    "evidently_html": "evidently",
    "evidently_json": "evidently",
}
_GIT_TIMEOUT_S = 5


def current_git_sha(root: Path) -> str:
    """Hash curto do commit em ``root``, ou ``unknown`` fora de um repositório.

    Args:
        root: Raiz do projeto.

    Returns:
        Saída de ``git rev-parse --short HEAD``.
    """
    try:
        done = subprocess.run(  # noqa: S603 - comando fixo, sem entrada externa
            ["git", "-C", str(root), "rev-parse", "--short", "HEAD"],  # noqa: S607
            capture_output=True,
            text=True,
            check=False,
            timeout=_GIT_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"
    return done.stdout.strip() or "unknown"


def artifact_location(tracking_uri: str, name: str) -> str | None:
    """Pasta de artefatos ao lado do banco SQLite, por experimento.

    Args:
        tracking_uri: URI de tracking.
        name: Nome do experimento.

    Returns:
        URI ``file://`` da pasta, ou ``None`` para backends que não são SQLite.
    """
    if not tracking_uri.startswith(_SQLITE_PREFIX):
        return None
    root = Path(tracking_uri.removeprefix(_SQLITE_PREFIX)).parent / "artifacts"
    return (root / name.replace("/", "__")).as_uri()


def experiment_id(client: MlflowClient, name: str, tracking_uri: str) -> str:
    """Obtém ou cria o experimento.

    Args:
        client: Cliente MLflow.
        name: Nome do experimento.
        tracking_uri: URI de tracking (define a pasta de artefatos).

    Returns:
        Id do experimento.
    """
    existing = client.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    location = artifact_location(tracking_uri, name)
    return client.create_experiment(name, artifact_location=location)


def start_pipeline_parent_run(
    client: MlflowClient, experiment: str, run_id: str, model_version: str
) -> str:
    """Cria o run pai de uma execução do pipeline.

    Args:
        client: Cliente MLflow.
        experiment: Id do experimento.
        run_id: Identificador da execução do pipeline.
        model_version: Versão do modelo avaliado.

    Returns:
        Id do run pai no MLflow.
    """
    parent = client.create_run(
        experiment,
        run_name=f"pipeline-{run_id}",
        tags={"stage": STAGE_TAG, "model_version": model_version},
    )
    client.log_param(parent.info.run_id, "pipeline_run_id", run_id)
    client.log_param(parent.info.run_id, "model_version", model_version)
    return parent.info.run_id


def _drift_metrics(result: WindowRunResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if result.drift is not None:
        for item in result.drift.numeric:
            metrics[f"psi_{item.feature}"] = item.psi
            metrics[f"ks_p_{item.feature}"] = item.ks_p_value
        for cat in result.drift.categorical:
            metrics[f"chi2_p_{cat.feature}"] = cat.chi2_p_value
            metrics[f"js_{cat.feature}"] = cat.jensen_shannon
    if result.share_drifted_evidently is not None:
        metrics["share_drifted_evidently"] = result.share_drifted_evidently
    return metrics


def _model_metrics(result: WindowRunResult) -> dict[str, float]:
    metrics: dict[str, float] = {}
    if result.predictions is not None:
        metrics["psi_score"] = result.predictions.psi_score_vs_reference
        metrics["approval_rate"] = result.predictions.approval_rate
    if result.performance is not None:
        perf = result.performance
        metrics |= {
            "roc_auc": perf.roc_auc,
            "f1": perf.f1,
            "brier": perf.brier,
            "base_rate": perf.base_rate,
            "auc_delta_vs_reference": perf.auc_delta_vs_reference,
        }
    return metrics


def window_metrics(result: WindowRunResult) -> dict[str, float]:
    """Métricas numéricas de uma janela (qualidade, drift, modelo, latência, budget).

    Args:
        result: Resultado da janela no pipeline.

    Returns:
        Nome da métrica → valor.
    """
    validation, slo = result.validation, result.slo
    metrics: dict[str, float] = {
        "validated_records": validation.valid_records,
        "rejected_records": validation.rejected_records,
        "rejected_ratio": validation.rejected_ratio,
        "remaining_budget": slo.remaining_budget,
    }
    metrics |= _drift_metrics(result) | _model_metrics(result)
    metrics |= {f"latency_{k}_ms": v for k, v in result.stage_latencies_ms.items()}
    metrics |= {f"budget_{r.slo_id}": r.consumption for r in slo.results}
    for severity in _SEVERITIES:
        metrics[f"n_alerts_{severity}"] = sum(
            a.severity == severity for a in slo.alerts
        )
    return {name: float(value) for name, value in metrics.items()}


def window_params(result: WindowRunResult) -> dict[str, str]:
    """Parâmetros de uma janela: identificação e limites dos SLOs avaliados.

    Args:
        result: Resultado da janela no pipeline.

    Returns:
        Nome do parâmetro → valor textual.
    """
    params = {
        "window": result.window,
        "model_version": result.model_version,
        "schema": result.validation.schema_name,
    }
    limits = {
        f"limite_{r.slo_id}": f"{r.op} {r.threshold:g}" for r in result.slo.results
    }
    return params | limits


class MlflowTrackingHook:
    """Registra cada janela do pipeline como um run aninhado no MLflow."""

    def __init__(
        self,
        tracking_uri: str,
        experiment: str,
        run_id: str,
        model_version: str,
        git_sha: str = "unknown",
    ) -> None:
        """Guarda os parâmetros; nenhuma conexão é aberta até a primeira janela.

        Args:
            tracking_uri: URI do backend de tracking.
            experiment: Nome do experimento de monitoramento.
            run_id: Identificador da execução do pipeline.
            model_version: Versão do modelo avaliado.
            git_sha: Hash do commit, gravado como tag.
        """
        self._uri = tracking_uri
        self._experiment = experiment
        self._run_id = run_id
        self._model_version = model_version
        self._git_sha = git_sha
        self._client: MlflowClient | None = None
        self._experiment_id: str | None = None
        self._parent_id: str | None = None
        self._done = 0
        self._blocked = 0

    def _prepare(self) -> tuple[MlflowClient, str, str]:
        if self._client is None:
            self._client = MlflowClient(tracking_uri=self._uri)
        if self._experiment_id is None:
            self._experiment_id = experiment_id(
                self._client, self._experiment, self._uri
            )
        if self._parent_id is None:
            self._parent_id = start_pipeline_parent_run(
                self._client, self._experiment_id, self._run_id, self._model_version
            )
        return self._client, self._experiment_id, self._parent_id

    def _tags(self, result: WindowRunResult, parent_id: str) -> dict[str, str]:
        return {
            MLFLOW_PARENT_RUN_ID: parent_id,
            "stage": STAGE_TAG,
            "decision": result.decision,
            "window": result.window,
            "model_version": result.model_version,
            "git_sha": self._git_sha,
            "release": CURRENT_RELEASE,
        }

    @staticmethod
    def _log(client: MlflowClient, child: str, result: WindowRunResult) -> None:
        stamp = int(utc_now().timestamp() * 1000)
        metrics = [Metric(k, v, stamp, 0) for k, v in window_metrics(result).items()]
        params = [Param(k, v) for k, v in window_params(result).items()]
        client.log_batch(child, metrics=metrics, params=params)

    @staticmethod
    def _attach(client: MlflowClient, child: str, artifacts: dict[str, Path]) -> None:
        for name, path in artifacts.items():
            folder = _ARTIFACT_FOLDERS.get(name)
            if folder is not None and path.exists():
                client.log_artifact(child, str(path), folder)

    def _update_parent(
        self, client: MlflowClient, parent_id: str, result: WindowRunResult
    ) -> None:
        self._done += 1
        self._blocked += result.decision == "BLOCKED"
        client.log_metric(parent_id, "windows_done", self._done)
        client.log_metric(parent_id, "windows_blocked", self._blocked)
        client.set_terminated(parent_id)

    def on_window_complete(
        self, result: WindowRunResult, artifacts: dict[str, Path]
    ) -> None:
        """Cria o run aninhado da janela e anexa métricas e artefatos.

        Args:
            result: Resultado da janela.
            artifacts: Arquivos que acompanham o resultado.
        """
        client, experiment, parent_id = self._prepare()
        child = client.create_run(
            experiment, run_name=result.window, tags=self._tags(result, parent_id)
        ).info.run_id
        try:
            self._log(client, child, result)
            self._attach(client, child, artifacts)
        except Exception:
            client.set_terminated(child, status="FAILED")
            raise
        client.set_terminated(child)
        self._update_parent(client, parent_id, result)


def hook_factory(
    tracking_uri: str, project_root: Path
) -> Callable[[str, str], Sequence[TrackingHook]]:
    """Fábrica de hooks para ``run_pipeline``.

    Args:
        tracking_uri: URI do backend de tracking já resolvida.
        project_root: Raiz do projeto (origem do hash do commit).

    Returns:
        Função que recebe ``(run_id, model_version)`` e devolve o hook do MLflow.
    """
    sha = current_git_sha(project_root)

    def build(run_id: str, model_version: str) -> Sequence[TrackingHook]:
        hook = MlflowTrackingHook(
            tracking_uri, EXPERIMENT_MONITORING, run_id, model_version, sha
        )
        return [hook]

    return build


class TrainingRunSpec(BaseModel):
    """O que registrar, além do próprio modelo, em um run de treino.

    Attributes:
        experiment: Nome do experimento (ex.: ``credit-sentinel/retraining``).
        run_name: Nome do run.
        tags: Tags do run.
        params: Parâmetros adicionais do run.
        metrics: Métricas adicionais (ex.: deltas contra o champion).
        artifacts: Pasta de destino no run → arquivo ou diretório local.
    """

    experiment: str
    run_name: str
    tags: dict[str, str] = Field(default_factory=dict)
    params: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    artifacts: dict[str, Path] = Field(default_factory=dict)


def _attach_run_artifacts(artifacts: dict[str, Path]) -> None:
    for folder, path in artifacts.items():
        if path.is_dir():
            mlflow.log_artifacts(str(path), folder)
        else:
            mlflow.log_artifact(str(path), folder)


def log_training_run(
    bundle: ModelBundle,
    spec: TrainingRunSpec,
    settings: Settings,
    example: pd.DataFrame,
) -> str:
    """Registra um run de treino e a nova versão do modelo no Model Registry.

    Args:
        bundle: Modelo treinado e analisado.
        spec: Experimento, nome, tags, métricas e artefatos adicionais.
        settings: Configuração (URI de tracking e semente).
        example: Algumas linhas de features (``input_example``).

    Returns:
        Id do run criado.
    """
    uri = resolve_tracking_uri(settings)
    mlflow.set_tracking_uri(uri)
    experiment = ensure_experiment(spec.experiment, uri)
    with mlflow.start_run(experiment_id=experiment, run_name=spec.run_name) as run:
        mlflow.set_tags({"model_version": bundle.version} | spec.tags)
        mlflow.log_params(bundle_params(bundle, settings) | spec.params)
        mlflow.log_metrics(bundle_metrics(bundle) | spec.metrics)
        _attach_run_artifacts(spec.artifacts)
        log_registered_model(bundle, example)
    return run.info.run_id
