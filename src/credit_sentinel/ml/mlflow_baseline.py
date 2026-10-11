"""Registro do baseline no MLflow (experimento e Model Registry locais)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import mlflow
import mlflow.sklearn

from credit_sentinel.ml.train_baseline import model_signature

if TYPE_CHECKING:
    import pandas as pd

    from credit_sentinel.ml.train_baseline import ModelBundle
    from credit_sentinel.settings import Settings

EXPERIMENT_NAME = "credit-sentinel/baseline"
REGISTERED_MODEL = "credit-scoring-model"
_SQLITE_PREFIX = "sqlite:///"
_LOGGED_PARAMS = ("max_iter", "learning_rate", "max_leaf_nodes", "l2_regularization")
# Tipos revisados do pipeline HGB próprio, liberados para a serialização skops do
# MLflow 3.x (o modelo é gerado por este projeto, nunca baixado de terceiros).
SKOPS_TRUSTED_TYPES = [
    "functools.partial",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.utils.validation.check_array",
]


def resolve_tracking_uri(settings: Settings) -> str:
    """Converte URIs SQLite relativas em absolutas sob a raiz do projeto.

    Args:
        settings: Configuração do projeto.

    Returns:
        URI de tracking pronta para ``mlflow.set_tracking_uri``.
    """
    uri = settings.mlflow_tracking_uri
    if (
        not uri.startswith(_SQLITE_PREFIX)
        or Path(uri.removeprefix(_SQLITE_PREFIX)).is_absolute()
    ):
        return uri
    path = settings.project_root / uri.removeprefix(_SQLITE_PREFIX)
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"{_SQLITE_PREFIX}{path}"


def ensure_experiment(name: str, tracking_uri: str) -> str:
    """Obtém ou cria o experimento, com artefatos ao lado do banco SQLite.

    Args:
        name: Nome do experimento.
        tracking_uri: URI de tracking já resolvida.

    Returns:
        Id do experimento.
    """
    existing = mlflow.get_experiment_by_name(name)
    if existing is not None:
        return existing.experiment_id
    location = None
    if tracking_uri.startswith(_SQLITE_PREFIX):
        root = Path(tracking_uri.removeprefix(_SQLITE_PREFIX)).parent / "artifacts"
        location = (root / name.replace("/", "__")).as_uri()
    return mlflow.create_experiment(name, artifact_location=location)


def _metrics(bundle: ModelBundle) -> dict[str, float]:
    data = bundle.metrics.model_dump(
        exclude={"confusion_matrix", "trained_at", "model_version"}
    )
    if bundle.threshold is not None:
        data["cost_threshold"] = bundle.threshold.cost_threshold
    return {key: float(value) for key, value in data.items()}


def log_baseline_run(
    bundle: ModelBundle, example: pd.DataFrame, settings: Settings
) -> str:
    """Registra params, métricas, artefatos e o modelo com assinatura.

    O modelo é registrado no Model Registry como ``credit-scoring-model``.

    Args:
        bundle: Modelo treinado e analisado.
        example: Algumas linhas de features (``input_example``).
        settings: Configuração (URI de tracking).

    Returns:
        Id do run criado.
    """
    uri = resolve_tracking_uri(settings)
    mlflow.set_tracking_uri(uri)
    experiment_id = ensure_experiment(EXPERIMENT_NAME, uri)
    params = bundle.pipeline.named_steps["clf"].get_params()
    with mlflow.start_run(experiment_id=experiment_id, run_name=bundle.version) as run:
        mlflow.set_tags({"model_version": bundle.version, "stage": "baseline"})
        mlflow.log_params(
            {key: params[key] for key in _LOGGED_PARAMS} | {"seed": settings.seed}
        )
        mlflow.log_metrics(_metrics(bundle))
        mlflow.log_dict(
            [i.model_dump() for i in bundle.importance], "feature_importance.json"
        )
        mlflow.sklearn.log_model(
            bundle.pipeline,
            name="model",
            signature=model_signature(),
            input_example=example,
            registered_model_name=REGISTERED_MODEL,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
    return run.info.run_id
