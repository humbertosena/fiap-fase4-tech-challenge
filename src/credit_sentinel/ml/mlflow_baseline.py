"""Registro do baseline no MLflow (experimento e Model Registry locais)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import mlflow
import mlflow.sklearn

from credit_sentinel.ml.train_baseline import model_signature

if TYPE_CHECKING:
    import pandas as pd

    from credit_sentinel.ml.train_baseline import ModelBundle
    from credit_sentinel.settings import Settings

EXPERIMENT_NAME = "credit-sentinel/baseline"
REGISTERED_MODEL = "credit-scoring-model"
BUNDLE_FOLDER = "bundle"
BASELINE_DIR = "baseline"
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


def bundle_metrics(bundle: ModelBundle) -> dict[str, float]:
    """Métricas numéricas de validação do bundle, prontas para o MLflow.

    Args:
        bundle: Modelo treinado e analisado.

    Returns:
        Nome da métrica → valor (inclui o limiar de custo, se houver).
    """
    data = bundle.metrics.model_dump(
        exclude={"confusion_matrix", "trained_at", "model_version"}
    )
    if bundle.threshold is not None:
        data["cost_threshold"] = bundle.threshold.cost_threshold
    return {key: float(value) for key, value in data.items()}


def bundle_params(bundle: ModelBundle, settings: Settings) -> dict[str, Any]:
    """Hiperparâmetros do classificador e a semente, prontos para o MLflow.

    Args:
        bundle: Modelo treinado.
        settings: Configuração (semente).

    Returns:
        Nome do parâmetro → valor.
    """
    params = bundle.pipeline.named_steps["clf"].get_params()
    return {key: params[key] for key in _LOGGED_PARAMS} | {"seed": settings.seed}


def log_registered_model(bundle: ModelBundle, example: pd.DataFrame) -> None:
    """Registra o pipeline como nova versão de ``credit-scoring-model``.

    Deve ser chamada dentro de um run ativo.

    Args:
        bundle: Modelo treinado e analisado.
        example: Algumas linhas de features (``input_example``).
    """
    mlflow.sklearn.log_model(
        bundle.pipeline,
        name="model",
        signature=model_signature(),
        input_example=example,
        registered_model_name=REGISTERED_MODEL,
        skops_trusted_types=SKOPS_TRUSTED_TYPES,
    )


def _log_bundle_dir(settings: Settings) -> None:
    """Anexa ao run a pasta do bundle, para recarregar o modelo pelo registry."""
    bundle_dir = settings.models_dir / BASELINE_DIR
    if bundle_dir.exists():
        mlflow.log_artifacts(str(bundle_dir), BUNDLE_FOLDER)


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
    with mlflow.start_run(experiment_id=experiment_id, run_name=bundle.version) as run:
        mlflow.set_tags({"model_version": bundle.version, "stage": "baseline"})
        mlflow.log_params(bundle_params(bundle, settings))
        mlflow.log_metrics(bundle_metrics(bundle))
        mlflow.log_dict(
            [i.model_dump() for i in bundle.importance], "feature_importance.json"
        )
        _log_bundle_dir(settings)
        log_registered_model(bundle, example)
    return run.info.run_id
