"""Treino, avaliação e serialização do modelo baseline (ADR 003).

Pipeline sklearn: ``ColumnTransformer`` (numéricas em *passthrough*, categóricas
com ``OrdinalEncoder``) seguido de ``HistGradientBoostingClassifier``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import joblib
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from credit_sentinel.data import dictionary as dd
from credit_sentinel.ml.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
)
from credit_sentinel.utils.io import read_json, write_json

if TYPE_CHECKING:
    from pathlib import Path

    import numpy as np
    import pandas as pd

    from credit_sentinel.settings import Settings

MODEL_VERSION = "baseline-1.0.0"
VALID_FRACTION = 0.2
_MLFLOW_TYPES = {
    "int64": "long",
    "float64": "double",
    "string": "string",
    "bool": "long",
}


class TrainingMetrics(BaseModel):
    """Métricas offline do modelo no split de validação."""

    roc_auc: float
    n_train: int
    n_valid: int
    base_rate: float
    trained_at: datetime
    model_version: str


@dataclass(frozen=True)
class ModelBundle:
    """Modelo treinado e seus metadados versionados.

    Attributes:
        pipeline: Pipeline sklearn ajustado.
        metrics: Métricas de validação.
        signature: Assinatura de entrada/saída (formato JSON simplificado).
        version: Versão semântica do modelo.
    """

    pipeline: Pipeline
    metrics: TrainingMetrics
    signature: dict[str, Any] = field(default_factory=dict)
    version: str = MODEL_VERSION


def split_reference(
    df: pd.DataFrame, *, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Divide a referência em treino/validação (80/20, estratificado).

    Args:
        df: Dataset de referência.
        seed: Semente do sorteio.

    Returns:
        Partições de treino e de validação.
    """
    train, valid = train_test_split(
        df, test_size=VALID_FRACTION, stratify=df[dd.TARGET], random_state=seed
    )
    return train, valid


def build_pipeline(seed: int) -> Pipeline:
    """Monta o pipeline não ajustado do baseline.

    Args:
        seed: Semente do HGB (early stopping usa partição interna).

    Returns:
        Pipeline sklearn.
    """
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    features = ColumnTransformer(
        [
            ("num", "passthrough", NUMERIC_FEATURES),
            ("cat", encoder, CATEGORICAL_FEATURES),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.1,
        categorical_features=CATEGORICAL_FEATURES,
        random_state=seed,
    )
    return Pipeline([("features", features), ("clf", model)])


def model_signature() -> ModelSignature:
    """Assinatura MLflow: 14 colunas de entrada e probabilidade ``double``."""
    dtypes = {spec.name: spec.dtype for spec in dd.COLUMN_SPECS}
    dtypes["comprometimento_renda"] = "float64"
    inputs = [ColSpec(_MLFLOW_TYPES[dtypes[name]], name) for name in MODEL_FEATURES]
    return ModelSignature(inputs=Schema(inputs), outputs=Schema([ColSpec("double")]))


def _signature_json() -> dict[str, Any]:
    signature = model_signature()
    inputs = [{"name": c.name, "type": c.type.name} for c in signature.inputs.inputs]
    return {"inputs": inputs, "outputs": [{"type": "double"}]}


def train_baseline(
    df_reference: pd.DataFrame, *, settings: Settings
) -> tuple[Pipeline, TrainingMetrics]:
    """Treina o baseline e mede o desempenho no split de validação.

    Args:
        df_reference: Dataset de referência já validado pelo contrato.
        settings: Configuração (semente).

    Returns:
        Pipeline ajustado e métricas de validação.
    """
    train, valid = split_reference(df_reference, seed=settings.seed)
    pipeline = build_pipeline(settings.seed)
    pipeline.fit(build_features(train), train[dd.TARGET])
    scores = pipeline.predict_proba(build_features(valid))[:, 1]
    metrics = TrainingMetrics(
        roc_auc=float(roc_auc_score(valid[dd.TARGET], scores)),
        n_train=len(train),
        n_valid=len(valid),
        base_rate=float(df_reference[dd.TARGET].mean()),
        trained_at=datetime.now(UTC),
        model_version=MODEL_VERSION,
    )
    return pipeline, metrics


def predict_proba(bundle: ModelBundle, df: pd.DataFrame) -> np.ndarray:
    """Probabilidade de inadimplência para um lote bruto.

    Args:
        bundle: Modelo carregado.
        df: Lote com as colunas do dicionário (o alvo é ignorado).

    Returns:
        Vetor de probabilidades da classe positiva.
    """
    return bundle.pipeline.predict_proba(build_features(df))[:, 1]


def save_model_bundle(bundle: ModelBundle, out_dir: Path) -> None:
    """Grava ``model.joblib``, ``metrics.json`` e ``signature.json``.

    Args:
        bundle: Modelo e metadados.
        out_dir: Diretório de destino (criado se necessário).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.pipeline, out_dir / "model.joblib")
    write_json(bundle.metrics, out_dir / "metrics.json")
    write_json(_signature_json(), out_dir / "signature.json")


def load_model_bundle(directory: Path) -> ModelBundle:
    """Carrega um bundle gravado por ``save_model_bundle``.

    Args:
        directory: Diretório do bundle.

    Returns:
        Bundle pronto para ``predict_proba``.
    """
    metrics = TrainingMetrics.model_validate(read_json(directory / "metrics.json"))
    return ModelBundle(
        pipeline=joblib.load(directory / "model.joblib"),
        metrics=metrics,
        signature=read_json(directory / "signature.json"),
        version=metrics.model_version,
    )
