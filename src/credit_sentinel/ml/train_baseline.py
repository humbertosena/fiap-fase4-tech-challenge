"""Treino, avaliação e serialização do modelo baseline (ADR 003).

Pipeline sklearn: ``ColumnTransformer`` (numéricas em *passthrough*, categóricas
com ``OrdinalEncoder``) seguido de ``HistGradientBoostingClassifier``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import joblib
from mlflow.models import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from pydantic import BaseModel
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from credit_sentinel.data import dictionary as dd
from credit_sentinel.ml.evaluation import (
    FeatureImportance,
    ThresholdReport,
    best_f1_threshold,
    classification_metrics,
    feature_importance,
    threshold_report,
)
from credit_sentinel.ml.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
)
from credit_sentinel.utils.clock import utc_now
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
    """Métricas offline do modelo no split de validação (limiar de F1)."""

    roc_auc: float
    average_precision: float
    f1: float
    precision: float
    recall: float
    brier: float
    ks_statistic: float
    threshold: float
    confusion_matrix: list[list[int]]
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
        threshold: Limiares de F1 e de custo, com histograma de scores.
        importance: Importância por permutação, em ordem decrescente.
        signature: Assinatura de entrada/saída (formato JSON simplificado).
        version: Versão semântica do modelo.
    """

    pipeline: Pipeline
    metrics: TrainingMetrics
    threshold: ThresholdReport | None = None
    importance: list[FeatureImportance] = field(default_factory=list)
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


def signature_json() -> dict[str, Any]:
    """Assinatura em JSON simplificado (nomes e tipos das colunas)."""
    signature = model_signature()
    inputs = [{"name": c.name, "type": c.type.name} for c in signature.inputs.inputs]
    return {"inputs": inputs, "outputs": [{"type": "double"}]}


def _validation_scores(
    pipeline: Pipeline, valid: pd.DataFrame
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    features = build_features(valid)
    scores = pipeline.predict_proba(features)[:, 1]
    return features, valid[dd.TARGET].to_numpy(), scores


def train_baseline(
    df_reference: pd.DataFrame, *, settings: Settings, version: str = MODEL_VERSION
) -> tuple[Pipeline, TrainingMetrics]:
    """Treina o baseline e mede o desempenho no split de validação.

    Args:
        df_reference: Dataset de referência já validado pelo contrato.
        settings: Configuração (semente).
        version: Versão registrada nas métricas (``challenger-*`` no retreino).

    Returns:
        Pipeline ajustado e métricas de validação no limiar de F1.
    """
    train, valid = split_reference(df_reference, seed=settings.seed)
    pipeline = build_pipeline(settings.seed)
    pipeline.fit(build_features(train), train[dd.TARGET])
    _, y, scores = _validation_scores(pipeline, valid)
    stats = classification_metrics(y, scores, threshold=best_f1_threshold(y, scores))
    metrics = TrainingMetrics(
        **stats,
        n_train=len(train),
        n_valid=len(valid),
        base_rate=float(df_reference[dd.TARGET].mean()),
        trained_at=utc_now(),
        model_version=version,
    )
    return pipeline, metrics


def train_and_analyze(
    df_reference: pd.DataFrame, *, settings: Settings, version: str = MODEL_VERSION
) -> ModelBundle:
    """Treina o baseline e calcula limiares e importância de features.

    Args:
        df_reference: Dataset de referência já validado pelo contrato.
        settings: Configuração (semente).
        version: Versão do modelo (``challenger-*`` no retreino).

    Returns:
        Bundle completo, pronto para ``save_model_bundle``.
    """
    pipeline, metrics = train_baseline(df_reference, settings=settings, version=version)
    _, valid = split_reference(df_reference, seed=settings.seed)
    features, y, scores = _validation_scores(pipeline, valid)
    return ModelBundle(
        pipeline=pipeline,
        metrics=metrics,
        threshold=threshold_report(y, scores),
        importance=feature_importance(pipeline, features, y, seed=settings.seed),
        signature=signature_json(),
        version=version,
    )


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
    """Grava modelo, métricas, assinatura, limiares e importância.

    Args:
        bundle: Modelo e metadados.
        out_dir: Diretório de destino (criado se necessário).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle.pipeline, out_dir / "model.joblib")
    write_json(bundle.metrics, out_dir / "metrics.json")
    write_json(signature_json(), out_dir / "signature.json")
    if bundle.threshold is not None:
        write_json(bundle.threshold, out_dir / "threshold.json")
    if bundle.importance:
        top = [item.feature for item in bundle.importance[:3]]
        items = [item.model_dump() for item in bundle.importance]
        payload = {"metric": "roc_auc", "n_repeats": 10, "top3": top, "features": items}
        write_json(payload, out_dir / "feature_importance.json")


def _optional_threshold(directory: Path) -> ThresholdReport | None:
    path = directory / "threshold.json"
    return ThresholdReport.model_validate(read_json(path)) if path.exists() else None


def _optional_importance(directory: Path) -> list[FeatureImportance]:
    path = directory / "feature_importance.json"
    if not path.exists():
        return []
    return [FeatureImportance.model_validate(i) for i in read_json(path)["features"]]


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
        threshold=_optional_threshold(directory),
        importance=_optional_importance(directory),
        signature=read_json(directory / "signature.json"),
        version=metrics.model_version,
    )
