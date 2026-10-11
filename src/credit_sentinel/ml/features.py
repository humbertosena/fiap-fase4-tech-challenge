"""Engenharia de features única, compartilhada por treino e inferência."""

from __future__ import annotations

from typing import TYPE_CHECKING

from credit_sentinel.data import dictionary as dd

if TYPE_CHECKING:
    import pandas as pd

DERIVED_FEATURES = ["comprometimento_renda"]
NUMERIC_FEATURES = [*dd.NUMERIC_COLUMNS, *DERIVED_FEATURES]
CATEGORICAL_FEATURES = list(dd.CATEGORICAL_COLUMNS)
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Seleciona e deriva as 14 features do modelo.

    ``genero``, ``regiao``, identificadores, datas e o alvo nunca entram.

    Args:
        df: Lote com as colunas do dicionário de dados.

    Returns:
        DataFrame com as colunas de ``MODEL_FEATURES``, na ordem fixa.
    """
    features = df[dd.FEATURE_COLUMNS].copy()
    features["possui_imovel"] = features["possui_imovel"].astype("int64")
    features["comprometimento_renda"] = (
        features["valor_parcela"] / features["renda_mensal"]
    )
    return features[MODEL_FEATURES]
