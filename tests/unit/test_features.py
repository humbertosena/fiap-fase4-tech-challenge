"""Testes da engenharia de features do modelo."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from credit_sentinel.data import dictionary as dd
from credit_sentinel.ml.features import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
)

pytestmark = pytest.mark.unit


def test_model_uses_fourteen_features_in_fixed_order(
    df_reference_small: pd.DataFrame,
) -> None:
    features = build_features(df_reference_small)
    assert list(features.columns) == MODEL_FEATURES
    assert len(MODEL_FEATURES) == 14
    assert MODEL_FEATURES == NUMERIC_FEATURES + CATEGORICAL_FEATURES


def test_sensitive_identifiers_and_target_are_excluded() -> None:
    forbidden = {*dd.SENSITIVE_COLUMNS, dd.ID_COLUMN, dd.DATE_COLUMN, dd.TARGET}
    assert forbidden.isdisjoint(MODEL_FEATURES)


def test_income_commitment_is_installment_over_income(
    df_reference_small: pd.DataFrame,
) -> None:
    features = build_features(df_reference_small)
    expected = df_reference_small["valor_parcela"] / df_reference_small["renda_mensal"]
    np.testing.assert_allclose(features["comprometimento_renda"], expected)


def test_boolean_owner_flag_becomes_integer(df_reference_small: pd.DataFrame) -> None:
    features = build_features(df_reference_small)
    assert features["possui_imovel"].dtype == np.int64


def test_input_frame_is_not_mutated(df_reference_small: pd.DataFrame) -> None:
    before = df_reference_small.copy()
    build_features(df_reference_small)
    pd.testing.assert_frame_equal(df_reference_small, before)
