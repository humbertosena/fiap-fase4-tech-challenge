"""Testes do dicionário de dados."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from credit_sentinel.data import dictionary as dd

if TYPE_CHECKING:
    import pandas as pd

pytestmark = pytest.mark.unit


def test_column_groups_partition_all_columns() -> None:
    grouped = [
        dd.ID_COLUMN,
        dd.DATE_COLUMN,
        *dd.NUMERIC_COLUMNS,
        *dd.CATEGORICAL_COLUMNS,
        *dd.SENSITIVE_COLUMNS,
        dd.TARGET,
    ]
    assert sorted(grouped) == sorted(dd.ALL_COLUMNS)
    assert len(dd.ALL_COLUMNS) == 18


def test_feature_columns_exclude_sensitive_id_date_and_target() -> None:
    forbidden = {*dd.SENSITIVE_COLUMNS, dd.ID_COLUMN, dd.DATE_COLUMN, dd.TARGET}
    assert forbidden.isdisjoint(dd.FEATURE_COLUMNS)


def test_dictionary_matches_generated_frame(df_reference_small: pd.DataFrame) -> None:
    assert list(df_reference_small.columns) == dd.ALL_COLUMNS
    assert [spec.name for spec in dd.COLUMN_SPECS] == dd.ALL_COLUMNS


def test_render_markdown_lists_every_column() -> None:
    markdown = dd.render_markdown()
    assert markdown.startswith("# Dicionário de dados")
    for column in dd.ALL_COLUMNS:
        assert f"`{column}`" in markdown


def test_versioned_dictionary_is_in_sync_with_code() -> None:
    from credit_sentinel.settings import Settings

    versioned = Settings(_env_file=None).docs_dir / "data_dictionary.md"
    assert versioned.read_text(encoding="utf-8") == dd.render_markdown()
