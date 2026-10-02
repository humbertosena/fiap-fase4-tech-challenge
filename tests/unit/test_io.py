"""Testes dos utilitários de leitura e escrita."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pydantic import BaseModel

from credit_sentinel.utils.io import (
    read_json,
    read_parquet,
    read_yaml,
    write_json,
    write_parquet,
    write_text,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


class _Sample(BaseModel):
    nome: str
    valor: float


def test_parquet_roundtrip_creates_parent_dirs(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    path = write_parquet(df, tmp_path / "nested" / "file.parquet")
    pd.testing.assert_frame_equal(read_parquet(path), df)


def test_write_json_accepts_pydantic_and_keeps_accents(tmp_path: Path) -> None:
    path = write_json(_Sample(nome="ação", valor=1.5), tmp_path / "out" / "s.json")
    assert "ação" in path.read_text(encoding="utf-8")
    assert read_json(path) == {"nome": "ação", "valor": 1.5}


def test_write_json_accepts_mapping(tmp_path: Path) -> None:
    path = write_json({"chave": [1, 2]}, tmp_path / "m.json")
    assert read_json(path) == {"chave": [1, 2]}


def test_read_yaml_and_write_text(tmp_path: Path) -> None:
    path = write_text("a: 1\nb: [x, y]\n", tmp_path / "cfg" / "c.yaml")
    assert read_yaml(path) == {"a": 1, "b": ["x", "y"]}
