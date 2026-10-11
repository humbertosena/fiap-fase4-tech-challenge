"""Leitura e escrita de artefatos com criação automática de diretórios."""

from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING, Any

import pandas as pd
import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path


def read_parquet(path: Path) -> pd.DataFrame:
    """Lê um arquivo parquet.

    Args:
        path: Caminho do arquivo.

    Returns:
        DataFrame carregado.
    """
    return pd.read_parquet(path)


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Grava um DataFrame em parquet (sem índice), criando diretórios.

    Args:
        df: DataFrame a gravar.
        path: Caminho de destino.

    Returns:
        O próprio ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def write_json(data: BaseModel | Mapping[str, Any] | list[Any], path: Path) -> Path:
    """Grava JSON indentado em UTF-8 preservando acentos.

    Args:
        data: Modelo pydantic, mapeamento ou lista serializável.
        path: Caminho de destino.

    Returns:
        O próprio ``path``.
    """
    payload = data.model_dump(mode="json") if isinstance(data, BaseModel) else data
    text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    return write_text(text + "\n", path)


def read_json(path: Path) -> Any:  # noqa: ANN401 - JSON arbitrário
    """Lê um arquivo JSON.

    Args:
        path: Caminho do arquivo.

    Returns:
        Estrutura Python correspondente ao JSON.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def read_yaml(path: Path) -> dict[str, Any]:
    """Lê um arquivo YAML com ``safe_load``.

    Args:
        path: Caminho do arquivo.

    Returns:
        Dicionário com o conteúdo.
    """
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def write_text(text: str, path: Path) -> Path:
    """Grava texto UTF-8 criando diretórios.

    Args:
        text: Conteúdo a gravar.
        path: Caminho de destino.

    Returns:
        O próprio ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def file_sha256(path: Path) -> str:
    """Calcula o SHA-256 de um arquivo (prova de reprodutibilidade).

    Args:
        path: Arquivo a resumir.

    Returns:
        Digest hexadecimal de 64 caracteres.
    """
    return hashlib.sha256(path.read_bytes()).hexdigest()
