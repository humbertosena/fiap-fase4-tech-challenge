"""Subcomando ``generate-data``: materializa os datasets versionados."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.corruptor import corrupt_batch
from credit_sentinel.data.generator import (
    SCENARIOS_FILE,
    generate_portfolio,
    load_period,
    load_scenarios,
)
from credit_sentinel.settings import Settings, get_settings
from credit_sentinel.utils.io import file_sha256, write_parquet, write_text
from credit_sentinel.utils.seeds import make_rng

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    import pandas as pd

REFERENCE_FILE = "credit_ref.parquet"
BAD_BATCH_FILE = "bad_batch.parquet"
DEFAULT_ROWS = 12_000
BAD_BATCH_ROWS = 600
BAD_BATCH_SEED = 7


class DatasetSummary(BaseModel):
    """Resumo de um dataset materializado.

    Attributes:
        path: Caminho relativo à raiz do projeto.
        rows: Número de linhas.
        columns: Número de colunas.
        base_rate: Taxa de inadimplência (ou ``None`` se não houver alvo).
        sha256: Digest do arquivo.
    """

    path: str
    rows: int
    columns: int
    base_rate: float | None
    sha256: str


def reference_path(settings: Settings) -> Path:
    """Caminho do dataset de referência."""
    return settings.reference_dir / REFERENCE_FILE


def summarize_dataset(
    frame: pd.DataFrame, path: Path, settings: Settings
) -> DatasetSummary:
    """Resume um dataset gravado (linhas, colunas, taxa base e hash).

    Args:
        frame: DataFrame gravado.
        path: Arquivo correspondente.
        settings: Configuração (para caminho relativo).

    Returns:
        Resumo serializável.
    """
    target = frame.get(dd.TARGET)
    return DatasetSummary(
        path=str(path.relative_to(settings.project_root)),
        rows=len(frame),
        columns=frame.shape[1],
        base_rate=None if target is None else float(target.mean()),
        sha256=file_sha256(path),
    )


def materialize_reference(
    settings: Settings, n_rows: int
) -> tuple[pd.DataFrame, DatasetSummary]:
    """Gera e grava ``credit_ref.parquet`` com a semente das settings.

    Args:
        settings: Configuração do projeto.
        n_rows: Quantidade de linhas.

    Returns:
        DataFrame gerado e resumo com hash e taxa base.
    """
    config = settings.config_dir / SCENARIOS_FILE
    params = load_scenarios(config)["reference"]
    period = load_period(config, "reference")
    frame = generate_portfolio(n_rows, params, make_rng(settings.seed), period)
    path = write_parquet(frame, reference_path(settings))
    return frame, summarize_dataset(frame, path, settings)


def materialize_bad_batch(
    reference: pd.DataFrame, settings: Settings
) -> DatasetSummary:
    """Amostra a referência (seed 7), injeta as anomalias e grava o lote.

    Args:
        reference: Dataset de referência limpo.
        settings: Configuração do projeto.

    Returns:
        Resumo do ``bad_batch.parquet``.
    """
    rng = make_rng(BAD_BATCH_SEED)
    size = min(BAD_BATCH_ROWS, len(reference))
    picks = np.sort(rng.choice(len(reference), size=size, replace=False))
    frame = corrupt_batch(reference.iloc[picks], rng)
    path = write_parquet(frame, settings.corrupted_dir / BAD_BATCH_FILE)
    return summarize_dataset(frame, path, settings)


def _print_summary(label: str, summary: DatasetSummary) -> None:
    print(f"{label}: {summary.path} ({summary.rows} linhas, {summary.columns} colunas)")
    if summary.base_rate is not None:
        print(f"  taxa base: {summary.base_rate:.2%}")
    print(f"  sha256: {summary.sha256}")


def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    if reference_path(settings).exists() and not args.force:
        print("erro: a referência já existe e é imutável; use --force para regerar")
        return 1
    reference, summary = materialize_reference(settings, args.rows)
    _print_summary("referência", summary)
    _print_summary("lote corrompido", materialize_bad_batch(reference, settings))
    dictionary_path = write_text(
        dd.render_markdown(), settings.docs_dir / "data_dictionary.md"
    )
    print(f"dicionário: {dictionary_path.relative_to(settings.project_root)}")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Registra o subcomando ``generate-data``.

    Args:
        subparsers: Grupo de subcomandos da CLI principal.
    """
    parser = subparsers.add_parser(
        "generate-data", help="gera a referência, o lote corrompido e o dicionário"
    )
    parser.add_argument(
        "--rows", type=int, default=DEFAULT_ROWS, help="linhas da referência"
    )
    parser.add_argument("--force", action="store_true", help="sobrescreve a referência")
    parser.set_defaults(handler=_run)
