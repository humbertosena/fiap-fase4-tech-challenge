"""Motor de janelas temporais de produção (ADR 004).

Cada janela reutiliza o gerador da referência com parâmetros sobrescritos em
``config/simulation_scenarios.yaml``. A semente de cada janela é a semente base
mais a posição da janela em ``WINDOWS`` (``mes_1`` → 43, ``mes_2`` → 44...), de
modo que os lotes são distintos entre si e idênticos a cada execução.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.generator import (
    ScenarioParams,
    generate_portfolio,
    load_period,
    load_scenarios,
)
from credit_sentinel.ml.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
)
from credit_sentinel.utils.io import read_yaml, write_parquet
from credit_sentinel.utils.seeds import DEFAULT_SEED, make_rng

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pandas as pd

WINDOWS = ("mes_1", "mes_2", "mes_3_crise")
DEFAULT_ROWS = 5000


class WindowSpec(BaseModel):
    """Especificação completa de uma janela de produção.

    Attributes:
        name: Nome do cenário (``mes_1``...).
        period: Datas inicial e final (inclusivas) de ``data_referencia``.
        n_rows: Quantidade de clientes da janela.
        params: Parâmetros do gerador já mesclados com a referência.
        seed: Semente do gerador desta janela.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    period: tuple[date, date]
    n_rows: int
    params: ScenarioParams
    seed: int


class NumericShift(BaseModel):
    """Deslocamento de uma feature numérica entre referência e janela.

    Attributes:
        feature: Nome da feature.
        mean_ref: Média na referência.
        mean_cur: Média na janela.
        median_ref: Mediana na referência.
        median_cur: Mediana na janela.
        relative_change: Variação relativa da média (``cur / ref - 1``).
    """

    feature: str
    mean_ref: float
    mean_cur: float
    median_ref: float
    median_cur: float
    relative_change: float


class CategoryShift(BaseModel):
    """Deslocamento da participação de uma categoria.

    Attributes:
        feature: Nome da feature categórica.
        category: Valor da categoria.
        share_ref: Participação na referência (0 a 1).
        share_cur: Participação na janela (0 a 1).
        delta_pp: Diferença em pontos percentuais.
    """

    feature: str
    category: str
    share_ref: float
    share_cur: float
    delta_pp: float


class ShiftDescription(BaseModel):
    """Resumo "antes e depois" de uma janela (README e páginas HTML).

    Attributes:
        reference_rows: Linhas da referência.
        window_rows: Linhas da janela.
        base_rate_ref: Taxa de inadimplência da referência.
        base_rate_cur: Taxa de inadimplência da janela.
        numeric: Deslocamento das 12 features numéricas.
        categorical: Deslocamento de cada categoria das 2 features categóricas.
    """

    reference_rows: int
    window_rows: int
    base_rate_ref: float
    base_rate_cur: float
    numeric: list[NumericShift]
    categorical: list[CategoryShift]


def _scenario_entry(config_path: Path, name: str) -> dict:
    return read_yaml(config_path)["scenarios"][name]


def load_windows(config_path: Path, seed: int = DEFAULT_SEED) -> list[WindowSpec]:
    """Carrega as janelas de produção declaradas em ``WINDOWS``.

    Args:
        config_path: Caminho de ``simulation_scenarios.yaml``.
        seed: Semente base; cada janela soma a sua posição (1, 2, ...).

    Returns:
        Especificações na ordem cronológica.
    """
    scenarios = load_scenarios(config_path)
    return [
        WindowSpec(
            name=name,
            period=load_period(config_path, name),
            n_rows=int(_scenario_entry(config_path, name).get("rows", DEFAULT_ROWS)),
            params=scenarios[name],
            seed=seed + position,
        )
        for position, name in enumerate(WINDOWS, start=1)
    ]


def simulate_window(spec: WindowSpec) -> pd.DataFrame:
    """Gera o lote de uma janela com o mesmo DGP da referência.

    Args:
        spec: Especificação da janela.

    Returns:
        DataFrame com as 18 colunas do dicionário, inclusive ``inadimplente``.
    """
    rng = make_rng(spec.seed)
    return generate_portfolio(spec.n_rows, spec.params, rng, spec.period)


def simulate_all(
    config_path: Path,
    out_dir: Path,
    windows: Sequence[str] | None = None,
    seed: int = DEFAULT_SEED,
) -> dict[str, Path]:
    """Gera e grava as janelas em ``<out_dir>/<janela>.parquet``.

    Args:
        config_path: Caminho de ``simulation_scenarios.yaml``.
        out_dir: Diretório de saída (``data/production``).
        windows: Janelas a gerar; ``None`` gera todas.
        seed: Semente base das janelas.

    Returns:
        Mapa nome da janela → arquivo gravado.

    Raises:
        ValueError: Se alguma janela pedida não existir.
    """
    specs = {spec.name: spec for spec in load_windows(config_path, seed)}
    selected = list(windows) if windows is not None else list(specs)
    unknown = [name for name in selected if name not in specs]
    if unknown:
        msg = f"janela(s) desconhecida(s): {', '.join(unknown)}; válidas: {WINDOWS}"
        raise ValueError(msg)
    return {
        name: write_parquet(simulate_window(specs[name]), out_dir / f"{name}.parquet")
        for name in selected
    }


def _numeric_shifts(ref: pd.DataFrame, cur: pd.DataFrame) -> list[NumericShift]:
    shifts = []
    for feature in NUMERIC_FEATURES:
        mean_ref, mean_cur = float(ref[feature].mean()), float(cur[feature].mean())
        shifts.append(
            NumericShift(
                feature=feature,
                mean_ref=mean_ref,
                mean_cur=mean_cur,
                median_ref=float(ref[feature].median()),
                median_cur=float(cur[feature].median()),
                relative_change=mean_cur / mean_ref - 1 if mean_ref else 0.0,
            )
        )
    return shifts


def _category_shifts(ref: pd.DataFrame, cur: pd.DataFrame) -> list[CategoryShift]:
    shifts = []
    for feature in CATEGORICAL_FEATURES:
        share_ref = ref[feature].value_counts(normalize=True)
        share_cur = cur[feature].value_counts(normalize=True)
        for category in dd.ENUMS[feature]:
            before = float(share_ref.get(category, 0.0))
            after = float(share_cur.get(category, 0.0))
            shifts.append(
                CategoryShift(
                    feature=feature,
                    category=category,
                    share_ref=before,
                    share_cur=after,
                    delta_pp=100 * (after - before),
                )
            )
    return shifts


def describe_shift(reference: pd.DataFrame, window: pd.DataFrame) -> ShiftDescription:
    """Compara médias, medianas e participações entre referência e janela.

    Args:
        reference: Dataset de referência (18 colunas).
        window: Lote da janela (18 colunas).

    Returns:
        Descrição serializável do deslocamento.
    """
    return ShiftDescription(
        reference_rows=len(reference),
        window_rows=len(window),
        base_rate_ref=float(reference[dd.TARGET].mean()),
        base_rate_cur=float(window[dd.TARGET].mean()),
        numeric=_numeric_shifts(build_features(reference), build_features(window)),
        categorical=_category_shifts(reference, window),
    )
