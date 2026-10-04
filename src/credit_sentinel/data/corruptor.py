"""Corruptor do lote de ingestão: injeta anomalias propositais e contáveis.

Cada anomalia recebe um bloco **disjunto** de linhas, sorteado por permutação,
para que as contagens sejam exatas e o inventário do contrato seja legível.
A coluna ``cpf`` é PII fictícia: formato válido, dígito verificador inválido.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from credit_sentinel.data import dictionary as dd

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

CPF_COLUMN = "cpf"
_CPF_DIGITS = 11
_UNEMPLOYED_TENURE = 24


@dataclass(frozen=True)
class AnomalySpec:
    """Anomalia injetada no lote corrompido.

    Attributes:
        key: Identificador estável da anomalia.
        description: Descrição em pt-BR.
        n_rows: Linhas afetadas (``0`` significa todas as linhas).
        rule_id: Regra do contrato que deve interceptá-la.
    """

    key: str
    description: str
    n_rows: int
    rule_id: str


ANOMALY_PLAN: tuple[AnomalySpec, ...] = (
    AnomalySpec("idade_fora_dominio", "idade negativa ou acima de 100", 30, "R04"),
    AnomalySpec("renda_nula", "renda_mensal nula", 24, "R03"),
    AnomalySpec("score_nulo", "score_bureau nulo", 18, "R03"),
    AnomalySpec("id_duplicado", "id_cliente duplicado", 18, "R02"),
    AnomalySpec("id_nulo", "id_cliente nulo", 6, "R02"),
    AnomalySpec("emprego_fora_enum", "situacao_emprego fora do enum", 18, "R08"),
    AnomalySpec("endividamento_acima_1", "taxa_endividamento maior que 1", 12, "R07"),
    AnomalySpec(
        "parcela_maior_renda", "valor_parcela maior que renda_mensal", 12, "R10"
    ),
    AnomalySpec(
        "desempregado_com_tempo", "tempo_emprego_meses > 0 com desempregado", 6, "R11"
    ),
    AnomalySpec("coluna_cpf", "coluna extra cpf (PII não prevista)", 0, "R01"),
)


def _invalid_age(df: pd.DataFrame, rows: np.ndarray, rng: np.random.Generator) -> None:
    half = len(rows) // 2
    df.loc[rows[:half], "idade"] = -rng.integers(1, 6, size=half)
    df.loc[rows[half:], "idade"] = rng.integers(101, 121, size=len(rows) - half)


def _null_income(df: pd.DataFrame, rows: np.ndarray, _rng: np.random.Generator) -> None:
    df.loc[rows, "renda_mensal"] = np.nan


def _null_score(df: pd.DataFrame, rows: np.ndarray, _rng: np.random.Generator) -> None:
    df["score_bureau"] = df["score_bureau"].astype(float)
    df.loc[rows, "score_bureau"] = np.nan


def _null_id(df: pd.DataFrame, rows: np.ndarray, _rng: np.random.Generator) -> None:
    df.loc[rows, dd.ID_COLUMN] = None


def _off_enum(df: pd.DataFrame, rows: np.ndarray, _rng: np.random.Generator) -> None:
    values = np.resize(np.array(["freelancer", "CLT "], dtype=object), len(rows))
    df.loc[rows, "situacao_emprego"] = values


def _debt_above_one(
    df: pd.DataFrame, rows: np.ndarray, rng: np.random.Generator
) -> None:
    df.loc[rows, "taxa_endividamento"] = np.round(
        1 + rng.uniform(0.05, 0.8, len(rows)), 4
    )


def _installment_above_income(
    df: pd.DataFrame, rows: np.ndarray, rng: np.random.Generator
) -> None:
    factor = rng.uniform(1.1, 1.5, len(rows))
    df.loc[rows, "valor_parcela"] = np.round(df.loc[rows, "renda_mensal"] * factor, 2)


def _unemployed_with_tenure(
    df: pd.DataFrame, rows: np.ndarray, _rng: np.random.Generator
) -> None:
    df.loc[rows, "situacao_emprego"] = "desempregado"
    df.loc[rows, "tempo_emprego_meses"] = _UNEMPLOYED_TENURE


_INJECTORS: dict[
    str, Callable[[pd.DataFrame, np.ndarray, np.random.Generator], None]
] = {
    "idade_fora_dominio": _invalid_age,
    "renda_nula": _null_income,
    "score_nulo": _null_score,
    "id_nulo": _null_id,
    "emprego_fora_enum": _off_enum,
    "endividamento_acima_1": _debt_above_one,
    "parcela_maior_renda": _installment_above_income,
    "desempregado_com_tempo": _unemployed_with_tenure,
}


def _check_digit(digits: list[int]) -> int:
    weight = len(digits) + 1
    total = sum(d * (weight - i) for i, d in enumerate(digits))
    remainder = (total * 10) % 11
    return 0 if remainder == 10 else remainder  # noqa: PLR2004 - regra do módulo 11


def is_valid_cpf(value: str) -> bool:
    """Valida um CPF pelos dígitos verificadores (módulo 11).

    Args:
        value: CPF com ou sem máscara.

    Returns:
        ``True`` se os dois dígitos verificadores conferem.
    """
    digits = [int(ch) for ch in re.sub(r"\D", "", value)]
    if len(digits) != _CPF_DIGITS or len(set(digits)) == 1:
        return False
    first = _check_digit(digits[:9])
    return digits[9:] == [first, _check_digit([*digits[:9], first])]


def _fake_cpf(rng: np.random.Generator) -> str:
    base = [int(d) for d in rng.integers(0, 10, size=9)]
    first = _check_digit(base)
    second = _check_digit([*base, first])
    digits = "".join(map(str, [*base, (first + 1) % 10, second]))
    return f"{digits[:3]}.{digits[3:6]}.{digits[6:9]}-{digits[9:]}"


def _blocks(order: np.ndarray) -> dict[str, np.ndarray]:
    blocks, cursor = {}, 0
    for spec in ANOMALY_PLAN:
        if spec.n_rows:
            blocks[spec.key] = order[cursor : cursor + spec.n_rows]
            cursor += spec.n_rows
    return blocks


def corrupt_batch(clean: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Gera o lote corrompido a partir de uma amostra limpa.

    Args:
        clean: Amostra limpa com as 18 colunas do dicionário.
        rng: Gerador aleatório (seed 7 no lote versionado).

    Returns:
        Cópia com as anomalias de ``ANOMALY_PLAN`` e a coluna ``cpf``.
    """
    df = clean.reset_index(drop=True).copy()
    order = rng.permutation(len(df))
    blocks = _blocks(order)
    duplicates = blocks.pop("id_duplicado")
    sources = order[-len(duplicates) :]
    df.loc[duplicates, dd.ID_COLUMN] = df.loc[sources, dd.ID_COLUMN].to_numpy()
    for key, rows in blocks.items():
        _INJECTORS[key](df, rows, rng)
    df[CPF_COLUMN] = [_fake_cpf(rng) for _ in range(len(df))]
    return df
