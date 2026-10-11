"""Processo gerador de dados (DGP) da carteira de crédito sintética (ADR 001).

O mesmo gerador produz o dataset de referência e as janelas de produção: os
cenários de ``config/simulation_scenarios.yaml`` apenas sobrescrevem parâmetros
de ``reference``. A ordem de consumo do ``rng`` é fixa, garantindo arquivos
idênticos para a mesma semente.
"""

from __future__ import annotations

from datetime import date
from typing import TYPE_CHECKING, Any, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, model_validator
from scipy.optimize import bisect
from scipy.stats import truncnorm

from credit_sentinel.data import dictionary as dd
from credit_sentinel.utils.io import read_yaml

if TYPE_CHECKING:
    from pathlib import Path

SCENARIOS_FILE = "simulation_scenarios.yaml"
SCORE_BAND = (550, 700)
CONCEPT_WEIGHT = 0.9
_ID_SPACE = 16**10
_SHARE_TOLERANCE = 1e-6
_OWNER_AGE_THRESHOLD = 40
_SHARE_FIELDS = ("region_shares", "employment_shares", "purpose_shares", "term_shares")


class ScenarioParams(BaseModel):
    """Parâmetros do DGP de um cenário (espelha ``simulation_scenarios.yaml``).

    Attributes:
        beta0: Intercepto do logit, calibrado para a taxa base da referência.
        beta0_adjustment: Choque aditivo no intercepto (prior probability shift).
        gamma_concept: Intensidade do concept drift na faixa intermediária de score.
        coef_commitment: Peso do comprometimento da renda (parcela/renda) no logit.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    female_share: float = 0.51
    region_shares: dict[str, float]
    region_income_factor: dict[str, float]
    employment_shares: dict[str, float]
    employment_income_factor: dict[str, float]
    purpose_shares: dict[str, float]
    term_shares: dict[int, float]
    income_median: float = 3800.0
    income_sigma: float = 0.55
    income_real_factor: float = 1.0
    income_min: float = 800.0
    income_max: float = 60_000.0
    score_noise_sd: float = 60.0
    debt_noise_sd: float = 0.5
    debt_logit_shift: float = 0.0
    utilization_shift: float = 0.0
    monthly_interest: float = 0.025
    max_installment_ratio: float = 0.6
    beta0: float
    beta0_adjustment: float = 0.0
    gamma_concept: float = 0.0
    coef_commitment: float = 0.8

    @model_validator(mode="after")
    def _shares_sum_to_one(self) -> Self:
        for field in _SHARE_FIELDS:
            total = sum(getattr(self, field).values())
            if abs(total - 1.0) > _SHARE_TOLERANCE:
                msg = f"{field} deve somar 1 (soma atual {total:.4f})"
                raise ValueError(msg)
        return self


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _categorical(rng: np.random.Generator, shares: dict[Any, float], n: int) -> Any:  # noqa: ANN401
    keys = list(shares)
    probs = np.array([shares[key] for key in keys], dtype=float)
    picks = rng.choice(len(keys), size=n, p=probs / probs.sum())
    return np.asarray(keys, dtype=object)[picks]


def _demographics(n: int, params: ScenarioParams, rng: np.random.Generator) -> dict:
    low, high = (18 - 42) / 13, (85 - 42) / 13
    ages = truncnorm.rvs(low, high, loc=42, scale=13, size=n, random_state=rng)
    female = rng.random(n) < params.female_share
    return {
        "idade": np.rint(ages).astype(np.int64),
        "genero": np.where(female, "F", "M").astype(object),
        "regiao": _categorical(rng, params.region_shares, n),
    }


def _employment(
    ages: np.ndarray, params: ScenarioParams, rng: np.random.Generator
) -> dict:
    status = _categorical(rng, params.employment_shares, len(ages))
    tenure = np.floor(rng.gamma(2.0, 36.0, size=len(ages)))
    tenure = np.minimum(tenure, np.minimum((ages - 16) * 12, 600))
    tenure = np.where(status == "desempregado", 0, tenure)
    return {"situacao_emprego": status, "tempo_emprego_meses": tenure.astype(np.int64)}


def _factor(values: np.ndarray, factors: dict[str, float]) -> np.ndarray:
    return pd.Series(values).map(factors).fillna(1.0).to_numpy(dtype=float)


def _income(
    status: np.ndarray,
    regions: np.ndarray,
    params: ScenarioParams,
    rng: np.random.Generator,
) -> np.ndarray:
    base = rng.lognormal(np.log(params.income_median), params.income_sigma, len(status))
    income = base * _factor(status, params.employment_income_factor)
    income *= _factor(regions, params.region_income_factor) * params.income_real_factor
    return np.round(np.clip(income, params.income_min, params.income_max), 2)


def _risk_profile(
    z: np.ndarray, params: ScenarioParams, rng: np.random.Generator
) -> dict:
    n = len(z)
    score = 650 - 120 * z + rng.normal(0, params.score_noise_sd, n)
    debt_logit = -0.6 + params.debt_logit_shift + 0.7 * z
    debt = _sigmoid(debt_logit + rng.normal(0, params.debt_noise_sd, n))
    usage = rng.beta(2, 3, n) + 0.15 * z + params.utilization_shift
    late = np.minimum(rng.poisson(np.exp(-1.5 + 0.8 * z)), 12)
    return {
        "score_bureau": np.rint(np.clip(score, 0, 1000)).astype(np.int64),
        "taxa_endividamento": np.round(debt, 4),
        "utilizacao_credito": np.round(np.clip(usage, 0, 1), 4),
        "qtd_parcelas_atraso_12m": late.astype(np.int64),
    }


def _loan(income: np.ndarray, params: ScenarioParams, rng: np.random.Generator) -> dict:
    n = len(income)
    principal = np.clip(rng.lognormal(np.log(12_000), 0.7, n), 1_000, 200_000)
    term = _categorical(rng, params.term_shares, n).astype(np.int64)
    rate = params.monthly_interest
    annuity = (1 - (1 + rate) ** (-term)) / rate
    installment = np.minimum(principal / annuity, params.max_installment_ratio * income)
    return {
        "valor_emprestimo": np.round(installment * annuity, 2),
        "prazo_meses": term,
        "valor_parcela": np.round(installment, 2),
        "finalidade_credito": _categorical(rng, params.purpose_shares, n),
    }


def generate_features(
    n_rows: int, params: ScenarioParams, rng: np.random.Generator
) -> pd.DataFrame:
    """Gera as colunas explicativas (sem id, data e alvo).

    Args:
        n_rows: Quantidade de clientes.
        params: Parâmetros do cenário.
        rng: Gerador aleatório (consumido em ordem fixa).

    Returns:
        DataFrame com features e atributos sensíveis.
    """
    z = rng.standard_normal(n_rows)
    demo = _demographics(n_rows, params, rng)
    work = _employment(demo["idade"], params, rng)
    income = _income(work["situacao_emprego"], demo["regiao"], params, rng)
    risk = _risk_profile(z, params, rng)
    loan = _loan(income, params, rng)
    owner = rng.random(n_rows) < 0.38 + 0.10 * (demo["idade"] > _OWNER_AGE_THRESHOLD)
    columns = {**demo, **work, "renda_mensal": income, **risk, **loan}
    return pd.DataFrame({**columns, "possui_imovel": owner})


def compute_logit(df: pd.DataFrame, params: ScenarioParams) -> np.ndarray:
    """Calcula o logit verdadeiro de inadimplência (``genero``/``regiao`` fora).

    Args:
        df: DataFrame com as features do DGP.
        params: Parâmetros do cenário.

    Returns:
        Vetor de logits por cliente.
    """
    commitment = df["valor_parcela"] / df["renda_mensal"]
    in_band = df["score_bureau"].between(*SCORE_BAND).astype(float)
    logit = (
        params.beta0
        + params.beta0_adjustment
        - 0.0045 * (df["score_bureau"] - 650)
        + 2.0 * (df["taxa_endividamento"] - 0.40)
        + 1.2 * (df["utilizacao_credito"] - 0.45)
        + 0.30 * df["qtd_parcelas_atraso_12m"]
        + 0.9 * (df["situacao_emprego"] == "desempregado")
        - 0.5 * df["possui_imovel"]
        - 0.35 * np.log(df["renda_mensal"] / 3_800)
        + params.coef_commitment * (commitment - 0.25)
        + CONCEPT_WEIGHT * params.gamma_concept * in_band
    )
    return logit.to_numpy(dtype=float)


def _client_ids(n_rows: int, rng: np.random.Generator) -> np.ndarray:
    codes = rng.choice(_ID_SPACE, size=n_rows, replace=False)
    return np.array([f"CLI-{code:010X}" for code in codes], dtype=object)


def _reference_dates(
    n_rows: int, period: tuple[date, date], rng: np.random.Generator
) -> pd.DatetimeIndex:
    span = (period[1] - period[0]).days
    offsets = rng.integers(0, span + 1, size=n_rows)
    return pd.Timestamp(period[0]) + pd.to_timedelta(offsets, unit="D")


def generate_portfolio(
    n_rows: int,
    params: ScenarioParams,
    rng: np.random.Generator,
    period: tuple[date, date],
) -> pd.DataFrame:
    """Gera a carteira completa (18 colunas) para um cenário e período.

    Args:
        n_rows: Quantidade de clientes.
        params: Parâmetros do cenário.
        rng: Gerador aleatório determinístico.
        period: Datas inicial e final (inclusivas) de ``data_referencia``.

    Returns:
        DataFrame na ordem de ``dictionary.ALL_COLUMNS``.
    """
    df = generate_features(n_rows, params, rng)
    df[dd.ID_COLUMN] = _client_ids(n_rows, rng)
    df[dd.DATE_COLUMN] = _reference_dates(n_rows, period, rng).astype("datetime64[ns]")
    probability = _sigmoid(compute_logit(df, params))
    df[dd.TARGET] = (rng.random(n_rows) < probability).astype(np.int64)
    return df[dd.ALL_COLUMNS].reset_index(drop=True)


def calibrate_intercept(
    params: ScenarioParams,
    target_rate: float,
    rng: np.random.Generator,
    n_rows: int = 50_000,
) -> float:
    """Encontra por bisseção o ``beta0`` que produz a taxa base desejada.

    Args:
        params: Parâmetros do cenário (``beta0`` atual é ignorado).
        target_rate: Taxa de inadimplência esperada (ex.: 0,06).
        rng: Gerador para a amostra de calibração.
        n_rows: Tamanho da amostra de calibração.

    Returns:
        Intercepto calibrado.
    """
    zeroed = params.model_copy(update={"beta0": 0.0, "beta0_adjustment": 0.0})
    base = compute_logit(generate_features(n_rows, zeroed, rng), zeroed)
    gap = lambda b: float(_sigmoid(base + b).mean()) - target_rate  # noqa: E731
    return float(bisect(gap, -15.0, 5.0, xtol=1e-6))


def _raw_scenarios(path: Path) -> dict[str, Any]:
    return read_yaml(path)["scenarios"]


def load_scenarios(path: Path) -> dict[str, ScenarioParams]:
    """Carrega os cenários, aplicando as sobrescritas sobre ``reference``.

    Args:
        path: Caminho de ``simulation_scenarios.yaml``.

    Returns:
        Mapa nome do cenário → parâmetros completos.
    """
    raw = _raw_scenarios(path)
    base = raw["reference"]["params"]
    return {
        name: ScenarioParams.model_validate({**base, **spec.get("params", {})})
        for name, spec in raw.items()
    }


def load_period(path: Path, name: str) -> tuple[date, date]:
    """Lê o período (início, fim) de um cenário.

    Args:
        path: Caminho de ``simulation_scenarios.yaml``.
        name: Nome do cenário.

    Returns:
        Tupla com as datas inicial e final.
    """
    period = _raw_scenarios(path)[name]["period"]
    start = date.fromisoformat(str(period["start"]))
    return start, date.fromisoformat(str(period["end"]))
