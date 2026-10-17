"""Testes estatísticos nativos de drift: KS, PSI, qui-quadrado e Jensen-Shannon.

Esta é a fonte das decisões de drift do projeto; o Evidently entrega o relatório
visual e uma segunda opinião. O p-valor sozinho engana com milhares de linhas, então
cada feature numérica leva também a magnitude (PSI e Wasserstein normalizado).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy import stats
from scipy.spatial.distance import jensenshannon

if TYPE_CHECKING:
    from collections.abc import Sequence

ALPHA = 0.05
PSI_BINS = 10
PSI_EPSILON = 1e-6
PSI_ATTENTION = 0.10
PSI_CRITICAL = 0.25
JS_ALERT = 0.10

PsiLevel = Literal["estavel", "atencao", "critico"]


class NumericDriftResult(BaseModel):
    """Resultado dos testes de drift de uma feature numérica.

    Attributes:
        feature: Nome da feature.
        ks_statistic: Estatística D do Kolmogorov-Smirnov.
        ks_p_value: P-valor do KS de duas amostras.
        ks_drift: Se ``ks_p_value`` é menor que ``alpha``.
        ks_drift_bonferroni: Se ``ks_p_value`` é menor que ``alpha / k``.
        psi: Population Stability Index (bins de quantil da referência).
        psi_level: Faixa do PSI (estável, atenção ou crítico).
        wasserstein_norm: Distância de Wasserstein dividida pelo desvio da referência.
        mean_ref: Média na referência.
        mean_cur: Média na janela.
    """

    feature: str
    ks_statistic: float
    ks_p_value: float
    ks_drift: bool
    ks_drift_bonferroni: bool
    psi: float
    psi_level: PsiLevel
    wasserstein_norm: float
    mean_ref: float
    mean_cur: float


class CategoricalDriftResult(BaseModel):
    """Resultado dos testes de drift de uma feature categórica.

    Attributes:
        feature: Nome da feature.
        chi2_statistic: Estatística do qui-quadrado de independência.
        chi2_p_value: P-valor do qui-quadrado.
        chi2_drift: Se ``chi2_p_value`` é menor que ``alpha``.
        jensen_shannon: Distância de Jensen-Shannon (base 2, de 0 a 1).
        shares_ref: Participação de cada categoria na referência.
        shares_cur: Participação de cada categoria na janela.
    """

    feature: str
    chi2_statistic: float
    chi2_p_value: float
    chi2_drift: bool
    jensen_shannon: float
    shares_ref: dict[str, float]
    shares_cur: dict[str, float]


class DriftSummary(BaseModel):
    """Resumo dos testes nativos de uma janela contra a referência.

    Attributes:
        window: Nome da janela (``mes_1``...).
        reference_rows: Linhas da referência.
        current_rows: Linhas da janela.
        alpha: Nível de significância dos testes.
        alpha_bonferroni: ``alpha`` dividido pelo número de features numéricas.
        numeric: Resultado de cada feature numérica.
        categorical: Resultado de cada feature categórica.
        n_drifted_features: Quantidade de features com drift significativo.
        drifted_features: KS corrigido (numéricas) ou qui-quadrado (categóricas).
        psi_critical_features: Features numéricas com PSI de 0,25 ou mais.
    """

    window: str
    reference_rows: int
    current_rows: int
    alpha: float
    alpha_bonferroni: float
    numeric: list[NumericDriftResult]
    categorical: list[CategoricalDriftResult]
    n_drifted_features: int
    drifted_features: list[str]
    psi_critical_features: list[str]


def _clean(values: pd.Series) -> np.ndarray:
    array = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    return array[~np.isnan(array)]


def _require_samples(reference: np.ndarray, current: np.ndarray) -> None:
    if len(reference) == 0 or len(current) == 0:
        msg = "amostra vazia: o PSI precisa de valores nas duas distribuições"
        raise ValueError(msg)


def ks_test(
    reference: pd.Series, current: pd.Series, alpha: float = ALPHA
) -> tuple[float, float, bool]:
    """Compara duas amostras numéricas com o Kolmogorov-Smirnov.

    Args:
        reference: Valores da referência.
        current: Valores da janela.
        alpha: Nível de significância.

    Returns:
        Estatística D, p-valor e se há drift (``p < alpha``).
    """
    result = stats.ks_2samp(_clean(reference), _clean(current))
    return float(result.statistic), float(result.pvalue), bool(result.pvalue < alpha)


def _psi_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    unique = np.unique(reference)
    if len(unique) == 1:
        # Referência constante: isola o valor para que qualquer outro conte como drift.
        value = unique[0]
        inner = np.array([np.nextafter(value, -np.inf), np.nextafter(value, np.inf)])
    elif len(unique) <= bins:
        inner = (unique[:-1] + unique[1:]) / 2
    else:
        inner = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)[1:-1]))
    return np.concatenate(([-np.inf], inner, [np.inf]))


def population_stability_index(
    reference: pd.Series, current: pd.Series, *, bins: int = PSI_BINS
) -> float:
    """Calcula o PSI entre a distribuição de referência e a atual.

    Os limites dos bins são quantis da referência, para que a métrica reflita a
    migração de população e não a escala absoluta. Features com até ``bins`` valores
    distintos (``prazo_meses``, ``possui_imovel``) usam um bin por valor, já que os
    quantis colapsariam em poucos cortes.

    Args:
        reference: Valores numéricos observados no dataset de referência.
        current: Valores numéricos observados na janela de produção.
        bins: Quantidade de bins de quantil derivados da referência.

    Returns:
        Valor do PSI; acima de 0,25 indica drift crítico.

    Raises:
        ValueError: Se alguma das amostras não tiver valores.
    """
    ref, cur = _clean(reference), _clean(current)
    _require_samples(ref, cur)
    edges = _psi_edges(ref, bins)
    ref_share = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_share = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_share = np.clip(ref_share, PSI_EPSILON, None)
    cur_share = np.clip(cur_share, PSI_EPSILON, None)
    return float(np.sum((cur_share - ref_share) * np.log(cur_share / ref_share)))


def classify_psi(value: float) -> PsiLevel:
    """Classifica o PSI: abaixo de 0,10 estável; de 0,10 a 0,25 atenção; acima, crítico.

    Args:
        value: Valor do PSI.

    Returns:
        ``estavel``, ``atencao`` ou ``critico``.
    """
    if value < PSI_ATTENTION:
        return "estavel"
    return "atencao" if value < PSI_CRITICAL else "critico"


def _shares(reference: pd.Series, current: pd.Series) -> tuple[list[str], np.ndarray]:
    categories = sorted(set(reference.dropna()) | set(current.dropna()))
    counts = [
        series.value_counts().reindex(categories, fill_value=0).to_numpy()
        for series in (reference, current)
    ]
    return [str(c) for c in categories], np.array(counts, dtype=float)


def chi_square_test(
    reference: pd.Series, current: pd.Series, alpha: float = ALPHA
) -> tuple[float, float, bool]:
    """Qui-quadrado de independência entre a categoria e o período da amostra.

    Args:
        reference: Categorias observadas na referência.
        current: Categorias observadas na janela.
        alpha: Nível de significância.

    Returns:
        Estatística, p-valor e se há drift (``p < alpha``). Com uma única categoria
        não há o que testar e o retorno é ``(0.0, 1.0, False)``.
    """
    categories, counts = _shares(reference, current)
    if len(categories) < 2:  # noqa: PLR2004 - uma categoria não gera tabela de contingência
        return 0.0, 1.0, False
    statistic, p_value, _, _ = stats.chi2_contingency(counts.T)
    return float(statistic), float(p_value), bool(p_value < alpha)


def jensen_shannon_divergence(reference: pd.Series, current: pd.Series) -> float:
    """Distância de Jensen-Shannon (base 2) entre as participações das categorias.

    É a raiz da divergência, portanto vai de 0 (iguais) a 1 (categorias disjuntas).

    Args:
        reference: Categorias observadas na referência.
        current: Categorias observadas na janela.

    Returns:
        Distância de Jensen-Shannon.
    """
    _, counts = _shares(reference, current)
    ref_share, cur_share = counts / counts.sum(axis=1, keepdims=True)
    return float(jensenshannon(ref_share, cur_share, base=2))


def _wasserstein_norm(reference: np.ndarray, current: np.ndarray) -> float:
    scale = float(np.std(reference)) or 1.0
    return float(stats.wasserstein_distance(reference, current) / scale)


def _numeric_result(
    feature: str,
    reference: pd.Series,
    current: pd.Series,
    alphas: tuple[float, float],
) -> NumericDriftResult:
    alpha, alpha_bonferroni = alphas
    ref, cur = _clean(reference), _clean(current)
    statistic, p_value, drift = ks_test(reference, current, alpha)
    psi = population_stability_index(reference, current)
    return NumericDriftResult(
        feature=feature,
        ks_statistic=statistic,
        ks_p_value=p_value,
        ks_drift=drift,
        ks_drift_bonferroni=bool(p_value < alpha_bonferroni),
        psi=psi,
        psi_level=classify_psi(psi),
        wasserstein_norm=_wasserstein_norm(ref, cur),
        mean_ref=float(np.mean(ref)),
        mean_cur=float(np.mean(cur)),
    )


def _category_shares(series: pd.Series, categories: list[str]) -> dict[str, float]:
    shares = series.astype(str).value_counts(normalize=True)
    return {name: float(shares.get(name, 0.0)) for name in categories}


def _categorical_result(
    feature: str, reference: pd.Series, current: pd.Series, alpha: float
) -> CategoricalDriftResult:
    statistic, p_value, drift = chi_square_test(reference, current, alpha)
    categories, _ = _shares(reference, current)
    return CategoricalDriftResult(
        feature=feature,
        chi2_statistic=statistic,
        chi2_p_value=p_value,
        chi2_drift=drift,
        jensen_shannon=jensen_shannon_divergence(reference, current),
        shares_ref=_category_shares(reference, categories),
        shares_cur=_category_shares(current, categories),
    )


def run_drift_suite(  # noqa: PLR0913 - interface do spec, tudo nomeado
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numeric: Sequence[str],
    categorical: Sequence[str],
    window: str,
    alpha: float = ALPHA,
) -> DriftSummary:
    """Roda todos os testes nativos de uma janela contra a referência.

    Uma feature numérica tem drift quando o KS passa pela correção de Bonferroni
    (``alpha`` dividido pelas features numéricas); uma categórica, quando o
    qui-quadrado rejeita ``alpha``. A magnitude fica no PSI e no Wasserstein.

    Args:
        reference: Features da referência.
        current: Features da janela.
        numeric: Colunas numéricas a testar.
        categorical: Colunas categóricas a testar.
        window: Nome da janela.
        alpha: Nível de significância.

    Returns:
        Sumário serializável com os resultados por feature.
    """
    alphas = (alpha, alpha / max(len(numeric), 1))
    numeric_results = [
        _numeric_result(name, reference[name], current[name], alphas)
        for name in numeric
    ]
    categorical_results = [
        _categorical_result(name, reference[name], current[name], alpha)
        for name in categorical
    ]
    drifted = [r.feature for r in numeric_results if r.ks_drift_bonferroni]
    drifted += [r.feature for r in categorical_results if r.chi2_drift]
    return DriftSummary(
        window=window,
        reference_rows=len(reference),
        current_rows=len(current),
        alpha=alpha,
        alpha_bonferroni=alphas[1],
        numeric=numeric_results,
        categorical=categorical_results,
        n_drifted_features=len(drifted),
        drifted_features=drifted,
        psi_critical_features=[
            r.feature for r in numeric_results if r.psi_level == "critico"
        ],
    )
