"""Relatório consolidado de drift e a tabela de leitura (``drift_table.md``).

``DriftReport`` é o conteúdo de ``reports/evidently_drift_summary.json`` e também o
payload da página didática da Etapa 2: nada do que a página mostra é digitado à mão.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from credit_sentinel.drift.predictions import (
    PerformanceSummary,
    PredictionSummary,
)
from credit_sentinel.drift.statistical_tests import (
    DriftSummary,
)

_LEVEL_LABEL = {"estavel": "estável", "atencao": "atenção", "critico": "crítico"}
_SCIENTIFIC_BELOW = 1e-3


class ReferenceInfo(BaseModel):
    """Referência usada nas comparações.

    Attributes:
        rows: Linhas da referência completa (testes nativos).
        holdout_rows: Linhas do holdout de validação (score e Evidently).
        roc_auc_validation: ROC-AUC do baseline na validação.
    """

    rows: int
    holdout_rows: int
    roc_auc_validation: float


class EvidentlyWindow(BaseModel):
    """Veredito do Evidently para uma janela.

    Attributes:
        share_drifted: Fração das features com drift.
        flags: Coluna → drift segundo o Evidently (features, alvo e score).
        html: Caminho do relatório HTML, relativo à raiz do projeto.
        json_report: Caminho do relatório JSON, relativo à raiz do projeto.
    """

    share_drifted: float
    flags: dict[str, bool]
    html: str
    json_report: str


class WindowReport(BaseModel):
    """Tudo o que se mediu em uma janela.

    Attributes:
        drift: Testes nativos por feature.
        predictions: Comportamento dos scores.
        performance: Performance contra o alvo observado.
        evidently: Veredito do Evidently.
    """

    drift: DriftSummary
    predictions: PredictionSummary
    performance: PerformanceSummary
    evidently: EvidentlyWindow


class DriftReport(BaseModel):
    """Conteúdo de ``reports/evidently_drift_summary.json``.

    Attributes:
        generated_at: Instante da geração (respeita ``SOURCE_DATE_EPOCH``).
        model_version: Versão do baseline avaliado.
        reference: Dados da referência.
        windows: Resultado por janela, na ordem cronológica.
    """

    generated_at: datetime
    model_version: str
    reference: ReferenceInfo
    windows: dict[str, WindowReport]


def _decimal(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _p_value(value: float) -> str:
    if value < _SCIENTIFIC_BELOW:
        return f"{value:.1e}".replace(".", ",")
    return _decimal(value, 4)


def _percent(value: float) -> str:
    return f"{_decimal(100 * value, 1)} %"


def _yes(flag: bool) -> str:  # noqa: FBT001 - formatação de um único booleano
    return "sim" if flag else "não"


def _table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---", *["---:"] * (len(headers) - 1)]) + " |")
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def _numeric_section(report: DriftReport) -> list[str]:
    names = list(report.windows)
    features = [r.feature for r in report.windows[names[0]].drift.numeric]
    headers = ["Feature"]
    for name in names:
        headers += [f"{name} p(KS)", f"{name} PSI"]
    rows = []
    for index, feature in enumerate(features):
        row = [f"`{feature}`"]
        for name in names:
            result = report.windows[name].drift.numeric[index]
            p_value = _p_value(result.ks_p_value)
            row.append(f"**{p_value}**" if result.ks_drift_bonferroni else p_value)
            row.append(f"{_decimal(result.psi)} · {_LEVEL_LABEL[result.psi_level]}")
        rows.append(row)
    return _table(headers, rows)


def _categorical_section(report: DriftReport) -> list[str]:
    names = list(report.windows)
    features = [r.feature for r in report.windows[names[0]].drift.categorical]
    headers = ["Feature"]
    for name in names:
        headers += [f"{name} p(qui²)", f"{name} JS"]
    rows = []
    for index, feature in enumerate(features):
        row = [f"`{feature}`"]
        for name in names:
            result = report.windows[name].drift.categorical[index]
            p_value = _p_value(result.chi2_p_value)
            row.append(f"**{p_value}**" if result.chi2_drift else p_value)
            row.append(_decimal(result.jensen_shannon))
        rows.append(row)
    return _table(headers, rows)


def _score_section(report: DriftReport) -> list[str]:
    headers = [
        "Janela", "P50", "P90", "P99", "Aprovação", "PSI do score", "AUC",
        "Δ AUC vs validação", "Δ AUC vs controle", "Brier", "Taxa base",
    ]  # fmt: skip
    rows = []
    for name, window in report.windows.items():
        predictions, performance = window.predictions, window.performance
        versus_control = performance.auc_delta_vs_control
        rows.append(
            [
                name,
                _decimal(predictions.p50),
                _decimal(predictions.p90),
                _decimal(predictions.p99),
                _percent(predictions.approval_rate),
                _decimal(predictions.psi_score_vs_reference),
                _decimal(performance.roc_auc),
                _decimal(performance.auc_delta_vs_reference),
                "—" if versus_control is None else _decimal(versus_control),
                _decimal(performance.brier),
                _percent(performance.base_rate),
            ]
        )
    return _table(headers, rows)


def _evidently_section(report: DriftReport) -> list[str]:
    names = list(report.windows)
    columns = list(report.windows[names[0]].evidently.flags)
    rows = [
        [
            f"`{column}`",
            *[_yes(report.windows[n].evidently.flags[column]) for n in names],
        ]
        for column in columns
    ]
    share = [_percent(report.windows[name].evidently.share_drifted) for name in names]
    rows.append(["**Participação de colunas com drift**", *share])
    return _table(["Coluna", *names], rows)


def _thousands(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _intro(report: DriftReport) -> str:
    first = next(iter(report.windows.values())).drift
    reference = report.reference
    return (
        f"Referência de {_thousands(reference.rows)} linhas "
        f"(holdout de validação com {_thousands(reference.holdout_rows)} linhas "
        "para o score e o Evidently). "
        f"Significância alfa = {_decimal(first.alpha, 2)}; com a correção de "
        f"Bonferroni, alfa = {_decimal(first.alpha_bonferroni, 4)}. "
        "**Negrito** marca drift significativo (KS corrigido ou qui-quadrado)."
    )


def render_drift_table(report: DriftReport) -> str:
    """Monta o Markdown com as tabelas de drift, scores, performance e Evidently.

    Args:
        report: Relatório consolidado das janelas.

    Returns:
        Texto Markdown terminado em uma única quebra de linha.
    """
    sections = [
        "# Drift por feature e janela",
        _intro(report),
        "## Features numéricas: p-valor do KS e PSI",
        *_numeric_section(report),
        "PSI: menor que 0,10 estável; de 0,10 a 0,25 atenção; 0,25 ou mais crítico.",
        "## Features categóricas: p-valor do qui-quadrado e Jensen-Shannon",
        *_categorical_section(report),
        "## Scores e performance do baseline",
        *_score_section(report),
        "## Veredito do Evidently AI (drift por coluna)",
        *_evidently_section(report),
    ]
    return "\n\n".join(_join(sections)) + "\n"


def _join(sections: list[str]) -> list[str]:
    """Agrupa linhas de tabela consecutivas em um único bloco."""
    blocks: list[str] = []
    for line in sections:
        if blocks and line.startswith("|") and blocks[-1].startswith("|"):
            blocks[-1] += "\n" + line
        else:
            blocks.append(line)
    return blocks
