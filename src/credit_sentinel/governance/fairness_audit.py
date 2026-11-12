"""Auditoria de fairness por subgrupos (Etapa 4), também usada pelo G4 do retreino.

``genero``, ``regiao`` e a faixa etária **nunca** entram no modelo (ADR 003); aqui eles
só servem para medir se o desfecho da decisão é parecido entre os grupos. Para cada
atributo, modelo e conjunto de dados, a auditoria compara cada grupo com o **maior
grupo** do atributo (a referência):

- **taxa de aprovação** (``score < limiar``) e a **razão de impacto desigual**: a taxa
  do grupo dividida pela da referência, aceitável em [0,80, 1,25] (regra dos 4/5);
- **TPR**: fração dos inadimplentes que o modelo nega; **FPR**: fração dos bons
  pagadores que o modelo nega (o custo de um grupo ser tratado como risco sem ser);
- **diferença de oportunidade igual**: TPR do grupo menos o TPR da referência.

Só pandas e numpy: a auditoria não depende do ``fairlearn``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel

from credit_sentinel.data import dictionary as dd
from credit_sentinel.utils.clock import utc_now

if TYPE_CHECKING:
    import numpy as np

FAIRNESS_ATTRIBUTES = ("genero", "regiao", "faixa_etaria")
IMPACT_LOWER = 0.80
IMPACT_UPPER = 1.25
AGE_EDGES = (0, 30, 45, 60, 200)
AGE_LABELS = ("18-29", "30-44", "45-59", "60+")
_EPS = 1e-9
_MISSING = "—"


class GroupImpact(BaseModel):
    """Taxa de aprovação de um grupo e sua razão contra o grupo de referência.

    É a visão compacta guardada no ``gates_report.json`` (guardrail G4).

    Attributes:
        attribute: Atributo protegido (``genero``, ``regiao`` ou ``faixa_etaria``).
        group: Valor do atributo.
        n: Clientes do grupo.
        approval_rate: Fração aprovada no grupo.
        reference_group: Maior grupo do atributo, usado como base.
        ratio: ``approval_rate`` dividido pela taxa do grupo de referência.
    """

    attribute: str
    group: str
    n: int
    approval_rate: float
    reference_group: str
    ratio: float


class GroupMetrics(BaseModel):
    """Métricas de decisão de um grupo, lidas contra o grupo de referência.

    Attributes:
        group: Valor do atributo.
        n: Clientes do grupo.
        approval_rate: Fração aprovada (``score < limiar``).
        tpr: Fração dos inadimplentes negados; ``None`` sem inadimplentes no grupo.
        fpr: Fração dos bons pagadores negados; ``None`` sem bons pagadores.
        observed_default_rate: Taxa de inadimplência observada no grupo.
        disparate_impact: Aprovação do grupo dividida pela da referência.
        equal_opportunity_diff: TPR do grupo menos o TPR da referência, se ambos
            existirem.
        within_four_fifths: ``True`` se ``disparate_impact`` está em [0,80, 1,25].
    """

    group: str
    n: int
    approval_rate: float
    tpr: float | None
    fpr: float | None
    observed_default_rate: float
    disparate_impact: float
    equal_opportunity_diff: float | None
    within_four_fifths: bool


class FairnessReport(BaseModel):
    """Auditoria de um atributo para um modelo em um conjunto de dados.

    Attributes:
        model_version: Versão do modelo auditado.
        role: Papel do modelo (``champion`` ou ``challenger``).
        dataset: Identificador do conjunto de dados.
        attribute: Atributo protegido auditado.
        threshold: Limiar operacional: scores a partir dele são negados.
        n: Clientes auditados.
        reference_group: Maior grupo do atributo.
        groups: Métricas por grupo, em ordem alfabética.
        passed: ``True`` se todos os grupos ficaram dentro de 4/5.
    """

    model_version: str
    role: str
    dataset: str
    attribute: str
    threshold: float
    n: int
    reference_group: str
    groups: list[GroupMetrics]
    passed: bool


class FairnessAudit(BaseModel):
    """Conteúdo de ``reports/fairness_audit.json``.

    Attributes:
        generated_at: Instante da geração (respeita ``SOURCE_DATE_EPOCH``).
        lower: Limite inferior da razão de impacto.
        upper: Limite superior da razão de impacto.
        datasets: Descrição de cada conjunto de dados auditado.
        passed: ``True`` somente se todos os relatórios passaram.
        reports: Um relatório por atributo, modelo e conjunto de dados.
    """

    generated_at: datetime
    lower: float
    upper: float
    datasets: dict[str, str]
    passed: bool
    reports: list[FairnessReport]


@dataclass(frozen=True)
class AuditSubject:
    """O que está sendo auditado: modelo, papel, conjunto de dados e limiar.

    Attributes:
        model_version: Versão do modelo.
        role: Papel do modelo (``champion`` ou ``challenger``).
        dataset: Identificador do conjunto de dados.
        threshold: Limiar operacional do modelo (negar quando ``score >= limiar``).
    """

    model_version: str
    role: str
    dataset: str
    threshold: float


@dataclass(frozen=True)
class _Rates:
    n: int
    approval: float
    tpr: float | None
    fpr: float | None
    default_rate: float


def age_band(age: pd.Series) -> pd.Series:
    """Converte a idade em faixa etária.

    Args:
        age: Idades em anos.

    Returns:
        Faixas ``18-29``, ``30-44``, ``45-59`` e ``60+`` como texto.
    """
    bands = pd.cut(age, bins=list(AGE_EDGES), labels=list(AGE_LABELS), right=False)
    return bands.astype(str)


def _attribute_values(frame: pd.DataFrame, attribute: str) -> pd.Series:
    if attribute == "faixa_etaria":
        return age_band(frame["idade"])
    return frame[attribute].astype(str)


def _mean_or_none(flags: pd.Series) -> float | None:
    return float(flags.mean()) if len(flags) else None


def _rates(part: pd.DataFrame) -> _Rates:
    denied, defaulted = part["denied"], part["defaulted"]
    return _Rates(
        n=len(part),
        approval=float(1.0 - denied.mean()),
        tpr=_mean_or_none(denied[defaulted]),
        fpr=_mean_or_none(denied[~defaulted]),
        default_rate=float(defaulted.mean()),
    )


def _within_band(ratio: float) -> bool:
    return IMPACT_LOWER - _EPS <= ratio <= IMPACT_UPPER + _EPS


def _group_metrics(group: str, rates: _Rates, reference: _Rates) -> GroupMetrics:
    impact = rates.approval / reference.approval if reference.approval > 0 else 1.0
    gap = None
    if rates.tpr is not None and reference.tpr is not None:
        gap = rates.tpr - reference.tpr
    return GroupMetrics(
        group=group,
        n=rates.n,
        approval_rate=rates.approval,
        tpr=rates.tpr,
        fpr=rates.fpr,
        observed_default_rate=rates.default_rate,
        disparate_impact=impact,
        equal_opportunity_diff=gap,
        within_four_fifths=_within_band(impact),
    )


def audit_fairness(
    frame: pd.DataFrame,
    scores: np.ndarray,
    subject: AuditSubject,
    *,
    attribute: str,
) -> FairnessReport:
    """Audita um atributo protegido para um modelo em um conjunto de dados.

    Args:
        frame: Clientes com ``idade``, ``genero``, ``regiao`` e o alvo observado.
        scores: Probabilidade de inadimplência por cliente, na ordem de ``frame``.
        subject: Modelo, papel, conjunto de dados e limiar operacional.
        attribute: Um de ``FAIRNESS_ATTRIBUTES``.

    Returns:
        Métricas por grupo contra o maior grupo do atributo.

    Raises:
        ValueError: Se ``attribute`` não for um atributo auditado.
    """
    if attribute not in FAIRNESS_ATTRIBUTES:
        msg = f"atributo {attribute!r} não auditado; use um de {FAIRNESS_ATTRIBUTES}"
        raise ValueError(msg)
    data = pd.DataFrame(
        {
            "group": _attribute_values(frame, attribute).to_numpy(),
            "denied": scores >= subject.threshold,
            "defaulted": frame[dd.TARGET].to_numpy().astype(bool),
        }
    )
    rates = {str(name): _rates(part) for name, part in data.groupby("group")}
    reference = max(rates, key=lambda name: rates[name].n)
    groups = [_group_metrics(g, r, rates[reference]) for g, r in rates.items()]
    return FairnessReport(
        model_version=subject.model_version,
        role=subject.role,
        dataset=subject.dataset,
        attribute=attribute,
        threshold=subject.threshold,
        n=len(data),
        reference_group=reference,
        groups=groups,
        passed=all(g.within_four_fifths for g in groups),
    )


def audit_subject(
    frame: pd.DataFrame, scores: np.ndarray, subject: AuditSubject
) -> list[FairnessReport]:
    """Audita os três atributos protegidos de um modelo em um conjunto de dados.

    Args:
        frame: Clientes com ``idade``, ``genero``, ``regiao`` e o alvo observado.
        scores: Probabilidade de inadimplência por cliente.
        subject: Modelo, papel, conjunto de dados e limiar operacional.

    Returns:
        Um relatório por atributo de ``FAIRNESS_ATTRIBUTES``, nessa ordem.
    """
    return [
        audit_fairness(frame, scores, subject, attribute=attribute)
        for attribute in FAIRNESS_ATTRIBUTES
    ]


def to_group_impacts(reports: list[FairnessReport]) -> list[GroupImpact]:
    """Reduz os relatórios à visão compacta usada pelo guardrail G4.

    Args:
        reports: Relatórios de um mesmo modelo e conjunto de dados.

    Returns:
        Uma entrada por grupo de cada atributo.
    """
    return [
        GroupImpact(
            attribute=report.attribute,
            group=group.group,
            n=group.n,
            approval_rate=group.approval_rate,
            reference_group=report.reference_group,
            ratio=group.disparate_impact,
        )
        for report in reports
        for group in report.groups
    ]


def build_audit(
    reports: list[FairnessReport],
    datasets: dict[str, str],
    *,
    generated_at: datetime | None = None,
) -> FairnessAudit:
    """Reúne os relatórios no documento gravado em ``fairness_audit.json``.

    Args:
        reports: Relatórios por atributo, modelo e conjunto de dados.
        datasets: Descrição de cada conjunto de dados.
        generated_at: Carimbo do documento; ``None`` usa o relógio do projeto.

    Returns:
        Auditoria completa, com o veredito geral.
    """
    return FairnessAudit(
        generated_at=generated_at or utc_now(),
        lower=IMPACT_LOWER,
        upper=IMPACT_UPPER,
        datasets=datasets,
        passed=all(report.passed for report in reports),
        reports=reports,
    )


def _br(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _percent(value: float | None) -> str:
    return _MISSING if value is None else f"{value * 100:.1f}%".replace(".", ",")


def _signed(value: float | None) -> str:
    return _MISSING if value is None else f"{value:+.3f}".replace(".", ",")


def _verdict(*, within: bool) -> str:
    return "dentro de 4/5" if within else "**fora de 4/5**"


def worst_group(report: FairnessReport) -> GroupMetrics:
    """Grupo cuja razão de impacto mais se afasta de 1,0.

    Args:
        report: Auditoria de um atributo.

    Returns:
        Métricas do grupo mais distante da paridade.
    """
    return max(report.groups, key=lambda g: abs(g.disparate_impact - 1.0))


def _intro(audit: FairnessAudit) -> str:
    overall = (
        "Todos os grupos ficaram dentro de 4/5."
        if audit.passed
        else "Há grupo(s) fora de 4/5; as linhas estão em negrito."
    )
    datasets = "\n".join(f"- `{name}`: {text}" for name, text in audit.datasets.items())
    return (
        "# Auditoria de fairness\n\n"
        f"{overall} `genero`, `regiao` e a faixa etária (derivada de `idade`) nunca"
        " entram nas features do modelo (ADR 003); aqui servem só para medir se a"
        " decisão trata os grupos de forma parecida.\n\n"
        "A **razão de impacto** é a taxa de aprovação do grupo dividida pela do maior"
        f" grupo do atributo (a referência) e deve ficar entre {_br(audit.lower, 2)}"
        f" e {_br(audit.upper, 2)} (regra dos 4/5). **TPR** é a fração dos"
        " inadimplentes que o modelo nega; **FPR**, a dos bons pagadores negados; a"
        " **diferença de oportunidade** é o TPR do grupo menos o da referência."
        " O modelo nega quando o score é maior ou igual ao limiar operacional (F1).\n\n"
        f"Conjuntos de dados:\n\n{datasets}"
    )


def _summary_row(report: FairnessReport) -> str:
    worst = worst_group(report)
    cells = [
        f"`{report.dataset}`",
        f"{report.role} (`{report.model_version}`)",
        f"`{report.attribute}`",
        f"`{report.reference_group}`",
        str(report.n),
        f"`{worst.group}` · {_br(worst.disparate_impact)}",
        _verdict(within=report.passed),
    ]
    return "| " + " | ".join(cells) + " |"


def _summary(audit: FairnessAudit) -> str:
    header = (
        "## Resumo\n\n"
        "| Conjunto | Modelo | Atributo | Referência | n | Pior razão | Situação |\n"
        "| --- | --- | --- | --- | ---: | --- | --- |"
    )
    return "\n".join([header, *(_summary_row(r) for r in audit.reports)])


def _group_row(report: FairnessReport, group: GroupMetrics) -> str:
    cells = [
        f"`{report.attribute}`",
        f"`{group.group}`",
        str(group.n),
        _percent(group.approval_rate),
        _percent(group.tpr),
        _percent(group.fpr),
        _percent(group.observed_default_rate),
        _br(group.disparate_impact),
        _signed(group.equal_opportunity_diff),
        _verdict(within=group.within_four_fifths),
    ]
    return "| " + " | ".join(cells) + " |"


def _detail(key: tuple[str, str, str], reports: list[FairnessReport]) -> str:
    dataset, role, version = key
    header = (
        f"### `{dataset}` · {role} (`{version}`)\n\n"
        "| Atributo | Grupo | n | Aprovação | TPR | FPR | Inadimpl. obs. |"
        " Razão de impacto | Δ oportunidade | Regra dos 4/5 |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |"
    )
    rows = (_group_row(r, g) for r in reports for g in r.groups)
    return "\n".join([header, *rows])


def _by_subject(
    reports: list[FairnessReport],
) -> dict[tuple[str, str, str], list[FairnessReport]]:
    grouped: dict[tuple[str, str, str], list[FairnessReport]] = {}
    for report in reports:
        key = (report.dataset, report.role, report.model_version)
        grouped.setdefault(key, []).append(report)
    return grouped


def render_audit_markdown(audit: FairnessAudit) -> str:
    """Gera ``fairness_audit.md``: resumo por atributo e detalhe por grupo.

    Args:
        audit: Auditoria completa.

    Returns:
        Documento Markdown em pt-BR, com vírgula decimal.
    """
    details = [
        _detail(key, reports) for key, reports in _by_subject(audit.reports).items()
    ]
    return (
        "\n\n".join([_intro(audit), _summary(audit), "## Detalhe por grupo", *details])
        + "\n"
    )
