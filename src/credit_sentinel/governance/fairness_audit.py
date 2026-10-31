"""Auditoria de fairness por subgrupos, versão mínima da Etapa 3.

``genero``, ``regiao`` e a faixa etária **nunca** entram no modelo (ADR 003); aqui
eles só servem para medir se o desfecho da decisão é parecido entre os grupos. O
critério da Etapa 3 é a razão de impacto desigual na taxa de aprovação: a taxa de cada
grupo dividida pela do grupo de referência (o maior), que deve ficar em [0,80, 1,25].
A Etapa 4 amplia a auditoria com métricas de erro por grupo.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from pydantic import BaseModel

if TYPE_CHECKING:
    import numpy as np

FAIRNESS_ATTRIBUTES = ("genero", "regiao", "faixa_etaria")
IMPACT_LOWER = 0.80
IMPACT_UPPER = 1.25
AGE_EDGES = (0, 30, 45, 60, 200)
AGE_LABELS = ("18-29", "30-44", "45-59", "60+")


class GroupImpact(BaseModel):
    """Taxa de aprovação de um grupo e sua razão contra o grupo de referência.

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


def age_band(age: pd.Series) -> pd.Series:
    """Converte a idade em faixa etária.

    Args:
        age: Idades em anos.

    Returns:
        Faixas ``18-29``, ``30-44``, ``45-59`` e ``60+`` como texto.
    """
    bands = pd.cut(age, bins=list(AGE_EDGES), labels=list(AGE_LABELS), right=False)
    return bands.astype(str)


def _protected_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "genero": frame["genero"].astype(str).to_numpy(),
            "regiao": frame["regiao"].astype(str).to_numpy(),
            "faixa_etaria": age_band(frame["idade"]).to_numpy(),
        }
    )


def _impacts_for(frame: pd.DataFrame, attribute: str) -> list[GroupImpact]:
    stats = frame.groupby(attribute)["approved"].agg(["size", "mean"])
    reference = str(stats["size"].idxmax())
    base = float(stats.loc[reference, "mean"])
    return [
        GroupImpact(
            attribute=attribute,
            group=str(group),
            n=int(row["size"]),
            approval_rate=float(row["mean"]),
            reference_group=reference,
            ratio=float(row["mean"]) / base if base > 0 else 1.0,
        )
        for group, row in stats.iterrows()
    ]


def approval_impact_ratios(
    frame: pd.DataFrame, approved: np.ndarray
) -> list[GroupImpact]:
    """Calcula a razão de impacto na aprovação por grupo.

    Args:
        frame: Clientes com ``genero``, ``regiao`` e ``idade``.
        approved: Decisão por cliente, na mesma ordem (``True`` = aprovado).

    Returns:
        Uma entrada por grupo de cada atributo de ``FAIRNESS_ATTRIBUTES``.
    """
    data = _protected_frame(frame).assign(approved=approved)
    return [
        impact
        for attribute in FAIRNESS_ATTRIBUTES
        for impact in _impacts_for(data, attribute)
    ]
