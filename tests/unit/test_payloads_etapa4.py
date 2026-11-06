"""Payload da Etapa 4: leitura do roteiro, faixa intermediária e inventário."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from credit_sentinel.data import dictionary as dd
from credit_sentinel.explainers.payloads import build_payload
from credit_sentinel.explainers.payloads_etapa4 import (
    band_check,
    inventory_rows,
    parse_star,
)

if TYPE_CHECKING:
    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.unit

ROTEIRO = """# Roteiro

## Cronograma

| Bloco | Início | Fim | Duração | Tela | Artefato |
| :--- | :---: | :---: | ---: | :--- | :--- |
| S · Situation | 0:00 | 0:45 | 45 s | Abertura | `README.md` |
| T · Task | 0:45 | 1:20 | 35 s | Índice | `docs/index.html`, `README.md` |

## Roteiro por bloco

### S · Situation (0:00-0:45, 45 s)

- **Fala** (3 palavras):

> Primeira linha.
> Segunda linha.

### T · Task (0:45-1:20, 35 s)

> Só a tarefa.
"""


def test_parse_star_reads_times_screens_artifacts_and_speech() -> None:
    star = parse_star(ROTEIRO)
    assert star.total_seconds == 80
    first, second = star.blocks
    assert (first.start, first.end, first.duration) == (0, 45, 45)
    assert first.speech == "Primeira linha. Segunda linha."
    assert second.artifacts == ["docs/index.html", "README.md"]
    assert second.screen == "Índice"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("# sem cronograma", "Cronograma"),
        ("## Cronograma\n\nnada aqui\n", "vazio"),
    ],
)
def test_parse_star_rejects_a_script_without_a_schedule(
    text: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_star(text)


def test_band_check_counts_false_approvals_in_the_middle_band() -> None:
    frame = pd.DataFrame(
        {
            "score_bureau": [500, 560, 600, 650, 700, 710, 650, 600],
            "inadimplente": [0, 1, 0, 1, 1, 1, 0, 0],
        }
    )
    scores = pd.Series([0.05, 0.05, 0.30, 0.05, 0.05, 0.05, 0.40, 0.05])
    row = band_check(frame, scores, threshold=0.20, model="m-1", role="champion")
    # Na faixa: 560, 600 (0,30), 650, 700, 650 (0,40) e o último 600.
    assert row.clients == 6
    assert row.observed_rate == pytest.approx(3 / 6)
    assert row.approved == 4  # os dois de score 0,30 e 0,40 foram negados
    assert row.defaulters_approved == 3  # 560, 650 e 700
    assert row.expected_defaulters == pytest.approx(4 * 0.05)
    assert row.all_defaulters_approved == 4  # inclui o 710, fora da faixa


def test_inventory_covers_the_columns_and_keeps_sensitive_ones_out_of_the_model() -> (
    None
):
    rows = inventory_rows({spec.name: "x" for spec in dd.COLUMN_SPECS})
    assert [r.column for r in rows] == dd.ALL_COLUMNS
    uses = {r.column: r.use for r in rows}
    assert uses["genero"] == uses["regiao"] == "fora do modelo (só auditoria)"
    assert sum(u == "feature" for u in uses.values()) == len(dd.FEATURE_COLUMNS)
    assert uses["inadimplente"].startswith("rótulo")


def test_etapa4_missing_reports_point_to_the_command(settings_tmp: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="fairness-audit"):
        build_payload(4, settings=settings_tmp)
