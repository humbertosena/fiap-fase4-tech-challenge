"""Relatório de drift do Evidently AI (API 0.7: ``Report``, ``Dataset`` e presets).

Isola a biblioteca em um único módulo: se a API mudar, só este arquivo muda, e os
testes nativos (``statistical_tests``) seguem entregando as decisões de drift.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from evidently import BinaryClassification, DataDefinition, Dataset, Report
from evidently.metrics import ValueDrift
from evidently.presets import ClassificationPreset, DataDriftPreset

from credit_sentinel.utils.io import write_json

if TYPE_CHECKING:
    from collections.abc import Sequence
    from datetime import datetime
    from pathlib import Path

    import pandas as pd
    from evidently.core.report import Snapshot

DRIFT_TEST_PREFIX = "Value Drift for column "
DRIFTED_SHARE_TYPE = "evidently:metric_v2:DriftedColumnsCount"
TARGET_TEST = "z"
_UUID7 = re.compile(
    r"\b([0-9a-f]{8})-([0-9a-f]{4})(-7[0-9a-f]{3}-[0-9a-f]{4}-[0-9a-f]{12})\b"
)
_MS_PER_SECOND = 1000


def _definition(
    numeric: Sequence[str], categorical: Sequence[str], target: str, prediction: str
) -> DataDefinition:
    return DataDefinition(
        numerical_columns=[*numeric, prediction],
        categorical_columns=[*categorical, target],
        classification=[
            BinaryClassification(
                target=target, prediction_probas=prediction, pos_label=1
            )
        ],
    )


def build_report(  # noqa: PLR0913 - interface do spec, tudo nomeado
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    numeric: Sequence[str],
    categorical: Sequence[str],
    target: str,
    prediction: str,
    timestamp: datetime | None = None,
    threshold: float = 0.5,
) -> Snapshot:
    """Gera o relatório do Evidently de uma janela contra a referência.

    O relatório combina o ``DataDriftPreset`` nas features, o ``ValueDrift`` do alvo
    (teste z, que enxerga o *prior shift*) e do score predito, e o
    ``ClassificationPreset`` com a qualidade do modelo.

    Args:
        reference: Referência com features, alvo e score.
        current: Janela com as mesmas colunas.
        numeric: Features numéricas.
        categorical: Features categóricas.
        target: Coluna do alvo (``inadimplente``).
        prediction: Coluna da probabilidade prevista (``score``).
        timestamp: Instante registrado no relatório; ``None`` usa o relógio.
        threshold: Limiar de decisão da classificação (o do modelo, não o 0,5 padrão).

    Returns:
        Snapshot do Evidently, pronto para ``save_report``.
    """
    definition = _definition(numeric, categorical, target, prediction)
    report = Report(
        [
            DataDriftPreset(columns=[*numeric, *categorical]),
            ValueDrift(column=target, method=TARGET_TEST),
            ValueDrift(column=prediction),
            ClassificationPreset(probas_threshold=threshold),
        ],
        include_tests=True,
    )
    return report.run(
        Dataset.from_pandas(current, data_definition=definition),
        Dataset.from_pandas(reference, data_definition=definition),
        timestamp=timestamp,
    )


def shift_uuid7_times(text: str, moment: datetime) -> str:
    """Move o relógio embutido nos UUIDs v7 do Evidently para ``moment``.

    Os 48 bits iniciais de um UUID v7 são o instante de criação em milissegundos, e o
    Evidently os usa nos ids do HTML. Sem o ajuste, um relatório gerado hoje carregaria
    a data de hoje mesmo quando o commit que o guarda é de outra data. O ajuste é uma
    translação única: a ordem e os bits aleatórios ficam intactos.

    Args:
        text: HTML ou JSON do relatório.
        moment: Instante (com fuso) que o primeiro identificador deve passar a ter.

    Returns:
        O texto com os UUIDs v7 transladados.
    """
    stamps = [int(m.group(1) + m.group(2), 16) for m in _UUID7.finditer(text)]
    if not stamps:
        return text
    delta = int(moment.timestamp() * _MS_PER_SECOND) - min(stamps)

    def move(match: re.Match[str]) -> str:
        stamp = f"{int(match.group(1) + match.group(2), 16) + delta:012x}"
        return f"{stamp[:8]}-{stamp[8:]}{match.group(3)}"

    return _UUID7.sub(move, text)


def save_report(
    snapshot: Snapshot,
    html_path: Path,
    json_path: Path,
    *,
    uuid_time: datetime | None = None,
) -> None:
    """Grava o relatório em HTML e JSON.

    Args:
        snapshot: Resultado de ``build_report``.
        html_path: Destino do HTML autocontido.
        json_path: Destino do JSON com métricas e testes.
        uuid_time: Instante para os UUIDs v7 (ver ``shift_uuid7_times``); ``None``
            mantém os ids originais.
    """
    html_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(html_path))
    payload = snapshot.json()
    if uuid_time is not None:
        html = html_path.read_text(encoding="utf-8")
        html_path.write_text(shift_uuid7_times(html, uuid_time), encoding="utf-8")
        payload = shift_uuid7_times(payload, uuid_time)
    import json  # noqa: PLC0415 - só aqui, para reformatar o JSON do Evidently

    write_json(json.loads(payload), json_path)


def extract_flags(snapshot_json: dict[str, Any]) -> dict[str, bool]:
    """Lê o veredito de drift que o Evidently deu a cada coluna.

    Args:
        snapshot_json: JSON do relatório (``json.loads(snapshot.json())``).

    Returns:
        Mapa coluna → ``True`` quando o teste de drift da coluna reprovou.
    """
    return {
        test["name"].removeprefix(DRIFT_TEST_PREFIX): test["status"] == "FAIL"
        for test in snapshot_json["tests"]
        if test["name"].startswith(DRIFT_TEST_PREFIX)
    }


def share_drifted(snapshot_json: dict[str, Any]) -> float:
    """Fração das colunas de features com drift segundo o Evidently.

    Args:
        snapshot_json: JSON do relatório.

    Returns:
        Valor entre 0 e 1.

    Raises:
        ValueError: Se o relatório não tiver a contagem de colunas com drift.
    """
    for metric in snapshot_json["metrics"]:
        if metric["config"]["type"] == DRIFTED_SHARE_TYPE:
            return float(metric["value"]["share"])
    msg = "relatório sem a métrica DriftedColumnsCount"
    raise ValueError(msg)
