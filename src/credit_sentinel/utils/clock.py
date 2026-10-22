"""Relógio do projeto com suporte a ``SOURCE_DATE_EPOCH`` (artefatos reproduzíveis).

Com a variável definida, todo carimbo de tempo gravado em relatórios usa o instante
informado em vez do relógio de parede. É assim que os artefatos versionados ficam
coerentes com a data do commit que os guarda.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime

ENV_VAR = "SOURCE_DATE_EPOCH"


def fixed_epoch() -> int | None:
    """Lê ``SOURCE_DATE_EPOCH`` (segundos desde 1970), se definida.

    Returns:
        O instante fixado ou ``None`` quando a variável não existe.

    Raises:
        ValueError: Se o valor não for um inteiro.
    """
    raw = os.environ.get(ENV_VAR)
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        msg = f"{ENV_VAR} deve ser um inteiro de segundos desde 1970, recebido {raw!r}"
        raise ValueError(msg) from exc


def utc_now() -> datetime:
    """Instante atual em UTC, ou o fixado por ``SOURCE_DATE_EPOCH``.

    Returns:
        Data e hora com fuso UTC.
    """
    epoch = fixed_epoch()
    if epoch is None:
        return datetime.now(UTC)
    return datetime.fromtimestamp(epoch, UTC)
