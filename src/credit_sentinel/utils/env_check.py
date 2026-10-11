"""Checagem do ambiente local (somente biblioteca padrão).

Também é usada por ``scripts/validate_env.py``, que roda antes do ``uv sync``.
"""

from __future__ import annotations

import shutil
import socket
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

REQUIRED_PYTHON = (3, 12)
MLFLOW_PORT = 5001


@dataclass(frozen=True)
class CheckResult:
    """Resultado de uma checagem de ambiente.

    Attributes:
        name: Nome curto do item verificado.
        ok: Se a checagem passou.
        detail: Valor observado ou explicação.
        required: Se a falha impede o uso do projeto.
    """

    name: str
    ok: bool
    detail: str
    required: bool = True


def check_python() -> CheckResult:
    """Verifica se o interpretador é Python 3.12."""
    version = sys.version_info
    found = f"{version.major}.{version.minor}.{version.micro}"
    return CheckResult("python", version[:2] == REQUIRED_PYTHON, found)


def check_binary(name: str, *, required: bool) -> CheckResult:
    """Verifica se um executável está no ``PATH``.

    Args:
        name: Nome do executável.
        required: Se a ausência deve reprovar o ambiente.

    Returns:
        Resultado com o caminho encontrado ou ``ausente``.
    """
    path = shutil.which(name)
    return CheckResult(name, path is not None, path or "ausente", required=required)


def check_port(port: int = MLFLOW_PORT) -> CheckResult:
    """Informa se a porta do MLflow está livre (ocupada não reprova).

    Args:
        port: Porta TCP local.

    Returns:
        Resultado informativo; ``required`` é sempre ``False``.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        busy = sock.connect_ex(("127.0.0.1", port)) == 0
    detail = "em uso (MLflow já ativo?)" if busy else "livre"
    return CheckResult(f"porta {port}", not busy, detail, required=False)


def run_checks() -> list[CheckResult]:
    """Executa todas as checagens na ordem de exibição."""
    return [
        check_python(),
        check_binary("uv", required=True),
        check_binary("docker", required=False),
        check_port(),
    ]


def summarize(results: Sequence[CheckResult]) -> int:
    """Imprime a tabela de resultados e devolve o exit code.

    Args:
        results: Resultados das checagens.

    Returns:
        ``0`` se todas as checagens obrigatórias passaram, senão ``1``.
    """
    for result in results:
        mark = "OK " if result.ok else ("ERR" if result.required else "AVS")
        sys.stdout.write(f"[{mark}] {result.name:<12} {result.detail}\n")
    return int(any(r.required and not r.ok for r in results))
