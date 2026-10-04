"""Ponto de entrada da CLI ``credit-sentinel`` (argparse com subcomandos).

Cada subpacote expõe ``commands.register(subparsers)``; o módulo é listado em
``COMMAND_MODULES`` e importado dinamicamente. Handlers devolvem o exit code:
``0`` sucesso, ``1`` erro, ``2`` lote bloqueado, ``3`` guardrail reprovado.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from typing import TYPE_CHECKING

from credit_sentinel import __version__
from credit_sentinel.utils.env_check import run_checks, summarize

if TYPE_CHECKING:
    from collections.abc import Sequence

COMMAND_MODULES: tuple[str, ...] = ("credit_sentinel.data.commands",)


def _env_check(_args: argparse.Namespace) -> int:
    return summarize(run_checks())


def build_parser() -> argparse.ArgumentParser:
    """Monta o parser com todos os subcomandos registrados.

    Returns:
        Parser pronto para ``parse_args``.
    """
    parser = argparse.ArgumentParser(
        prog="credit-sentinel",
        description="Sustentação e confiabilidade de um modelo de credit scoring.",
    )
    parser.add_argument(
        "--version", action="version", version=f"credit-sentinel {__version__}"
    )
    subparsers = parser.add_subparsers(dest="command", metavar="<comando>")
    env = subparsers.add_parser("env-check", help="verifica python, uv e docker")
    env.set_defaults(handler=_env_check)
    for module_name in COMMAND_MODULES:
        importlib.import_module(module_name).register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Executa a CLI.

    Args:
        argv: Argumentos (sem o nome do programa); ``None`` usa ``sys.argv``.

    Returns:
        Exit code do subcomando executado.
    """
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 1
    handler = getattr(args, "handler", None)
    if handler is None:
        parser.print_help()
        return 1
    try:
        return int(handler(args))
    except Exception as exc:  # noqa: BLE001 - fronteira da CLI converte erro em exit 1
        print(f"erro: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
