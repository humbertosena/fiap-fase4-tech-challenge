"""Subcomando ``build-explainers``: gera as páginas didáticas das releases."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from credit_sentinel.contracts.commands import display_path
from credit_sentinel.explainers.build import build_all
from credit_sentinel.settings import get_settings

if TYPE_CHECKING:
    import argparse


def _parse_etapas(raw: str | None) -> list[int] | None:
    return [int(item) for item in raw.split(",")] if raw else None


def _run(args: argparse.Namespace) -> int:
    settings = get_settings()
    etapas = _parse_etapas(args.etapa)
    overrides = {k: v for k, v in {"tag": args.tag, "date": args.date}.items() if v}
    if overrides and (etapas is None or len(etapas) != 1):
        print("erro: --tag/--date exigem exatamente uma etapa em --etapa")
        return 1
    out_dir = Path(args.out) if args.out else None
    pages = build_all(
        settings=settings, etapas=etapas, out_dir=out_dir, overrides=overrides
    )
    for page in pages:
        size = page.stat().st_size / 1024
        print(f"página: {display_path(page, settings.project_root)} ({size:.0f} KB)")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Registra o subcomando ``build-explainers``.

    Args:
        subparsers: Grupo de subcomandos da CLI principal.
    """
    parser = subparsers.add_parser(
        "build-explainers", help="gera docs/releases/*.html e docs/index.html"
    )
    parser.add_argument("--etapa", default=None, help="etapas separadas por vírgula")
    parser.add_argument("--tag", default=None, help="tag exibida (uma etapa)")
    parser.add_argument("--date", default=None, help="data AAAA-MM-DD (uma etapa)")
    parser.add_argument("--out", default=None, help="diretório de saída (padrão docs/)")
    parser.set_defaults(handler=_run)
