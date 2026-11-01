"""Subcomando ``fairness-audit`` (Etapa 4)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from credit_sentinel.contracts.commands import display_path
from credit_sentinel.drift.detection import MissingInputError
from credit_sentinel.governance.audit_runner import (
    DATASET_DESCRIPTIONS,
    run_full_audit,
)
from credit_sentinel.governance.fairness_audit import (
    build_audit,
    render_audit_markdown,
    worst_group,
)
from credit_sentinel.settings import get_settings
from credit_sentinel.utils.io import write_json, write_text

if TYPE_CHECKING:
    import argparse

    from credit_sentinel.governance.fairness_audit import FairnessAudit
    from credit_sentinel.settings import Settings

AUDIT_JSON = "fairness_audit.json"
AUDIT_MARKDOWN = "fairness_audit.md"
EXIT_FAIRNESS_FAILED = 3  # o mesmo código de guardrail reprovado do retreino


def _print_summary(audit: FairnessAudit, settings: Settings) -> None:
    print(f"fairness-audit: {len(audit.reports)} relatórios (atributo, modelo e dados)")
    for report in audit.reports:
        worst = worst_group(report)
        ratio = f"{worst.disparate_impact:.3f}".replace(".", ",")
        verdict = "dentro de 4/5" if report.passed else "FORA de 4/5"
        print(
            f"  {report.dataset} · {report.role} · {report.attribute}:"
            f" pior razão {ratio} ({worst.group}) · {verdict}"
        )
    failed = sum(not report.passed for report in audit.reports)
    print(
        "resultado: todos os grupos dentro de 4/5"
        if not failed
        else f"resultado: {failed} relatório(s) com grupo fora de 4/5"
    )
    for name in (AUDIT_JSON, AUDIT_MARKDOWN):
        shown = display_path(settings.reports_dir / name, settings.project_root)
        print(f"relatório: {shown}")


def _run_audit(args: argparse.Namespace) -> int:
    settings = get_settings()
    try:
        reports = run_full_audit(settings=settings)
    except MissingInputError as exc:
        print(f"erro: {exc}")
        return 1
    audit = build_audit(reports, DATASET_DESCRIPTIONS)
    write_json(audit, settings.reports_dir / AUDIT_JSON)
    write_text(render_audit_markdown(audit), settings.reports_dir / AUDIT_MARKDOWN)
    _print_summary(audit, settings)
    return EXIT_FAIRNESS_FAILED if args.strict and not audit.passed else 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Registra o subcomando ``fairness-audit``.

    Args:
        subparsers: Grupo de subcomandos da CLI principal.
    """
    parser = subparsers.add_parser(
        "fairness-audit",
        help="audita genero, regiao e faixa etária de champion e challenger",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 3 se algum grupo ficar fora de 4/5 (padrão: exit 0)",
    )
    parser.set_defaults(handler=_run_audit)
