"""Subcomando ``run-pipeline`` (Etapa 3)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from credit_sentinel.contracts.commands import EXIT_BLOCKED, display_path
from credit_sentinel.drift.detection import MissingInputError
from credit_sentinel.ml.mlflow_baseline import resolve_tracking_uri
from credit_sentinel.observability.pipeline_runner import (
    WINDOW_ORDER,
    AuditReport,
    WindowRunResult,
    run_pipeline,
)
from credit_sentinel.settings import get_settings
from credit_sentinel.tracking.mlflow_tracking import hook_factory
from credit_sentinel.utils.io import write_json

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from credit_sentinel.observability.pipeline_runner import TrackingHook
    from credit_sentinel.settings import Settings

    HookFactory = Callable[[str, str], Sequence[TrackingHook]]

AUDIT_FILE = "slo_audit_report.json"
_MS = 1000


def _output_dir(args: argparse.Namespace, settings: Settings) -> Path:
    """Execuções completas gravam em ``reports/``; as parciais, fora do versionado."""
    if args.output_dir:
        return settings.project_root / args.output_dir
    if args.window == "all" and not args.quick:
        return settings.reports_dir
    return settings.logs_dir / "last_run"


def _hook_factory(settings: Settings) -> HookFactory:
    uri = resolve_tracking_uri(settings)
    return hook_factory(uri, settings.project_root)


def _print_window(result: WindowRunResult) -> None:
    severities = ", ".join(sorted({a.severity for a in result.slo.alerts}))
    seconds = sum(result.stage_latencies_ms.values()) / _MS
    budget = result.slo.remaining_budget
    print(
        f"janela {result.window}: {result.decision} · alertas {severities}"
        f" · saldo de orçamento {budget:+.0%} · {seconds:.1f} s"
    )
    for alert in result.slo.alerts:
        print(f"  [{alert.severity}] {alert.message}")


def _print_summary(report: AuditReport, audit: Path, settings: Settings) -> None:
    for result in report.windows:
        _print_window(result)
    totals = report.totals
    print(
        f"{totals['windows']} janela(s): {totals['success']} com sucesso,"
        f" {totals['blocked']} bloqueada(s)"
    )
    print(f"auditoria: {display_path(audit, settings.project_root)}")
    print(f"logs: {display_path(settings.logs_dir, settings.project_root)}/")


def _run_pipeline(args: argparse.Namespace) -> int:
    settings = get_settings()
    windows = list(WINDOW_ORDER) if args.window == "all" else [args.window]
    try:
        report = run_pipeline(
            windows,
            settings=settings,
            quick=args.quick,
            hook_factory=None if args.no_mlflow else _hook_factory(settings),
        )
    except MissingInputError as exc:
        print(f"erro: {exc}")
        return 1
    audit = write_json(report, _output_dir(args, settings) / AUDIT_FILE)
    _print_summary(report, audit, settings)
    blocked = report.totals["blocked"]
    return EXIT_BLOCKED if args.strict and blocked else 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Registra o subcomando ``run-pipeline``.

    Args:
        subparsers: Grupo de subcomandos da CLI principal.
    """
    parser = subparsers.add_parser(
        "run-pipeline",
        help="executa ingestão, contrato, inferência, drift e SLOs por janela",
    )
    parser.add_argument(
        "--window",
        choices=("all", *WINDOW_ORDER),
        default="all",
        help="janela a processar (padrão: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="amostra de 1.000 linhas e sem Evidently (uso na CI)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit 2 se alguma janela for bloqueada (padrão: exit 0)",
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="não registra a execução no MLflow"
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="pasta (relativa à raiz) do slo_audit_report.json",
    )
    parser.set_defaults(handler=_run_pipeline)
