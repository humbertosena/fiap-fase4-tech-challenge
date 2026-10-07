"""Subcomandos do contrato de dados: ``validate``."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from credit_sentinel.contracts.schema import (
    build_inference_schema,
    build_reference_schema,
)
from credit_sentinel.contracts.validator import (
    BatchContext,
    CircuitBreakerOpenError,
    ContractViolationError,
    ValidationPolicy,
    ValidationReport,
    validate_batch,
)
from credit_sentinel.settings import get_settings
from credit_sentinel.utils.io import file_sha256, read_parquet, write_json

if TYPE_CHECKING:
    import argparse

EXIT_BLOCKED = 2


def display_path(path: Path, root: Path) -> str:
    """Mostra o caminho relativo à raiz do projeto quando possível."""
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path)


def run_validation(path: Path, schema_kind: str) -> ValidationReport:
    """Valida um parquet e devolve o relatório (bloqueado ou não).

    Args:
        path: Arquivo parquet do lote.
        schema_kind: ``reference`` ou ``inference``.

    Returns:
        Relatório de validação.
    """
    settings = get_settings()
    builder = (
        build_reference_schema if schema_kind == "reference" else build_inference_schema
    )
    context = BatchContext(
        batch_name=path.stem,
        run_id=f"val-{file_sha256(path)[:10]}",
        quarantine_dir=settings.quarantine_dir,
    )
    try:
        _, report = validate_batch(
            read_parquet(path),
            schema=builder(),
            policy=ValidationPolicy(),
            context=context,
        )
    except (ContractViolationError, CircuitBreakerOpenError) as exc:
        return exc.report
    return report


def _print_report(report: ValidationReport, report_path: str) -> None:
    print(f"lote: {report.batch_name} ({report.total_records} linhas)")
    print(f"schema: {report.schema_name}")
    suffix = f" — {report.blocked_reason}" if report.blocked_reason else ""
    print(f"decisão: {report.decision}{suffix}")
    if report.violations:
        print(f"{'regra':<6} {'coluna':<24} {'falhas':>6}  check")
    for v in report.violations:
        print(f"{v.rule_id:<6} {v.column or '—':<24} {v.n_failures:>6}  {v.check}")
    print(
        f"rejeitadas: {report.rejected_records} ({report.rejected_ratio:.2%})"
        f" · válidas: {report.valid_records}"
    )
    print(f"relatório: {report_path}")


def _run_validate(args: argparse.Namespace) -> int:
    settings = get_settings()
    path = Path(args.input)
    report = run_validation(path, args.schema)
    default = settings.reports_dir / "validation" / f"{path.stem}_report.json"
    out = write_json(report, Path(args.report) if args.report else default)
    _print_report(report, display_path(out, settings.project_root))
    return 0 if report.decision == "SUCCESS" else EXIT_BLOCKED


def register(subparsers: argparse._SubParsersAction) -> None:
    """Registra o subcomando ``validate``.

    Args:
        subparsers: Grupo de subcomandos da CLI principal.
    """
    parser = subparsers.add_parser(
        "validate", help="valida um lote contra o contrato (exit 2 = BLOCKED)"
    )
    parser.add_argument("--input", required=True, help="arquivo parquet do lote")
    parser.add_argument(
        "--schema", choices=("reference", "inference"), default="inference"
    )
    parser.add_argument("--report", default=None, help="caminho do relatório JSON")
    parser.set_defaults(handler=_run_validate)
