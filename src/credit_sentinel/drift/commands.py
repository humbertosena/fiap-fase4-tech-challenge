"""Subcomandos ``simulate`` e ``detect-drift`` (Etapa 2)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from credit_sentinel.contracts.commands import EXIT_BLOCKED
from credit_sentinel.data.commands import reference_path, summarize_dataset
from credit_sentinel.data.generator import SCENARIOS_FILE
from credit_sentinel.drift.detection import (
    MissingInputError,
    WindowBlockedError,
    run_detection,
)
from credit_sentinel.drift.simulator import (
    WINDOWS,
    NumericShift,
    ShiftDescription,
    describe_shift,
    simulate_all,
)
from credit_sentinel.settings import get_settings
from credit_sentinel.utils.io import read_parquet

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    import pandas as pd

    from credit_sentinel.drift.report import WindowReport
    from credit_sentinel.settings import Settings

KEY_NUMERIC = (
    ("renda_mensal", "median"),
    ("taxa_endividamento", "mean"),
    ("utilizacao_credito", "mean"),
    ("comprometimento_renda", "mean"),
)
KEY_CATEGORIES = (
    ("situacao_emprego", "desempregado"),
    ("finalidade_credito", "consolidacao_dividas"),
)
_LABEL_WIDTH = 42
_CELL_WIDTH = 12


def _csv(text: str) -> list[str]:
    return [item.strip() for item in text.split(",") if item.strip()]


def _statistic(shift: NumericShift, stat: str) -> tuple[float, float]:
    if stat == "median":
        return shift.median_ref, shift.median_cur
    return shift.mean_ref, shift.mean_cur


def _row(label: str, cells: list[str]) -> str:
    return f"  {label:<{_LABEL_WIDTH}}" + "".join(f"{c:>{_CELL_WIDTH}}" for c in cells)


def _table_lines(shifts: dict[str, ShiftDescription]) -> list[str]:
    first = next(iter(shifts.values()))
    lines = [_row("indicador", ["referência", *shifts])]
    rates = [f"{s.base_rate_cur:.2%}" for s in shifts.values()]
    lines.append(_row("taxa base", [f"{first.base_rate_ref:.2%}", *rates]))
    for feature, stat in KEY_NUMERIC:
        pairs = [
            _statistic({n.feature: n for n in s.numeric}[feature], stat)
            for s in shifts.values()
        ]
        cells = [f"{pairs[0][0]:.3f}", *(f"{cur:.3f}" for _, cur in pairs)]
        lines.append(
            _row(f"{feature} ({'mediana' if stat == 'median' else 'média'})", cells)
        )
    for feature, category in KEY_CATEGORIES:
        found = [
            next(
                c
                for c in s.categorical
                if (c.feature, c.category) == (feature, category)
            )
            for s in shifts.values()
        ]
        cells = [f"{found[0].share_ref:.3f}", *(f"{c.share_cur:.3f}" for c in found)]
        lines.append(_row(f"{feature}={category}", cells))
    return lines


def _existing(settings: Settings, names: list[str]) -> list[str]:
    return [n for n in names if (settings.production_dir / f"{n}.parquet").exists()]


def _print_windows(
    paths: dict[str, Path], reference: pd.DataFrame, settings: Settings
) -> None:
    shifts = {}
    for name, path in paths.items():
        frame = read_parquet(path)
        summary = summarize_dataset(frame, path, settings)
        print(f"janela {name}: {summary.path} ({summary.rows} linhas)")
        print(f"  taxa base: {summary.base_rate:.2%}")
        print(f"  sha256: {summary.sha256}")
        shifts[name] = describe_shift(reference, frame)
    print("deslocamentos em relação à referência:")
    print("\n".join(_table_lines(shifts)))


def _run_simulate(args: argparse.Namespace) -> int:
    settings = get_settings()
    path = reference_path(settings)
    if not path.exists():
        print("erro: a referência não existe; rode `credit-sentinel generate-data`")
        return 1
    names = args.windows or list(WINDOWS)
    existing = _existing(settings, names)
    if existing and not args.force:
        print(f"erro: janelas já existem ({', '.join(existing)}); use --force")
        return 1
    config = settings.config_dir / SCENARIOS_FILE
    paths = simulate_all(config, settings.production_dir, names, settings.seed)
    _print_windows(paths, read_parquet(path), settings)
    return 0


def _print_detection(name: str, window: WindowReport) -> None:
    drift, predictions = window.drift, window.predictions
    performance, evidently = window.performance, window.evidently
    print(f"janela {name} ({drift.current_rows} linhas)")
    drifted = ", ".join(drift.drifted_features) or "nenhuma"
    critical = ", ".join(drift.psi_critical_features) or "nenhuma"
    print(f"  drift nativo: {drift.n_drifted_features} feature(s): {drifted}")
    print(f"  PSI crítico: {critical}")
    print(
        f"  score: P50 {predictions.p50:.3f} · P90 {predictions.p90:.3f}"
        f" · P99 {predictions.p99:.3f} · PSI {predictions.psi_score_vs_reference:.3f}"
    )
    control = performance.auc_delta_vs_control
    versus = "" if control is None else f" · {control:+.3f} vs controle"
    print(
        f"  performance: AUC {performance.roc_auc:.4f}"
        f" ({performance.auc_delta_vs_reference:+.3f} vs validação{versus})"
        f" · taxa base {performance.base_rate:.2%}"
    )
    print(f"  Evidently: {evidently.share_drifted:.1%} das features · {evidently.html}")


def _run_detect(args: argparse.Namespace) -> int:
    settings = get_settings()
    names = list(WINDOWS) if args.window == "all" else [args.window]
    try:
        report = run_detection(settings, names)
    except MissingInputError as exc:
        print(f"erro: {exc}")
        return 1
    except WindowBlockedError as exc:
        print(f"janela {exc.window} bloqueada pelo contrato: {exc.reason}")
        return EXIT_BLOCKED
    for name, window in report.windows.items():
        _print_detection(name, window)
    print("relatórios: reports/evidently_drift_summary.json, reports/drift_table.md")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Registra os subcomandos ``simulate`` e ``detect-drift``.

    Args:
        subparsers: Grupo de subcomandos da CLI principal.
    """
    parser = subparsers.add_parser(
        "simulate", help="gera as janelas de produção mes_1, mes_2 e mes_3_crise"
    )
    parser.add_argument(
        "--windows",
        type=_csv,
        default=None,
        help="janelas separadas por vírgula (padrão: todas)",
    )
    parser.add_argument("--force", action="store_true", help="sobrescreve janelas")
    parser.set_defaults(handler=_run_simulate)
    detect = subparsers.add_parser(
        "detect-drift",
        help="detecta drift (testes nativos e Evidently) e mede a degradação",
    )
    detect.add_argument(
        "--window",
        choices=("all", *WINDOWS),
        default="all",
        help="janela a analisar (padrão: all)",
    )
    detect.set_defaults(handler=_run_detect)
