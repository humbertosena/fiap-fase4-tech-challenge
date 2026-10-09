"""Subcomando ``train``: valida a referência e treina o baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from credit_sentinel.contracts.commands import (
    EXIT_BLOCKED,
    display_path,
    run_validation,
)
from credit_sentinel.data.commands import reference_path
from credit_sentinel.ml.train_baseline import (
    ModelBundle,
    save_model_bundle,
    train_baseline,
)
from credit_sentinel.settings import get_settings
from credit_sentinel.utils.io import read_parquet

if TYPE_CHECKING:
    import argparse

BASELINE_DIR = "baseline"


def _run_train(_args: argparse.Namespace) -> int:
    settings = get_settings()
    path = reference_path(settings)
    report = run_validation(path, "reference")
    if report.decision != "SUCCESS":
        print(f"treino abortado: referência bloqueada ({report.blocked_reason})")
        return EXIT_BLOCKED
    pipeline, metrics = train_baseline(read_parquet(path), settings=settings)
    out_dir = settings.models_dir / BASELINE_DIR
    save_model_bundle(ModelBundle(pipeline=pipeline, metrics=metrics), out_dir)
    where = display_path(out_dir, settings.project_root)
    print(f"modelo {metrics.model_version} salvo em {where}")
    print(f"  roc_auc: {metrics.roc_auc:.4f} (validação, n={metrics.n_valid})")
    return 0


def register(subparsers: argparse._SubParsersAction) -> None:
    """Registra o subcomando ``train``.

    Args:
        subparsers: Grupo de subcomandos da CLI principal.
    """
    parser = subparsers.add_parser(
        "train", help="valida a referência e treina o baseline"
    )
    parser.add_argument(
        "--no-mlflow", action="store_true", help="não registra o run no MLflow"
    )
    parser.set_defaults(handler=_run_train)
