"""Subcomando ``train``: valida a referência, treina e registra o baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from credit_sentinel.contracts.commands import (
    EXIT_BLOCKED,
    display_path,
    run_validation,
)
from credit_sentinel.data.commands import reference_path
from credit_sentinel.ml.features import build_features
from credit_sentinel.ml.model_card import render_metrics_section, update_model_card
from credit_sentinel.ml.train_baseline import (
    ModelBundle,
    save_model_bundle,
    train_and_analyze,
)
from credit_sentinel.settings import Settings, get_settings
from credit_sentinel.utils.io import read_parquet

if TYPE_CHECKING:
    import argparse

    import pandas as pd

BASELINE_DIR = "baseline"
MODEL_CARD = "model_card.md"


def _print_metrics(bundle: ModelBundle) -> None:
    m = bundle.metrics
    print(
        f"  roc_auc: {m.roc_auc:.4f} · brier: {m.brier:.4f} · ks: {m.ks_statistic:.4f}"
    )
    print(f"  f1: {m.f1:.4f} no limiar {m.threshold:.3f} (validação, n={m.n_valid})")
    if bundle.threshold is not None:
        print(f"  limiar de custo 5:1: {bundle.threshold.cost_threshold:.3f}")
    print(f"  top-3 features: {', '.join(i.feature for i in bundle.importance[:3])}")


def _publish(
    bundle: ModelBundle, reference: pd.DataFrame, settings: Settings, *, mlflow: bool
) -> None:
    card = settings.docs_dir / MODEL_CARD
    if card.exists():
        update_model_card(card, render_metrics_section(bundle))
    if mlflow:
        from credit_sentinel.ml.mlflow_baseline import log_baseline_run  # noqa: PLC0415

        run_id = log_baseline_run(bundle, build_features(reference.head(5)), settings)
        print(f"  mlflow run: {run_id} (experimento credit-sentinel/baseline)")


def _run_train(args: argparse.Namespace) -> int:
    settings = get_settings()
    path = reference_path(settings)
    report = run_validation(path, "reference")
    if report.decision != "SUCCESS":
        print(f"treino abortado: referência bloqueada ({report.blocked_reason})")
        return EXIT_BLOCKED
    reference = read_parquet(path)
    bundle = train_and_analyze(reference, settings=settings)
    out_dir = settings.models_dir / BASELINE_DIR
    save_model_bundle(bundle, out_dir)
    where = display_path(out_dir, settings.project_root)
    print(f"modelo {bundle.version} salvo em {where}")
    _print_metrics(bundle)
    _publish(bundle, reference, settings, mlflow=not args.no_mlflow)
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
