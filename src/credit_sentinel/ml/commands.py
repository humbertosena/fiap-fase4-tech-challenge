"""Subcomando ``train``: valida a referência, treina e registra o baseline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from credit_sentinel.contracts.commands import (
    EXIT_BLOCKED,
    display_path,
    run_validation,
)
from credit_sentinel.data.commands import reference_path
from credit_sentinel.drift.detection import MissingInputError
from credit_sentinel.ml.evaluate_gates import GatesReport
from credit_sentinel.ml.features import build_features
from credit_sentinel.ml.mlflow_baseline import log_baseline_run, resolve_tracking_uri
from credit_sentinel.ml.model_card import render_metrics_section, update_model_card
from credit_sentinel.ml.retrain_challenger import CHALLENGER_DIR, retrain_challenger
from credit_sentinel.ml.train_baseline import (
    ModelBundle,
    save_model_bundle,
    train_and_analyze,
)
from credit_sentinel.settings import Settings, get_settings
from credit_sentinel.tracking.registry import ModelRegistry, PromotionRefusedError
from credit_sentinel.tracking.retraining import register_challenger
from credit_sentinel.utils.io import read_parquet, write_json

if TYPE_CHECKING:
    import argparse

    import pandas as pd


BASELINE_DIR = "baseline"
MODEL_CARD = "model_card.md"
GATES_REPORT = "gates_report.json"
EXIT_GATES_FAILED = 3


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
        run_id = log_baseline_run(bundle, build_features(reference.head(5)), settings)
        print(f"  mlflow run: {run_id} (experimento credit-sentinel/baseline)")
        registry = ModelRegistry(resolve_tracking_uri(settings))
        version = registry.version_for_run(run_id)
        registry.set_alias("champion", version)
        print(f"  registry: credit-scoring-model v{version} @champion")


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


def _print_gates(report: GatesReport) -> None:
    for gate in report.results:
        mark = "aprovado" if gate.passed else "REPROVADO"
        print(f"  {gate.gate_id}: {mark} · {gate.detail}")
    holdout = f"{report.holdout_window} ({report.holdout_rows} linhas)"
    print(f"holdout temporal: {holdout}")
    verdict = "aprovado" if report.passed else "reprovado"
    print(f"challenger {report.challenger_model} {verdict} pelos guardrails")


def _promote(registry: ModelRegistry, report: GatesReport) -> int:
    try:
        version = registry.promote_challenger(gates=report)
    except PromotionRefusedError as exc:
        print(f"promoção recusada: {exc}")
        return EXIT_GATES_FAILED
    print(f"promoção concluída: credit-scoring-model v{version} agora é @champion")
    _print_registry(registry)
    return 0


def _print_registry(registry: ModelRegistry) -> None:
    for info in registry.describe():
        aliases = ", ".join(f"@{a}" for a in info.aliases) or "sem alias"
        retired = info.tags.get("retired_reason")
        note = f" · retired_reason={retired}" if retired else ""
        print(f"  credit-scoring-model v{info.version}: {aliases}{note}")


def _register_and_promote(
    challenger: ModelBundle, report: GatesReport, args: argparse.Namespace
) -> tuple[GatesReport, int]:
    settings = get_settings()
    path = settings.reports_dir / GATES_REPORT
    example = build_features(read_parquet(reference_path(settings)).head(5))
    final, registry = register_challenger(challenger, report, settings, example, path)
    champion, new = final.champion_version, final.challenger_version
    print(f"registry: v{new} @challenger (champion atual: v{champion})")
    _print_registry(registry)
    code = _promote(registry, final) if args.promote else 0
    return final, code


def _run_promote(_args: argparse.Namespace) -> int:
    settings = get_settings()
    path = settings.reports_dir / GATES_REPORT
    if not path.exists():
        print("erro: relatório ausente; rode `credit-sentinel retrain-challenger`")
        return 1
    report = GatesReport.model_validate_json(path.read_text(encoding="utf-8"))
    registry = ModelRegistry(resolve_tracking_uri(settings))
    current = registry.version_of("challenger")
    if report.challenger_version is None or report.challenger_version != current:
        print(
            f"erro: o relatório de guardrails (v{report.challenger_version}) não é do"
            f" challenger atual (v{current}); rode `credit-sentinel retrain-challenger`"
        )
        return 1
    return _promote(registry, report)


def _run_retrain(args: argparse.Namespace) -> int:
    settings = get_settings()
    if args.promote and args.no_mlflow:
        print("erro: --promote exige o MLflow (retire --no-mlflow)")
        return 1
    try:
        challenger, report = retrain_challenger(settings=settings)
    except MissingInputError as exc:
        print(f"erro: {exc}")
        return 1
    out_dir = settings.models_dir / CHALLENGER_DIR
    save_model_bundle(challenger, out_dir)
    path = write_json(report, settings.reports_dir / GATES_REPORT)
    where = display_path(out_dir, settings.project_root)
    print(f"challenger {challenger.version} salvo em {where}")
    _print_gates(report)
    code = 0
    if not args.no_mlflow:
        report, code = _register_and_promote(challenger, report, args)
    print(f"relatório: {display_path(path, settings.project_root)}")
    return code or (0 if report.passed else EXIT_GATES_FAILED)


def register(subparsers: argparse._SubParsersAction) -> None:
    """Registra ``train``, ``retrain-challenger`` e ``promote-challenger``.

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
    retrain = subparsers.add_parser(
        "retrain-challenger",
        help="retreina o challenger e avalia os guardrails (exit 3 = reprovado)",
    )
    retrain.add_argument(
        "--promote",
        action="store_true",
        help="promove o challenger a champion se os guardrails aprovarem",
    )
    retrain.add_argument(
        "--no-mlflow", action="store_true", help="não usa o MLflow nem o registry"
    )
    retrain.set_defaults(handler=_run_retrain)
    promote = subparsers.add_parser(
        "promote-challenger",
        help="promove o challenger registrado a champion (exige gates aprovados)",
    )
    promote.set_defaults(handler=_run_promote)
