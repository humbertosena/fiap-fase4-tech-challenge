"""Orquestra a detecção de drift de ponta a ponta (contrato, scores, testes, Evidently).

Para cada janela: valida o lote com o ``inference_schema``, deriva as features,
pontua com o baseline, roda os testes nativos, gera o relatório do Evidently e grava
os artefatos em ``reports/``. Qualquer lote bloqueado pelo contrato interrompe a
execução antes de gravar qualquer coisa.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from typing import TYPE_CHECKING

from credit_sentinel.contracts.commands import display_path, run_validation
from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.commands import reference_path
from credit_sentinel.drift.evidently_runner import (
    build_report,
    extract_flags,
    save_report,
    share_drifted,
)
from credit_sentinel.drift.predictions import (
    SCORE_COLUMN,
    decision_threshold,
    evaluate_performance,
    score_window,
    summarize_predictions,
)
from credit_sentinel.drift.report import (
    DriftReport,
    EvidentlyWindow,
    ReferenceInfo,
    WindowReport,
    render_drift_table,
)
from credit_sentinel.drift.statistical_tests import run_drift_suite
from credit_sentinel.ml.features import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    build_features,
)
from credit_sentinel.ml.train_baseline import (
    ModelBundle,
    load_model_bundle,
    split_reference,
)
from credit_sentinel.utils.clock import fixed_epoch, utc_now
from credit_sentinel.utils.io import read_parquet, write_json, write_parquet, write_text

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pandas as pd

    from credit_sentinel.settings import Settings

BASELINE_DIR = "baseline"
CONTROL_WINDOW = "mes_1"
CRISIS_WINDOW = "mes_3_crise"
SUMMARY_FILE = "evidently_drift_summary.json"
TABLE_FILE = "drift_table.md"
NAMED_REPORT = "evidently_drift_report.html"


class MissingInputError(Exception):
    """Insumo ausente (modelo, referência ou janela), com a dica de como gerá-lo."""


class WindowBlockedError(Exception):
    """Janela reprovada pelo contrato de dados (circuit breaker)."""

    def __init__(self, window: str, reason: str) -> None:
        """Guarda a janela e o motivo do bloqueio.

        Args:
            window: Nome da janela bloqueada.
            reason: Motivo informado pelo validador.
        """
        super().__init__(f"{window}: {reason}")
        self.window = window
        self.reason = reason


@dataclass(frozen=True)
class Baseline:
    """Modelo e referências carregados uma única vez por execução.

    Attributes:
        bundle: Baseline treinado.
        reference_features: Features da referência completa (testes nativos).
        holdout_frame: Features, alvo e score do holdout (Evidently).
        holdout_scores: Scores do holdout (PSI do score).
        reference_rows: Linhas da referência completa.
    """

    bundle: ModelBundle
    reference_features: pd.DataFrame
    holdout_frame: pd.DataFrame
    holdout_scores: pd.DataFrame
    reference_rows: int


def model_frame(frame: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Features do modelo acrescidas do alvo observado e do score previsto.

    Args:
        frame: Lote bruto com as colunas do dicionário.
        scores: Saída de ``score_window`` para o mesmo lote.

    Returns:
        DataFrame no formato esperado pelo relatório do Evidently.
    """
    features = build_features(frame)
    features[dd.TARGET] = frame[dd.TARGET].to_numpy()
    features[SCORE_COLUMN] = scores[SCORE_COLUMN].to_numpy()
    return features


def load_baseline(settings: Settings) -> Baseline:
    """Carrega o baseline e prepara a referência e o holdout.

    Args:
        settings: Configuração do projeto.

    Returns:
        Insumos comuns a todas as janelas.

    Raises:
        MissingInputError: Se o modelo ou a referência não existirem.
    """
    model_dir = settings.models_dir / BASELINE_DIR
    if not (model_dir / "model.joblib").exists():
        msg = "modelo baseline não encontrado; rode `credit-sentinel train`"
        raise MissingInputError(msg)
    path = reference_path(settings)
    if not path.exists():
        msg = "referência não encontrada; rode `credit-sentinel generate-data`"
        raise MissingInputError(msg)
    bundle, reference = load_model_bundle(model_dir), read_parquet(path)
    _, valid = split_reference(reference, seed=settings.seed)
    scores = score_window(bundle, valid)
    return Baseline(
        bundle=bundle,
        reference_features=build_features(reference),
        holdout_frame=model_frame(valid, scores),
        holdout_scores=scores,
        reference_rows=len(reference),
    )


def window_path(settings: Settings, name: str) -> Path:
    """Caminho do parquet de uma janela de produção.

    Args:
        settings: Configuração do projeto.
        name: Nome da janela.

    Returns:
        Arquivo em ``data/production``.
    """
    return settings.production_dir / f"{name}.parquet"


def check_windows(settings: Settings, names: Sequence[str]) -> None:
    """Confere a existência e o contrato de todas as janelas antes de analisar.

    Args:
        settings: Configuração do projeto.
        names: Janelas a analisar.

    Raises:
        MissingInputError: Se algum parquet não existir.
        WindowBlockedError: Se alguma janela violar o ``inference_schema``.
    """
    for name in names:
        if not window_path(settings, name).exists():
            msg = f"janela {name} não encontrada; rode `credit-sentinel simulate`"
            raise MissingInputError(msg)
    for name in names:
        report = run_validation(window_path(settings, name), "inference")
        if report.decision != "SUCCESS":
            raise WindowBlockedError(name, report.blocked_reason or "contrato violado")


def control_auc(settings: Settings, baseline: Baseline) -> float | None:
    """AUC do baseline na janela de controle, se ela existir.

    Args:
        settings: Configuração do projeto.
        baseline: Modelo e referências.

    Returns:
        ROC-AUC de ``mes_1`` ou ``None`` sem o arquivo.
    """
    path = window_path(settings, CONTROL_WINDOW)
    if not path.exists():
        return None
    frame = read_parquet(path)
    scores = score_window(baseline.bundle, frame)
    return evaluate_performance(frame, scores, baseline.bundle, CONTROL_WINDOW).roc_auc


def _evidently(
    settings: Settings,
    baseline: Baseline,
    name: str,
    current: pd.DataFrame,
) -> EvidentlyWindow:
    now = utc_now()
    snapshot = build_report(
        baseline.holdout_frame,
        current,
        numeric=NUMERIC_FEATURES,
        categorical=CATEGORICAL_FEATURES,
        target=dd.TARGET,
        prediction=SCORE_COLUMN,
        timestamp=now,
        threshold=decision_threshold(baseline.bundle),
    )
    folder = settings.reports_dir / "evidently"
    html, data = (
        folder / f"{name}_drift_report.html",
        folder / f"{name}_drift_report.json",
    )
    fixed = now if fixed_epoch() is not None else None
    save_report(snapshot, html, data, uuid_time=fixed)
    payload = json.loads(snapshot.json())
    return EvidentlyWindow(
        share_drifted=share_drifted(payload),
        flags=extract_flags(payload),
        html=display_path(html, settings.project_root),
        json_report=display_path(data, settings.project_root),
    )


def analyze_window(
    settings: Settings, baseline: Baseline, name: str, control: float | None
) -> WindowReport:
    """Analisa uma janela e grava os scores e o relatório do Evidently dela.

    Args:
        settings: Configuração do projeto.
        baseline: Modelo e referências.
        name: Nome da janela.
        control: AUC da janela de controle (``None`` para o próprio controle).

    Returns:
        Resultados nativos, de scores, de performance e do Evidently.
    """
    frame = read_parquet(window_path(settings, name))
    scores = score_window(baseline.bundle, frame)
    drift = run_drift_suite(
        baseline.reference_features,
        build_features(frame),
        numeric=NUMERIC_FEATURES,
        categorical=CATEGORICAL_FEATURES,
        window=name,
    )
    write_parquet(
        scores, settings.reports_dir / "predictions" / f"{name}_scores.parquet"
    )
    return WindowReport(
        drift=drift,
        predictions=summarize_predictions(baseline.holdout_scores, scores, name),
        performance=evaluate_performance(
            frame,
            scores,
            baseline.bundle,
            name,
            None if name == CONTROL_WINDOW else control,
        ),
        evidently=_evidently(settings, baseline, name, model_frame(frame, scores)),
    )


def _write_outputs(settings: Settings, report: DriftReport) -> None:
    reports = settings.reports_dir
    write_json(report, reports / SUMMARY_FILE)
    write_text(render_drift_table(report), reports / TABLE_FILE)
    if CRISIS_WINDOW in report.windows:
        source = settings.project_root / report.windows[CRISIS_WINDOW].evidently.html
        shutil.copy2(source, reports / NAMED_REPORT)


def run_detection(settings: Settings, names: Sequence[str]) -> DriftReport:
    """Executa a detecção nas janelas pedidas e grava todos os artefatos.

    Args:
        settings: Configuração do projeto.
        names: Janelas a analisar, em ordem cronológica.

    Returns:
        Relatório consolidado (o mesmo gravado em ``evidently_drift_summary.json``).

    Raises:
        MissingInputError: Se faltar modelo, referência ou janela.
        WindowBlockedError: Se uma janela violar o contrato de dados.
    """
    baseline = load_baseline(settings)
    check_windows(settings, names)
    control = control_auc(settings, baseline)
    windows = {n: analyze_window(settings, baseline, n, control) for n in names}
    report = DriftReport(
        generated_at=utc_now(),
        model_version=baseline.bundle.version,
        reference=ReferenceInfo(
            rows=baseline.reference_rows,
            holdout_rows=len(baseline.holdout_frame),
            roc_auc_validation=baseline.bundle.metrics.roc_auc,
        ),
        windows=windows,
    )
    _write_outputs(settings, report)
    return report
