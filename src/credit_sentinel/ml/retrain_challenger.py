"""Retreino do challenger com janela deslizante e avaliação em holdout temporal.

Protocolo (ADR 005, seção 5): depois que o drift é detectado, o challenger é treinado
com os últimos seis meses da referência, ``mes_1``, ``mes_2`` e a **primeira quinzena**
da crise (dias 1 a 15 de ``mes_3_crise``, já rotulados). A segunda quinzena é o
holdout temporal em que champion e challenger são comparados. Hiperparâmetros e
features são os do baseline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.commands import reference_path
from credit_sentinel.drift.detection import MissingInputError, window_path
from credit_sentinel.ml.evaluate_gates import (
    GatesReport,
    TrainingProtocol,
    evaluate_promotion_gates,
)
from credit_sentinel.ml.train_baseline import (
    ModelBundle,
    load_model_bundle,
    train_and_analyze,
)
from credit_sentinel.utils.io import read_parquet

if TYPE_CHECKING:
    from credit_sentinel.settings import Settings

CHALLENGER_VERSION = "challenger-1.0.0"
CHALLENGER_DIR = "challenger"
BASELINE_DIR = "baseline"
SLIDING_MONTHS = 6
TRAINING_WINDOWS = ("mes_1", "mes_2")
CRISIS_WINDOW = "mes_3_crise"
CRISIS_TRAIN_UNTIL_DAY = 15
_EARLY_CRISIS = "mes_3_crise_ate_dia_15"


def _recent_reference(reference: pd.DataFrame, months: int) -> pd.DataFrame:
    dates = pd.to_datetime(reference[dd.DATE_COLUMN])
    first_month = dates.max().to_period("M").start_time
    return reference[dates >= first_month - pd.DateOffset(months=months - 1)]


def build_sliding_window(
    reference: pd.DataFrame,
    windows: dict[str, pd.DataFrame],
    *,
    months: int = SLIDING_MONTHS,
) -> pd.DataFrame:
    """Monta o conjunto de treino: referência recente mais as janelas informadas.

    Args:
        reference: Dataset de referência completo.
        windows: Janelas de produção rotuladas que entram no treino.
        months: Quantos meses finais da referência manter.

    Returns:
        Linhas da referência dos últimos ``months`` meses seguidas das janelas.
    """
    recent = _recent_reference(reference, months)
    return pd.concat([recent, *windows.values()], ignore_index=True)


def split_crisis(
    crisis: pd.DataFrame, *, until_day: int = CRISIS_TRAIN_UNTIL_DAY
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separa a crise em treino (até o dia ``until_day``) e holdout (restante).

    Args:
        crisis: Janela ``mes_3_crise``.
        until_day: Último dia do mês incluído no treino.

    Returns:
        Primeira parte (treino) e segunda parte (holdout temporal).
    """
    day = pd.to_datetime(crisis[dd.DATE_COLUMN]).dt.day
    return crisis[day <= until_day], crisis[day > until_day]


def _load_inputs(settings: Settings) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    path = reference_path(settings)
    if not path.exists():
        msg = "referência não encontrada; rode `credit-sentinel generate-data`"
        raise MissingInputError(msg)
    names = (*TRAINING_WINDOWS, CRISIS_WINDOW)
    missing = [n for n in names if not window_path(settings, n).exists()]
    if missing:
        msg = f"janela {missing[0]} não encontrada; rode `credit-sentinel simulate`"
        raise MissingInputError(msg)
    return read_parquet(path), {
        n: read_parquet(window_path(settings, n)) for n in names
    }


def _protocol(
    recent: pd.DataFrame, parts: dict[str, pd.DataFrame], holdout: pd.DataFrame
) -> TrainingProtocol:
    dates = pd.to_datetime(holdout[dd.DATE_COLUMN])
    reference_dates = pd.to_datetime(recent[dd.DATE_COLUMN])
    return TrainingProtocol(
        sliding_months=SLIDING_MONTHS,
        reference_start=f"{reference_dates.min():%Y-%m-%d}",
        reference_rows=len(recent),
        windows={name: len(frame) for name, frame in parts.items()},
        training_rows=len(recent) + sum(len(f) for f in parts.values()),
        crisis_train_until=f"{CRISIS_TRAIN_UNTIL_DAY:02d} do mês da crise",
        holdout_window=CRISIS_WINDOW,
        holdout_rows=len(holdout),
        holdout_start=f"{dates.min():%Y-%m-%d}",
        holdout_end=f"{dates.max():%Y-%m-%d}",
    )


def retrain_challenger(*, settings: Settings) -> tuple[ModelBundle, GatesReport]:
    """Treina o challenger e avalia os guardrails contra o champion.

    Args:
        settings: Configuração do projeto.

    Returns:
        Challenger treinado e o relatório dos cinco gates.

    Raises:
        MissingInputError: Se faltar a referência, uma janela ou o baseline.
    """
    reference, windows = _load_inputs(settings)
    early, holdout = split_crisis(windows[CRISIS_WINDOW])
    parts = {name: windows[name] for name in TRAINING_WINDOWS}
    parts[_EARLY_CRISIS] = early
    train = build_sliding_window(reference, parts)
    challenger = train_and_analyze(train, settings=settings, version=CHALLENGER_VERSION)
    model_dir = settings.models_dir / BASELINE_DIR
    if not (model_dir / "model.joblib").exists():
        msg = "modelo baseline não encontrado; rode `credit-sentinel train`"
        raise MissingInputError(msg)
    champion = load_model_bundle(model_dir)
    gates = evaluate_promotion_gates(champion, challenger, holdout, settings=settings)
    protocol = _protocol(_recent_reference(reference, SLIDING_MONTHS), parts, holdout)
    return challenger, gates.model_copy(
        update={"protocol": protocol, "holdout_window": CRISIS_WINDOW}
    )
