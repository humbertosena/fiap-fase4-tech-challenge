"""Payloads das páginas didáticas: números lidos dos artefatos versionados.

Nenhum número é digitado nos templates; tudo vem de ``reports/``, ``models/``,
``config/`` e ``data/``. Se um artefato faltar, o erro diz como gerá-lo.
"""

from __future__ import annotations

import json
from itertools import pairwise
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel

from credit_sentinel.contracts.privacy_sanitizer import hash_identifier
from credit_sentinel.contracts.schema import RULES, RuleSpec
from credit_sentinel.contracts.validator import ValidationReport
from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.corruptor import ANOMALY_PLAN
from credit_sentinel.data.generator import (
    SCENARIOS_FILE,
    ScenarioParams,
    load_scenarios,
)
from credit_sentinel.drift.report import DriftReport
from credit_sentinel.drift.simulator import WINDOWS
from credit_sentinel.ml.evaluation import FeatureImportance, ThresholdReport
from credit_sentinel.ml.train_baseline import TrainingMetrics
from credit_sentinel.utils.io import file_sha256, read_json, read_parquet

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from pathlib import Path

    from credit_sentinel.settings import Settings

HISTOGRAM_FEATURES = [
    "renda_mensal",
    "score_bureau",
    "taxa_endividamento",
    "utilizacao_credito",
]
HISTOGRAM_BINS = 20
VALIDATOR_FIELDS = [
    "idade",
    "renda_mensal",
    "score_bureau",
    "taxa_endividamento",
    "utilizacao_credito",
    "valor_parcela",
    "tempo_emprego_meses",
    "situacao_emprego",
]
PSEUDONYM_DEMO_INPUT = "CLI-XXXXXXXXXX"
MIN_RULES = 3
AUC_MIN = 0.80
BRIER_MAX = 0.06
KS_MIN = 0.45


class ReleaseInfo(BaseModel):
    """Metadados de uma release e da sua página didática."""

    etapa: int
    tag: str
    date: str
    title: str
    slug: str
    rubric: str
    bullets: list[str]
    commit_sha: str = "local"
    status: Literal["publicada", "em breve"] = "publicada"


RELEASES: dict[int, ReleaseInfo] = {
    1: ReleaseInfo(
        etapa=1,
        tag="v0.1.0-alpha",
        date="2026-10-11",
        title="Contratos de dados e modelo baseline",
        slug="etapa-1-contratos",
        rubric="Validação de Dados (25 %)",
        bullets=[
            "Contrato Pandera com 12 regras ICRA",
            "Circuit breaker bloqueia o lote corrompido",
            "Baseline HistGradientBoosting com AUC 0,83",
        ],
    ),
    2: ReleaseInfo(
        etapa=2,
        tag="v0.2.0-beta",
        date="2026-10-21",
        title="Simulação temporal e detecção de drift",
        slug="etapa-2-drift",
        rubric="Detecção de Drift (25 %)",
        bullets=[
            "Três janelas de produção",
            "KS, PSI, χ² e Evidently",
            "Degradação medida",
        ],
    ),
    3: ReleaseInfo(
        etapa=3,
        tag="v0.3.0-rc",
        date="2026-10-31",
        title="Observabilidade, Data SLOs e MLflow",
        slug="etapa-3-observabilidade",
        rubric="Observabilidade e Logs (20 %)",
        bullets=[
            "Logs JSON com run_id",
            "Error budget e alertas",
            "Champion vs. challenger",
        ],
    ),
    4: ReleaseInfo(
        etapa=4,
        tag="v1.0.0-ga",
        date="2026-11-14",
        title="Governança LGPD, análise causal e entrega final",
        slug="etapa-4-governanca",
        rubric="Governança LGPD (15 %) e Vídeo STAR (15 %)",
        bullets=["Plano LGPD e retenção", "Fairness e DAG causal", "Roteiro STAR"],
    ),
}


def release_info(
    etapa: int,
    *,
    tag: str | None = None,
    date: str | None = None,
    commit_sha: str = "local",
) -> ReleaseInfo:
    """Metadados da release, com sobrescritas opcionais da CLI.

    Args:
        etapa: Número da etapa (1 a 4).
        tag: Tag alternativa.
        date: Data alternativa (``AAAA-MM-DD``).
        commit_sha: Commit de origem dos artefatos.

    Returns:
        Cópia de ``RELEASES[etapa]`` com as sobrescritas.
    """
    base = RELEASES[etapa]
    update = {
        "tag": tag or base.tag,
        "date": date or base.date,
        "commit_sha": commit_sha,
    }
    return base.model_copy(update=update)


class DatasetCard(BaseModel):
    """Resumo do dataset de referência."""

    rows: int
    columns: int
    base_rate: float
    period_start: str
    period_end: str
    sha256: str
    n_features: int


class Histogram(BaseModel):
    """Histograma pré-agregado com taxa de inadimplência por faixa."""

    edges: list[float]
    counts: list[int]
    default_rate: list[float | None]


class CategoryShare(BaseModel):
    """Participação e taxa de inadimplência de uma categoria."""

    category: str
    share: float
    default_rate: float


class AnomalyRow(BaseModel):
    """Anomalia injetada no lote corrompido."""

    key: str
    description: str
    n_rows: int
    rule_id: str


class ValidatorExample(BaseModel):
    """Linha de exemplo para o validador interativo (sem identificadores)."""

    label: str
    values: dict[str, Any]


class AcceptanceItem(BaseModel):
    """Critério de aceitação verificado a partir dos artefatos."""

    criterion: str
    ok: bool
    evidence: str


class Etapa1Payload(BaseModel):
    """Dados da página da Etapa 1 (contratos de dados e baseline)."""

    release: ReleaseInfo
    dataset: DatasetCard
    histograms: dict[str, Histogram]
    employment: list[CategoryShare]
    rules: list[RuleSpec]
    anomalies: list[AnomalyRow]
    bad_batch: ValidationReport
    reference_report: ValidationReport
    validator_fields: list[str]
    validator_examples: list[ValidatorExample]
    metrics: TrainingMetrics
    thresholds: ThresholdReport
    importance: list[FeatureImportance]
    signature: dict[str, Any]
    pseudonymization: dict[str, str]
    acceptance: list[AcceptanceItem]
    commands: list[str]


def require(path: Path, hint: str) -> Path:
    """Garante que um artefato existe.

    Args:
        path: Caminho esperado.
        hint: Comando que gera o artefato.

    Returns:
        O próprio ``path``.

    Raises:
        FileNotFoundError: Com a orientação de como gerar o artefato.
    """
    if not path.exists():
        msg = f"artefato ausente: {path} (gere com `{hint}`)"
        raise FileNotFoundError(msg)
    return path


def _histogram(values: pd.Series, target: pd.Series) -> Histogram:
    low, high = np.quantile(values, [0.005, 0.995])
    edges = np.linspace(low, high, HISTOGRAM_BINS + 1)
    bins = np.clip(np.digitize(values, edges[1:-1]), 0, HISTOGRAM_BINS - 1)
    grouped = target.groupby(bins)
    counts = grouped.size().reindex(range(HISTOGRAM_BINS), fill_value=0)
    rates = grouped.mean().reindex(range(HISTOGRAM_BINS))
    return Histogram(
        edges=[round(float(e), 4) for e in edges],
        counts=counts.astype(int).tolist(),
        default_rate=[None if pd.isna(r) else round(float(r), 4) for r in rates],
    )


def _employment(reference: pd.DataFrame) -> list[CategoryShare]:
    grouped = reference.groupby("situacao_emprego")[dd.TARGET]
    shares = grouped.size() / len(reference)
    return [
        CategoryShare(
            category=str(cat), share=float(shares[cat]), default_rate=float(rate)
        )
        for cat, rate in grouped.mean().sort_values(ascending=False).items()
    ]


def _row_values(row: pd.Series) -> dict[str, Any]:
    return json.loads(row[VALIDATOR_FIELDS].to_json())


def _examples(reference: pd.DataFrame, bad: pd.DataFrame) -> list[ValidatorExample]:
    masks = {
        "Idade fora do domínio (R04)": ~bad["idade"].between(18, 100),
        "Renda nula (R03)": bad["renda_mensal"].isna(),
        "Parcela maior que a renda (R10)": bad["valor_parcela"] > bad["renda_mensal"],
        "Desempregado com tempo de emprego (R11)": (
            bad["situacao_emprego"] == "desempregado"
        )
        & (bad["tempo_emprego_meses"] > 0),
        "Emprego fora do domínio (R08)": ~bad["situacao_emprego"].isin(dd.EMPLOYMENT),
    }
    examples = [
        ValidatorExample(
            label="Linha válida (referência)", values=_row_values(reference.iloc[0])
        )
    ]
    examples += [
        ValidatorExample(label=label, values=_row_values(bad[mask].iloc[0]))
        for label, mask in masks.items()
        if mask.any()
    ]
    return examples


def _dataset_card(reference: pd.DataFrame, path: Path) -> DatasetCard:
    dates = reference[dd.DATE_COLUMN]
    return DatasetCard(
        rows=len(reference),
        columns=reference.shape[1],
        base_rate=float(reference[dd.TARGET].mean()),
        period_start=dates.min().date().isoformat(),
        period_end=dates.max().date().isoformat(),
        sha256=file_sha256(path),
        n_features=14,
    )


def _acceptance(payload: dict[str, Any]) -> list[AcceptanceItem]:
    bad, ref, m = payload["bad_batch"], payload["reference_report"], payload["metrics"]
    found = {v.rule_id for v in bad.violations}
    expected = {a.rule_id for a in payload["anomalies"]}
    inputs = {item["name"] for item in payload["signature"]["inputs"]}
    return [
        AcceptanceItem(
            criterion="Contrato com ≥ 3 regras rígidas",
            ok=len(payload["rules"]) >= MIN_RULES,
            evidence=f"{len(payload['rules'])} regras (R01 a R12)",
        ),
        AcceptanceItem(
            criterion="Referência aprovada em todas as regras",
            ok=ref.decision == "SUCCESS" and ref.rejected_records == 0,
            evidence=f"{ref.valid_records} de {ref.total_records} linhas válidas",
        ),
        AcceptanceItem(
            criterion="Lote corrompido bloqueado na ingestão",
            ok=bad.decision == "BLOCKED",
            evidence=bad.blocked_reason or "",
        ),
        AcceptanceItem(
            criterion="Inventário cobre as 10 anomalias do lote",
            ok=expected <= found,
            evidence=f"regras acionadas: {', '.join(sorted(found))}",
        ),
        AcceptanceItem(
            criterion="ROC-AUC de validação ≥ 0,80",
            ok=m.roc_auc >= AUC_MIN,
            evidence=f"{m.roc_auc:.4f}",
        ),
        AcceptanceItem(
            criterion="Brier ≤ 0,06 e KS ≥ 0,45",
            ok=m.brier <= BRIER_MAX and m.ks_statistic >= KS_MIN,
            evidence=f"Brier {m.brier:.4f} · KS {m.ks_statistic:.4f}",
        ),
        AcceptanceItem(
            criterion="Atributos sensíveis fora do modelo",
            ok=inputs.isdisjoint(dd.SENSITIVE_COLUMNS),
            evidence=f"{len(inputs)} features de entrada",
        ),
    ]


def _baseline_artifacts(settings: Settings) -> dict[str, Any]:
    folder = settings.models_dir / "baseline"
    hint = "uv run credit-sentinel train"
    return {
        "metrics": TrainingMetrics.model_validate(
            read_json(require(folder / "metrics.json", hint))
        ),
        "thresholds": ThresholdReport.model_validate(
            read_json(require(folder / "threshold.json", hint))
        ),
        "importance": [
            FeatureImportance.model_validate(i)
            for i in read_json(require(folder / "feature_importance.json", hint))[
                "features"
            ]
        ],
        "signature": read_json(require(folder / "signature.json", hint)),
    }


def _validation_reports(settings: Settings) -> dict[str, ValidationReport]:
    folder = settings.reports_dir / "validation"
    hint = "uv run credit-sentinel validate --input <lote>"
    return {
        "bad_batch": ValidationReport.model_validate(
            read_json(require(folder / "bad_batch_report.json", hint))
        ),
        "reference_report": ValidationReport.model_validate(
            read_json(require(folder / "credit_ref_report.json", hint))
        ),
    }


def build_etapa1(settings: Settings, release: ReleaseInfo) -> Etapa1Payload:
    """Monta o payload da Etapa 1 a partir dos artefatos.

    Args:
        settings: Configuração do projeto.
        release: Metadados da release.

    Returns:
        Payload validado.
    """
    ref_path = require(
        settings.reference_dir / "credit_ref.parquet",
        "uv run credit-sentinel generate-data",
    )
    bad_path = require(
        settings.corrupted_dir / "bad_batch.parquet",
        "uv run credit-sentinel generate-data",
    )
    reference, bad = read_parquet(ref_path), read_parquet(bad_path)
    data: dict[str, Any] = {
        "release": release,
        "dataset": _dataset_card(reference, ref_path),
        "histograms": {
            f: _histogram(reference[f], reference[dd.TARGET])
            for f in HISTOGRAM_FEATURES
        },
        "employment": _employment(reference),
        "rules": RULES,
        "anomalies": [AnomalyRow(**vars(spec)) for spec in ANOMALY_PLAN],
        "validator_fields": VALIDATOR_FIELDS,
        "validator_examples": _examples(reference, bad),
        "pseudonymization": {
            "input": PSEUDONYM_DEMO_INPUT,
            "output": hash_identifier(
                PSEUDONYM_DEMO_INPUT, settings.pii_salt.get_secret_value()
            ),
        },
        "commands": ETAPA1_COMMANDS,
        **_validation_reports(settings),
        **_baseline_artifacts(settings),
    }
    return Etapa1Payload(**data, acceptance=_acceptance(data))


# --- Etapa 2: simulação temporal e detecção de drift ---------------------------------

OVERLAY_BINS = 24
PSI_FINE_BINS = 60
SCORE_BANDS = [0, 500, 550, 600, 650, 700, 750, 800, 1001]
DATASET_KEYS = ("reference", *WINDOWS)
DATASET_LABELS = {
    "reference": "Referência (jul/2025 a jun/2026)",
    "mes_1": "Mês 1 · jul/2026 (controle)",
    "mes_2": "Mês 2 · ago/2026 (inflação)",
    "mes_3_crise": "Mês 3 · set/2026 (crise)",
}
CATEGORY_FEATURES = ["situacao_emprego", "finalidade_credito"]
PSI_ATTENTION = 0.10
PSI_CRITICAL = 0.25
WINDOW_ROWS = 5000
CONTROL_PSI_MAX = 0.10
MIN_CRITICAL_FEATURES = 2
AUC_DROP_MIN = 0.05
BASE_RATE_MIN = 0.12
EVIDENTLY_CONTROL_MAX = 0.10
ScenarioKind = Literal["covariate", "concept", "prior"]
SCENARIO_ROWS: list[tuple[str, str, ScenarioKind, str, str | None]] = [
    (
        "income_real_factor",
        "Fator da renda real",
        "covariate",
        "income_real_factor",
        None,
    ),
    (
        "debt_logit_shift",
        "Deslocamento do endividamento (logit)",
        "covariate",
        "debt_logit_shift",
        None,
    ),
    (
        "utilization_shift",
        "Deslocamento da utilização de crédito",
        "covariate",
        "utilization_shift",
        None,
    ),
    (
        "monthly_interest",
        "Juros mensais (tabela Price)",
        "covariate",
        "monthly_interest",
        None,
    ),
    (
        "unemployed_share",
        "Participação de desempregados",
        "covariate",
        "employment_shares",
        "desempregado",
    ),
    (
        "consolidation_share",
        "Finalidade: consolidação de dívidas",
        "covariate",
        "purpose_shares",
        "consolidacao_dividas",
    ),
    ("coef_commitment", "Peso da parcela no risco", "concept", "coef_commitment", None),
    (
        "gamma_concept",
        "Concept drift na faixa de score 550 a 700",
        "concept",
        "gamma_concept",
        None,
    ),
    (
        "beta0_adjustment",
        "Ajuste residual do intercepto",
        "prior",
        "beta0_adjustment",
        None,
    ),
]
ETAPA2_COMMANDS = [
    "uv sync --all-groups",
    "uv run credit-sentinel generate-data --force",
    "uv run credit-sentinel train",
    "uv run credit-sentinel simulate --force",
    "uv run credit-sentinel detect-drift --window all",
    "uv run credit-sentinel scan-pii reports docs",
    "uv run credit-sentinel build-explainers --etapa 2",
]


class OverlayHistogram(BaseModel):
    """Contagens por faixa de uma feature em cada conjunto (bins da referência)."""

    feature: str
    edges: list[float]
    counts: dict[str, list[int]]


class PsiCalculator(BaseModel):
    """Distribuição fina da referência, base da calculadora de PSI em JavaScript."""

    feature: str
    edges: list[float]
    counts: list[int]
    mean: float
    std: float
    observed_shift_sd: dict[str, float]


class BandCurve(BaseModel):
    """Taxa de inadimplência por faixa de ``score_bureau`` em cada conjunto."""

    edges: list[int]
    counts: dict[str, list[int]]
    default_rate: dict[str, list[float | None]]


class CategoryShares(BaseModel):
    """Participação de cada categoria em cada conjunto."""

    feature: str
    categories: list[str]
    shares: dict[str, list[float]]


class ScenarioRow(BaseModel):
    """Um parâmetro do gerador nos quatro conjuntos (lido de ``config/``)."""

    parameter: str
    label: str
    kind: ScenarioKind
    values: dict[str, float]


class WindowCard(BaseModel):
    """Identificação e hash de um conjunto de dados (referência ou janela)."""

    key: str
    label: str
    period_start: str
    period_end: str
    rows: int
    base_rate: float
    sha256: str


class Etapa2Payload(BaseModel):
    """Dados da página da Etapa 2 (simulação de drift e sua detecção)."""

    release: ReleaseInfo
    windows: list[WindowCard]
    scenarios: list[ScenarioRow]
    histograms: dict[str, OverlayHistogram]
    psi_calculator: dict[str, PsiCalculator]
    categories: dict[str, CategoryShares]
    band_curve: BandCurve
    report: DriftReport
    acceptance: list[AcceptanceItem]
    commands: list[str]


def _bin_counts(values: pd.Series, edges: np.ndarray) -> list[int]:
    bins = len(edges) - 1
    index = np.clip(np.digitize(values.to_numpy(dtype=float), edges[1:-1]), 0, bins - 1)
    return np.bincount(index, minlength=bins).astype(int).tolist()


def _edges(reference: pd.Series, bins: int, tails: tuple[float, float]) -> np.ndarray:
    low, high = np.quantile(reference.to_numpy(dtype=float), list(tails))
    return np.linspace(low, high, bins + 1)


def overlay_histogram(
    frames: Mapping[str, pd.DataFrame], feature: str, *, bins: int = OVERLAY_BINS
) -> OverlayHistogram:
    """Conta a feature por faixa em todos os conjuntos, com os bins da referência.

    Args:
        frames: Conjuntos por chave; ``reference`` é obrigatório.
        feature: Coluna numérica.
        bins: Número de faixas.

    Returns:
        Histograma sobreposto, com as pontas (0,5 %) agregadas nas faixas extremas.
    """
    edges = _edges(frames["reference"][feature], bins, (0.005, 0.995))
    return OverlayHistogram(
        feature=feature,
        edges=[round(float(e), 4) for e in edges],
        counts={
            key: _bin_counts(frame[feature], edges) for key, frame in frames.items()
        },
    )


def psi_calculator(
    frames: Mapping[str, pd.DataFrame], feature: str, *, bins: int = PSI_FINE_BINS
) -> PsiCalculator:
    """Distribuição fina da referência e o deslocamento real de cada janela.

    Args:
        frames: Conjuntos por chave; ``reference`` é obrigatório.
        feature: Coluna numérica.
        bins: Número de faixas finas.

    Returns:
        Insumos da calculadora de PSI (a mesma fórmula de ``statistical_tests``).
    """
    reference = frames["reference"][feature]
    edges = _edges(reference, bins, (0.001, 0.999))
    mean, std = float(reference.mean()), float(reference.std(ddof=0))
    shifts = {
        key: (float(frame[feature].mean()) - mean) / std
        for key, frame in frames.items()
        if key != "reference"
    }
    return PsiCalculator(
        feature=feature,
        edges=[round(float(e), 4) for e in edges],
        counts=_bin_counts(reference, edges),
        mean=mean,
        std=std,
        observed_shift_sd=shifts,
    )


def score_band_curve(frames: Mapping[str, pd.DataFrame]) -> BandCurve:
    """Taxa de inadimplência por faixa de ``score_bureau`` em cada conjunto.

    Args:
        frames: Conjuntos por chave, com ``score_bureau`` e ``inadimplente``.

    Returns:
        Contagens e taxas por faixa de ``SCORE_BANDS`` (a curva do concept drift).
    """
    size = len(SCORE_BANDS) - 1
    counts: dict[str, list[int]] = {}
    rates: dict[str, list[float | None]] = {}
    for key, frame in frames.items():
        band = pd.cut(
            frame["score_bureau"], bins=SCORE_BANDS, right=False, labels=False
        )
        grouped = frame.groupby(band, observed=False)[dd.TARGET]
        counts[key] = (
            grouped.size().reindex(range(size), fill_value=0).astype(int).tolist()
        )
        mean = grouped.mean().reindex(range(size))
        rates[key] = [None if pd.isna(r) else round(float(r), 4) for r in mean]
    return BandCurve(edges=SCORE_BANDS, counts=counts, default_rate=rates)


def category_shares(frames: Mapping[str, pd.DataFrame], feature: str) -> CategoryShares:
    """Participação de cada categoria em cada conjunto.

    Args:
        frames: Conjuntos por chave.
        feature: Coluna categórica do dicionário de dados.

    Returns:
        Participações na ordem de ``dictionary.ENUMS``.
    """
    categories = list(dd.ENUMS[feature])
    shares = {
        key: frame[feature]
        .value_counts(normalize=True)
        .reindex(categories, fill_value=0.0)
        .astype(float)
        .tolist()
        for key, frame in frames.items()
    }
    return CategoryShares(feature=feature, categories=categories, shares=shares)


def _br(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _psi(report: DriftReport, window: str, feature: str) -> float:
    numeric = report.windows[window].drift.numeric
    return next(r.psi for r in numeric if r.feature == feature)


def _item(criterion: str, *, ok: bool, evidence: str) -> AcceptanceItem:
    return AcceptanceItem(criterion=criterion, ok=ok, evidence=evidence)


def _rises(values: list[float]) -> bool:
    return all(a <= b for a, b in pairwise(values))


def _stability_items(
    report: DriftReport, cards: list[WindowCard]
) -> list[AcceptanceItem]:
    control = report.windows["mes_1"]
    top = max(r.psi for r in control.drift.numeric)
    share = control.evidently.share_drifted
    return [
        _item(
            "Três janelas de 5.000 linhas, separadas da referência",
            ok=sum(c.rows == WINDOW_ROWS for c in cards) == len(WINDOWS),
            evidence="data/production/{mes_1,mes_2,mes_3_crise}.parquet",
        ),
        _item(
            "Controle estável: nenhuma feature de mes_1 com PSI ≥ 0,10",
            ok=top < CONTROL_PSI_MAX,
            evidence=f"maior PSI de mes_1: {_br(top)}",
        ),
        _item(
            "Evidently: controle com no máximo 10 % das colunas em drift",
            ok=share <= EVIDENTLY_CONTROL_MAX,
            evidence=f"{_br(100 * share, 1)} % em mes_1",
        ),
    ]


def _crisis_items(report: DriftReport) -> list[AcceptanceItem]:
    crisis = report.windows["mes_3_crise"]
    critical = crisis.drift.psi_critical_features
    flagged = {k for k, v in crisis.evidently.flags.items() if v}
    psi_score = crisis.predictions.psi_score_vs_reference
    delta = crisis.performance.auc_delta_vs_reference
    named = ", ".join(f"{f} {_br(_psi(report, 'mes_3_crise', f))}" for f in critical)
    return [
        _item(
            "Crise: pelo menos 2 features com PSI crítico (≥ 0,25)",
            ok=len(critical) >= MIN_CRITICAL_FEATURES,
            evidence=named,
        ),
        _item(
            "Score predito com PSI crítico em mes_3_crise",
            ok=psi_score >= PSI_CRITICAL,
            evidence=f"PSI do score {_br(psi_score)}",
        ),
        _item(
            "Testes nativos e Evidently concordam nas features críticas",
            ok=bool(critical) and set(critical) <= flagged,
            evidence=f"críticas: {', '.join(critical)}; o Evidently marca todas",
        ),
        _item(
            "Taxa de inadimplência de mes_3_crise ≥ 12 %",
            ok=crisis.performance.base_rate >= BASE_RATE_MIN,
            evidence=f"{_br(100 * crisis.performance.base_rate, 2)} %",
        ),
        _item(
            "AUC de mes_3_crise pelo menos 0,05 abaixo da validação",
            ok=delta <= -AUC_DROP_MIN,
            evidence=(
                f"{_br(crisis.performance.roc_auc)} contra "
                f"{_br(report.reference.roc_auc_validation)} ({_br(delta)})"
            ),
        ),
    ]


def _gradation_items(report: DriftReport) -> list[AcceptanceItem]:
    windows = [report.windows[name].predictions for name in WINDOWS]
    p90 = [w.p90 for w in windows]
    renda = _psi(report, "mes_2", "renda_mensal")
    divida = _psi(report, "mes_2", "taxa_endividamento")
    return [
        _item(
            "P90 e P99 do score sobem de mes_1 a mes_3_crise",
            ok=_rises(p90) and _rises([w.p99 for w in windows]),
            evidence=" → ".join(_br(v) for v in p90).join(["P90 ", ""]),
        ),
        _item(
            "mes_2: renda_mensal e taxa_endividamento com PSI ≥ 0,10",
            ok=min(renda, divida) >= PSI_ATTENTION,
            evidence=(
                f"renda_mensal {_br(renda)} "
                f"(KS p = {_p(report, 'mes_2', 'renda_mensal')}), "
                f"taxa_endividamento {_br(divida)}"
            ),
        ),
    ]


def _acceptance_etapa2(
    report: DriftReport, cards: list[WindowCard]
) -> list[AcceptanceItem]:
    return [
        *_stability_items(report, cards),
        *_crisis_items(report),
        *_gradation_items(report),
    ]


def _p(report: DriftReport, window: str, feature: str) -> str:
    result = next(
        r for r in report.windows[window].drift.numeric if r.feature == feature
    )
    return f"{result.ks_p_value:.1e}".replace(".", ",")


def _datasets(settings: Settings) -> tuple[dict[str, pd.DataFrame], dict[str, Path]]:
    paths = {
        "reference": require(
            settings.reference_dir / "credit_ref.parquet",
            "uv run credit-sentinel generate-data",
        )
    }
    for name in WINDOWS:
        paths[name] = require(
            settings.production_dir / f"{name}.parquet",
            "uv run credit-sentinel simulate",
        )
    return {key: read_parquet(path) for key, path in paths.items()}, paths


def _window_cards(
    frames: Mapping[str, pd.DataFrame], paths: Mapping[str, Path]
) -> list[WindowCard]:
    return [
        WindowCard(
            key=key,
            label=DATASET_LABELS[key],
            period_start=frames[key][dd.DATE_COLUMN].min().date().isoformat(),
            period_end=frames[key][dd.DATE_COLUMN].max().date().isoformat(),
            rows=len(frames[key]),
            base_rate=float(frames[key][dd.TARGET].mean()),
            sha256=file_sha256(paths[key]),
        )
        for key in DATASET_KEYS
    ]


def _read(params: ScenarioParams, attr: str, member: str | None) -> float:
    value = getattr(params, attr)
    return float(value if member is None else value[member])


def _scenario_rows(settings: Settings) -> list[ScenarioRow]:
    scenarios = load_scenarios(settings.config_dir / SCENARIOS_FILE)
    return [
        ScenarioRow(
            parameter=name,
            label=label,
            kind=kind,
            values={key: _read(scenarios[key], attr, member) for key in DATASET_KEYS},
        )
        for name, label, kind, attr, member in SCENARIO_ROWS
    ]


def build_etapa2(settings: Settings, release: ReleaseInfo) -> Etapa2Payload:
    """Monta o payload da Etapa 2 a partir dos relatórios e dos conjuntos de dados.

    Args:
        settings: Configuração do projeto.
        release: Metadados da release.

    Returns:
        Payload validado.
    """
    summary = require(
        settings.reports_dir / "evidently_drift_summary.json",
        "uv run credit-sentinel detect-drift --window all",
    )
    report = DriftReport.model_validate(read_json(summary))
    frames, paths = _datasets(settings)
    cards = _window_cards(frames, paths)
    return Etapa2Payload(
        release=release,
        windows=cards,
        scenarios=_scenario_rows(settings),
        histograms={f: overlay_histogram(frames, f) for f in HISTOGRAM_FEATURES},
        psi_calculator={f: psi_calculator(frames, f) for f in HISTOGRAM_FEATURES},
        categories={f: category_shares(frames, f) for f in CATEGORY_FEATURES},
        band_curve=score_band_curve(frames),
        report=report,
        acceptance=_acceptance_etapa2(report, cards),
        commands=ETAPA2_COMMANDS,
    )


ETAPA1_COMMANDS = [
    "uv sync --all-groups",
    "uv run credit-sentinel generate-data --force",
    "uv run credit-sentinel validate --input data/corrupted/bad_batch.parquet",
    (
        "uv run credit-sentinel validate --input data/reference/credit_ref.parquet"
        " --schema reference"
    ),
    "uv run credit-sentinel train",
    "uv run credit-sentinel scan-pii reports docs",
    "uv run credit-sentinel build-explainers --etapa 1",
]


def _builders() -> dict[int, Callable[[Settings, ReleaseInfo], BaseModel]]:
    """Construtores por etapa (a Etapa 3 fica em módulo próprio, sem ciclo)."""
    from credit_sentinel.explainers.payloads_etapa3 import build_etapa3  # noqa: PLC0415

    return {1: build_etapa1, 2: build_etapa2, 3: build_etapa3}


def build_payload(
    etapa: int, *, settings: Settings, release: ReleaseInfo | None = None
) -> BaseModel:
    """Monta o payload de uma etapa.

    Args:
        etapa: Número da etapa.
        settings: Configuração do projeto.
        release: Metadados da release (padrão: ``RELEASES[etapa]``).

    Returns:
        Payload pydantic da etapa.

    Raises:
        ValueError: Se a etapa ainda não tiver página.
    """
    builders = _builders()
    if etapa not in builders:
        msg = f"a etapa {etapa} ainda não possui página didática"
        raise ValueError(msg)
    return builders[etapa](settings, release or release_info(etapa))


class IndexPayload(BaseModel):
    """Dados da página índice (cards das quatro releases)."""

    releases: list[ReleaseInfo]
