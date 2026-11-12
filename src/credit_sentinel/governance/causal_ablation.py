"""Ablação causal por intervenção no simulador (README, seção 8.3).

O gerador de dados permite intervir: cada choque da crise é um parâmetro do cenário.
Para cada cenário do ``CATALOGUE``, a janela ``mes_3_crise`` é regerada com as mesmas
sementes (números aleatórios comuns), ora com só alguns choques ligados (``only``,
partindo da referência), ora com alguns desligados (``without``, partindo da crise), e
o baseline é medido nela. Como as sementes são as mesmas em todos os cenários, a queda
de AUC é pareada com a do controle, o que isola o efeito do choque do ruído amostral.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Literal

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import roc_auc_score

from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.generator import (
    SCENARIOS_FILE,
    generate_portfolio,
    load_scenarios,
)
from credit_sentinel.drift.detection import MissingInputError
from credit_sentinel.drift.simulator import load_windows
from credit_sentinel.ml.retrain_challenger import BASELINE_DIR, CRISIS_WINDOW
from credit_sentinel.ml.train_baseline import load_model_bundle, predict_proba
from credit_sentinel.utils.clock import utc_now
from credit_sentinel.utils.seeds import make_rng

if TYPE_CHECKING:
    from collections.abc import Iterable

    from credit_sentinel.data.generator import ScenarioParams
    from credit_sentinel.drift.simulator import WindowSpec
    from credit_sentinel.ml.train_baseline import ModelBundle
    from credit_sentinel.settings import Settings

Mode = Literal["only", "without"]
EN_DASH = "\N{EN DASH}"
MINUS = "\N{MINUS SIGN}"
BAND_TEXT = f"550{EN_DASH}700"
DEFAULT_SEEDS = 10
CONTROL_KEY = "controle"
COVARIATE = (
    "income_real_factor",
    "debt_logit_shift",
    "utilization_shift",
    "monthly_interest",
    "employment_shares",
    "purpose_shares",
)
STRESS = ("income_real_factor", "debt_logit_shift", "utilization_shift")
CONCEPT = ("coef_commitment", "gamma_concept")
BAND = ("gamma_concept",)
PRICE = ("monthly_interest", "coef_commitment")
UNEMPLOYMENT = ("employment_shares",)
TIGHTENING = ("beta0_adjustment",)


@dataclass(frozen=True)
class ScenarioSpec:
    """Um cenário de intervenção.

    Attributes:
        key: Identificador estável.
        label: Texto exibido nas tabelas.
        mode: ``only`` liga os choques sobre a referência; ``without`` os desliga
            na crise.
        fields: Parâmetros do cenário envolvidos.
    """

    key: str
    label: str
    mode: Mode
    fields: tuple[str, ...]


CATALOGUE = (
    ScenarioSpec(CONTROL_KEY, "Controle (sem nenhum choque)", "only", ()),
    ScenarioSpec("crise", "Crise completa", "without", ()),
    ScenarioSpec("so_covariate", "Só covariate shift (6 choques)", "only", COVARIATE),
    ScenarioSpec(
        "so_renda_endividamento_utilizacao",
        "Só renda + endividamento + utilização",
        "only",
        STRESS,
    ),
    ScenarioSpec(
        "so_concept",
        f"Só concept drift (peso da parcela + faixa {BAND_TEXT})",
        "only",
        CONCEPT,
    ),
    ScenarioSpec("so_faixa_550_700", f"Só a faixa {BAND_TEXT}", "only", BAND),
    ScenarioSpec("so_juros_peso_parcela", "Só juros + peso da parcela", "only", PRICE),
    ScenarioSpec("so_desemprego", "Só desemprego", "only", UNEMPLOYMENT),
    ScenarioSpec("crise_sem_concept", "Crise sem concept drift", "without", CONCEPT),
    ScenarioSpec(
        "crise_sem_faixa_550_700",
        f"Crise sem o risco extra da faixa {BAND_TEXT}",
        "without",
        BAND,
    ),
    ScenarioSpec(
        "crise_sem_aperto", "Crise sem o aperto da concessão", "without", TIGHTENING
    ),
)


class ScenarioResult(BaseModel):
    """Resultado de um cenário, agregado nas sementes.

    Attributes:
        key: Identificador do cenário.
        label: Texto exibido.
        mode: ``only`` ou ``without``.
        fields: Parâmetros envolvidos.
        base_rate: Taxa de inadimplência média.
        auc: ROC-AUC médio do baseline.
        delta_auc: Variação média do AUC contra o controle da mesma semente.
        delta_auc_std: Desvio-padrão (amostral) dessa variação.
    """

    key: str
    label: str
    mode: Mode
    fields: list[str]
    base_rate: float
    auc: float
    delta_auc: float
    delta_auc_std: float


class CausalAblation(BaseModel):
    """Relatório da ablação (``reports/causal_ablation.json``).

    Attributes:
        generated_at: Instante da geração (UTC).
        model_version: Versão do modelo medido (o champion).
        window: Janela de crise regerada.
        rows: Clientes por amostra.
        seeds: Sementes usadas (as mesmas em todos os cenários).
        scenarios: Um resultado por item do ``CATALOGUE``.
    """

    generated_at: datetime
    model_version: str
    window: str
    rows: int
    seeds: list[int]
    scenarios: list[ScenarioResult]


@dataclass(frozen=True)
class _Setup:
    bundle: ModelBundle
    reference: ScenarioParams
    crisis: ScenarioParams
    window: WindowSpec
    rows: int


def params_for(
    spec: ScenarioSpec, reference: ScenarioParams, crisis: ScenarioParams
) -> ScenarioParams:
    """Parâmetros do cenário de intervenção.

    Args:
        spec: Cenário.
        reference: Parâmetros da referência (sem choques).
        crisis: Parâmetros da janela de crise (todos os choques).

    Returns:
        Referência com os choques de ``spec.fields`` ligados (``only``) ou crise com
        eles desligados (``without``).
    """
    if spec.mode == "only":
        return reference.model_copy(
            update={name: getattr(crisis, name) for name in spec.fields}
        )
    return crisis.model_copy(
        update={name: getattr(reference, name) for name in spec.fields}
    )


def summarize(
    spec: ScenarioSpec,
    rates: np.ndarray,
    aucs: np.ndarray,
    control_aucs: np.ndarray,
) -> ScenarioResult:
    """Agrega as sementes de um cenário.

    Args:
        spec: Cenário.
        rates: Taxa de inadimplência por semente.
        aucs: ROC-AUC do baseline por semente.
        control_aucs: ROC-AUC do controle nas mesmas sementes.

    Returns:
        Médias e variação pareada do AUC contra o controle.
    """
    deltas = aucs - control_aucs
    spread = float(deltas.std(ddof=1)) if len(deltas) > 1 else 0.0
    return ScenarioResult(
        key=spec.key,
        label=spec.label,
        mode=spec.mode,
        fields=list(spec.fields),
        base_rate=float(rates.mean()),
        auc=float(aucs.mean()),
        delta_auc=float(deltas.mean()),
        delta_auc_std=spread,
    )


def _measure(setup: _Setup, params: ScenarioParams, seed: int) -> tuple[float, float]:
    frame = generate_portfolio(setup.rows, params, make_rng(seed), setup.window.period)
    target = frame[dd.TARGET]
    scores = predict_proba(setup.bundle, frame)
    return float(target.mean()), float(roc_auc_score(target, scores))


def _measure_all(
    setup: _Setup, seeds: Iterable[int]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    measured = {}
    for spec in CATALOGUE:
        params = params_for(spec, setup.reference, setup.crisis)
        pairs = [_measure(setup, params, seed) for seed in seeds]
        rates, aucs = (np.array(column) for column in zip(*pairs, strict=True))
        measured[spec.key] = (rates, aucs)
    return measured


def _load_setup(settings: Settings, rows: int | None) -> _Setup:
    directory = settings.models_dir / BASELINE_DIR
    if not (directory / "model.joblib").exists():
        msg = "modelo champion não encontrado; rode `credit-sentinel train`"
        raise MissingInputError(msg)
    config = settings.config_dir / SCENARIOS_FILE
    scenarios = load_scenarios(config)
    window = next(w for w in load_windows(config) if w.name == CRISIS_WINDOW)
    return _Setup(
        bundle=load_model_bundle(directory),
        reference=scenarios["reference"],
        crisis=window.params,
        window=window,
        rows=rows or window.n_rows,
    )


def run_ablation(
    *, settings: Settings, n_seeds: int = DEFAULT_SEEDS, rows: int | None = None
) -> CausalAblation:
    """Executa todos os cenários do ``CATALOGUE`` nas mesmas sementes.

    Args:
        settings: Configuração do projeto.
        n_seeds: Quantidade de sementes, a partir da semente da janela de crise.
        rows: Clientes por amostra (padrão: o tamanho da janela de crise).

    Returns:
        O relatório da ablação.

    Raises:
        MissingInputError: Se o baseline não estiver treinado.
    """
    setup = _load_setup(settings, rows)
    seeds = list(range(setup.window.seed, setup.window.seed + n_seeds))
    measured = _measure_all(setup, seeds)
    control = measured[CONTROL_KEY][1]
    return CausalAblation(
        generated_at=utc_now(),
        model_version=setup.bundle.version,
        window=setup.window.name,
        rows=setup.rows,
        seeds=seeds,
        scenarios=[summarize(spec, *measured[spec.key], control) for spec in CATALOGUE],
    )


def _br(value: float, digits: int) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _delta_text(result: ScenarioResult) -> str:
    if result.key == CONTROL_KEY:
        return "—"
    sign = MINUS if result.delta_auc < 0 else "+"
    return f"{sign} {_br(abs(result.delta_auc), 3)} ± {_br(result.delta_auc_std, 3)}"


def render_ablation_markdown(ablation: CausalAblation) -> str:
    """Tabela pt-BR da ablação, para o README e a revisão humana.

    Args:
        ablation: Relatório da ablação.

    Returns:
        Documento Markdown.
    """
    seeds = ablation.seeds
    intro = (
        f"Cada linha regera a janela `{ablation.window}` ({ablation.rows} clientes) "
        f"com as mesmas {len(seeds)} sementes ({seeds[0]} a {seeds[-1]}), mudando só "
        f"os choques indicados, e mede o modelo `{ablation.model_version}` (sem "
        "retreino). `only` parte da referência e liga os choques; `without` parte "
        "da crise e os desliga. A variação de AUC é pareada com o controle da mesma "
        "semente (média ± desvio-padrão)."
    )
    lines = [
        "# Ablação causal por intervenção",
        "",
        intro,
        "",
        "| Cenário | Modo | Parâmetros | Taxa base | AUC | Δ AUC vs. controle |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]
    for result in ablation.scenarios:
        names = ", ".join(f"`{name}`" for name in result.fields) or "—"
        lines.append(
            f"| {result.label} | {result.mode} | {names} | "
            f"{_br(100 * result.base_rate, 2)} % | {_br(result.auc, 3)} | "
            f"{_delta_text(result)} |"
        )
    return "\n".join(lines) + "\n"
