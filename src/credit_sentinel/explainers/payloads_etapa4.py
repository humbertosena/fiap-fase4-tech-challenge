"""Payload da página da Etapa 4 (governança, LGPD, fairness e análise causal).

Os números vêm dos artefatos versionados: ``reports/fairness_audit.json``,
``reports/causal_ablation.json``, ``reports/evidently_drift_summary.json``,
``reports/slo_audit_report.json``, ``config/simulation_scenarios.yaml``, os modelos e o
roteiro em ``docs/video_star_roteiro.md``. O texto jurídico e a política de retenção
estão em ``static/etapa4_content.yaml`` (e espelham as seções 7.3 e 7.4 do README),
nunca nos templates.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from pydantic import BaseModel

from credit_sentinel.contracts.privacy_sanitizer import scan_for_pii
from credit_sentinel.data import dictionary as dd
from credit_sentinel.data.generator import SCENARIOS_FILE, load_scenarios
from credit_sentinel.drift.predictions import decision_threshold
from credit_sentinel.drift.report import DriftReport
from credit_sentinel.explainers.payloads import AcceptanceItem, ReleaseInfo, require
from credit_sentinel.governance.causal_ablation import CausalAblation
from credit_sentinel.governance.fairness_audit import FairnessAudit
from credit_sentinel.ml.train_baseline import load_model_bundle, predict_proba
from credit_sentinel.observability.pipeline_runner import AuditReport
from credit_sentinel.utils.io import read_json, read_parquet, read_yaml

if TYPE_CHECKING:
    from collections.abc import Callable

    from credit_sentinel.data.generator import ScenarioParams
    from credit_sentinel.settings import Settings

STATIC_DIR = Path(__file__).resolve().parent / "static"
CONTENT_FILE = "etapa4_content.yaml"
ROTEIRO_FILE = "video_star_roteiro.md"
CRISIS = "mes_3_crise"
WINDOW_NAMES = ("mes_1", "mes_2", CRISIS)
HOLDOUT_START = "2026-09-16"
BAND = (550, 700)
CELLS_PER_ROW = 6
MAX_VIDEO_SECONDS = 300
MIN_CONCEPT_DROP = 0.02
EXPECTED_REPORTS = 12
README_SECTIONS = tuple(f"## {n}. " for n in range(7, 13))
PREDICTION_COLUMNS = ["id_cliente_hash", "score", "pred"]
AUDIT_HINT = "uv run credit-sentinel fairness-audit"
CAUSAL_HINT = "uv run credit-sentinel causal-ablation"
DRIFT_HINT = "uv run credit-sentinel detect-drift --window all"
PIPELINE_HINT = "uv run credit-sentinel run-pipeline --window all"
ROTEIRO_HINT = "git restore docs/video_star_roteiro.md"
ETAPA4_COMMANDS = [
    "uv sync --all-groups",
    "uv run credit-sentinel generate-data --force",
    "uv run credit-sentinel train",
    "uv run credit-sentinel simulate --force",
    "uv run credit-sentinel run-pipeline --window all",
    "uv run credit-sentinel retrain-challenger",
    "uv run credit-sentinel fairness-audit",
    "uv run credit-sentinel causal-ablation",
    "uv run credit-sentinel scan-pii reports docs logs",
    "uv run credit-sentinel build-explainers --etapa 4",
]
_CLOCK = re.compile(r"^(\d+):([0-5]\d)$")
_PATH = re.compile(r"`([^`\s]+)`")
# (rótulo, parâmetro do cenário, formato) por nó do grafo causal.
_PARAM_EVIDENCE: dict[str, list[tuple[str, str, str]]] = {
    "INF": [("Fator da renda real", "income_real_factor", "num")],
    "REN": [("Fator da renda real", "income_real_factor", "num")],
    "SEL": [("Juros mensais", "monthly_interest", "pct")],
    "JUR": [("Juros mensais", "monthly_interest", "pct")],
    "EST": [
        ("Deslocamento do endividamento (logit)", "debt_logit_shift", "num"),
        ("Deslocamento da utilização", "utilization_shift", "num"),
    ],
    "APE": [("Ajuste do intercepto", "beta0_adjustment", "num")],
    "INT": [("Ajuste do intercepto", "beta0_adjustment", "num")],
    "PES": [("Peso da parcela no risco", "coef_commitment", "num")],
    "FAI": [("Risco extra na faixa 550-700", "gamma_concept", "num")],
}
_PSI_EVIDENCE: dict[str, list[tuple[str, str]]] = {
    "REN": [("PSI de renda_mensal", "renda_mensal")],
    "JUR": [("PSI de valor_parcela", "valor_parcela")],
    "DIV": [
        ("PSI de taxa_endividamento", "taxa_endividamento"),
        ("PSI de utilizacao_credito", "utilizacao_credito"),
    ],
    "COM": [("PSI de comprometimento_renda", "comprometimento_renda")],
    "FAI": [("PSI de score_bureau (a feature não se move)", "score_bureau")],
}


class PrivacyStep(BaseModel):
    """Um passo do fluxo de dados com a medida de privacidade aplicada."""

    key: str
    title: str
    measure: str
    artifact: str
    verify: str
    status: Literal["implementado", "política"]


class InventoryRow(BaseModel):
    """Uma coluna do contrato, com finalidade e uso no modelo."""

    column: str
    role: str
    purpose: str
    use: str


class LegalBasis(BaseModel):
    """Um cartão de base legal ou direito do titular."""

    article: str
    title: str
    summary: str
    use: str


class RetentionItem(BaseModel):
    """Um tipo de dado, o prazo-alvo de retenção e o que acontece depois."""

    key: str
    label: str
    where: str
    personal: bool
    days: int | None
    until: str
    after: str
    basis: str


class Evidence(BaseModel):
    """Uma evidência numérica de um nó do grafo causal."""

    label: str
    value: float | None = None
    text: str


class DagNode(BaseModel):
    """Um nó do grafo causal, com posição na grade e evidências por janela."""

    id: str
    label: str
    kind: Literal["narrativa", "mecanismo", "efeito", "consequencia"]
    column: int
    row: int
    text: str
    static: list[Evidence]
    by_window: dict[str, list[Evidence]]


class DagEdge(BaseModel):
    """Uma aresta do grafo causal (tracejada quando é narrativa)."""

    source: str
    target: str
    dashed: bool


class CausalDag(BaseModel):
    """O grafo causal da degradação."""

    nodes: list[DagNode]
    edges: list[DagEdge]
    windows: list[str]


class BandRow(BaseModel):
    """Aprovações falsas na faixa intermediária de score, por modelo."""

    model: str
    role: str
    clients: int
    observed_rate: float
    mean_score: float
    approved: int
    defaulters_approved: int
    expected_defaulters: float
    all_defaulters_approved: int


class StarBlock(BaseModel):
    """Um bloco do roteiro do vídeo STAR."""

    name: str
    start: int
    end: int
    duration: int
    screen: str
    artifacts: list[str]
    speech: str


class StarScript(BaseModel):
    """O roteiro do vídeo, lido de ``docs/video_star_roteiro.md``."""

    blocks: list[StarBlock]
    total_seconds: int


class Etapa4Payload(BaseModel):
    """Dados da página da Etapa 4."""

    release: ReleaseInfo
    steps: list[PrivacyStep]
    inventory: list[InventoryRow]
    legal: list[LegalBasis]
    retention: list[RetentionItem]
    fairness: FairnessAudit
    causal: CausalAblation
    dag: CausalDag
    band: list[BandRow]
    star: StarScript
    acceptance: list[AcceptanceItem]
    commands: list[str]


@dataclass(frozen=True)
class _Ctx:
    drift: DriftReport
    scenarios: dict[str, ScenarioParams]
    causal: CausalAblation
    audit: AuditReport
    bands: list[BandRow]


def _br(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}".replace(".", ",")


def _pct(value: float, digits: int = 1) -> str:
    return f"{_br(100 * value, digits)} %"


def _delta_text(delta: float, spread: float) -> str:
    sign = "\N{MINUS SIGN}" if delta < 0 else "+"
    return f"{sign} {_br(abs(delta), 3)} \N{PLUS-MINUS SIGN} {_br(spread, 3)}"


def _ev(label: str, value: float, text: str) -> Evidence:
    return Evidence(label=label, value=value, text=text)


def load_content() -> dict[str, Any]:
    """Lê o conteúdo textual da página (``static/etapa4_content.yaml``).

    Returns:
        Dicionário com passos, bases legais, retenção, finalidades e o grafo.
    """
    return read_yaml(STATIC_DIR / CONTENT_FILE)


def inventory_rows(purposes: dict[str, str]) -> list[InventoryRow]:
    """Inventário das 18 colunas do contrato, na ordem do dicionário de dados.

    Args:
        purposes: Finalidade de cada coluna (do conteúdo textual).

    Returns:
        Uma linha por coluna, com finalidade e uso no modelo.
    """
    rows = []
    for spec in dd.COLUMN_SPECS:
        if spec.name in dd.SENSITIVE_COLUMNS:
            use = "fora do modelo (só auditoria)"
        elif spec.name in dd.FEATURE_COLUMNS:
            use = "feature"
        elif spec.name == dd.TARGET:
            use = "rótulo de treino e avaliação"
        else:
            use = "não é feature"
        rows.append(
            InventoryRow(
                column=spec.name,
                role=spec.role,
                purpose=purposes[spec.name],
                use=use,
            )
        )
    return rows


def _seconds(text: str) -> int:
    match = _CLOCK.match(text.strip())
    if match is None:
        msg = f"horário inválido no roteiro: {text!r}"
        raise ValueError(msg)
    return int(match[1]) * 60 + int(match[2])


def _speeches(text: str) -> dict[str, str]:
    """Fala de cada bloco: o texto citado (``>``) sob o título ``### nome``."""
    speeches = {}
    for section in re.split(r"^### ", text, flags=re.MULTILINE)[1:]:
        lines = section.splitlines()
        quote = " ".join(
            line.removeprefix("> ").strip() for line in lines if line.startswith("> ")
        )
        speeches[lines[0].split(" (")[0]] = quote
    return speeches


def _block(cells: list[str], speeches: dict[str, str]) -> StarBlock:
    name, start, end, duration, screen, artifact = cells
    return StarBlock(
        name=name,
        start=_seconds(start),
        end=_seconds(end),
        duration=int(duration.removesuffix(" s")),
        screen=screen,
        artifacts=_PATH.findall(artifact),
        speech=speeches.get(name, ""),
    )


def parse_star(text: str) -> StarScript:
    """Lê o cronograma e as falas de ``docs/video_star_roteiro.md``.

    Args:
        text: Conteúdo do roteiro.

    Returns:
        Blocos em ordem, com horários, tela, artefatos e fala.

    Raises:
        ValueError: Se o cronograma estiver ausente, vazio ou com horário inválido.
    """
    if "## Cronograma" not in text:
        msg = "o roteiro não tem a seção `## Cronograma`"
        raise ValueError(msg)
    section = text.split("## Cronograma", 1)[1].split("\n## ", 1)[0]
    speeches = _speeches(text)
    rows = (
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in section.splitlines()
    )
    blocks = [
        _block(cells, speeches)
        for cells in rows
        if len(cells) == CELLS_PER_ROW and _CLOCK.match(cells[1])
    ]
    if not blocks:
        msg = "o cronograma do roteiro está vazio"
        raise ValueError(msg)
    return StarScript(blocks=blocks, total_seconds=blocks[-1].end)


def band_check(
    frame: pd.DataFrame, scores: pd.Series, *, threshold: float, model: str, role: str
) -> BandRow:
    """Aprovações falsas na faixa intermediária de ``score_bureau``.

    Args:
        frame: Clientes com ``score_bureau`` e ``inadimplente``.
        scores: Probabilidade de inadimplência prevista, alinhada a ``frame``.
        threshold: Limiar operacional (nega quando ``score >= threshold``).
        model: Versão do modelo.
        role: ``champion`` ou ``challenger``.

    Returns:
        Contagens da faixa e dos inadimplentes aprovados.
    """
    in_band = frame["score_bureau"].between(*BAND)
    approved = scores < threshold
    approved_band = in_band & approved
    defaulters = frame["inadimplente"] == 1
    return BandRow(
        model=model,
        role=role,
        clients=int(in_band.sum()),
        observed_rate=float(frame.loc[in_band, "inadimplente"].mean()),
        mean_score=float(scores[in_band].mean()),
        approved=int(approved_band.sum()),
        defaulters_approved=int((approved_band & defaulters).sum()),
        expected_defaulters=float(scores[approved_band].sum()),
        all_defaulters_approved=int((approved & defaulters).sum()),
    )


def _band_rows(settings: Settings) -> list[BandRow]:
    crisis = read_parquet(settings.production_dir / f"{CRISIS}.parquet")
    holdout = crisis[crisis["data_referencia"] >= HOLDOUT_START].reset_index(drop=True)
    rows = []
    for role, folder in (("champion", "baseline"), ("challenger", "challenger")):
        bundle = load_model_bundle(settings.models_dir / folder)
        scores = pd.Series(predict_proba(bundle, holdout))
        rows.append(
            band_check(
                holdout,
                scores,
                threshold=decision_threshold(bundle),
                model=bundle.version,
                role=role,
            )
        )
    return rows


def _scenario(ctx: _Ctx, window: str, name: str) -> Any:  # noqa: ANN401 - campo dinâmico
    return getattr(ctx.scenarios[window], name)


def _format(value: float, kind: str) -> str:
    return _pct(value) if kind == "pct" else _br(value, 2)


def _unemployment(window: str, ctx: _Ctx) -> list[Evidence]:
    share = float(_scenario(ctx, window, "employment_shares")["desempregado"])
    return [_ev("Desempregados na carteira", share, _pct(share))]


def _covariate(window: str, ctx: _Ctx) -> list[Evidence]:
    summary = ctx.drift.windows[window].drift
    critical = len(summary.psi_critical_features)
    return [
        _ev("Features com PSI crítico", critical, str(critical)),
        _ev(
            "Features com drift significativo",
            summary.n_drifted_features,
            str(summary.n_drifted_features),
        ),
    ]


def _prior(window: str, ctx: _Ctx) -> list[Evidence]:
    rate = ctx.drift.windows[window].performance.base_rate
    return [_ev("Taxa de inadimplência", rate, _pct(rate, 2))]


def _alerts(window: str, ctx: _Ctx) -> list[Evidence]:
    result = next(w for w in ctx.audit.windows if w.window == window)
    names = ", ".join(a.severity for a in result.slo.alerts)
    balance = result.slo.remaining_budget
    psi = ctx.drift.windows[window].predictions.psi_score_vs_reference
    return [
        _ev("PSI do score", psi, _br(psi, 3)),
        _ev("Saldo do orçamento", balance, _pct(balance, 0)),
        Evidence(label="Alertas", text=names),
    ]


def _performance(window: str, ctx: _Ctx) -> list[Evidence]:
    report = ctx.drift.windows[window]
    auc, approval = report.performance.roc_auc, report.predictions.approval_rate
    return [
        _ev("ROC-AUC do champion", auc, _br(auc, 3)),
        _ev("Taxa de aprovação", approval, _pct(approval)),
    ]


def _calibration(window: str, ctx: _Ctx) -> list[Evidence]:
    brier = ctx.drift.windows[window].performance.brier
    return [_ev("Brier", brier, _br(brier, 3))]


_SPECIAL: dict[str, Callable[[str, _Ctx], list[Evidence]]] = {
    "DES": _unemployment,
    "EMP": _unemployment,
    "COV": _covariate,
    "PRI": _prior,
    "SLO": _alerts,
    "AUC": _performance,
    "BRI": _calibration,
}


def _window_evidence(node: str, window: str, ctx: _Ctx) -> list[Evidence]:
    items = []
    for label, name, kind in _PARAM_EVIDENCE.get(node, ()):
        value = float(_scenario(ctx, window, name))
        items.append(_ev(label, value, _format(value, kind)))
    for label, feature in _PSI_EVIDENCE.get(node, ()):
        numeric = ctx.drift.windows[window].drift.numeric
        psi = next(item.psi for item in numeric if item.feature == feature)
        items.append(_ev(label, psi, _br(psi, 3)))
    if node in _SPECIAL:
        items.extend(_SPECIAL[node](window, ctx))
    return items


def _ablation(ctx: _Ctx, key: str, label: str) -> list[Evidence]:
    item = next(s for s in ctx.causal.scenarios if s.key == key)
    delta = _delta_text(item.delta_auc, item.delta_auc_std)
    return [
        _ev(f"{label}: taxa base", item.base_rate, _pct(item.base_rate, 2)),
        _ev(f"{label}: variação do AUC", item.delta_auc, delta),
    ]


def _false_approvals(ctx: _Ctx) -> list[Evidence]:
    evidence = []
    for row in ctx.bands:
        expected = _br(row.expected_defaulters, 0)
        evidence.append(
            _ev(
                f"Faixa 550-700, {row.role}: inadimplentes aprovados",
                row.defaulters_approved,
                f"{row.defaulters_approved} (o modelo esperava {expected})",
            )
        )
    return evidence


def _static_evidence(node: str, ctx: _Ctx) -> list[Evidence]:
    if node == "INT":
        return _ablation(ctx, "crise_sem_aperto", "Crise sem o aperto")
    if node == "COV":
        return _ablation(ctx, "so_covariate", "Só covariate shift")
    if node == "CON":
        return _ablation(ctx, "so_concept", "Só concept drift")
    if node == "FAI":
        return _ablation(ctx, "so_faixa_550_700", "Só a faixa 550-700")
    return _false_approvals(ctx) if node == "AUC" else []


def build_dag(content: dict[str, Any], ctx: _Ctx) -> CausalDag:
    """Monta o grafo causal com as evidências numéricas por janela.

    Args:
        content: Conteúdo textual (``dag.nodes`` e ``dag.edges``).
        ctx: Relatórios e cenários de onde saem os números.

    Returns:
        Grafo com nós, arestas e as janelas disponíveis.
    """
    nodes = [
        DagNode(
            id=node_id,
            **spec,
            static=_static_evidence(node_id, ctx),
            by_window={w: _window_evidence(node_id, w, ctx) for w in WINDOW_NAMES},
        )
        for node_id, spec in content["dag"]["nodes"].items()
    ]
    edges = [
        DagEdge(source=s, target=t, dashed=dashed)
        for s, t, dashed in content["dag"]["edges"]
    ]
    return CausalDag(nodes=nodes, edges=edges, windows=list(WINDOW_NAMES))


def _item(criterion: str, *, ok: bool, evidence: str) -> AcceptanceItem:
    return AcceptanceItem(criterion=criterion, ok=ok, evidence=evidence)


def _fairness_items(audit: FairnessAudit, settings: Settings) -> list[AcceptanceItem]:
    worst = min(g.disparate_impact for r in audit.reports for g in r.groups)
    signature = read_json(settings.models_dir / "baseline" / "signature.json")
    inputs = {entry["name"] for entry in signature["inputs"]}
    return [
        _item(
            "A auditoria cobre 3 atributos, 2 modelos e 2 conjuntos de dados",
            ok=len(audit.reports) == EXPECTED_REPORTS,
            evidence=f"{len(audit.reports)} relatórios em fairness_audit.json",
        ),
        _item(
            "Todos os grupos ficam dentro da regra dos 4/5",
            ok=audit.passed,
            evidence=f"menor razão de impacto: {_br(worst, 3)}",
        ),
        _item(
            "genero e regiao não estão nas entradas do modelo",
            ok=inputs.isdisjoint(dd.SENSITIVE_COLUMNS),
            evidence=f"{len(inputs)} entradas na assinatura do baseline",
        ),
    ]


def _privacy_items(settings: Settings) -> list[AcceptanceItem]:
    report = read_json(settings.reports_dir / "validation" / "bad_batch_report.json")
    blocked = any(
        v["rule_id"] == "R01" and v["column"] == "cpf" for v in report["violations"]
    )
    scores = read_parquet(
        settings.reports_dir / "predictions" / f"{CRISIS}_scores.parquet"
    )
    return [
        _item(
            "A coluna cpf é barrada pela regra R01 e o lote termina BLOCKED",
            ok=blocked and report["decision"] == "BLOCKED",
            evidence=f"decisão: {report['decision']}",
        ),
        _item(
            "As predições guardam só o hash do id, o score e a decisão",
            ok=list(scores.columns) == PREDICTION_COLUMNS,
            evidence=", ".join(scores.columns),
        ),
    ]


def _causal_items(causal: CausalAblation) -> list[AcceptanceItem]:
    drop = {s.key: s.delta_auc for s in causal.scenarios}
    return [
        _item(
            "O concept drift sozinho derruba o AUC e retirá-lo recupera o AUC",
            ok=drop["so_concept"] <= -MIN_CONCEPT_DROP
            and abs(drop["crise_sem_concept"]) < MIN_CONCEPT_DROP,
            evidence=f"só concept: {_br(drop['so_concept'], 3)}; crise sem concept:"
            f" {_br(drop['crise_sem_concept'], 3)}",
        ),
        _item(
            "O covariate shift sozinho não derruba o AUC",
            ok=abs(drop["so_covariate"]) < MIN_CONCEPT_DROP,
            evidence=f"só covariate: {_br(drop['so_covariate'], 3)}",
        ),
    ]


def _document_items(
    star: StarScript, audit_md: str, settings: Settings
) -> list[AcceptanceItem]:
    readme = (settings.project_root / "README.md").read_text(encoding="utf-8")
    names = " ".join(b.name for b in star.blocks)
    missing = [s.strip(" #.") for s in README_SECTIONS if s not in readme]
    return [
        _item(
            "O roteiro STAR tem os quatro blocos e cabe em 5 minutos",
            ok=star.total_seconds <= MAX_VIDEO_SECONDS
            and all(w in names for w in ("Situation", "Task", "Action", "Result")),
            evidence=f"{star.total_seconds} s em {len(star.blocks)} blocos",
        ),
        _item(
            "O README traz as seções 7 a 12, da governança aos agradecimentos",
            ok=not missing,
            evidence="seções 7 a 12 presentes" if not missing else f"faltam: {missing}",
        ),
        _item(
            "O ADR 006 e o relatório de fairness não contêm dados pessoais",
            ok=(settings.docs_dir / "adr" / "006-privacidade-e-governanca.md").exists()
            and scan_for_pii(audit_md) == [],
            evidence="scan-pii sem achados em fairness_audit.md",
        ),
    ]


def build_etapa4(settings: Settings, release: ReleaseInfo) -> Etapa4Payload:
    """Monta o payload da Etapa 4 a partir dos relatórios, do roteiro e dos modelos.

    Args:
        settings: Configuração do projeto.
        release: Metadados da release.

    Returns:
        Payload validado.
    """
    reports = settings.reports_dir
    fairness = FairnessAudit.model_validate(
        read_json(require(reports / "fairness_audit.json", AUDIT_HINT))
    )
    causal = CausalAblation.model_validate(
        read_json(require(reports / "causal_ablation.json", CAUSAL_HINT))
    )
    drift = DriftReport.model_validate(
        read_json(require(reports / "evidently_drift_summary.json", DRIFT_HINT))
    )
    audit = AuditReport.model_validate(
        read_json(require(reports / "slo_audit_report.json", PIPELINE_HINT))
    )
    roteiro = require(settings.docs_dir / ROTEIRO_FILE, ROTEIRO_HINT)
    star = parse_star(roteiro.read_text(encoding="utf-8"))
    content = load_content()
    bands = _band_rows(settings)
    scenarios = load_scenarios(settings.config_dir / SCENARIOS_FILE)
    ctx = _Ctx(drift, scenarios, causal, audit, bands)
    audit_md = (reports / "fairness_audit.md").read_text(encoding="utf-8")
    return Etapa4Payload(
        release=release,
        steps=[PrivacyStep.model_validate(s) for s in content["steps"]],
        inventory=inventory_rows(content["purpose"]),
        legal=[LegalBasis.model_validate(x) for x in content["legal"]],
        retention=[RetentionItem.model_validate(x) for x in content["retention"]],
        fairness=fairness,
        causal=causal,
        dag=build_dag(content, ctx),
        band=bands,
        star=star,
        acceptance=[
            *_fairness_items(fairness, settings),
            *_privacy_items(settings),
            *_causal_items(causal),
            *_document_items(star, audit_md, settings),
        ],
        commands=ETAPA4_COMMANDS,
    )


__all__ = ["Etapa4Payload", "band_check", "build_etapa4", "parse_star"]
