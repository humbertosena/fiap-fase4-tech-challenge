"""Atualização automática da seção de métricas do ``docs/model_card.md``."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

    from credit_sentinel.ml.train_baseline import ModelBundle

START_MARKER = "<!-- metrics:start -->"
END_MARKER = "<!-- metrics:end -->"


def br(value: float, digits: int = 4) -> str:
    """Formata um decimal no padrão pt-BR (vírgula decimal).

    Args:
        value: Número a formatar.
        digits: Casas decimais.

    Returns:
        Texto como ``0,8291``.
    """
    return f"{value:.{digits}f}".replace(".", ",")


def render_metrics_section(bundle: ModelBundle) -> str:
    """Gera a tabela Markdown de métricas, limiares e importância.

    Args:
        bundle: Modelo treinado e analisado.

    Returns:
        Bloco Markdown.
    """
    m = bundle.metrics
    (tn, fp), (fn, tp) = m.confusion_matrix
    header = (
        f"Versão `{m.model_version}` · validação com {m.n_valid} propostas"
        f" (treino: {m.n_train}) · taxa base {br(100 * m.base_rate, 2)} %"
    )
    classification = f"{br(m.f1, 3)} / {br(m.precision, 3)} / {br(m.recall, 3)}"
    matrix = f"VN={tn}, FP={fp}, FN={fn}, VP={tp}"
    lines = [
        header,
        "",
        "| Métrica | Valor | Critério |",
        "| :--- | ---: | :--- |",
        f"| ROC-AUC | {br(m.roc_auc)} | ≥ 0,80 |",
        f"| Brier | {br(m.brier)} | ≤ 0,06 |",
        f"| KS | {br(m.ks_statistic)} | ≥ 0,45 |",
        f"| Average precision | {br(m.average_precision)} | — |",
        f"| F1 / precisão / recall | {classification} | limiar de F1 |",
        "",
        f"Matriz de confusão no limiar {br(m.threshold, 3)}: {matrix}.",
    ]
    if bundle.threshold is not None:
        lines.append(
            f"Limiar de custo (FN custa {bundle.threshold.fn_cost:g} vezes um FP):"
            f" {br(bundle.threshold.cost_threshold, 3)}."
        )
    if bundle.importance:
        top = ", ".join(f"`{i.feature}`" for i in bundle.importance[:3])
        lines.append(f"Top-3 por importância de permutação (ROC-AUC): {top}.")
    return "\n".join(lines)


def update_model_card(path: Path, section: str) -> None:
    """Substitui o conteúdo entre os marcadores de métricas.

    Args:
        path: Arquivo do model card.
        section: Novo conteúdo Markdown.

    Raises:
        ValueError: Se os marcadores não existirem no arquivo.
    """
    text = path.read_text(encoding="utf-8")
    if START_MARKER not in text or END_MARKER not in text:
        msg = f"marcadores {START_MARKER} / {END_MARKER} ausentes em {path}"
        raise ValueError(msg)
    head, rest = text.split(START_MARKER, 1)
    _, tail = rest.split(END_MARKER, 1)
    path.write_text(
        f"{head}{START_MARKER}\n{section}\n{END_MARKER}{tail}", encoding="utf-8"
    )
