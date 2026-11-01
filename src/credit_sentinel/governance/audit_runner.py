"""Executa a auditoria de fairness completa sobre os modelos e dados do projeto.

Audita **champion** (``models/baseline``) e **challenger** (``models/challenger``) em
dois conjuntos de dados, cada um com os mesmos clientes para os dois modelos:

- ``referencia_validacao``: a validação do baseline (20 % da referência). O challenger
  treinou com os últimos seis meses da referência, então viu parte desses clientes; a
  leitura decisiva para ele é o conjunto seguinte;
- ``mes_3_crise_holdout``: a segunda quinzena de ``mes_3_crise``, o holdout temporal
  do retreino. O champion nunca viu a crise e o challenger só viu a primeira quinzena.
"""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from credit_sentinel.data.commands import reference_path
from credit_sentinel.drift.detection import MissingInputError, window_path
from credit_sentinel.drift.predictions import decision_threshold
from credit_sentinel.governance.fairness_audit import (
    AuditSubject,
    FairnessReport,
    audit_subject,
)
from credit_sentinel.ml.retrain_challenger import (
    BASELINE_DIR,
    CHALLENGER_DIR,
    CRISIS_TRAIN_UNTIL_DAY,
    CRISIS_WINDOW,
    split_crisis,
)
from credit_sentinel.ml.train_baseline import (
    ModelBundle,
    load_model_bundle,
    predict_proba,
    split_reference,
)
from credit_sentinel.utils.io import read_parquet

if TYPE_CHECKING:
    import pandas as pd

    from credit_sentinel.settings import Settings

DATASET_REFERENCE = "referencia_validacao"
DATASET_CRISIS = "mes_3_crise_holdout"
ROLE_CHAMPION = "champion"
ROLE_CHALLENGER = "challenger"
DATASET_DESCRIPTIONS = {
    DATASET_REFERENCE: (
        "validação do baseline (20 % da referência); o challenger treinou com os"
        " últimos seis meses da referência e viu parte destes clientes"
    ),
    DATASET_CRISIS: (
        f"dias {CRISIS_TRAIN_UNTIL_DAY + 1} a 30 de `{CRISIS_WINDOW}` (holdout temporal"
        " do retreino); nenhum dos modelos treinou com estes clientes"
    ),
}
_MODEL_FOLDERS = {
    ROLE_CHAMPION: (BASELINE_DIR, "train"),
    ROLE_CHALLENGER: (CHALLENGER_DIR, "retrain-challenger"),
}


def _load_models(settings: Settings) -> dict[str, ModelBundle]:
    models = {}
    for role, (folder, command) in _MODEL_FOLDERS.items():
        directory = settings.models_dir / folder
        if not (directory / "model.joblib").exists():
            msg = f"modelo {role} não encontrado; rode `credit-sentinel {command}`"
            raise MissingInputError(msg)
        models[role] = load_model_bundle(directory)
    return models


def _load_datasets(settings: Settings) -> dict[str, pd.DataFrame]:
    reference = reference_path(settings)
    if not reference.exists():
        msg = "referência não encontrada; rode `credit-sentinel generate-data`"
        raise MissingInputError(msg)
    crisis = window_path(settings, CRISIS_WINDOW)
    if not crisis.exists():
        msg = f"janela {CRISIS_WINDOW} não encontrada; rode `credit-sentinel simulate`"
        raise MissingInputError(msg)
    _, validation = split_reference(read_parquet(reference), seed=settings.seed)
    _, holdout = split_crisis(read_parquet(crisis))
    return {DATASET_REFERENCE: validation, DATASET_CRISIS: holdout}


def run_full_audit(*, settings: Settings) -> list[FairnessReport]:
    """Audita champion e challenger nos dois conjuntos de dados.

    Args:
        settings: Configuração do projeto (diretórios e semente do split).

    Returns:
        Doze relatórios: conjunto de dados, depois modelo, depois atributo.

    Raises:
        MissingInputError: Se faltar um modelo, a referência ou a janela da crise.
    """
    models = _load_models(settings)
    datasets = _load_datasets(settings)
    reports: list[FairnessReport] = []
    for (dataset, frame), (role, bundle) in product(datasets.items(), models.items()):
        subject = AuditSubject(
            model_version=bundle.version,
            role=role,
            dataset=dataset,
            threshold=decision_threshold(bundle),
        )
        reports += audit_subject(frame, predict_proba(bundle, frame), subject)
    return reports
