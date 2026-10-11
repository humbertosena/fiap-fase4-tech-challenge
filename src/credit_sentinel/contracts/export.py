"""Exportação formal do contrato de dados para ``config/data_contract.yaml``.

O documento combina metadados de governança (``info``), a política de qualidade
(``quality``), as 12 regras legíveis (``rules``) e o schema Pandera serializado
(``schema``), que pode ser recarregado com ``load_contract_yaml``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandera.pandas as pa
import yaml

from credit_sentinel.contracts.schema import RULES, build_reference_schema
from credit_sentinel.contracts.validator import ValidationPolicy
from credit_sentinel.utils.io import read_yaml, write_text

if TYPE_CHECKING:
    from pathlib import Path

CONTRACT_FILE = "data_contract.yaml"
CONTRACT_INFO: dict[str, str] = {
    "name": "credit_scoring_input",
    "version": "1.0.0",
    "owner": "time-de-credito@fintech.exemplo",
    "purpose": "concessão de crédito pessoa física (LGPD art. 7º, incisos V, IX e X)",
    "classification": "dados pessoais pseudonimizados",
}


def contract_document(policy: ValidationPolicy | None = None) -> dict[str, Any]:
    """Monta o documento do contrato.

    Args:
        policy: Política do circuit breaker (padrão: ``ValidationPolicy()``).

    Returns:
        Dicionário com as seções ``info``, ``quality``, ``rules`` e ``schema``.
    """
    policy = policy or ValidationPolicy()
    return {
        "info": CONTRACT_INFO,
        "quality": {
            "max_rejected_ratio": policy.max_rejected_ratio,
            "fail_fast": True,
            "quarantine": policy.quarantine,
        },
        "rules": [
            rule.model_dump(mode="json", exclude_defaults=True) for rule in RULES
        ],
        "schema": yaml.safe_load(build_reference_schema().to_yaml()),
    }


def export_contract_yaml(path: Path, policy: ValidationPolicy | None = None) -> Path:
    """Grava o contrato em YAML legível (UTF-8, ordem preservada).

    Args:
        path: Arquivo de destino.
        policy: Política do circuit breaker.

    Returns:
        O próprio ``path``.
    """
    header = "# Gerado por `credit-sentinel export-contract`; não edite à mão.\n"
    body = yaml.safe_dump(
        contract_document(policy), allow_unicode=True, sort_keys=False, width=100
    )
    return write_text(header + body, path)


def load_contract_yaml(path: Path) -> pa.DataFrameSchema:
    """Recarrega o schema Pandera a partir do contrato exportado.

    Args:
        path: Arquivo ``data_contract.yaml``.

    Returns:
        Schema equivalente a ``build_reference_schema()``.
    """
    document = read_yaml(path)
    return pa.DataFrameSchema.from_yaml(
        yaml.safe_dump(document["schema"], sort_keys=False)
    )
