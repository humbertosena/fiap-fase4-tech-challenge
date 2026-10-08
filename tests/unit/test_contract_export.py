"""Testes da exportação do contrato de dados para YAML."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from credit_sentinel.cli import main
from credit_sentinel.contracts.export import (
    CONTRACT_FILE,
    export_contract_yaml,
    load_contract_yaml,
)
from credit_sentinel.contracts.schema import build_reference_schema
from credit_sentinel.settings import Settings
from credit_sentinel.utils.io import read_yaml

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def test_yaml_round_trip_rebuilds_the_same_schema(tmp_path: Path) -> None:
    path = export_contract_yaml(tmp_path / "contract.yaml")
    assert load_contract_yaml(path).to_json() == build_reference_schema().to_json()


def test_document_sections_and_twelve_rules(tmp_path: Path) -> None:
    document = read_yaml(export_contract_yaml(tmp_path / "contract.yaml"))
    assert list(document) == ["info", "quality", "rules", "schema"]
    assert document["quality"]["max_rejected_ratio"] == 0.01
    assert [rule["id"] for rule in document["rules"]] == [
        f"R{n:02d}" for n in range(1, 13)
    ]


def test_yaml_keeps_accents_readable(tmp_path: Path) -> None:
    text = export_contract_yaml(tmp_path / "contract.yaml").read_text(encoding="utf-8")
    assert "Consistência" in text
    assert "\\xB7" not in text


def test_versioned_contract_is_in_sync_with_code(tmp_path: Path) -> None:
    versioned = Settings(_env_file=None).config_dir / CONTRACT_FILE
    fresh = export_contract_yaml(tmp_path / CONTRACT_FILE)
    assert versioned.read_text(encoding="utf-8") == fresh.read_text(encoding="utf-8")


def test_export_contract_cli(tmp_path: Path) -> None:
    output = tmp_path / "out.yaml"
    assert main(["export-contract", "--output", str(output)]) == 0
    assert load_contract_yaml(output).name == build_reference_schema().name
