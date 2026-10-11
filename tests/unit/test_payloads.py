"""Testes dos payloads das páginas didáticas."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from credit_sentinel.cli import main
from credit_sentinel.contracts.privacy_sanitizer import scan_for_pii
from credit_sentinel.explainers.payloads import (
    RELEASES,
    Etapa1Payload,
    build_payload,
    release_info,
)

if TYPE_CHECKING:
    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.unit


def test_missing_artifacts_raise_clear_error(settings_tmp: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="artefato ausente"):
        build_payload(1, settings=settings_tmp)


def test_release_catalog_has_four_tags() -> None:
    assert [RELEASES[e].tag for e in (1, 2, 3, 4)] == [
        "v0.1.0-alpha",
        "v0.2.0-beta",
        "v0.3.0-rc",
        "v1.0.0-ga",
    ]


def test_release_info_accepts_overrides() -> None:
    info = release_info(1, tag="v0.1.1", date="2026-10-12")
    assert (info.tag, info.date) == ("v0.1.1", "2026-10-12")
    assert info.slug == "etapa-1-contratos"


@pytest.fixture
def etapa1(project_tmp: Settings) -> Etapa1Payload:
    assert main(["generate-data", "--rows", "1500"]) == 0
    assert main(["train", "--no-mlflow"]) == 0
    reference = project_tmp.reference_dir / "credit_ref.parquet"
    bad = project_tmp.corrupted_dir / "bad_batch.parquet"
    assert main(["validate", "--input", str(reference), "--schema", "reference"]) == 0
    assert main(["validate", "--input", str(bad)]) == 2
    payload = build_payload(1, settings=project_tmp)
    assert isinstance(payload, Etapa1Payload)
    return payload


def test_etapa1_payload_is_built_from_artifacts(etapa1: Etapa1Payload) -> None:
    assert etapa1.dataset.rows == 1500
    assert len(etapa1.rules) == 12
    assert etapa1.bad_batch.decision == "BLOCKED"
    expected = {"renda_mensal", "score_bureau", "taxa_endividamento"}
    assert set(etapa1.histograms) >= expected
    assert len(etapa1.thresholds.score_histogram.edges) == 51
    assert all(item.ok for item in etapa1.acceptance if "lote" in item.criterion)


def test_etapa1_payload_has_no_personal_data(etapa1: Etapa1Payload) -> None:
    assert scan_for_pii(etapa1.model_dump_json()) == []
    assert all("id_cliente" not in ex.values for ex in etapa1.validator_examples)
