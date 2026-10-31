"""Payload da Etapa 3: registry, campos de log, demo de PII e prints embutidos."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import pytest

from credit_sentinel.contracts.privacy_sanitizer import (
    pii_patterns,
    redact_text,
    scan_for_pii,
)
from credit_sentinel.explainers.payloads import build_payload
from credit_sentinel.explainers.payloads_etapa3 import (
    MAX_SCREENSHOT_BYTES,
    log_fields,
    pii_demo,
    registry_states,
    screenshot_cards,
)
from credit_sentinel.ml.evaluate_gates import GatesReport
from credit_sentinel.observability.logger import REQUIRED_FIELDS, PipelineEvent

if TYPE_CHECKING:
    from pathlib import Path

    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.unit


def _gates(*, passed: bool) -> GatesReport:
    base = {"roc_auc": 0.8, "brier": 0.05, "f1": 0.3, "approval_rate": 0.8}
    return GatesReport(
        passed=passed,
        results=[],
        champion_model="baseline-1.0.0",
        challenger_model="challenger-1.0.0",
        champion_version=1,
        challenger_version=2,
        holdout_window="mes_3_crise",
        holdout_rows=10,
        champion=base,
        challenger=base,
        fairness=[],
    )


def test_registry_after_promotion_moves_the_champion_and_retires_v1() -> None:
    before, after = registry_states(_gates(passed=True))
    assert [(v.version, v.aliases) for v in before.versions] == [
        (1, ["champion"]),
        (2, ["challenger"]),
    ]
    assert [(v.version, v.aliases) for v in after.versions] == [
        (1, []),
        (2, ["champion"]),
    ]
    assert after.versions[0].tag == "retired_reason=drift_mes_3_crise"
    assert after.versions[1].tag is None


def test_registry_is_unchanged_when_the_gates_failed() -> None:
    before, after = registry_states(_gates(passed=False))
    assert after.versions == before.versions
    assert "recusada" in after.note


def test_registry_needs_the_versions_from_the_gates_report() -> None:
    report = _gates(passed=True).model_copy(update={"challenger_version": None})
    with pytest.raises(ValueError, match="versões"):
        registry_states(report)


def test_log_fields_mark_the_eight_required_ones() -> None:
    fields = {f.name: f for f in log_fields()}
    assert {n for n, f in fields.items() if f.required} == set(REQUIRED_FIELDS)
    assert set(fields) == set(PipelineEvent.model_fields)
    assert all(f.description for f in fields.values())


def test_pii_demo_fragments_never_contain_personal_data_on_their_own() -> None:
    demo = pii_demo()
    assert scan_for_pii(" ".join(demo.fragments)) == []
    assert scan_for_pii('"' + '", "'.join(demo.fragments) + '"') == []


def test_pii_demo_text_is_redacted_by_the_real_patterns() -> None:
    demo = pii_demo()
    text = "".join(demo.fragments)
    assert {f.kind for f in scan_for_pii(text)} == {"id_cliente", "cpf", "email"}
    assert scan_for_pii(redact_text(text)) == []
    assert redact_text(text).count("***") >= 3


def test_pii_patterns_are_the_ones_used_by_the_sanitizer() -> None:
    patterns = pii_patterns()
    assert {kind for kind, _ in patterns} == {"cpf", "email", "telefone", "id_cliente"}
    for _, source in patterns:
        assert re.compile(source)


def _jpeg(path: Path, size: tuple[int, int]) -> None:
    """Escreve bytes com cabeçalho JPEG; o gerador só os codifica em base64."""
    path.write_bytes(b"\xff\xd8\xff\xe0" + bytes(size))


def test_screenshot_cards_embed_base64_in_the_requested_order(tmp_path: Path) -> None:
    for name in ("a.jpg", "b.jpg"):
        _jpeg(tmp_path / name, (120, 80))
    cards = screenshot_cards(
        tmp_path,
        [("b.jpg", "Segundo", "legenda b"), ("a.jpg", "Primeiro", "legenda a")],
    )
    assert [c.title for c in cards] == ["Segundo", "Primeiro"]
    assert all(c.data_uri.startswith("data:image/jpeg;base64,") for c in cards)
    assert all(0 < c.size_bytes < MAX_SCREENSHOT_BYTES for c in cards)


def test_screenshot_over_the_limit_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "grande.jpg").write_bytes(b"0" * (MAX_SCREENSHOT_BYTES + 1))
    with pytest.raises(ValueError, match=r"grande\.jpg"):
        screenshot_cards(tmp_path, [("grande.jpg", "T", "c")])


def test_missing_artifacts_raise_a_clear_error(settings_tmp: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="run-pipeline"):
        build_payload(3, settings=settings_tmp)
