"""Testes do sanitizador de privacidade (Privacy by Default)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest
from pydantic import SecretStr

from credit_sentinel.cli import main
from credit_sentinel.contracts.privacy_sanitizer import (
    HASH_COLUMN,
    drop_direct_identifiers,
    hash_identifier,
    mask_record,
    pseudonymize_ids,
    scan_for_pii,
    scan_paths,
)
from credit_sentinel.contracts.schema import build_reference_schema
from credit_sentinel.contracts.validator import (
    BatchContext,
    ValidationPolicy,
    validate_batch,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def test_hash_is_deterministic_per_salt() -> None:
    first = hash_identifier("CLI-0A1B2C3D4E", "salt-a")
    assert first == hash_identifier("CLI-0A1B2C3D4E", "salt-a")
    assert first != hash_identifier("CLI-0A1B2C3D4E", "salt-b")
    assert len(first) == 64
    assert "CLI-" not in first


def test_pseudonymize_replaces_id_column_in_place() -> None:
    df = pd.DataFrame({"id_cliente": ["CLI-0A1B2C3D4E", None], "idade": [30, 40]})
    out = pseudonymize_ids(df, salt="s")
    assert list(out.columns) == [HASH_COLUMN, "idade"]
    assert out.loc[0, HASH_COLUMN] == hash_identifier("CLI-0A1B2C3D4E", "s")
    assert pd.isna(out.loc[1, HASH_COLUMN])
    assert "id_cliente" in df.columns  # entrada não é alterada


def test_drop_direct_identifiers_reports_dropped_columns() -> None:
    df = pd.DataFrame({"CPF": ["1"], "email": ["a"], "idade": [30]})
    out, dropped = drop_direct_identifiers(df)
    assert list(out.columns) == ["idade"]
    assert dropped == ["CPF", "email"]


def test_mask_record_hides_identifiers_and_pii_in_text() -> None:
    record = {
        "id_cliente": "CLI-0A1B2C3D4E",
        "cpf": "529.982.247-25",
        "mensagem": "contato fulano@gmail.com",
        "idade": 30,
    }
    masked = mask_record(record)
    assert masked["id_cliente"] == "***"
    assert masked["cpf"] == "***"
    assert "fulano@gmail.com" not in masked["mensagem"]
    assert masked["idade"] == 30


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("cpf 529.982.247-25 no texto", "cpf"),
        ("documento 52998224725.", "cpf"),
        ("escreva para fulano.silva@gmail.com", "email"),
        ("ligue (11) 98765-4321", "telefone"),
        ("whatsapp +55 21 91234-5678", "telefone"),
        ('{"id_cliente": "CLI-0A1B2C3D4E"}', "id_cliente"),
    ],
)
def test_scan_detects_positive_cases(text: str, kind: str) -> None:
    findings = scan_for_pii(text, location="x")
    assert [finding.kind for finding in findings] == [kind]
    assert findings[0].sample_masked != text


@pytest.mark.parametrize(
    "text",
    [
        "data 2026-10-01 e período 2025-2026",
        "probabilidade 0.12345678901 e versão 3.18.0",
        "sha256 cb41f7d3b239c3ea43e100f478ef8975c6c815abbcf1f89ad018b2f2a4224ec4",
        "formato CLI-XXXXXXXXXX e regex ^CLI-[0-9A-F]{10}$",
        "dono time-de-credito@fintech.exemplo",
        "valor R$ 1.000.000,00 e 12000 linhas",
    ],
)
def test_scan_ignores_negative_cases(text: str) -> None:
    assert scan_for_pii(text) == []


def test_scan_ignores_base64_data_uris() -> None:
    text = 'src="data:image/jpeg;base64,QUJD12345678901REVG"'
    assert scan_for_pii(text) == []


def test_scan_paths_walks_text_files_with_line_numbers(tmp_path: Path) -> None:
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "log.jsonl").write_text(
        '{"ok": 1}\n{"cpf": "529.982.247-25"}\n'
    )
    (tmp_path / "clean.md").write_text("nada aqui\n")
    (tmp_path / "image.png").write_bytes(b"529.982.247-25")
    findings = scan_paths([tmp_path])
    assert len(findings) == 1
    assert findings[0].location.endswith("log.jsonl:2")


def test_validation_report_samples_are_pseudonymized(
    df_reference_small: pd.DataFrame, tmp_path: Path
) -> None:
    batch = df_reference_small.copy()
    batch.loc[0, "idade"] = 15
    context = BatchContext(
        batch_name="b", run_id="r", quarantine_dir=tmp_path, pii_salt=SecretStr("s")
    )
    _, report = validate_batch(
        batch,
        schema=build_reference_schema(),
        policy=ValidationPolicy(),
        context=context,
    )
    (sample,) = report.sample_failures
    assert sample[HASH_COLUMN] == hash_identifier(batch.loc[0, "id_cliente"], "s")
    assert scan_for_pii(report.model_dump_json()) == []


def test_scan_pii_cli_exit_codes(tmp_path: Path) -> None:
    clean = tmp_path / "clean"
    clean.mkdir()
    (clean / "a.json").write_text('{"ok": true}')
    assert main(["scan-pii", str(clean)]) == 0
    (clean / "b.log").write_text("fulano@gmail.com")
    assert main(["scan-pii", str(clean)]) == 1
