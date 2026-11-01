"""Subcomando ``fairness-audit``: 3 atributos, 2 modelos e 2 conjuntos de dados."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from credit_sentinel.cli import main
from credit_sentinel.data.commands import reference_path
from credit_sentinel.drift.predictions import decision_threshold
from credit_sentinel.governance.fairness_audit import (
    FAIRNESS_ATTRIBUTES,
    FairnessAudit,
)
from credit_sentinel.ml.retrain_challenger import split_crisis
from credit_sentinel.ml.train_baseline import load_model_bundle, split_reference
from credit_sentinel.utils.io import read_parquet

if TYPE_CHECKING:
    from pathlib import Path

    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.smoke

EXPECTED_REPORTS = 12
NOVEMBER_1_EPOCH = "1793502000"  # 2026-11-01T00:00:00-03:00 = 03:00 UTC


def _read(settings: Settings) -> FairnessAudit:
    text = (settings.reports_dir / "fairness_audit.json").read_text("utf-8")
    return FairnessAudit.model_validate_json(text)


@pytest.fixture
def audited(project_clean: Settings) -> Settings:
    """Projeto com o challenger treinado e a auditoria já executada."""
    assert main(["retrain-challenger", "--no-mlflow"]) in {0, 3}
    assert main(["fairness-audit"]) == 0
    return project_clean


def test_audit_covers_every_attribute_model_and_dataset(audited: Settings) -> None:
    audit = _read(audited)
    combos = {(r.dataset, r.role, r.attribute) for r in audit.reports}
    assert len(audit.reports) == len(combos) == EXPECTED_REPORTS
    assert {r.dataset for r in audit.reports} == {
        "referencia_validacao",
        "mes_3_crise_holdout",
    }
    assert {r.role for r in audit.reports} == {"champion", "challenger"}
    assert {r.attribute for r in audit.reports} == set(FAIRNESS_ATTRIBUTES)
    assert set(audit.datasets) == {"referencia_validacao", "mes_3_crise_holdout"}


def test_audit_uses_each_models_own_threshold_and_the_documented_datasets(
    audited: Settings,
) -> None:
    audit = _read(audited)
    champion = load_model_bundle(audited.models_dir / "baseline")
    challenger = load_model_bundle(audited.models_dir / "challenger")
    reference = read_parquet(reference_path(audited))
    _, validation = split_reference(reference, seed=audited.seed)
    _, holdout = split_crisis(
        read_parquet(audited.production_dir / "mes_3_crise.parquet")
    )
    sizes = {
        "referencia_validacao": len(validation),
        "mes_3_crise_holdout": len(holdout),
    }
    for report in audit.reports:
        bundle = champion if report.role == "champion" else challenger
        assert report.model_version == bundle.version
        assert report.threshold == pytest.approx(decision_threshold(bundle))
        assert report.n == sizes[report.dataset]


def test_audit_writes_json_and_markdown(audited: Settings) -> None:
    markdown = (audited.reports_dir / "fairness_audit.md").read_text("utf-8")
    assert markdown.startswith("# Auditoria de fairness")
    assert "baseline-1.0.0" in markdown
    assert "challenger-1.0.0" in markdown
    assert "id_cliente" not in markdown
    assert "id_cliente" not in (audited.reports_dir / "fairness_audit.json").read_text()


def test_audit_is_reproducible_with_a_fixed_clock(
    audited: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", NOVEMBER_1_EPOCH)
    path: Path = audited.reports_dir / "fairness_audit.json"
    assert main(["fairness-audit"]) == 0
    first = path.read_bytes()
    assert main(["fairness-audit"]) == 0
    assert path.read_bytes() == first
    assert _read(audited).generated_at.isoformat().startswith("2026-11-01T03:00:00")


def test_strict_exit_code_follows_the_verdict(audited: Settings) -> None:
    passed = _read(audited).passed
    assert main(["fairness-audit", "--strict"]) == (0 if passed else 3)


def test_missing_challenger_points_to_retrain(
    project_clean: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    capsys.readouterr()
    assert main(["fairness-audit"]) == 1
    assert "retrain-challenger" in capsys.readouterr().out
    assert not (project_clean.reports_dir / "fairness_audit.json").exists()


def test_missing_window_points_to_simulate(
    audited: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    (audited.production_dir / "mes_3_crise.parquet").unlink()
    capsys.readouterr()
    assert main(["fairness-audit"]) == 1
    assert "simulate" in capsys.readouterr().out


def test_missing_reference_points_to_generate_data(
    audited: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    reference_path(audited).unlink()
    capsys.readouterr()
    assert main(["fairness-audit"]) == 1
    assert "generate-data" in capsys.readouterr().out
