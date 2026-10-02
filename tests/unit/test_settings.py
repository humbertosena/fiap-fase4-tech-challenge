"""Testes da configuração centralizada (pydantic-settings)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from credit_sentinel.settings import Settings, get_settings

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def test_defaults_use_seed_42_and_local_sqlite_tracking() -> None:
    settings = Settings(_env_file=None)
    assert settings.seed == 42
    assert settings.mlflow_tracking_uri == "sqlite:///mlruns/mlflow.db"


def test_env_prefix_cs_overrides_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CS_SEED", "7")
    assert Settings(_env_file=None).seed == 7


def test_unprefixed_env_is_ignored(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SEED", "99")
    assert Settings(_env_file=None).seed == 42


def test_derived_directories_live_under_project_root(tmp_path: Path) -> None:
    settings = Settings(_env_file=None, project_root=tmp_path)
    assert settings.reference_dir == tmp_path / "data" / "reference"
    assert settings.production_dir == tmp_path / "data" / "production"
    assert settings.corrupted_dir == tmp_path / "data" / "corrupted"
    assert settings.quarantine_dir == tmp_path / "data" / "quarantine"
    assert settings.models_dir == tmp_path / "models"
    assert settings.reports_dir == tmp_path / "reports"
    assert settings.logs_dir == tmp_path / "logs"
    assert settings.config_dir == tmp_path / "config"
    assert settings.docs_dir == tmp_path / "docs"


def test_pii_salt_is_secret_and_never_rendered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CS_PII_SALT", "salt-super-secreto")
    settings = Settings(_env_file=None)
    assert "salt-super-secreto" not in repr(settings)
    assert "salt-super-secreto" not in str(settings)
    assert settings.pii_salt.get_secret_value() == "salt-super-secreto"


def test_get_settings_is_cached() -> None:
    assert get_settings() is get_settings()
