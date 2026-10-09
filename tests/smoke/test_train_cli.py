"""Smoke tests do subcomando ``train``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from credit_sentinel.cli import main
from credit_sentinel.utils.io import read_parquet, write_parquet

if TYPE_CHECKING:
    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.smoke


def test_train_validates_reference_and_saves_bundle(
    project_tmp: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["generate-data", "--rows", "1500"]) == 0
    assert main(["train", "--no-mlflow"]) == 0
    bundle_dir = project_tmp.models_dir / "baseline"
    assert (bundle_dir / "model.joblib").exists()
    assert (bundle_dir / "metrics.json").exists()
    assert "roc_auc" in capsys.readouterr().out


def test_train_refuses_reference_that_breaks_the_contract(
    project_tmp: Settings,
) -> None:
    assert main(["generate-data", "--rows", "600"]) == 0
    path = project_tmp.reference_dir / "credit_ref.parquet"
    broken = read_parquet(path)
    broken["cpf"] = "nao-previsto"
    write_parquet(broken, path)
    assert main(["train", "--no-mlflow"]) == 2
    assert not (project_tmp.models_dir / "baseline" / "model.joblib").exists()
