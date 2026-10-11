"""Smoke tests do subcomando ``generate-data``."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from credit_sentinel.cli import main
from credit_sentinel.utils.io import file_sha256

if TYPE_CHECKING:
    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.smoke


def test_generate_data_writes_reference_and_dictionary(
    project_tmp: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["generate-data", "--rows", "300"]) == 0
    reference = pd.read_parquet(project_tmp.reference_dir / "credit_ref.parquet")
    assert reference.shape == (300, 18)
    assert (project_tmp.docs_dir / "data_dictionary.md").exists()
    bad_batch = pd.read_parquet(project_tmp.corrupted_dir / "bad_batch.parquet")
    assert bad_batch.shape == (300, 19)
    out = capsys.readouterr().out
    assert "sha256" in out
    assert "taxa base" in out


@pytest.mark.usefixtures("project_tmp")
def test_generate_data_refuses_overwrite_without_force() -> None:
    assert main(["generate-data", "--rows", "200"]) == 0
    assert main(["generate-data", "--rows", "200"]) == 1


def test_force_regenerates_identical_file(project_tmp: Settings) -> None:
    path = project_tmp.reference_dir / "credit_ref.parquet"
    main(["generate-data", "--rows", "200"])
    first = file_sha256(path)
    assert main(["generate-data", "--rows", "200", "--force"]) == 0
    assert file_sha256(path) == first
