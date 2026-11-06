"""Subcomando ``causal-ablation``: relatórios, reprodutibilidade e erros."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from credit_sentinel.cli import main
from credit_sentinel.governance.causal_ablation import CATALOGUE, CausalAblation

if TYPE_CHECKING:
    from credit_sentinel.settings import Settings

pytestmark = pytest.mark.smoke

FAST = ["causal-ablation", "--seeds", "2", "--rows", "500"]
NOVEMBER_6_EPOCH = "1793934000"  # 2026-11-06T00:00:00-03:00 = 03:00 UTC


def _read(settings: Settings) -> CausalAblation:
    text = (settings.reports_dir / "causal_ablation.json").read_text("utf-8")
    return CausalAblation.model_validate_json(text)


def test_cli_writes_the_json_and_the_markdown(
    project_clean: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(FAST) == 0
    ablation = _read(project_clean)
    assert [s.key for s in ablation.scenarios] == [spec.key for spec in CATALOGUE]
    assert len(ablation.seeds) == 2
    markdown = (project_clean.reports_dir / "causal_ablation.md").read_text("utf-8")
    assert markdown.startswith("# Ablação causal por intervenção")
    out = capsys.readouterr().out
    assert "causal-ablation: 11 cenários" in out
    assert "reports/causal_ablation.json" in out


def test_same_clock_gives_identical_reports(
    project_clean: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", NOVEMBER_6_EPOCH)
    assert main(FAST) == 0
    first = (project_clean.reports_dir / "causal_ablation.json").read_bytes()
    assert main(FAST) == 0
    assert (project_clean.reports_dir / "causal_ablation.json").read_bytes() == first


@pytest.mark.parametrize("flag", [["--seeds", "0"], ["--rows", "0"]])
def test_non_positive_options_are_rejected(
    project_clean: Settings, flag: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    assert main(["causal-ablation", *flag]) == 1
    assert "positivos" in capsys.readouterr().out
    assert not (project_clean.reports_dir / "causal_ablation.json").exists()


def test_missing_baseline_exits_with_a_hint(
    project_clean: Settings, capsys: pytest.CaptureFixture[str]
) -> None:
    (project_clean.models_dir / "baseline" / "model.joblib").unlink()
    assert main(FAST) == 1
    assert "train" in capsys.readouterr().out
