"""Model Registry: versões, aliases champion/challenger e promoção com guardrails."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

from credit_sentinel.cli import main
from credit_sentinel.ml import commands
from credit_sentinel.ml.evaluate_gates import GatesReport
from credit_sentinel.ml.features import build_features
from credit_sentinel.ml.mlflow_baseline import log_baseline_run
from credit_sentinel.ml.retrain_challenger import CHALLENGER_VERSION, retrain_challenger
from credit_sentinel.ml.train_baseline import (
    predict_proba,
    save_model_bundle,
    train_and_analyze,
)
from credit_sentinel.settings import Settings, get_settings
from credit_sentinel.tracking.mlflow_tracking import TrainingRunSpec, log_training_run
from credit_sentinel.tracking.registry import (
    MODEL_NAME,
    RETIRED_REASON,
    ModelRegistry,
    PromotionRefusedError,
)

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from credit_sentinel.ml.train_baseline import ModelBundle

pytestmark = pytest.mark.integration


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    """Configuração com o MLflow em um store temporário (nunca ``mlruns/`` do repo)."""
    monkeypatch.setenv("CS_PROJECT_ROOT", str(tmp_path))
    monkeypatch.setenv("CS_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/mlflow.db")
    get_settings.cache_clear()
    yield get_settings()
    get_settings.cache_clear()


def _bundle(frame: pd.DataFrame, settings: Settings, version: str) -> ModelBundle:
    return train_and_analyze(frame, settings=settings, version=version)


def _register_baseline(
    store: Settings, frame: pd.DataFrame
) -> tuple[ModelRegistry, ModelBundle, int]:
    bundle = _bundle(frame, store, "baseline-1.0.0")
    save_model_bundle(bundle, store.models_dir / "baseline")
    run_id = log_baseline_run(bundle, build_features(frame.head(5)), store)
    registry = ModelRegistry(store.mlflow_tracking_uri)
    version = registry.version_for_run(run_id)
    registry.set_alias("champion", version)
    return registry, bundle, version


def _register_challenger(
    store: Settings, frame: pd.DataFrame, registry: ModelRegistry
) -> tuple[ModelBundle, int]:
    bundle = _bundle(frame.sample(frac=0.8, random_state=1), store, CHALLENGER_VERSION)
    save_model_bundle(bundle, store.models_dir / "challenger")
    spec = TrainingRunSpec(
        experiment="credit-sentinel/retraining",
        run_name="challenger-teste",
        artifacts={"bundle": store.models_dir / "challenger"},
    )
    run_id = log_training_run(bundle, spec, store, build_features(frame.head(5)))
    version = registry.version_for_run(run_id)
    registry.set_alias("challenger", version)
    return bundle, version


def _report(*, passed: bool) -> GatesReport:
    base = {"roc_auc": 0.8, "brier": 0.05, "f1": 0.3, "approval_rate": 0.8}
    return GatesReport(
        passed=passed,
        results=[],
        champion_model="baseline-1.0.0",
        challenger_model=CHALLENGER_VERSION,
        holdout_window="mes_3_crise",
        holdout_rows=10,
        champion=base,
        challenger=base,
        fairness=[],
    )


def test_baseline_becomes_version_1_with_the_champion_alias(
    store: Settings, df_reference_small: pd.DataFrame
) -> None:
    registry, _, version = _register_baseline(store, df_reference_small)
    assert version == 1
    assert registry.version_of("champion") == 1
    assert registry.version_of("challenger") is None


def test_registry_lists_v1_champion_and_v2_challenger(
    store: Settings, df_reference_small: pd.DataFrame
) -> None:
    registry, _, _ = _register_baseline(store, df_reference_small)
    _, version = _register_challenger(store, df_reference_small, registry)
    assert version == 2
    listing = {v.version: v for v in registry.describe()}
    assert listing[1].aliases == ["champion"]
    assert listing[2].aliases == ["challenger"]
    assert MODEL_NAME == "credit-scoring-model"


def test_model_is_loaded_back_from_the_registry_by_alias(
    store: Settings, df_reference_small: pd.DataFrame, tmp_path: Path
) -> None:
    registry, baseline, _ = _register_baseline(store, df_reference_small)
    challenger, _ = _register_challenger(store, df_reference_small, registry)
    champion_back = registry.get_model_by_alias("champion", tmp_path / "dl1")
    challenger_back = registry.get_model_by_alias("challenger", tmp_path / "dl2")
    assert champion_back.version == "baseline-1.0.0"
    assert challenger_back.version == CHALLENGER_VERSION
    np.testing.assert_allclose(
        predict_proba(champion_back, df_reference_small),
        predict_proba(baseline, df_reference_small),
    )
    np.testing.assert_allclose(
        predict_proba(challenger_back, df_reference_small),
        predict_proba(challenger, df_reference_small),
    )


def test_unknown_alias_cannot_be_loaded(store: Settings, tmp_path: Path) -> None:
    registry = ModelRegistry(store.mlflow_tracking_uri)
    with pytest.raises(LookupError, match="champion"):
        registry.get_model_by_alias("champion", tmp_path / "dl")


def test_promotion_is_refused_when_the_gates_failed(
    store: Settings, df_reference_small: pd.DataFrame
) -> None:
    registry, _, _ = _register_baseline(store, df_reference_small)
    _register_challenger(store, df_reference_small, registry)
    with pytest.raises(PromotionRefusedError, match="guardrails"):
        registry.promote_challenger(gates=_report(passed=False))
    assert registry.version_of("champion") == 1
    assert registry.version_of("challenger") == 2


def test_promotion_without_a_challenger_is_refused(
    store: Settings, df_reference_small: pd.DataFrame
) -> None:
    registry, _, _ = _register_baseline(store, df_reference_small)
    with pytest.raises(PromotionRefusedError, match="challenger"):
        registry.promote_challenger(gates=_report(passed=True))


def test_promotion_swaps_the_aliases_and_retires_the_old_champion(
    store: Settings, df_reference_small: pd.DataFrame
) -> None:
    registry, _, _ = _register_baseline(store, df_reference_small)
    _register_challenger(store, df_reference_small, registry)
    promoted = registry.promote_challenger(gates=_report(passed=True))
    assert promoted == 2
    assert registry.version_of("champion") == 2
    assert registry.version_of("challenger") is None
    listing = {v.version: v for v in registry.describe()}
    assert listing[1].tags["retired_reason"] == RETIRED_REASON == "drift_mes_3_crise"
    assert listing[1].aliases == []
    assert "retired_reason" not in listing[2].tags


@pytest.mark.usefixtures("pipeline_project")
def test_cli_train_registers_the_baseline_as_champion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CS_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/m.db")
    get_settings.cache_clear()
    try:
        assert main(["train"]) == 0
        registry = ModelRegistry(get_settings().mlflow_tracking_uri)
        assert registry.version_of("champion") == 1
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def _force_gates(monkeypatch: pytest.MonkeyPatch, *, passed: bool) -> None:
    real = commands.retrain_challenger

    def forced(**kwargs: Settings) -> tuple[ModelBundle, GatesReport]:
        challenger, report = real(**kwargs)
        return challenger, report.model_copy(update={"passed": passed})

    monkeypatch.setattr(commands, "retrain_challenger", forced)


def test_cli_retrain_then_promote_follows_the_aliases_and_the_gates(
    project_clean: Settings, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CS_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/m.db")
    get_settings.cache_clear()
    try:
        assert main(["train"]) == 0
        registry = ModelRegistry(get_settings().mlflow_tracking_uri)
        _force_gates(monkeypatch, passed=True)
        assert main(["retrain-challenger"]) == 0
        assert (registry.version_of("champion"), registry.version_of("challenger")) == (
            1,
            2,
        )
        report = GatesReport.model_validate_json(
            (project_clean.reports_dir / "gates_report.json").read_text("utf-8")
        )
        assert (report.champion_version, report.challenger_version) == (1, 2)
        assert main(["retrain-challenger", "--promote"]) == 0
        assert registry.version_of("champion") == 3
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


@pytest.mark.usefixtures("project_clean")
def test_cli_promote_with_failed_gates_changes_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CS_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/m.db")
    get_settings.cache_clear()
    try:
        assert main(["train"]) == 0
        _force_gates(monkeypatch, passed=False)
        assert main(["retrain-challenger", "--promote"]) == 3
        registry = ModelRegistry(get_settings().mlflow_tracking_uri)
        assert registry.version_of("champion") == 1
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


@pytest.mark.usefixtures("project_clean")
def test_cli_without_mlflow_does_not_touch_the_registry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    uri_file = tmp_path / "ausente.db"
    monkeypatch.setenv("CS_MLFLOW_TRACKING_URI", f"sqlite:///{uri_file}")
    get_settings.cache_clear()
    try:
        assert main(["retrain-challenger", "--no-mlflow"]) in {0, 3}
        assert not uri_file.exists()
        assert main(["retrain-challenger", "--no-mlflow", "--promote"]) == 1
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


@pytest.mark.usefixtures("project_clean")
def test_real_retraining_registers_a_challenger_with_real_gates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CS_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/m.db")
    get_settings.cache_clear()
    try:
        assert main(["train"]) == 0
        challenger, _ = retrain_challenger(settings=get_settings())
        assert challenger.version == CHALLENGER_VERSION
        assert main(["retrain-challenger"]) in {0, 3}
        registry = ModelRegistry(get_settings().mlflow_tracking_uri)
        assert registry.version_of("challenger") == 2
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


def _train_and_retrain(
    monkeypatch: pytest.MonkeyPatch, *, passed: bool
) -> ModelRegistry:
    assert main(["train"]) == 0
    _force_gates(monkeypatch, passed=passed)
    assert main(["retrain-challenger"]) == (0 if passed else 3)
    return ModelRegistry(get_settings().mlflow_tracking_uri)


@pytest.mark.usefixtures("project_clean")
def test_promote_challenger_promotes_the_registered_version_without_retraining(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CS_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/m.db")
    get_settings.cache_clear()
    try:
        registry = _train_and_retrain(monkeypatch, passed=True)
        assert main(["promote-challenger"]) == 0
        assert registry.version_of("champion") == 2
        assert registry.version_of("challenger") is None
        listing = {v.version: v for v in registry.describe()}
        assert listing[1].tags["retired_reason"] == "drift_mes_3_crise"
        assert len(listing) == 2
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


@pytest.mark.usefixtures("project_clean")
def test_promote_challenger_is_refused_when_the_gates_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CS_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/m.db")
    get_settings.cache_clear()
    try:
        registry = _train_and_retrain(monkeypatch, passed=False)
        assert main(["promote-challenger"]) == 3
        assert registry.version_of("champion") == 1
        assert registry.version_of("challenger") == 2
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()


@pytest.mark.usefixtures("project_clean")
def test_promote_challenger_needs_a_report_of_the_current_challenger(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("CS_MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path}/m.db")
    get_settings.cache_clear()
    try:
        assert main(["promote-challenger"]) == 1
        assert "retrain-challenger" in capsys.readouterr().out
        registry = _train_and_retrain(monkeypatch, passed=True)
        registry.set_alias("challenger", 1)
        capsys.readouterr()
        assert main(["promote-challenger"]) == 1
        assert "não é do challenger atual" in capsys.readouterr().out
        assert registry.version_of("champion") == 1
    finally:
        monkeypatch.undo()
        get_settings.cache_clear()
