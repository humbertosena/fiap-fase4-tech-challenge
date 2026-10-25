"""Testes do relógio com suporte a SOURCE_DATE_EPOCH."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from credit_sentinel.contracts.schema import build_reference_schema
from credit_sentinel.contracts.validator import (
    BatchContext,
    ValidationPolicy,
    validate_batch,
)
from credit_sentinel.utils.clock import fixed_epoch, utc_now

if TYPE_CHECKING:
    import pandas as pd

pytestmark = pytest.mark.unit


def test_utc_now_follows_the_wall_clock_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SOURCE_DATE_EPOCH", raising=False)
    now = utc_now()
    assert now.tzinfo is UTC
    assert abs((datetime.now(UTC) - now).total_seconds()) < 5
    assert fixed_epoch() is None


def test_source_date_epoch_fixes_the_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1791727200")
    assert utc_now() == datetime(2026, 10, 11, 14, 0, tzinfo=UTC)
    assert fixed_epoch() == 1791727200


def test_invalid_source_date_epoch_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "ontem")
    with pytest.raises(ValueError, match="SOURCE_DATE_EPOCH"):
        utc_now()


def test_validation_report_is_stamped_by_the_project_clock(
    monkeypatch: pytest.MonkeyPatch, df_reference_small: pd.DataFrame
) -> None:
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "1791727200")
    _, report = validate_batch(
        df_reference_small,
        schema=build_reference_schema(),
        policy=ValidationPolicy(quarantine=False),
        context=BatchContext(batch_name="referencia", run_id="val-relogio"),
    )
    assert report.validated_at == datetime(2026, 10, 11, 14, 0, tzinfo=UTC)
