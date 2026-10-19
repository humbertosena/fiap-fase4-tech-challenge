"""Testes do relógio com suporte a SOURCE_DATE_EPOCH."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from credit_sentinel.utils.clock import fixed_epoch, utc_now

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
