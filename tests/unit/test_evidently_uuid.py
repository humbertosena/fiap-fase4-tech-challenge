"""Testes da normalização do relógio embutido nos UUIDs v7 do Evidently."""

from __future__ import annotations

import re
import uuid
from datetime import UTC, datetime

import pytest

from credit_sentinel.drift.evidently_runner import shift_uuid7_times

pytestmark = pytest.mark.unit

TARGET = datetime(2026, 10, 20, 21, 20, tzinfo=UTC)
TARGET_MS = int(TARGET.timestamp() * 1000)
UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}")


def _uuid7(ms: int, tail: str) -> str:
    stamp = f"{ms:012x}"
    return f"{stamp[:8]}-{stamp[8:]}-7{tail[:3]}-{tail[3:7]}-{tail[7:19]}"


def _time_of(identifier: str) -> int:
    return int(identifier.replace("-", "")[:12], 16)


def test_first_identifier_moves_to_the_target_and_order_is_preserved() -> None:
    base = 1_794_000_000_000
    ids = [_uuid7(base, "abc12340123456789ab"), _uuid7(base + 5, "def5678ba9876543210")]
    text = f'{{"id": "{ids[0]}", "other": "{ids[1]}"}}'
    shifted = UUID_RE.findall(shift_uuid7_times(text, TARGET))
    assert _time_of(shifted[0]) == TARGET_MS
    assert _time_of(shifted[1]) == TARGET_MS + 5


def test_everything_but_the_timestamp_is_kept() -> None:
    original = _uuid7(1_794_000_000_000, "abc12340123456789ab")
    shifted = UUID_RE.findall(shift_uuid7_times(original, TARGET))[0]
    assert shifted.replace("-", "")[12:] == original.replace("-", "")[12:]


def test_other_uuid_versions_and_plain_text_are_untouched() -> None:
    random_id = str(uuid.UUID(int=0x1234_5678_9ABC_4DEF_8123_4567_89AB_CDEF))
    text = f"id={random_id} sem horario, valor 1794000000000"
    assert shift_uuid7_times(text, TARGET) == text


def test_text_without_identifiers_is_returned_as_is() -> None:
    assert shift_uuid7_times("nada a ajustar", TARGET) == "nada a ajustar"
