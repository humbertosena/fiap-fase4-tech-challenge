"""Testes da atualização automática do model card."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from credit_sentinel.ml.model_card import (
    END_MARKER,
    START_MARKER,
    update_model_card,
)

if TYPE_CHECKING:
    from pathlib import Path

pytestmark = pytest.mark.unit


def test_only_the_marked_section_is_replaced(tmp_path: Path) -> None:
    card = tmp_path / "model_card.md"
    card.write_text(f"# Card\n\nantes\n{START_MARKER}\nvelho\n{END_MARKER}\ndepois\n")
    update_model_card(card, "| métrica | valor |")
    text = card.read_text()
    assert "velho" not in text
    assert "| métrica | valor |" in text
    assert text.startswith("# Card\n\nantes\n")
    assert text.endswith("depois\n")


def test_missing_markers_raise_clear_error(tmp_path: Path) -> None:
    card = tmp_path / "model_card.md"
    card.write_text("# Card sem marcadores\n")
    with pytest.raises(ValueError, match="marcadores"):
        update_model_card(card, "x")
