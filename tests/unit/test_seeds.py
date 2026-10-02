"""Testes das sementes determinísticas."""

from __future__ import annotations

import random

import numpy as np
import pytest

from credit_sentinel.utils.seeds import DEFAULT_SEED, make_rng, set_global_seed

pytestmark = pytest.mark.unit


def test_same_seed_produces_same_sequence() -> None:
    first = make_rng(123).normal(size=5)
    second = make_rng(123).normal(size=5)
    np.testing.assert_array_equal(first, second)


def test_different_seeds_produce_different_sequences() -> None:
    assert not np.array_equal(make_rng(1).normal(size=5), make_rng(2).normal(size=5))


def test_none_falls_back_to_default_seed() -> None:
    assert DEFAULT_SEED == 42
    np.testing.assert_array_equal(make_rng().random(3), make_rng(42).random(3))


def test_set_global_seed_fixes_stdlib_random() -> None:
    set_global_seed(5)
    first = random.random()  # noqa: S311 - aleatoriedade não criptográfica
    set_global_seed(5)
    assert random.random() == first  # noqa: S311
