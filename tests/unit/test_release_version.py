"""A versão do pacote, a tag gravada no MLflow e o catálogo de releases andam juntos."""

from __future__ import annotations

import pytest

from credit_sentinel import __version__
from credit_sentinel.explainers.payloads import RELEASES
from credit_sentinel.tracking.mlflow_tracking import CURRENT_RELEASE

pytestmark = pytest.mark.unit


def test_the_mlflow_release_tag_is_the_latest_release() -> None:
    assert RELEASES[max(RELEASES)].tag == CURRENT_RELEASE


def test_the_package_version_matches_the_release_tag() -> None:
    assert CURRENT_RELEASE.startswith(f"v{__version__}")
