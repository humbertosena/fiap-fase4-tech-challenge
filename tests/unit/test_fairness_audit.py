"""Auditoria de fairness por subgrupos: métricas, regra dos 4/5 e relatório."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from credit_sentinel.governance.fairness_audit import (
    FAIRNESS_ATTRIBUTES,
    AuditSubject,
    FairnessAudit,
    FairnessReport,
    GroupMetrics,
    age_band,
    audit_fairness,
    audit_subject,
    build_audit,
    render_audit_markdown,
    to_group_impacts,
)

if TYPE_CHECKING:
    import numpy as np

pytestmark = pytest.mark.unit

THRESHOLD = 0.5
DENIED, APPROVED = 0.9, 0.1


def _subject(**overrides: object) -> AuditSubject:
    values = {
        "model_version": "m-1.0.0",
        "role": "champion",
        "dataset": "teste",
        "threshold": THRESHOLD,
    }
    return AuditSubject(**{**values, **overrides})


def _cohort(group: str, *, good: tuple[int, int], bad: tuple[int, int]) -> pd.DataFrame:
    """Um grupo de ``genero``: ``(aprovados, negados)`` entre bons e maus pagadores."""
    rows = []
    for y, (approved, denied) in ((0, good), (1, bad)):
        rows += [(group, y, APPROVED)] * approved + [(group, y, DENIED)] * denied
    frame = pd.DataFrame(rows, columns=["genero", "inadimplente", "score"])
    return frame.assign(regiao="Sul", idade=35)


def _controlled() -> tuple[pd.DataFrame, np.ndarray]:
    """F (100): aprovação 0,64, TPR 0,75, FPR 0,10; M (50): 0,56, 0,50, 0,40."""
    female = _cohort("F", good=(54, 6), bad=(10, 30))
    male = _cohort("M", good=(18, 12), bad=(10, 10))
    frame = pd.concat([female, male], ignore_index=True)
    return frame, frame["score"].to_numpy()


def _by_group(report: FairnessReport) -> dict[str, GroupMetrics]:
    return {g.group: g for g in report.groups}


def test_group_metrics_match_hand_computed_values() -> None:
    frame, scores = _controlled()
    report = audit_fairness(frame, scores, _subject(), attribute="genero")
    female, male = _by_group(report)["F"], _by_group(report)["M"]
    assert (female.n, male.n) == (100, 50)
    assert female.approval_rate == pytest.approx(0.64)
    assert female.tpr == pytest.approx(0.75)
    assert female.fpr == pytest.approx(0.10)
    assert female.observed_default_rate == pytest.approx(0.40)
    assert male.approval_rate == pytest.approx(0.56)
    assert male.tpr == pytest.approx(0.50)
    assert male.fpr == pytest.approx(0.40)
    assert male.observed_default_rate == pytest.approx(0.40)


def test_disparate_impact_and_equal_opportunity_use_the_largest_group() -> None:
    frame, scores = _controlled()
    report = audit_fairness(frame, scores, _subject(), attribute="genero")
    female, male = _by_group(report)["F"], _by_group(report)["M"]
    assert report.reference_group == "F"
    assert female.disparate_impact == 1.0
    assert female.equal_opportunity_diff == 0.0
    assert male.disparate_impact == pytest.approx(0.56 / 0.64)
    assert male.equal_opportunity_diff == pytest.approx(0.50 - 0.75)
    assert report.passed is True


def test_report_carries_the_subject_and_the_cohort_size() -> None:
    frame, scores = _controlled()
    subject = _subject(model_version="x-2", role="challenger", dataset="crise")
    report = audit_fairness(frame, scores, subject, attribute="genero")
    assert (report.model_version, report.role, report.dataset) == (
        "x-2",
        "challenger",
        "crise",
    )
    assert (report.attribute, report.threshold, report.n) == ("genero", THRESHOLD, 150)


def test_threshold_decides_who_is_approved() -> None:
    frame, scores = _controlled()
    lenient = _subject(threshold=0.95)
    report = audit_fairness(frame, scores, lenient, attribute="genero")
    assert all(g.approval_rate == 1.0 for g in report.groups)
    on_the_line = _subject(threshold=APPROVED)  # score == limiar nega (como o pred)
    report = audit_fairness(frame, scores, on_the_line, attribute="genero")
    assert all(g.approval_rate == 0.0 for g in report.groups)


@pytest.mark.parametrize(
    ("reference_approved", "group_approved", "within"),
    [
        (80, 32, True),  # razão 0,80 exata: limite inferior é inclusivo
        (80, 31, False),  # 0,775
        (80, 50, True),  # razão 1,25 exata: limite superior é inclusivo
        (60, 50, False),  # 1,67
    ],
)
def test_four_fifths_band_is_inclusive_at_both_edges(
    reference_approved: int, group_approved: int, *, within: bool
) -> None:
    big = _cohort("A", good=(reference_approved, 100 - reference_approved), bad=(0, 0))
    small = _cohort("B", good=(group_approved, 50 - group_approved), bad=(0, 0))
    frame = pd.concat([big, small], ignore_index=True)
    report = audit_fairness(
        frame, frame["score"].to_numpy(), _subject(), attribute="genero"
    )
    assert _by_group(report)["B"].within_four_fifths is within
    assert report.passed is within


def test_one_group_outside_the_band_fails_the_whole_attribute() -> None:
    big = _cohort("A", good=(90, 10), bad=(0, 0))
    mid = _cohort("B", good=(40, 10), bad=(0, 0))
    low = _cohort("C", good=(20, 30), bad=(0, 0))
    frame = pd.concat([big, mid, low], ignore_index=True)
    report = audit_fairness(
        frame, frame["score"].to_numpy(), _subject(), attribute="genero"
    )
    flags = {g.group: g.within_four_fifths for g in report.groups}
    assert flags == {"A": True, "B": True, "C": False}
    assert report.passed is False


def test_rates_are_none_when_a_class_is_absent() -> None:
    frame = pd.concat(
        [
            _cohort("A", good=(40, 10), bad=(5, 15)),
            _cohort("B", good=(10, 5), bad=(0, 0)),
            _cohort("C", good=(0, 0), bad=(2, 8)),
        ],
        ignore_index=True,
    )
    report = audit_fairness(
        frame, frame["score"].to_numpy(), _subject(), attribute="genero"
    )
    groups = _by_group(report)
    assert groups["B"].tpr is None
    assert groups["B"].equal_opportunity_diff is None
    assert groups["C"].fpr is None
    assert groups["C"].tpr == pytest.approx(0.8)
    assert groups["A"].tpr == pytest.approx(0.75)


def test_reference_without_approvals_keeps_the_ratio_neutral() -> None:
    big = _cohort("A", good=(0, 60), bad=(0, 40))
    small = _cohort("B", good=(10, 20), bad=(0, 0))
    frame = pd.concat([big, small], ignore_index=True)
    report = audit_fairness(
        frame, frame["score"].to_numpy(), _subject(), attribute="genero"
    )
    assert _by_group(report)["B"].disparate_impact == 1.0


def test_tied_groups_pick_a_deterministic_reference() -> None:
    frame = pd.concat(
        [_cohort("Z", good=(8, 2), bad=(0, 0)), _cohort("A", good=(5, 5), bad=(0, 0))],
        ignore_index=True,
    )
    report = audit_fairness(
        frame, frame["score"].to_numpy(), _subject(), attribute="genero"
    )
    assert report.reference_group == "A"


def test_age_band_uses_four_bands() -> None:
    bands = age_band(pd.Series([18, 29, 30, 44, 45, 59, 60, 85]))
    assert list(bands) == ["18-29"] * 2 + ["30-44"] * 2 + ["45-59"] * 2 + ["60+"] * 2


def test_age_attribute_audits_the_bands_not_the_raw_age() -> None:
    ages = [25] * 30 + [40] * 50 + [50] * 40 + [70] * 20
    frame = pd.DataFrame(
        {
            "genero": "F",
            "regiao": "Sul",
            "idade": ages,
            "inadimplente": 0,
            "score": APPROVED,
        }
    )
    report = audit_fairness(
        frame, frame["score"].to_numpy(), _subject(), attribute="faixa_etaria"
    )
    assert [g.group for g in report.groups] == ["18-29", "30-44", "45-59", "60+"]
    assert report.reference_group == "30-44"
    assert [g.n for g in report.groups] == [30, 50, 40, 20]


def test_unknown_attribute_is_rejected() -> None:
    frame, scores = _controlled()
    with pytest.raises(ValueError, match="atributo"):
        audit_fairness(frame, scores, _subject(), attribute="raca")


def test_audit_subject_covers_the_three_protected_attributes() -> None:
    frame, scores = _controlled()
    reports = audit_subject(frame, scores, _subject())
    assert tuple(r.attribute for r in reports) == FAIRNESS_ATTRIBUTES
    assert {r.n for r in reports} == {150}
    assert all(r.model_version == "m-1.0.0" for r in reports)


def test_group_impacts_are_the_compact_view_used_by_the_gate() -> None:
    frame, scores = _controlled()
    reports = audit_subject(frame, scores, _subject())
    impacts = {(i.attribute, i.group): i for i in to_group_impacts(reports)}
    male = impacts[("genero", "M")]
    assert male.n == 50
    assert male.reference_group == "F"
    assert male.approval_rate == pytest.approx(0.56)
    assert male.ratio == pytest.approx(0.56 / 0.64)
    assert len(impacts) == sum(len(r.groups) for r in reports)


def _audit(frame: pd.DataFrame, scores: np.ndarray) -> FairnessAudit:
    reports = audit_subject(frame, scores, _subject())
    stamp = datetime(2026, 11, 1, 14, 0, tzinfo=UTC)
    return build_audit(reports, {"teste": "dados controlados"}, generated_at=stamp)


def test_audit_passes_only_if_every_report_passes() -> None:
    frame, scores = _controlled()
    assert _audit(frame, scores).passed is True
    big = _cohort("A", good=(90, 10), bad=(0, 0))
    low = _cohort("C", good=(20, 30), bad=(0, 0))
    skewed = pd.concat([big, low], ignore_index=True)
    assert _audit(skewed, skewed["score"].to_numpy()).passed is False


def test_audit_round_trips_through_json() -> None:
    frame, scores = _controlled()
    audit = _audit(frame, scores)
    restored = FairnessAudit.model_validate_json(audit.model_dump_json())
    assert restored == audit
    assert (restored.lower, restored.upper) == (0.80, 1.25)
    assert restored.datasets == {"teste": "dados controlados"}


def test_markdown_lists_every_group_and_marks_the_failures() -> None:
    big = _cohort("A", good=(90, 10), bad=(0, 0))
    low = _cohort("C", good=(20, 30), bad=(0, 0))
    skewed = pd.concat([big, low], ignore_index=True)
    text = render_audit_markdown(_audit(skewed, skewed["score"].to_numpy()))
    assert text.startswith("# Auditoria de fairness")
    assert "## Resumo" in text
    assert "m-1.0.0" in text
    assert "| `genero` | `C` |" in text
    assert "fora de 4/5" in text
    assert "0,444" in text  # razão de C: 0,40 / 0,90, com vírgula decimal


def test_markdown_for_a_clean_audit_has_no_failure_marker() -> None:
    frame, scores = _controlled()
    text = render_audit_markdown(_audit(frame, scores))
    assert "fora de 4/5" not in text
    assert "dados controlados" in text
