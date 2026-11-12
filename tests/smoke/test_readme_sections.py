"""O README traz as seções exigidas, na ordem, e todos os links locais existem."""

from __future__ import annotations

import re

import pytest

from credit_sentinel.settings import Settings

pytestmark = pytest.mark.smoke

ROOT = Settings(_env_file=None).project_root
README = (ROOT / "README.md").read_text("utf-8")
SECTIONS = [
    "1. Resumo executivo",
    "2. Arquitetura",
    "3. Guia de reprodução",
    "4. Etapa 1",
    "5. Etapa 2",
    "6. Etapa 3",
    "7. Governança e LGPD",
    "8. Análise de inferência causal",
    "9. Runbook de incidentes",
    "10. Roteiro do vídeo STAR",
    "11. Decisões, estrutura e rastreabilidade",
    "12. Agradecimentos e transparência",
]
SUBSECTIONS = [
    *(f"7.{n}" for n in range(1, 6)),
    *(f"8.{n}" for n in range(1, 7)),
    *(f"11.{n}" for n in range(1, 6)),
]
LINK = re.compile(r"\]\(([^)\s]+)\)")


def _headings(level: int) -> list[str]:
    prefix = "#" * level + " "
    return [
        line.removeprefix(prefix)
        for line in README.splitlines()
        if line.startswith(prefix)
    ]


def test_sections_exist_in_the_required_order() -> None:
    titles = _headings(2)
    positions = []
    for wanted in SECTIONS:
        match = [i for i, title in enumerate(titles) if title.startswith(wanted)]
        assert match, f"falta a seção {wanted!r}"
        positions.append(match[0])
    assert positions == sorted(positions)


def test_governance_subsections_exist() -> None:
    titles = _headings(3)
    for number in SUBSECTIONS:
        assert any(t.startswith(f"{number} ") for t in titles), number


def test_the_video_link_is_highlighted_at_the_top() -> None:
    assert "**Vídeo STAR" in "\n".join(README.splitlines()[:12])


def test_every_release_page_is_linked_and_exists() -> None:
    pages = sorted((ROOT / "docs" / "releases").glob("etapa-*.html"))
    assert len(pages) == 4
    for page in pages:
        assert f"docs/releases/{page.name}" in README, page.name


def test_every_local_link_points_to_an_existing_file() -> None:
    broken = []
    for target in LINK.findall(README):
        if target.startswith(("http", "#", "mailto:")):
            continue
        path = target.split("#", 1)[0]
        if path and not (ROOT / path).exists():
            broken.append(target)
    assert broken == []


def test_the_ai_assistance_is_acknowledged() -> None:
    section = README.split("## 12. ", 1)[1]
    assert "Claude Code" in section
    assert "Anthropic" in section


def test_architecture_and_causal_diagrams_are_present() -> None:
    assert README.count("```mermaid") >= 2
