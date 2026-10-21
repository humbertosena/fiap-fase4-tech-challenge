"""Renderização das páginas didáticas em HTML único e autocontido.

CSS e JS do pacote são embutidos; o payload vai em
``<script type="application/json" id="payload">``. A única dependência externa é o
Chart.js no cdnjs com ``integrity``; sem ele, cada gráfico mostra sua tabela.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, FileSystemLoader, StrictUndefined, select_autoescape

from credit_sentinel.explainers.payloads import (
    RELEASES,
    IndexPayload,
    ReleaseInfo,
    build_payload,
    release_info,
)

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pydantic import BaseModel

    from credit_sentinel.settings import Settings

PACKAGE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = PACKAGE_DIR / "templates"
STATIC_DIR = PACKAGE_DIR / "static"
CHART_JS_SRC = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"
_SRI_HEAD = "sha512-CQBWl4fJHWbryGE+Pc7UAxWMUMNMWzWxF4SQo9CgkJIN1kx6d"
CHART_JS_INTEGRITY = _SRI_HEAD + "jDQZjh3Y8SZ1d+6I+1zze6Z7kHXO7q3UyZAWw=="
MAX_PAGE_BYTES = 2_500_000
REPO_URL = "https://github.com/humbertosena/fiap-fase4-tech-challenge"
PAGES_URL = "https://humbertosena.github.io/fiap-fase4-tech-challenge/"
_COPIED_REPORT_SUFFIXES = (".html", ".json", ".md")


def br(value: float, digits: int = 2) -> str:
    """Filtro Jinja: decimal com vírgula (``0,829``)."""
    return f"{value:.{digits}f}".replace(".", ",")


def pct(value: float, digits: int = 1) -> str:
    """Filtro Jinja: fração como porcentagem pt-BR (``6,45 %``)."""
    return f"{br(100 * value, digits)} %"


def thousands(value: float) -> str:
    """Filtro Jinja: inteiro com separador de milhar pt-BR (``12.000``)."""
    return f"{round(value):,}".replace(",", ".")


def _environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        autoescape=select_autoescape(["html", "j2"]),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update({"br": br, "pct": pct, "thousands": thousands})
    return env


def _static(name: str) -> str:
    path = STATIC_DIR / name
    return path.read_text(encoding="utf-8") if path.exists() else ""


def embed_json(payload: BaseModel) -> str:
    """Serializa o payload para embutir com segurança em ``<script>``.

    Args:
        payload: Modelo pydantic da página.

    Returns:
        JSON com ``<``, ``>`` e ``&`` escapados em unicode.
    """
    text = json.dumps(payload.model_dump(mode="json"), ensure_ascii=False)
    return text.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def commit_sha(root: Path) -> str:
    """Hash curto do commit atual (``local`` fora de um repositório git)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],  # noqa: S607 - comando fixo
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "local"
    return result.stdout.strip() or "local"


def _write(html: str, out: Path) -> Path:
    size = len(html.encode("utf-8"))
    if size > MAX_PAGE_BYTES:
        msg = f"{out.name} tem {size} bytes; o limite é {MAX_PAGE_BYTES}"
        raise ValueError(msg)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def render(
    template: str,
    payload: BaseModel,
    release: ReleaseInfo,
    out: Path,
    published: frozenset[int] = frozenset(),
) -> Path:
    """Renderiza um template com CSS/JS inline e o payload embutido.

    Args:
        template: Nome do template Jinja (ex.: ``etapa1.html.j2``).
        payload: Dados da página.
        release: Release exibida no cabeçalho.
        out: Arquivo HTML de destino.
        published: Etapas com página publicada (para a navegação).

    Returns:
        O caminho gravado.

    Raises:
        ValueError: Se a página passar de ``MAX_PAGE_BYTES``.
    """
    page_script = _static(template.replace(".html.j2", ".js"))
    context = {
        "payload": payload,
        "payload_json": embed_json(payload),
        "release": release,
        "releases": RELEASES,
        "css": _static("explainer.css")
        + "\n"
        + _static(template.replace(".html.j2", ".css")),
        "js": _static("explainer.js") + "\n" + page_script,
        "chart_src": CHART_JS_SRC,
        "chart_integrity": CHART_JS_INTEGRITY,
        "repo_url": REPO_URL,
        "pages_url": PAGES_URL,
        "depth": "../" if out.parent.name == "releases" else "",
        "published": published,
        "current_etapa": None if template == "index.html.j2" else release.etapa,
    }
    html = _environment().get_template(template).render(**context)
    return _write(html, out)


def available_etapas() -> list[int]:
    """Etapas que já possuem template de página."""
    return [e for e in RELEASES if (TEMPLATES_DIR / f"etapa{e}.html.j2").exists()]


def _published(docs: Path, building: list[int]) -> frozenset[int]:
    existing = {
        e
        for e, info in RELEASES.items()
        if (docs / "releases" / f"{info.slug}.html").exists()
    }
    return frozenset(existing | set(building))


def _index_payload(published: frozenset[int], sha: str) -> IndexPayload:
    releases = []
    for etapa, info in RELEASES.items():
        status = "publicada" if etapa in published else "em breve"
        releases.append(info.model_copy(update={"status": status, "commit_sha": sha}))
    return IndexPayload(releases=releases)


def _copy_reports(reports: Path, target: Path) -> None:
    if not reports.exists():
        return
    for path in reports.rglob("*"):
        if path.is_file() and path.suffix in _COPIED_REPORT_SUFFIXES:
            destination = target / path.relative_to(reports)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def build_all(
    *,
    settings: Settings,
    etapas: Iterable[int] | None = None,
    out_dir: Path | None = None,
    overrides: dict[str, str] | None = None,
) -> list[Path]:
    """Gera as páginas das etapas, o índice, ``.nojekyll`` e a cópia de ``reports/``.

    Args:
        settings: Configuração do projeto.
        etapas: Etapas a gerar (padrão: todas com template).
        out_dir: Diretório de saída (padrão: ``docs/``).
        overrides: ``tag``/``date`` alternativos (apenas com uma etapa).

    Returns:
        Caminhos das páginas geradas.
    """
    docs = out_dir or settings.docs_dir
    sha = commit_sha(settings.project_root)
    building = list(etapas or available_etapas())
    published = _published(docs, building)
    pages = []
    for etapa in building:
        info = release_info(etapa, commit_sha=sha, **(overrides or {}))
        payload = build_payload(etapa, settings=settings, release=info)
        out = docs / "releases" / f"{info.slug}.html"
        pages.append(render(f"etapa{etapa}.html.j2", payload, info, out, published))
    index = _index_payload(published, sha)
    home = release_info(1, commit_sha=sha)
    pages.append(render("index.html.j2", index, home, docs / "index.html", published))
    (docs / ".nojekyll").touch()
    _copy_reports(settings.reports_dir, docs / "reports")
    return pages
