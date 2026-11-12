"""Roteiro do vídeo STAR: cronometragem coerente e artefatos citados existem."""

from __future__ import annotations

import itertools
import re
from typing import NamedTuple

import pytest

from credit_sentinel.cli import build_parser
from credit_sentinel.settings import Settings

pytestmark = pytest.mark.smoke

ROOT = Settings(_env_file=None).project_root
ROTEIRO = ROOT / "docs" / "video_star_roteiro.md"
LIMITE_SEGUNDOS = 300
CLOCK = re.compile(r"^(\d+):([0-5]\d)$")
PATH_IN_BACKTICKS = re.compile(r"`([^`\s]+)`")
MAKE_TARGET = re.compile(r"\bmake ([a-z-]+)")
CLI_COMMAND = re.compile(r"\bcredit-sentinel ([a-z-]+)")


class Bloco(NamedTuple):
    """Uma linha do cronograma do roteiro."""

    nome: str
    inicio: int
    fim: int
    duracao: int
    artefatos: list[str]


def _segundos(texto: str) -> int:
    match = CLOCK.match(texto.strip())
    assert match, f"horário inválido: {texto!r}"
    return int(match[1]) * 60 + int(match[2])


def _cronograma() -> list[Bloco]:
    """Lê a tabela do cronograma: Bloco | Início | Fim | Duração | Tela | Artefato."""
    texto = ROTEIRO.read_text("utf-8")
    secao = texto.split("## Cronograma", 1)[1].split("\n## ", 1)[0]
    blocos = []
    for linha in secao.splitlines():
        celulas = [c.strip() for c in linha.strip().strip("|").split("|")]
        if len(celulas) == 6 and CLOCK.match(celulas[1]):
            nome, inicio, fim, duracao, _tela, artefato = celulas
            blocos.append(
                Bloco(
                    nome=nome,
                    inicio=_segundos(inicio),
                    fim=_segundos(fim),
                    duracao=int(duracao.removesuffix(" s")),
                    artefatos=PATH_IN_BACKTICKS.findall(artefato),
                )
            )
    return blocos


def test_roteiro_existe() -> None:
    assert ROTEIRO.is_file()


def test_blocos_sao_contiguos_e_cabem_em_cinco_minutos() -> None:
    blocos = _cronograma()
    assert blocos, "cronograma vazio"
    assert blocos[0].inicio == 0
    for anterior, atual in itertools.pairwise(blocos):
        assert atual.inicio == anterior.fim, f"lacuna ou sobreposição em {atual.nome}"
    for bloco in blocos:
        assert bloco.fim - bloco.inicio == bloco.duracao, bloco.nome
    assert sum(b.duracao for b in blocos) == blocos[-1].fim <= LIMITE_SEGUNDOS


def test_os_quatro_blocos_star_aparecem_em_ordem() -> None:
    nomes = " ".join(b.nome for b in _cronograma())
    posicoes = [
        nomes.index(parte) for parte in ("Situation", "Task", "Action", "Result")
    ]
    assert posicoes == sorted(posicoes)


def test_todo_bloco_cita_um_artefato_que_existe() -> None:
    for bloco in _cronograma():
        assert bloco.artefatos, f"{bloco.nome} não cita artefato"
        for artefato in bloco.artefatos:
            if artefato.startswith("http"):
                continue
            caminho = artefato.split("#", 1)[0]
            assert (ROOT / caminho).exists(), f"{bloco.nome}: {artefato} não existe"


def test_comandos_citados_existem() -> None:
    texto = ROTEIRO.read_text("utf-8")
    makefile = (ROOT / "Makefile").read_text("utf-8")
    alvos = set(re.findall(r"^([a-z-]+):", makefile, re.MULTILINE))
    assert set(MAKE_TARGET.findall(texto)) <= alvos
    ajuda = build_parser().format_help()
    subcomandos = set(re.findall(r"^\s{4}([a-z-]+)\s", ajuda, re.MULTILINE))
    assert subcomandos, "não consegui ler os subcomandos da CLI"
    assert set(CLI_COMMAND.findall(texto)) <= subcomandos
