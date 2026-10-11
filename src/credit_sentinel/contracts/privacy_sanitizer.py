"""Sanitização de dados pessoais (Privacy by Design e by Default).

- ``pseudonymize_ids``: troca ``id_cliente`` por HMAC-SHA-256 com salt secreto.
- ``drop_direct_identifiers``: remove colunas da deny-list (CPF, nome, e-mail...).
- ``mask_record`` / ``redact_text``: mascaram registros e textos de log.
- ``scan_for_pii`` / ``scan_paths``: varrem saídas textuais em busca de PII.
"""

from __future__ import annotations

import hashlib
import hmac
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import pandas as pd
from pydantic import BaseModel

from credit_sentinel.data import dictionary as dd

if TYPE_CHECKING:
    from collections.abc import Iterable

HASH_COLUMN = "id_cliente_hash"
MASK = "***"
DIRECT_IDENTIFIER_DENYLIST = ("cpf", "nome", "email", "telefone", "endereco", "rg")
SCANNED_SUFFIXES = (".json", ".jsonl", ".html", ".md", ".csv", ".txt", ".log")
_ALLOWED_EMAIL_DOMAINS = re.compile(r"(\.exemplo|example\.(com|org))$", re.IGNORECASE)
_DATA_URI = re.compile(r"data:[\w/+.-]+;base64,[A-Za-z0-9+/=]+")
PiiKind = Literal["cpf", "email", "telefone", "id_cliente"]
_PATTERNS: tuple[tuple[PiiKind, re.Pattern[str]], ...] = (
    ("cpf", re.compile(r"\b\d{3}\.\d{3}\.\d{3}-\d{2}\b")),
    ("cpf", re.compile(r"(?<![\w.])\d{11}(?!\w)(?!\.\d)")),
    ("email", re.compile(r"\b[\w.%+-]+@((?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})\b")),
    ("telefone", re.compile(r"(?:\+55\s?)?\(?\b\d{2}\)?\s?9?\d{4}-\d{4}\b")),
    ("id_cliente", re.compile(r"\bCLI-[0-9A-F]{10}\b")),
)


class PiiFinding(BaseModel):
    """Ocorrência de possível dado pessoal em uma saída textual.

    Attributes:
        kind: Tipo de dado pessoal.
        location: Arquivo e linha (ou rótulo informado).
        sample_masked: Trecho mascarado, seguro para exibir.
    """

    kind: PiiKind
    location: str
    sample_masked: str


def hash_identifier(value: str, salt: str) -> str:
    """Pseudonimiza um identificador com HMAC-SHA-256 (salt como chave).

    Args:
        value: Identificador em claro.
        salt: Segredo do ambiente (``CS_PII_SALT``).

    Returns:
        Digest hexadecimal de 64 caracteres.
    """
    return hmac.new(salt.encode(), value.encode(), hashlib.sha256).hexdigest()


def pseudonymize_ids(
    df: pd.DataFrame, *, salt: str, column: str = dd.ID_COLUMN
) -> pd.DataFrame:
    """Substitui a coluna de identificador por ``id_cliente_hash``.

    Args:
        df: DataFrame de entrada (não é alterado).
        salt: Segredo usado no HMAC.
        column: Coluna do identificador direto.

    Returns:
        Cópia com o hash na mesma posição da coluna original.
    """
    out = df.copy()
    position = out.columns.get_loc(column)
    original = out.pop(column)
    hashed = [None if pd.isna(v) else hash_identifier(str(v), salt) for v in original]
    out.insert(position, HASH_COLUMN, hashed)
    return out


def drop_direct_identifiers(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Remove colunas cujo nome está na deny-list de identificadores diretos.

    Args:
        df: DataFrame de entrada.

    Returns:
        DataFrame sem as colunas e a lista das colunas removidas.
    """
    dropped = [c for c in df.columns if str(c).lower() in DIRECT_IDENTIFIER_DENYLIST]
    return df.drop(columns=dropped), dropped


def redact_text(text: str) -> str:
    """Troca por ``***`` todo trecho que pareça dado pessoal.

    Args:
        text: Texto livre.

    Returns:
        Texto mascarado.
    """
    for _, pattern in _PATTERNS:
        text = pattern.sub(MASK, text)
    return text


def mask_record(record: dict[str, Any]) -> dict[str, Any]:
    """Mascara um registro antes de logar.

    Args:
        record: Registro chave → valor.

    Returns:
        Cópia com identificadores e PII textual mascarados.
    """
    sensitive = {*DIRECT_IDENTIFIER_DENYLIST, dd.ID_COLUMN}
    masked: dict[str, Any] = {}
    for key, value in record.items():
        if key.lower() in sensitive:
            masked[key] = MASK
        else:
            masked[key] = redact_text(value) if isinstance(value, str) else value
    return masked


def _mask_sample(value: str) -> str:
    keep = max(len(value) - 5, 0)
    return f"{value[:3]}{'*' * keep}{value[-2:]}"


def _is_allowed(kind: PiiKind, match: re.Match[str]) -> bool:
    return kind == "email" and bool(_ALLOWED_EMAIL_DOMAINS.search(match.group(1)))


def scan_for_pii(text: str, *, location: str = "") -> list[PiiFinding]:
    """Procura CPF, e-mail, telefone BR e ``id_cliente`` em claro.

    Blobs ``data:...;base64`` são ignorados; e-mails de domínios reservados
    para exemplos (``.exemplo``, ``example.com``) não contam.

    Args:
        text: Conteúdo a varrer.
        location: Rótulo (ex.: caminho do arquivo) usado nos achados.

    Returns:
        Achados ordenados por posição no texto.
    """
    clean = _DATA_URI.sub("", text)
    hits: list[tuple[int, PiiFinding]] = []
    for kind, pattern in _PATTERNS:
        for match in pattern.finditer(clean):
            if _is_allowed(kind, match):
                continue
            line = clean.count("\n", 0, match.start()) + 1
            where = f"{location}:{line}" if location else f"linha {line}"
            finding = PiiFinding(
                kind=kind, location=where, sample_masked=_mask_sample(match.group(0))
            )
            hits.append((match.start(), finding))
    return [finding for _, finding in sorted(hits, key=lambda hit: hit[0])]


def _expand(paths: Iterable[Path]) -> list[Path]:
    files: set[Path] = set()
    for path in map(Path, paths):
        candidates = path.rglob("*") if path.is_dir() else [path]
        files.update(
            p for p in candidates if p.is_file() and p.suffix in SCANNED_SUFFIXES
        )
    return sorted(files)


def scan_paths(paths: Iterable[Path]) -> list[PiiFinding]:
    """Varre arquivos textuais (recursivamente em diretórios).

    Args:
        paths: Arquivos ou diretórios.

    Returns:
        Todos os achados, com ``arquivo:linha`` em ``location``.
    """
    findings: list[PiiFinding] = []
    for file in _expand(paths):
        text = file.read_text(encoding="utf-8", errors="ignore")
        findings.extend(scan_for_pii(text, location=str(file)))
    return findings
