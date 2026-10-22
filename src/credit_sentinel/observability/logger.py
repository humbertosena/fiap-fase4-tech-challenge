"""Logger JSON Lines do pipeline, com PII mascarada antes de qualquer saída.

Cada evento é uma linha JSON gravada em ``logs/pipeline_events.jsonl`` (append) e
espelhada no stderr. ``PiiRedactingFilter`` roda **antes** dos dois destinos, então
nenhum ``CLI-xxxxxxxxxx``, CPF ou e-mail sai em claro. ``StageTimer`` mede cada
estágio e traduz o desfecho em ``SUCCESS``, ``FAILED`` ou ``BLOCKED``.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field

from credit_sentinel.contracts.privacy_sanitizer import MASK, mask_record, redact_text
from credit_sentinel.contracts.validator import (
    CircuitBreakerOpenError,
    ContractViolationError,
)
from credit_sentinel.utils.clock import utc_now

if TYPE_CHECKING:
    from pathlib import Path
    from types import TracebackType

Stage = Literal[
    "ingestion",
    "contract_validation",
    "inference",
    "drift_detection",
    "slo_evaluation",
    "tracking",
]
Status = Literal["STARTED", "SUCCESS", "FAILED", "BLOCKED", "SKIPPED"]
REQUIRED_FIELDS = (
    "timestamp",
    "run_id",
    "stage",
    "status",
    "records_processed",
    "rejected_records",
    "latency_ms",
    "model_version",
)
BLOCKING_ERRORS = (ContractViolationError, CircuitBreakerOpenError)
_MS = 1000


class PipelineEvent(BaseModel):
    """Um evento do pipeline (uma linha do JSONL).

    Attributes:
        timestamp: Instante UTC do evento.
        run_id: Identificador da execução do pipeline.
        stage: Estágio que emitiu o evento.
        status: Situação do estágio.
        records_processed: Registros tratados pelo estágio.
        rejected_records: Registros rejeitados pelo contrato.
        latency_ms: Duração do estágio em milissegundos.
        model_version: Versão do modelo em uso.
        window: Janela de dados (opcional).
        message: Texto livre, já mascarado (opcional).
        details: Dados adicionais, já mascarados.
    """

    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    run_id: str
    stage: Stage
    status: Status
    records_processed: int
    rejected_records: int
    latency_ms: int
    model_version: str
    window: str | None = None
    message: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


def _redact(value: Any) -> Any:  # noqa: ANN401 - estrutura JSON arbitrária
    if isinstance(value, dict):
        masked = mask_record(value)
        return {
            key: _redact(value[key]) if masked[key] != MASK else MASK for key in masked
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return redact_text(value) if isinstance(value, str) else value


class PiiRedactingFilter(logging.Filter):
    """Mascara ``message`` e ``details`` do evento anexado ao registro."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Substitui o evento do registro por uma cópia sem dados pessoais.

        Args:
            record: Registro com o atributo ``event`` (um ``PipelineEvent``).

        Returns:
            Sempre ``True``: o evento é mascarado, nunca descartado.
        """
        event: PipelineEvent = record.event  # type: ignore[attr-defined]
        message = None if event.message is None else redact_text(event.message)
        record.event = event.model_copy(  # type: ignore[attr-defined]
            update={"message": message, "details": _redact(event.details)}
        )
        return True


class _JsonLinesFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        event: PipelineEvent = record.event  # type: ignore[attr-defined]
        data = event.model_dump(mode="json", exclude_none=True)
        if not data["details"]:
            del data["details"]
        return json.dumps(data, ensure_ascii=False)


class PipelineLogger:
    """Escreve eventos do pipeline em JSONL e no stderr."""

    def __init__(self, run_id: str, model_version: str, log_path: Path) -> None:
        """Abre o arquivo de log (append) e o espelho no stderr.

        Args:
            run_id: Identificador da execução (uuid4 por execução).
            model_version: Versão do modelo registrada em cada evento.
            log_path: Arquivo JSONL de destino.
        """
        self.run_id = run_id
        self.model_version = model_version
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._base = utc_now()
        self._started = time.perf_counter()
        self._logger = logging.getLogger(f"credit_sentinel.pipeline.{id(self)}")
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False
        self._logger.addFilter(PiiRedactingFilter())
        for handler in (
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stderr),
        ):
            handler.setFormatter(_JsonLinesFormatter())
            self._logger.addHandler(handler)

    def _now(self) -> datetime:
        return self._base + timedelta(seconds=time.perf_counter() - self._started)

    def emit(self, **fields: Any) -> None:  # noqa: ANN401 - campos opcionais do evento
        """Valida e grava um evento.

        Args:
            **fields: ``stage`` e ``status`` obrigatórios; opcionais:
                ``records_processed``, ``rejected_records``, ``latency_ms``,
                ``window``, ``message`` e ``details``.
        """
        counters = {"records_processed": 0, "rejected_records": 0, "latency_ms": 0}
        event = PipelineEvent(
            timestamp=self._now(),
            run_id=self.run_id,
            model_version=self.model_version,
            **(counters | fields),
        )
        self._logger.info("pipeline_event", extra={"event": event})

    def close(self) -> None:
        """Fecha os destinos; pode ser chamado mais de uma vez."""
        for handler in list(self._logger.handlers):
            handler.close()
            self._logger.removeHandler(handler)

    def __enter__(self) -> Self:
        """Permite usar o logger como gerenciador de contexto."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Fecha o logger ao sair do contexto."""
        self.close()


def get_pipeline_logger(
    run_id: str, model_version: str, log_path: Path
) -> PipelineLogger:
    """Cria o logger de uma execução do pipeline.

    Args:
        run_id: Identificador da execução.
        model_version: Versão do modelo avaliado.
        log_path: Arquivo JSONL (criado com os diretórios, em modo append).

    Returns:
        Logger pronto para ``emit``.
    """
    return PipelineLogger(run_id, model_version, log_path)


def read_events(path: Path) -> list[PipelineEvent]:
    """Lê todos os eventos de um arquivo JSONL.

    Args:
        path: Arquivo gravado por ``PipelineLogger``.

    Returns:
        Eventos validados, na ordem do arquivo.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    return [PipelineEvent.model_validate_json(line) for line in lines if line.strip()]


class StageTimer:
    """Mede um estágio e emite ``STARTED`` e o desfecho.

    O desfecho é ``SUCCESS``, ``FAILED`` ou ``BLOCKED`` (erros do contrato).

    Dentro do bloco, o código informa ``records_processed``, ``rejected_records``,
    ``message`` e ``details``; exceções propagam depois de registradas.
    """

    def __init__(
        self, logger: PipelineLogger, stage: Stage, window: str | None = None
    ) -> None:
        """Prepara o cronômetro de um estágio.

        Args:
            logger: Destino dos eventos.
            stage: Estágio medido.
            window: Janela de dados em processamento.
        """
        self._logger = logger
        self.stage: Stage = stage
        self.window = window
        self.records_processed = 0
        self.rejected_records = 0
        self.latency_ms = 0
        self.message: str | None = None
        self.details: dict[str, Any] = {}
        self._started = 0.0

    def _emit(self, status: Status, message: str | None) -> None:
        self._logger.emit(
            stage=self.stage,
            status=status,
            window=self.window,
            records_processed=self.records_processed,
            rejected_records=self.rejected_records,
            latency_ms=self.latency_ms,
            message=message,
            details=self.details,
        )

    def __enter__(self) -> Self:
        """Emite ``STARTED`` e dispara o cronômetro."""
        self._started = time.perf_counter()
        self._emit("STARTED", None)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Emite o desfecho; a exceção, se houver, segue para quem chamou."""
        del exc_type, tb
        self.latency_ms = round((time.perf_counter() - self._started) * _MS)
        if exc is None:
            self._emit("SUCCESS", self.message)
        elif isinstance(exc, BLOCKING_ERRORS):
            self._emit("BLOCKED", str(exc))
        else:
            self._emit("FAILED", f"{type(exc).__name__}: {exc}")

    def skip(self, reason: str) -> None:
        """Registra o estágio como ``SKIPPED`` (sem ``STARTED``).

        Args:
            reason: Motivo do salto (ex.: lote bloqueado antes).
        """
        self._emit("SKIPPED", reason)
