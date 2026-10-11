"""Configuração centralizada do projeto via pydantic-settings (prefixo ``CS_``)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Parâmetros de execução lidos do ambiente e do arquivo ``.env``.

    Todos os caminhos do projeto derivam de ``project_root``; nenhum módulo
    deve montar caminhos literais por conta própria.

    Attributes:
        seed: Semente global usada por ``make_rng``.
        pii_salt: Salt da pseudonimização HMAC-SHA-256 (nunca exibido em ``repr``).
        mlflow_tracking_uri: URI do backend de tracking do MLflow.
        project_root: Raiz do repositório.
    """

    model_config = SettingsConfigDict(
        env_prefix="CS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    seed: int = 42
    pii_salt: SecretStr = SecretStr("dev-salt-nao-usar-em-producao")
    mlflow_tracking_uri: str = "sqlite:///mlruns/mlflow.db"
    project_root: Path = PROJECT_ROOT

    @property
    def data_dir(self) -> Path:
        """Diretório raiz dos datasets."""
        return self.project_root / "data"

    @property
    def reference_dir(self) -> Path:
        """Dataset de referência (treino), imutável após a v0.1.0-alpha."""
        return self.data_dir / "reference"

    @property
    def production_dir(self) -> Path:
        """Janelas de produção simuladas (onde o drift ocorre)."""
        return self.data_dir / "production"

    @property
    def corrupted_dir(self) -> Path:
        """Lotes corrompidos usados para provar o circuit breaker."""
        return self.data_dir / "corrupted"

    @property
    def quarantine_dir(self) -> Path:
        """Linhas rejeitadas pelo contrato, por ``run_id`` (fora do git)."""
        return self.data_dir / "quarantine"

    @property
    def models_dir(self) -> Path:
        """Bundles de modelos serializados."""
        return self.project_root / "models"

    @property
    def reports_dir(self) -> Path:
        """Relatórios versionados (validação, drift, SLO, fairness)."""
        return self.project_root / "reports"

    @property
    def logs_dir(self) -> Path:
        """Logs JSON Lines das execuções (fora do git)."""
        return self.project_root / "logs"

    @property
    def config_dir(self) -> Path:
        """Arquivos de configuração declarativa (contrato, SLOs, cenários)."""
        return self.project_root / "config"

    @property
    def docs_dir(self) -> Path:
        """Documentação, ADRs, evidências e páginas didáticas."""
        return self.project_root / "docs"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Retorna a instância única de ``Settings`` do processo.

    Returns:
        Settings carregadas do ambiente na primeira chamada.
    """
    return Settings()
