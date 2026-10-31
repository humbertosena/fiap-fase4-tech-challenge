"""Model Registry do MLflow: versões, aliases ``champion``/``challenger`` e promoção.

O baseline é a versão 1 com o alias ``champion``; o retreino registra a versão seguinte
com o alias ``challenger``. A troca de aliases (``promote_challenger``) só acontece se
o relatório de guardrails aprovou; a versão aposentada ganha a tag ``retired_reason``.
O ``MlflowClient`` recebe a URI explícita, sem estado global.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from pydantic import BaseModel

from credit_sentinel.ml.mlflow_baseline import BUNDLE_FOLDER, REGISTERED_MODEL
from credit_sentinel.ml.train_baseline import ModelBundle, load_model_bundle

if TYPE_CHECKING:
    from credit_sentinel.ml.evaluate_gates import GatesReport

MODEL_NAME = REGISTERED_MODEL
RETIRED_REASON = "drift_mes_3_crise"
Alias = Literal["champion", "challenger"]
_NOT_FOUND = "RESOURCE_DOES_NOT_EXIST"


class PromotionRefusedError(Exception):
    """A promoção do challenger foi recusada (gates reprovados ou sem candidato)."""


class VersionInfo(BaseModel):
    """Uma versão do modelo no registry.

    Attributes:
        version: Número da versão.
        run_id: Run de treino que a originou.
        aliases: Aliases que apontam para a versão.
        tags: Tags da versão (ex.: ``retired_reason``).
    """

    version: int
    run_id: str
    aliases: list[str]
    tags: dict[str, str]


class ModelRegistry:
    """Operações sobre o ``credit-scoring-model`` em um backend MLflow."""

    def __init__(self, tracking_uri: str) -> None:
        """Abre o cliente do registry.

        Args:
            tracking_uri: URI do backend (a mesma do tracking).
        """
        self.client = MlflowClient(tracking_uri=tracking_uri)

    def version_for_run(self, run_id: str) -> int:
        """Versão registrada para um run de treino.

        Args:
            run_id: Run que registrou o modelo.

        Returns:
            Número da versão mais recente criada por esse run.

        Raises:
            LookupError: Se o run não registrou nenhuma versão.
        """
        found = self.client.search_model_versions(
            f"name = '{MODEL_NAME}' and run_id = '{run_id}'"
        )
        if not found:
            msg = f"o run {run_id} não registrou versão de {MODEL_NAME}"
            raise LookupError(msg)
        return max(int(item.version) for item in found)

    def set_alias(self, alias: Alias, version: int) -> None:
        """Aponta um alias para uma versão.

        Args:
            alias: ``champion`` ou ``challenger``.
            version: Versão de destino.
        """
        self.client.set_registered_model_alias(MODEL_NAME, alias, str(version))

    def version_of(self, alias: Alias) -> int | None:
        """Versão para a qual o alias aponta.

        Args:
            alias: ``champion`` ou ``challenger``.

        Returns:
            Número da versão, ou ``None`` se o alias (ou o modelo) não existir.

        Raises:
            MlflowException: Para qualquer erro diferente de modelo inexistente.
        """
        try:
            model = self.client.get_registered_model(MODEL_NAME)
        except MlflowException as exc:
            if exc.error_code == _NOT_FOUND:
                return None
            raise
        version = model.aliases.get(alias)
        return None if version is None else int(version)

    def describe(self) -> list[VersionInfo]:
        """Lista as versões com seus aliases e tags.

        Returns:
            Versões em ordem crescente.
        """
        versions = self.client.search_model_versions(f"name = '{MODEL_NAME}'")
        owners: dict[int, list[str]] = {}
        if versions:
            for alias, version in self.client.get_registered_model(
                MODEL_NAME
            ).aliases.items():
                owners.setdefault(int(version), []).append(alias)
        return sorted(
            (
                VersionInfo(
                    version=int(v.version),
                    run_id=v.run_id,
                    aliases=sorted(owners.get(int(v.version), [])),
                    tags=dict(v.tags),
                )
                for v in versions
            ),
            key=lambda info: info.version,
        )

    def get_model_by_alias(self, alias: Alias, download_dir: Path) -> ModelBundle:
        """Baixa e carrega o bundle da versão apontada pelo alias.

        Args:
            alias: ``champion`` ou ``challenger``.
            download_dir: Pasta temporária para o download.

        Returns:
            Bundle pronto para ``predict_proba``.

        Raises:
            LookupError: Se o alias não apontar para nenhuma versão.
        """
        version = self.version_of(alias)
        if version is None:
            msg = f"o alias {alias} não aponta para nenhuma versão de {MODEL_NAME}"
            raise LookupError(msg)
        run_id = self.client.get_model_version(MODEL_NAME, str(version)).run_id
        download_dir.mkdir(parents=True, exist_ok=True)
        path = self.client.download_artifacts(run_id, BUNDLE_FOLDER, str(download_dir))
        return load_model_bundle(Path(path))

    def promote_challenger(self, *, gates: GatesReport) -> int:
        """Promove o challenger a champion, se os guardrails aprovaram.

        O alias ``champion`` passa para a versão do challenger, o alias
        ``challenger`` é removido e a versão antiga recebe ``retired_reason``.

        Args:
            gates: Relatório dos guardrails do challenger.

        Returns:
            Versão que virou champion.

        Raises:
            PromotionRefusedError: Guardrails reprovados ou nenhum challenger.
        """
        if not gates.passed:
            msg = "guardrails reprovados: o challenger não pode ser promovido"
            raise PromotionRefusedError(msg)
        challenger = self.version_of("challenger")
        if challenger is None:
            msg = "não há versão com o alias challenger para promover"
            raise PromotionRefusedError(msg)
        previous = self.version_of("champion")
        self.set_alias("champion", challenger)
        self.client.delete_registered_model_alias(MODEL_NAME, "challenger")
        if previous is not None and previous != challenger:
            self.client.set_model_version_tag(
                MODEL_NAME, str(previous), "retired_reason", RETIRED_REASON
            )
        return challenger
