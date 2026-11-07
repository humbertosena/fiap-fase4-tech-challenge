"""Credit Sentinel: sustentação e confiabilidade de um modelo de credit scoring."""

import os

# Silencia a dica informativa que o MLflow 3.x imprime ao ser importado.
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

__version__ = "1.0.0"
