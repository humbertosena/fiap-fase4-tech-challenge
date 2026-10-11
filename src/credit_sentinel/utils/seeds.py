"""Sementes aleatórias centralizadas e determinísticas.

Toda aleatoriedade do projeto recebe um ``np.random.Generator`` criado por
``make_rng``. ``PYTHONHASHSEED`` não é alterado em tempo de execução; o
projeto não depende da ordem de iteração de ``set``/``dict`` com hash de str.
"""

from __future__ import annotations

import random

import numpy as np

DEFAULT_SEED = 42


def make_rng(seed: int | None = None) -> np.random.Generator:
    """Cria um gerador aleatório determinístico.

    Args:
        seed: Semente desejada; ``None`` usa ``DEFAULT_SEED``.

    Returns:
        Gerador ``PCG64`` inicializado com a semente.
    """
    return np.random.default_rng(DEFAULT_SEED if seed is None else seed)


def set_global_seed(seed: int = DEFAULT_SEED) -> None:
    """Fixa as sementes globais legadas (``random`` e ``numpy.random``).

    Usado apenas para bibliotecas de terceiros que ainda leem o estado global.

    Args:
        seed: Semente a aplicar.
    """
    random.seed(seed)
    np.random.seed(seed)  # noqa: NPY002 - estado legado lido por bibliotecas de terceiros
