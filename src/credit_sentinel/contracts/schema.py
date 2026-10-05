"""Contrato de dados Pandera com 12 regras ICRA (ADR 002).

As regras vivem como metadados em ``RULES`` (fonte única para o schema, o YAML
exportado e a página didática). Todo check é nativo do Pandera ou registrado
via ``register_check_method``, para que o schema seja serializável em YAML. O id
da regra é embutido na mensagem de erro (``"R04 · ..."``) e recuperado por
``rule_id_for`` a partir dos ``failure_cases``.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

import pandera.pandas as pa
from pandera import extensions
from pydantic import BaseModel, ConfigDict

from credit_sentinel.data import dictionary as dd

if TYPE_CHECKING:
    import pandas as pd

ID_PATTERN = r"^CLI-[0-9A-F]{10}$"
REFERENCE_SCHEMA_NAME = "credit_scoring_reference"
INFERENCE_SCHEMA_NAME = "credit_scoring_inference"
_RULE_PREFIX = re.compile(r"^(R\d{2})\b")
_UNIQUENESS_CHECKS = {"field_uniqueness", "multiple_fields_uniqueness"}
_DTYPES = {
    "string": str,
    "int64": "int64",
    "float64": "float64",
    "bool": "bool",
    "datetime64[ns]": "datetime64[ns]",
}


class Bound(BaseModel):
    """Limites numéricos de uma coluna.

    Attributes:
        min: Limite inferior.
        max: Limite superior (``None`` = sem teto).
        include_min: Se o limite inferior é inclusivo.
    """

    model_config = ConfigDict(frozen=True)

    min: float
    max: float | None = None
    include_min: bool = True


class RuleSpec(BaseModel):
    """Metadados de uma regra do contrato.

    Attributes:
        id: Identificador estável (R01 a R12).
        dimension: Dimensão ICRA.
        description: Descrição em pt-BR.
        columns: Colunas envolvidas.
        kind: Tipo da regra (usado pelo validador interativo em JS).
        severity: ``estrutural`` bloqueia o lote; ``linha`` vai à quarentena.
        bounds: Limites por coluna (regras ``range``).
        allowed: Valores permitidos por coluna (regras ``isin``).
        expression: Expressão legível das regras cruzadas.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    dimension: str
    description: str
    columns: list[str]
    kind: Literal["schema", "unique", "notnull", "range", "isin", "cross"]
    severity: Literal["estrutural", "linha"]
    bounds: dict[str, Bound] = {}
    allowed: dict[str, list[str | int]] = {}
    expression: str | None = None


RULES: list[RuleSpec] = [
    RuleSpec(
        id="R01",
        dimension="Integridade",
        description="Schema estrito: conjunto exato de colunas e tipos exatos",
        columns=dd.ALL_COLUMNS,
        kind="schema",
        severity="estrutural",
    ),
    RuleSpec(
        id="R02",
        dimension="Unicidade",
        description="id_cliente único, não nulo e no formato CLI-XXXXXXXXXX",
        columns=[dd.ID_COLUMN],
        kind="unique",
        severity="estrutural",
        expression=ID_PATTERN,
    ),
    RuleSpec(
        id="R03",
        dimension="Completude",
        description="Campos críticos sem nulos (renda, score, idade, endividamento)",
        columns=["renda_mensal", "score_bureau", "idade", "taxa_endividamento"],
        kind="notnull",
        severity="linha",
    ),
    RuleSpec(
        id="R04",
        dimension="Acurácia",
        description="Idade entre 18 e 100 anos",
        columns=["idade"],
        kind="range",
        severity="linha",
        bounds={"idade": Bound(min=18, max=100)},
    ),
    RuleSpec(
        id="R05",
        dimension="Acurácia",
        description="Renda mensal positiva e até R$ 1.000.000",
        columns=["renda_mensal"],
        kind="range",
        severity="linha",
        bounds={"renda_mensal": Bound(min=0, max=1_000_000, include_min=False)},
    ),
    RuleSpec(
        id="R06",
        dimension="Acurácia",
        description="Score de bureau entre 0 e 1000",
        columns=["score_bureau"],
        kind="range",
        severity="linha",
        bounds={"score_bureau": Bound(min=0, max=1000)},
    ),
    RuleSpec(
        id="R07",
        dimension="Acurácia",
        description="Taxa de endividamento e utilização de crédito entre 0 e 1",
        columns=["taxa_endividamento", "utilizacao_credito"],
        kind="range",
        severity="linha",
        bounds={
            "taxa_endividamento": Bound(min=0, max=1),
            "utilizacao_credito": Bound(min=0, max=1),
        },
    ),
    RuleSpec(
        id="R08",
        dimension="Acurácia",
        description="Categorias dentro dos domínios e prazo/alvo nos valores válidos",
        columns=[*dd.ENUMS, "prazo_meses", dd.TARGET],
        kind="isin",
        severity="linha",
        allowed={**dd.ENUMS, "prazo_meses": dd.TERM_OPTIONS, dd.TARGET: [0, 1]},
    ),
    RuleSpec(
        id="R09",
        dimension="Razoabilidade",
        description="Valores da operação positivos e contadores em faixas plausíveis",
        columns=[
            "valor_emprestimo",
            "valor_parcela",
            "tempo_emprego_meses",
            "qtd_parcelas_atraso_12m",
        ],
        kind="range",
        severity="linha",
        bounds={
            "valor_emprestimo": Bound(min=0, include_min=False),
            "valor_parcela": Bound(min=0, include_min=False),
            "tempo_emprego_meses": Bound(min=0, max=600),
            "qtd_parcelas_atraso_12m": Bound(min=0, max=12),
        },
    ),
    RuleSpec(
        id="R10",
        dimension="Consistência",
        description="Parcela não pode exceder a renda mensal",
        columns=["valor_parcela", "renda_mensal"],
        kind="cross",
        severity="linha",
        expression="valor_parcela <= renda_mensal",
    ),
    RuleSpec(
        id="R11",
        dimension="Consistência",
        description="Desempregado não pode ter tempo de emprego",
        columns=["situacao_emprego", "tempo_emprego_meses"],
        kind="cross",
        severity="linha",
        expression="situacao_emprego == 'desempregado' => tempo_emprego_meses == 0",
    ),
    RuleSpec(
        id="R12",
        dimension="Consistência",
        description="Tempo de emprego compatível com a idade (a partir dos 16 anos)",
        columns=["tempo_emprego_meses", "idade"],
        kind="cross",
        severity="linha",
        expression="tempo_emprego_meses <= (idade - 16) * 12",
    ),
]
RULES_BY_ID = {rule.id: rule for rule in RULES}


def _has(df: pd.DataFrame, *columns: str) -> bool:
    return all(column in df.columns for column in columns)


@extensions.register_check_method(statistics=["left", "right"], check_type="vectorized")
def cs_column_le_column(df: pd.DataFrame, *, left: str, right: str) -> pd.Series | bool:
    """Check registrado: ``left <= right`` (nulos são tratados por R03)."""
    if not _has(df, left, right):
        return True
    return (df[left] <= df[right]) | df[left].isna() | df[right].isna()


@extensions.register_check_method(
    statistics=["status_column", "status_value", "value_column"],
    check_type="vectorized",
)
def cs_zero_when_status(
    df: pd.DataFrame, *, status_column: str, status_value: str, value_column: str
) -> pd.Series | bool:
    """Check registrado: ``status == valor`` implica ``value_column == 0``."""
    if not _has(df, status_column, value_column):
        return True
    return (df[status_column] != status_value) | (df[value_column] == 0)


@extensions.register_check_method(
    statistics=["tenure", "age", "start_age", "age_bounds"],
    check_type="vectorized",
)
def cs_tenure_within_age(
    df: pd.DataFrame, *, tenure: str, age: str, start_age: int, age_bounds: list[int]
) -> pd.Series | bool:
    """Check registrado: tempo de emprego <= (idade - início) * 12.

    Só avalia idades dentro do domínio; idades inválidas já falham em R04.
    """
    if not _has(df, tenure, age):
        return True
    in_domain = df[age].between(*age_bounds)
    limit = (df[age] - start_age) * 12
    return ~in_domain | (df[tenure] <= limit) | df[tenure].isna()


def _error(rule: RuleSpec, detail: str) -> str:
    return f"{rule.id} · {detail}"


def _range_check(rule: RuleSpec, column: str) -> pa.Check:
    bound = rule.bounds[column]
    error = _error(rule, f"{column} fora da faixa")
    if bound.max is None:
        compare = pa.Check.ge if bound.include_min else pa.Check.gt
        return compare(bound.min, name=rule.id, error=error)
    return pa.Check.in_range(
        bound.min, bound.max, include_min=bound.include_min, name=rule.id, error=error
    )


def _column_checks(column: str) -> list[pa.Check]:
    checks = [
        _range_check(rule, column)
        for rule in RULES
        if rule.kind == "range" and column in rule.bounds
    ]
    for rule in (r for r in RULES if r.kind == "isin" and column in r.allowed):
        error = _error(rule, f"{column} fora do domínio")
        checks.append(pa.Check.isin(rule.allowed[column], name=rule.id, error=error))
    if column == dd.ID_COLUMN:
        error = _error(RULES_BY_ID["R02"], "formato inválido")
        checks.append(pa.Check.str_matches(ID_PATTERN, name="R02", error=error))
    return checks


def _cross_checks() -> list[pa.Check]:
    rules = RULES_BY_ID
    return [
        pa.Check.cs_column_le_column(
            left="valor_parcela",
            right="renda_mensal",
            error=_error(rules["R10"], rules["R10"].expression or ""),
        ),
        pa.Check.cs_zero_when_status(
            status_column="situacao_emprego",
            status_value="desempregado",
            value_column="tempo_emprego_meses",
            error=_error(rules["R11"], rules["R11"].expression or ""),
        ),
        pa.Check.cs_tenure_within_age(
            tenure="tempo_emprego_meses",
            age="idade",
            start_age=16,
            age_bounds=[18, 100],
            error=_error(rules["R12"], rules["R12"].expression or ""),
        ),
    ]


def _column(spec: dd.ColumnSpec, *, optional: bool = False) -> pa.Column:
    return pa.Column(
        _DTYPES[spec.dtype],
        checks=_column_checks(spec.name),
        nullable=optional,
        required=not optional,
        unique=spec.name == dd.ID_COLUMN,
        title=spec.description,
    )


def _build_schema(name: str, *, target_required: bool) -> pa.DataFrameSchema:
    columns = {
        spec.name: _column(
            spec, optional=spec.name == dd.TARGET and not target_required
        )
        for spec in dd.COLUMN_SPECS
    }
    return pa.DataFrameSchema(
        columns, checks=_cross_checks(), strict=True, coerce=False, name=name
    )


def build_reference_schema() -> pa.DataFrameSchema:
    """Schema do dataset de referência (alvo obrigatório)."""
    return _build_schema(REFERENCE_SCHEMA_NAME, target_required=True)


def build_inference_schema() -> pa.DataFrameSchema:
    """Schema dos lotes de inferência (alvo opcional e anulável)."""
    return _build_schema(INFERENCE_SCHEMA_NAME, target_required=False)


def rule_id_for(check: object, column: object) -> str:
    """Mapeia uma linha de ``failure_cases`` do Pandera para o id da regra.

    Args:
        check: Valor da coluna ``check`` (mensagem do erro ou nome nativo).
        column: Coluna afetada (``NaN`` para falhas de schema).

    Returns:
        Id da regra (R01 a R12). Falhas estruturais nativas viram R01.
    """
    text = str(check)
    match = _RULE_PREFIX.match(text)
    if match:
        return match.group(1)
    if text in _UNIQUENESS_CHECKS or (
        text == "not_nullable" and column == dd.ID_COLUMN
    ):
        return "R02"
    return "R03" if text == "not_nullable" else "R01"
