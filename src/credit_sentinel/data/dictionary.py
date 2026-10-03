"""Dicionário de dados da carteira de crédito (fonte única de colunas e domínios)."""

from __future__ import annotations

from dataclasses import dataclass

ID_COLUMN = "id_cliente"
DATE_COLUMN = "data_referencia"
TARGET = "inadimplente"

NUMERIC_COLUMNS = [
    "idade",
    "renda_mensal",
    "score_bureau",
    "taxa_endividamento",
    "utilizacao_credito",
    "tempo_emprego_meses",
    "qtd_parcelas_atraso_12m",
    "valor_emprestimo",
    "prazo_meses",
    "valor_parcela",
    "possui_imovel",
]
CATEGORICAL_COLUMNS = ["situacao_emprego", "finalidade_credito"]
SENSITIVE_COLUMNS = ["genero", "regiao"]
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

EMPLOYMENT = ["CLT", "autonomo", "servidor_publico", "aposentado", "desempregado"]
PURPOSES = [
    "veiculo",
    "reforma",
    "educacao",
    "consolidacao_dividas",
    "saude",
    "outros",
]
GENDERS = ["F", "M"]
REGIONS = ["Norte", "Nordeste", "Centro-Oeste", "Sudeste", "Sul"]
TERM_OPTIONS = [6, 12, 24, 36, 48, 60]

ENUMS: dict[str, list[str]] = {
    "situacao_emprego": EMPLOYMENT,
    "finalidade_credito": PURPOSES,
    "genero": GENDERS,
    "regiao": REGIONS,
}


@dataclass(frozen=True)
class ColumnSpec:
    """Descrição de uma coluna do dataset.

    Attributes:
        name: Nome da coluna (pt-BR).
        dtype: Tipo pandas esperado.
        domain: Domínio válido em linguagem natural.
        role: Papel no projeto (identificador, feature, sensível, alvo...).
        description: Significado de negócio.
    """

    name: str
    dtype: str
    domain: str
    role: str
    description: str


COLUMN_SPECS: list[ColumnSpec] = [
    ColumnSpec(
        ID_COLUMN,
        "string",
        "`^CLI-[0-9A-F]{10}$`, único",
        "identificador direto",
        "Identificador do cliente; pseudonimizado (HMAC-SHA-256 com salt) antes de"
        " qualquer saída",
    ),
    ColumnSpec(
        DATE_COLUMN,
        "datetime64[ns]",
        "referência: 2025-07-01 a 2026-06-30",
        "janela temporal",
        "Data da proposta de crédito",
    ),
    ColumnSpec("idade", "int64", "18 a 100", "feature", "Idade do cliente em anos"),
    ColumnSpec(
        "renda_mensal",
        "float64",
        "> 0 e ≤ 1.000.000 (BRL)",
        "feature",
        "Renda mensal declarada em reais",
    ),
    ColumnSpec(
        "score_bureau", "int64", "0 a 1000", "feature", "Score de birô de crédito"
    ),
    ColumnSpec(
        "taxa_endividamento",
        "float64",
        "0 a 1",
        "feature",
        "Comprometimento da renda com dívidas existentes",
    ),
    ColumnSpec(
        "utilizacao_credito",
        "float64",
        "0 a 1",
        "feature",
        "Uso do limite de crédito rotativo disponível",
    ),
    ColumnSpec(
        "tempo_emprego_meses",
        "int64",
        "0 a 600; 0 se desempregado; ≤ (idade - 16) * 12",
        "feature",
        "Tempo no emprego atual em meses",
    ),
    ColumnSpec(
        "qtd_parcelas_atraso_12m",
        "int64",
        "0 a 12",
        "feature",
        "Parcelas pagas em atraso nos últimos 12 meses",
    ),
    ColumnSpec(
        "valor_emprestimo", "float64", "> 0 (BRL)", "feature", "Valor solicitado"
    ),
    ColumnSpec(
        "prazo_meses",
        "int64",
        "{6, 12, 24, 36, 48, 60}",
        "feature",
        "Prazo do empréstimo em meses",
    ),
    ColumnSpec(
        "valor_parcela",
        "float64",
        "> 0 e ≤ renda_mensal (BRL)",
        "feature",
        "Parcela mensal calculada pela tabela Price",
    ),
    ColumnSpec(
        "situacao_emprego",
        "string",
        ", ".join(EMPLOYMENT),
        "feature",
        "Vínculo de trabalho atual",
    ),
    ColumnSpec(
        "finalidade_credito",
        "string",
        ", ".join(PURPOSES),
        "feature",
        "Finalidade declarada do crédito",
    ),
    ColumnSpec("possui_imovel", "bool", "verdadeiro/falso", "feature", "Tem imóvel"),
    ColumnSpec(
        "genero",
        "string",
        ", ".join(GENDERS),
        "sensível (só auditoria de fairness)",
        "Gênero declarado; nunca usado como feature",
    ),
    ColumnSpec(
        "regiao",
        "string",
        ", ".join(REGIONS),
        "sensível (só auditoria de fairness)",
        "Região do país; nunca usada como feature",
    ),
    ColumnSpec(
        TARGET,
        "int64",
        "{0, 1}",
        "alvo",
        "1 se o cliente ficou inadimplente (atraso > 90 dias)",
    ),
]

ALL_COLUMNS = [spec.name for spec in COLUMN_SPECS]


def render_markdown() -> str:
    """Gera o dicionário de dados em Markdown (``docs/data_dictionary.md``).

    Returns:
        Documento Markdown com uma linha por coluna.
    """
    header = [
        "# Dicionário de dados — carteira de crédito sintética",
        "",
        (
            "> Gerado por `credit_sentinel.data.dictionary.render_markdown()`;"
            " não edite à mão."
        ),
        "",
        "| Coluna | Tipo | Domínio | Papel | Descrição |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ]
    rows = [
        f"| `{s.name}` | {s.dtype} | {s.domain} | {s.role} | {s.description} |"
        for s in COLUMN_SPECS
    ]
    footer = [
        "",
        (
            f"Features do modelo: {len(FEATURE_COLUMNS)} colunas"
            " (+ derivada `comprometimento_renda`)."
        ),
        (
            "Atributos sensíveis (`genero`, `regiao`) ficam fora do modelo e"
            " entram apenas na auditoria de fairness."
        ),
    ]
    return "\n".join(header + rows + footer) + "\n"
