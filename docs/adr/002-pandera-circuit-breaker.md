# ADR 002 — Contrato de dados em Pandera com circuit breaker em dois níveis

| Campo | Valor |
|---|---|
| Status | **Aceito** |
| Data | 2026-10-08 (Etapa 1) |
| Decisores | Humberto Sena Santos |
| Módulos afetados | `data-contracts`, `observability` (SLO 1), `baseline-model` |
| Relacionados | ADR 001 (dataset sintético) |

## 1. Contexto

O foco da Etapa 1 é **garantir que lixo não entre no modelo**. O enunciado pede ao
menos três regras rígidas (sem nulos, faixa de idade, limites de renda), um contrato
em Great Expectations ou Pandera e a demonstração de um lote com erros sendo
**bloqueado** na ingestão. Ao mesmo tempo, a Etapa 3 define um SLO de integridade
(até 1 % de registros rejeitados por lote), o que pressupõe que *alguns* registros
ruins sejam tolerados sem parar a operação. As duas exigências parecem conflitar:
*fail-fast* puro bloquearia qualquer lote com uma linha inválida.

## 2. Decisão

### 2.1 Ferramenta: Pandera

| Critério | Pandera | Great Expectations |
|---|---|---|
| Integração com pandas e pytest | Nativa (`DataFrameSchema`, `lazy=True`) | Exige *Data Context*, *checkpoints* e *suites* |
| Contrato como código revisável | Schema Python + export YAML (`to_yaml`/`from_yaml`) | JSON de *expectations* gerado por ferramenta |
| Inventário completo de falhas | `SchemaErrors.failure_cases` (DataFrame) | *Validation results* aninhados |
| Peso de dependências | Leve | Pesado (dezenas de dependências transitivas) |

Escolhemos **Pandera 0.35** (`import pandera.pandas as pa`), com `strict=True`
(conjunto exato de colunas) e `coerce=False` (tipos exatos, sem conversão silenciosa).

### 2.2 Doze regras ICRA, uma fonte única

As regras R01 a R12 cobrem Integridade, Completude, Razoabilidade e Acurácia
(tabela completa no README e em `config/data_contract.yaml`). Elas são declaradas uma
única vez como metadados (`contracts.schema.RULES`), e a partir delas são gerados: o
schema Pandera, o YAML exportado e o validador interativo da página didática.

Para que o schema seja **serializável**, só usamos checks nativos (`in_range`, `isin`,
`str_matches`, `gt`) e três checks registrados com `register_check_method` para as
regras cruzadas (R10 parcela ≤ renda, R11 desempregado sem tempo de emprego, R12
tempo de emprego compatível com a idade). O id da regra vai na mensagem de erro
(`"R04 · idade fora da faixa"`) e é recuperado dos `failure_cases`.

### 2.3 Circuit breaker em dois níveis

```
validate_batch(lote)
 ├─ violação estrutural (R01 schema/tipos, R02 chave primária) ─▶ BLOCKED (exit 2)
 ├─ violações de linha (R03–R12)
 │    ├─ fração rejeitada > 1 % ─▶ BLOCKED: circuit breaker aberto (exit 2)
 │    └─ fração rejeitada ≤ 1 % ─▶ linhas vão para data/quarantine/<run_id>.parquet
 │                                   e o lote segue com as linhas válidas (SUCCESS)
 └─ sem violações ─▶ SUCCESS
```

- **Estrutural bloqueia sempre**: coluna inesperada (ex.: `cpf`), coluna ausente,
  tipo errado, id nulo ou duplicado indicam que o *produtor* do dado mudou; seguir
  adiante contaminaria o modelo e, no caso da coluna `cpf`, violaria a LGPD.
- **Linha tolera até 1 %**: erros pontuais vão para a quarentena com o `run_id`, sem
  imputação; acima do limite, o problema é sistêmico e o lote inteiro para.
- O relatório (`ValidationReport`) é sempre completo (`lazy=True`), mesmo quando o lote
  é bloqueado, e traz amostras com `id_cliente_hash` (HMAC-SHA-256), nunca o id em claro.

## 3. Consequências

**Positivas**
- O lote corrompido é bloqueado com inventário das 10 anomalias
  (`reports/validation/bad_batch_report.json`), o que atende ao critério de 25 % da rubrica.
- O SLO 1 da Etapa 3 consome diretamente `rejected_ratio` do relatório.
- O contrato é versionado em YAML e testado por *round-trip* (`to_json` idêntico).

**Negativas e mitigações**
- Mensagens de erro carregam o id da regra por convenção: um teste garante o
  mapeamento (`rule_id_for`) e que cada regra quebrada isoladamente é detectada.
- Checks registrados precisam ser importados antes de `load_contract_yaml`; o módulo
  `export` importa `schema`, que os registra.
- `coerce=False` torna o contrato sensível a mudanças de tipo do pandas (ex.: nulos em
  coluna inteira viram `float64`); isso é intencional e aparece como R01 no inventário.
