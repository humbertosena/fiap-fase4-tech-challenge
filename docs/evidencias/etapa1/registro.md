# Registro de verificação — Etapa 1 (`v0.1.0-alpha`)

Evidências coletadas durante a Etapa 1. Cada bloco traz o comando executado e a saída
relevante, para que qualquer pessoa repita a verificação.

## 1. Ambiente

| Item | Valor |
| :--- | :--- |
| Python | 3.12.13 (`.python-version`) |
| uv | 0.12.22 |
| pandas / numpy | 2.3.3 / 2.5.4 (pandas fixado em `<3`: Evidently 0.7.23 não importa com pandas 3) |
| pandera / scikit-learn | 0.35.0 / 1.9.1 |
| evidently / mlflow | 0.7.23 / 3.18.0 |
| Docker Engine | 29.9.2 |

```console
$ uv run python scripts/validate_env.py
[OK ] python       3.12.13
[OK ] uv           /opt/homebrew/bin/uv
[OK ] docker       /usr/local/bin/docker
[OK ] porta 5001   livre
```

## 2. Infraestrutura local (T05)

MLflow Tracking Server via `docker compose` (porta 5001, pois a 5000 é do AirPlay no macOS):

```console
$ docker compose up -d mlflow
 Container credit-sentinel-mlflow Started
$ docker inspect -f '{{.State.Health.Status}}' credit-sentinel-mlflow
healthy
$ curl -s http://localhost:5001/health
OK
```

Imagem da CLI construída a partir do `Dockerfile` multi-stage, executando como usuário
não-root:

```console
$ docker build -t credit-sentinel:dev .
$ docker run --rm credit-sentinel:dev --version
credit-sentinel 0.1.0
$ docker run --rm --entrypoint id credit-sentinel:dev
uid=999(appuser) gid=999(appuser) groups=999(appuser)
```

## 3. Dataset de referência (T08)

```console
$ uv run credit-sentinel generate-data
referência: data/reference/credit_ref.parquet (12000 linhas, 18 colunas)
  taxa base: 6.45%
  sha256: cb41f7d3b239c3ea43e100f478ef8975c6c815abbcf1f89ad018b2f2a4224ec4
dicionário: docs/data_dictionary.md
$ uv run credit-sentinel generate-data          # sem --force: imutabilidade
erro: a referência já existe e é imutável; use --force para regerar
$ uv run credit-sentinel generate-data --force  # regeração bit a bit
  sha256: cb41f7d3b239c3ea43e100f478ef8975c6c815abbcf1f89ad018b2f2a4224ec4
```

| Verificação | Resultado |
| :--- | :--- |
| Linhas × colunas | 12.000 × 18 |
| Valores nulos | 0 |
| `id_cliente` único | sim |
| Taxa base (alvo 6 % ± 0,5 pp) | 6,45 % |
| `beta0` calibrado (bisseção, seed 2024) | −3,3067 |
| AUC do oráculo (logit verdadeiro) | 0,832 |
| SHA-256 em duas execuções | idêntico |

## 4. Lote corrompido (T09)

`generate-data` amostra 600 linhas da referência (seed 7) e injeta dez anomalias em
blocos disjuntos de linhas (`credit_sentinel.data.corruptor.ANOMALY_PLAN`):

```console
$ uv run credit-sentinel generate-data --force
lote corrompido: data/corrupted/bad_batch.parquet (600 linhas, 19 colunas)
  sha256: 914ef091aa341f50b9a010ed168f7391c31d7ca98e3f45fad9ea48fad5bb8398
```

| Anomalia | Linhas | Regra que deve interceptar |
| :--- | ---: | :--- |
| `idade` negativa ou acima de 100 | 30 | R04 |
| `renda_mensal` nula | 24 | R03 |
| `score_bureau` nulo | 18 | R03 |
| `id_cliente` duplicado | 18 | R02 |
| `id_cliente` nulo | 6 | R02 |
| `situacao_emprego` fora do enum (`freelancer`, `"CLT "`) | 18 | R08 |
| `taxa_endividamento` maior que 1 | 12 | R07 |
| `valor_parcela` maior que `renda_mensal` | 12 | R10 |
| `tempo_emprego_meses` > 0 com `desempregado` | 6 | R11 |
| coluna extra `cpf` (PII não prevista) | todas | R01 |

Os CPFs têm o formato `NNN.NNN.NNN-DD` com **dígito verificador propositalmente
inválido** (`is_valid_cpf` retorna falso para todos): são fictícios por construção.

## 5. Contrato de dados e circuit breaker (T10 a T13)

O lote corrompido é **bloqueado** (exit 2) por violação estrutural, e o relatório ainda
inventaria todas as violações de linha (`lazy=True`):

```console
$ uv run credit-sentinel validate --input data/corrupted/bad_batch.parquet
lote: bad_batch (600 linhas)
schema: credit_scoring_inference
decisão: BLOCKED — violação estrutural (R01, R02): lote bloqueado
regra  coluna                   falhas  check
R01    cpf                           1  column_in_schema
R01    score_bureau                 18  dtype('int64')
R02    id_cliente                   36  field_uniqueness
R02    id_cliente                    6  not_nullable
R03    renda_mensal                 24  not_nullable
R03    score_bureau                 18  not_nullable
R04    idade                        30  R04 · idade fora da faixa
R07    taxa_endividamento           12  R07 · taxa_endividamento fora da faixa
R08    situacao_emprego             18  R08 · situacao_emprego fora do domínio
R10    —                            12  R10 · valor_parcela <= renda_mensal
R11    —                             6  R11 · situacao_emprego == 'desempregado' => tempo_emprego_meses == 0
rejeitadas: 162 (27.00%) · válidas: 438
relatório: reports/validation/bad_batch_report.json
$ echo $?
2
```

A referência passa em 100 % das regras (exit 0):

```console
$ uv run credit-sentinel validate --input data/reference/credit_ref.parquet --schema reference
lote: credit_ref (12000 linhas)
schema: credit_scoring_reference
decisão: SUCCESS
rejeitadas: 0 (0.00%) · válidas: 12000
relatório: reports/validation/credit_ref_report.json
```

Leitura do inventário:

- **162 linhas rejeitadas** = 144 linhas com anomalia injetada + as 18 linhas
  originais cujos ids foram duplicados (a unicidade marca as duas ocorrências: 36).
- `score_bureau` aparece também em **R01** (`dtype('int64')`): os nulos transformam a
  coluna inteira em `float64`, o que viola o tipo exato do contrato.
- As amostras de falha do relatório trazem apenas `id_cliente_hash` (HMAC-SHA-256);
  o `id_cliente` em claro nunca é gravado. Os relatórios versionados usam o salt de
  desenvolvimento (`dev-salt-nao-usar-em-producao`), aceitável porque os dados são
  sintéticos; em produção o salt vem de `CS_PII_SALT` (segredo fora do git).

Varredura de PII nos relatórios versionados:

```console
$ uv run credit-sentinel scan-pii reports
nenhum dado pessoal encontrado em: reports
```
