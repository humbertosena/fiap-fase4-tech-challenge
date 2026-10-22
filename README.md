# Credit Sentinel — Tech Challenge Fase 4 (FIAP MLET)

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![Ruff](https://img.shields.io/badge/lint-ruff-261230)
![Pandera](https://img.shields.io/badge/contratos-pandera-0a7d5a)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

> **Vídeo STAR (5 min):** *link publicado na release `v1.0.0-ga`.*
> **Páginas didáticas:** [índice das releases](docs/index.html) ·
> [Etapa 1 — contratos de dados](docs/releases/etapa-1-contratos.html) ·
> [Etapa 2 — drift](docs/releases/etapa-2-drift.html)
> (também no GitHub Pages: <https://humbertosena.github.io/fiap-fase4-tech-challenge/>).

## 1. Resumo executivo

Uma fintech fictícia de crédito tem um modelo de *credit scoring* em produção. Com
inflação, juros altos e novos perfis de clientes, a diretoria suspeita de
**degradação silenciosa**: o modelo continua respondendo, mas decide pior. Este
repositório constrói a **camada de sustentação e confiabilidade** desse modelo:

1. **Barreira de qualidade na ingestão:** contrato de dados Pandera com 12 regras e
   *circuit breaker* que bloqueia lotes corrompidos (Etapa 1).
2. **Diagnóstico de degradação:** simulação de crise em três janelas mensais e
   detecção estatística de *drift* com KS, PSI e Evidently (Etapa 2).
3. **Observabilidade:** logs JSON, Data SLOs com *error budget*, alertas e MLflow
   (Etapa 3).
4. **Governança:** LGPD, auditoria de fairness e análise causal (Etapa 4).

Todos os dados são **sintéticos** (ADR 001); nenhum dado pessoal real é usado.

### Estado atual

| Etapa | Tema | Release | Status | Página |
| :--- | :--- | :--- | :--- | :--- |
| 1 | Validação de dados e contratos | `v0.1.0-alpha` | ✅ concluída | [etapa-1-contratos.html](docs/releases/etapa-1-contratos.html) |
| 2 | Simulação e detecção de drift | `v0.2.0-beta` | ✅ concluída | [etapa-2-drift.html](docs/releases/etapa-2-drift.html) |
| 3 | Observabilidade de pipelines e modelos | `v0.3.0-rc` | ⏳ planejada | — |
| 4 | Governança (LGPD) e vídeo STAR | `v1.0.0-ga` | ⏳ planejada | — |

## 2. Arquitetura

```mermaid
flowchart LR
    G[Gerador sintético<br/>DGP explícito] --> R[(Referência<br/>credit_ref.parquet)]
    G --> B[(Lote corrompido<br/>bad_batch.parquet)]
    R --> C{Contrato Pandera<br/>12 regras}
    B --> C
    C -- estrutural ou &gt; 1 % --> X[BLOCKED<br/>exit 2]
    C -- linhas inválidas ≤ 1 % --> Q[(Quarentena)]
    C -- válido --> M[Baseline HGB<br/>models/baseline]
    M --> F[MLflow<br/>tracking + registry]
    C --> P[Relatórios JSON<br/>reports/validation]
    P --> H[Páginas didáticas<br/>docs/]
    M --> H
    G --> W[(Janelas de produção<br/>mes_1, mes_2, mes_3_crise)]
    W --> C
    C -- janela válida --> D[Testes KS, PSI, χ², JS<br/>+ Evidently AI]
    M --> D
    D --> E[Relatórios de drift<br/>reports/]
    E --> H
```

O pacote `credit_sentinel` (em `src/`) é organizado por capacidade, e cada subpacote
expõe seus subcomandos na CLI `credit-sentinel`:

| Subpacote | Responsabilidade | Release |
| :--- | :--- | :--- |
| `data` | Gerador sintético, lote corrompido, dicionário de dados | v0.1.0-alpha |
| `contracts` | Schema Pandera, circuit breaker, sanitizador de PII, export YAML | v0.1.0-alpha |
| `ml` | Features, treino, métricas, limiares, model card, MLflow | v0.1.0-alpha |
| `drift` | Simulação das janelas, testes de drift, predições por janela, Evidently | v0.2.0-beta |
| `explainers` | Páginas HTML didáticas autocontidas | uma por release |
| `observability`, `tracking`, `governance` | Etapas 3 e 4 | próximas releases |

## 3. Guia de reprodução

Pré-requisitos: Python 3.12, [uv](https://docs.astral.sh/uv/) ≥ 0.12 e, opcionalmente,
Docker (MLflow em container).

```bash
git clone https://github.com/humbertosena/fiap-fase4-tech-challenge.git
cd fiap-fase4-tech-challenge
cp .env.example .env                 # troque CS_PII_SALT antes de qualquer uso real
make setup                           # uv sync --all-groups + validação do ambiente
make lint test                       # ruff estrito + pytest
make data                            # referência (12.000) + lote corrompido (600), bit a bit iguais
make validate                        # referência aprovada; lote corrompido BLOCKED (exit 2)
make train                           # valida, treina e registra o baseline no MLflow
make simulate                        # três janelas de produção (5.000 linhas cada)
make drift                           # testes de drift, scores e relatórios do Evidently
make explainers                      # gera docs/index.html e docs/releases/*.html
make mlflow                          # MLflow UI em http://localhost:5001
```

Tempos aproximados em um MacBook (Apple Silicon): `make setup` ≈ 1 min na primeira vez,
`make data` ≈ 2 s, `make train` ≈ 10 s, `make simulate` ≈ 3 s, `make drift` ≈ 6 s, suíte de testes ≈ 35 s.

Alternativa ao `make mlflow`: `docker compose up -d mlflow` sobe o mesmo servidor na
porta 5001 (a 5000 é usada pelo AirPlay no macOS), lendo o mesmo `mlruns/`.

| Comando da CLI | Efeito | Exit code |
| :--- | :--- | :--- |
| `credit-sentinel generate-data [--force]` | gera os datasets e `docs/data_dictionary.md` | 0, ou 1 se a referência já existir sem `--force` |
| `credit-sentinel validate --input <parquet> [--schema reference]` | valida um lote e grava `reports/validation/<lote>_report.json` | 0 aprovado · 2 bloqueado |
| `credit-sentinel train [--no-mlflow]` | treina o baseline e atualiza o model card | 0, ou 2 se a referência violar o contrato |
| `credit-sentinel simulate [--windows a,b] [--force]` | gera `data/production/<janela>.parquet` e imprime os deslocamentos | 0, ou 1 se a janela já existir sem `--force` |
| `credit-sentinel detect-drift [--window all\|<janela>]` | valida a janela, roda os testes de drift e o Evidently, grava `reports/` | 0 · 2 se o contrato bloquear a janela |
| `credit-sentinel scan-pii <caminhos…>` | procura CPF, e-mail, telefone e ids em claro | 0 limpo · 1 achados |
| `credit-sentinel export-contract` | exporta `config/data_contract.yaml` | 0 |
| `credit-sentinel build-explainers [--etapa 1,2]` | gera as páginas didáticas | 0 |

## 4. Etapa 1 — Validação de dados e contratos

### 4.1 Dataset de referência

`data/reference/credit_ref.parquet`: 12.000 propostas (jul/2025 a jun/2026), 18
colunas, taxa base de inadimplência de 6,45 %, SHA-256
`cb41f7d3b239c3ea43e100f478ef8975c6c815abbcf1f89ad018b2f2a4224ec4`. O processo gerador
está no [ADR 001](docs/adr/001-dataset-sintetico.md) e as colunas no
[dicionário de dados](docs/data_dictionary.md). O
[notebook de EDA](notebooks/01_eda_and_data_quality.ipynb) traz o diagnóstico ICRA.

### 4.2 Contrato de dados (Pandera)

O contrato tem 12 regras, quatro vezes o mínimo de 3 pedido pelo enunciado. Está em
`src/credit_sentinel/contracts/schema.py` e é exportado para
[`config/data_contract.yaml`](config/data_contract.yaml) (ADR 002).

| Regra | Dimensão ICRA | Descrição | Severidade |
| :--- | :--- | :--- | :--- |
| R01 | Integridade | Schema estrito: colunas exatas (`strict=True`) e tipos exatos (`coerce=False`) | estrutural |
| R02 | Unicidade | `id_cliente` único, não nulo e no formato `CLI-` + 10 hexadecimais | estrutural |
| R03 | Completude | Renda, score, idade e endividamento sem nulos | linha |
| R04 | Acurácia | Idade entre 18 e 100 anos | linha |
| R05 | Acurácia | Renda mensal positiva e até R$ 1.000.000 | linha |
| R06 | Acurácia | Score de bureau entre 0 e 1000 | linha |
| R07 | Acurácia | Endividamento e utilização de crédito entre 0 e 1 | linha |
| R08 | Acurácia | Categorias nos domínios; prazo e alvo nos valores válidos | linha |
| R09 | Razoabilidade | Valores positivos; tempo de emprego e atrasos em faixas plausíveis | linha |
| R10 | Consistência | Parcela não excede a renda | linha |
| R11 | Consistência | Desempregado não tem tempo de emprego | linha |
| R12 | Consistência | Tempo de emprego compatível com a idade (a partir dos 16 anos) | linha |

**Circuit breaker em dois níveis:** violações estruturais bloqueiam o lote; violações
de linha vão para a quarentena (`data/quarantine/<run_id>.parquet`) enquanto forem até
1 % do lote; acima disso o breaker abre. Nunca há imputação silenciosa, e o inventário
é sempre completo (`lazy=True`).

### 4.3 Prova de bloqueio do lote corrompido

O lote `data/corrupted/bad_batch.parquet` (600 linhas) recebe dez anomalias
propositais, inclusive uma coluna `cpf` fictícia:

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
```

Relatórios completos em [`reports/validation/`](reports/validation/). As amostras de
falha trazem apenas `id_cliente_hash` (HMAC-SHA-256 com salt secreto), e
`credit-sentinel scan-pii reports docs` confirma que não há dado pessoal em claro.

### 4.4 Modelo baseline

`HistGradientBoostingClassifier` em `Pipeline` sklearn, com 14 features, sem
`genero`/`regiao` (ADR 003, [model card](docs/model_card.md)). O treino só acontece
depois que a referência passa pelo contrato.

| Métrica (validação, 2.400 propostas) | Valor | Critério |
| :--- | ---: | :--- |
| ROC-AUC | 0,829 | ≥ 0,80 |
| Brier | 0,052 | ≤ 0,06 |
| KS | 0,541 | ≥ 0,45 |
| F1 no limiar 0,125 | 0,383 | — |

Top-3 features por importância de permutação: `score_bureau`,
`qtd_parcelas_atraso_12m` e `utilizacao_credito`. O modelo é registrado no MLflow como
`credit-scoring-model` (experimento `credit-sentinel/baseline`) com assinatura de 14
colunas de entrada.

## 5. Etapa 2 — Simulação temporal e detecção de drift

### 5.1 A crise que simulamos

O mesmo gerador da referência produz três lotes mensais de 5.000 clientes, em
`data/production/` (separados fisicamente da referência). Cada janela sobrescreve só os
parâmetros que a narrativa econômica move: a inflação corrói a renda real, a Selic alta
encarece a parcela e as famílias se endividam. A justificativa de cada parâmetro e a
calibração estão no [ADR 004](docs/adr/004-cenarios-de-drift.md).

| Parâmetro | Referência | `mes_1` (jul/2026) | `mes_2` (ago/2026) | `mes_3_crise` (set/2026) | Tipo |
| :--- | ---: | ---: | ---: | ---: | :--- |
| Fator da renda real | 1,00 | 1,00 | 0,90 | 0,80 | covariate |
| Deslocamento do endividamento (logit) | 0 | 0 | +0,35 | +0,70 | covariate |
| Deslocamento da utilização de crédito | 0 | 0 | +0,06 | +0,20 | covariate |
| Juros mensais (Price) | 2,5 % | 2,5 % | 3,0 % | 3,8 % | covariate |
| Participação de desempregados | 7 % | 7 % | 9 % | 12 % | covariate |
| Finalidade: consolidação de dívidas | 20 % | 20 % | 26 % | 34 % | covariate |
| Peso da parcela no risco | 0,8 | 0,8 | 1,0 | 1,6 | concept |
| Concept drift na faixa de score 550 a 700 | 0 | 0 | 0 | 1,5 | concept |
| Ajuste residual do intercepto | 0 | 0 | 0 | −0,64 | prior |
| **Taxa de inadimplência resultante** | **6,45 %** | **6,06 %** | **8,60 %** | **12,96 %** | prior shift |

`mes_1` é o **controle**: nova amostra da mesma população, sem nenhuma mudança. Os valores
iniciais do spec levavam `mes_3_crise` a 21,6 % de inadimplência; a calibração (ADR 004,
seção 2.3) ajustou `gamma_concept`, a utilização e o intercepto residual, e manteve os
demais choques como previstos. Os arquivos são reproduzíveis bit a bit
(`make simulate`; hashes em [`registro.md`](docs/evidencias/etapa2/registro.md)).

### 5.2 Como o drift é detectado

Os testes nativos (`src/credit_sentinel/drift/statistical_tests.py`) tomam as decisões; o
Evidently AI dá a segunda opinião e o relatório visual. Cada janela é validada no
`inference_schema` antes da análise: um lote bloqueado encerra a execução com exit 2.

| Teste | Aplicação | Regra |
| :--- | :--- | :--- |
| Kolmogorov-Smirnov | features numéricas | drift se p < 0,05 e, com Bonferroni (÷ 12), p < 0,0042 |
| PSI (decis da referência) | features numéricas e score predito | menor que 0,10 estável · de 0,10 a 0,25 atenção · 0,25 ou mais crítico |
| Qui-quadrado | features categóricas | drift se p < 0,05 |
| Jensen-Shannon | features categóricas | magnitude de 0 a 1; alerta a partir de 0,10 |
| Wasserstein normalizado | features numéricas | informativo |
| Evidently `DataDriftPreset` | 14 features | drift por coluna (Wasserstein e Jensen-Shannon, corte de 0,1) |
| Evidently `ValueDrift` | alvo (teste z) e score | prior shift e drift da predição |

Com 5.000 linhas o KS enxerga deslocamentos minúsculos, então o relatório sempre lê o
p-valor junto com a magnitude (PSI): o drift de uma feature numérica é o KS corrigido; a
gravidade é o PSI. O score e o Evidently usam como referência o **holdout de validação**
(2.400 linhas), porque os scores das linhas de treino são otimistas.

### 5.3 Resultado: tabela de KS e PSI por feature e janela

Gerada por `detect-drift` em [`reports/drift_table.md`](reports/drift_table.md). **Negrito**
marca drift significativo (KS com Bonferroni ou qui-quadrado).

#### Features numéricas

| Feature | mes_1 p(KS) | mes_1 PSI | mes_2 p(KS) | mes_2 PSI | mes_3_crise p(KS) | mes_3_crise PSI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `idade` | 0,7776 | 0,002 · estável | 0,1517 | 0,002 · estável | 0,9975 | 0,002 · estável |
| `renda_mensal` | 0,9595 | 0,002 · estável | **8,4e-18** | 0,029 · estável | **1,2e-79** | 0,161 · atenção |
| `score_bureau` | 0,8244 | 0,002 · estável | 0,3889 | 0,002 · estável | 0,9001 | 0,001 · estável |
| `taxa_endividamento` | 0,0489 | 0,004 · estável | **4,4e-103** | 0,171 · atenção | **3,8e-322** | 0,632 · crítico |
| `utilizacao_credito` | 0,3394 | 0,002 · estável | **9,2e-35** | 0,065 · estável | **4,4e-297** | 0,660 · crítico |
| `tempo_emprego_meses` | 0,8273 | 0,001 · estável | 0,1097 | 0,004 · estável | **1,2e-07** | 0,024 · estável |
| `qtd_parcelas_atraso_12m` | 1,0000 | 0,004 · estável | 1,0000 | 0,002 · estável | 1,0000 | 0,003 · estável |
| `valor_emprestimo` | 0,8619 | 0,002 · estável | 0,0312 | 0,006 · estável | **1,3e-10** | 0,027 · estável |
| `prazo_meses` | 1,0000 | 0,001 · estável | 0,8710 | 0,001 · estável | 1,0000 | 0,001 · estável |
| `valor_parcela` | 0,4096 | 0,003 · estável | **2,1e-05** | 0,014 · estável | **5,5e-15** | 0,033 · estável |
| `possui_imovel` | 1,0000 | 0,000 · estável | 1,0000 | 0,000 · estável | 0,9999 | 0,000 · estável |
| `comprometimento_renda` | 0,8786 | 0,003 · estável | **2,2e-15** | 0,031 · estável | **2,6e-77** | 0,145 · atenção |

PSI: menor que 0,10 estável; de 0,10 a 0,25 atenção; 0,25 ou mais crítico.

PSI: menor que 0,10 estável; de 0,10 a 0,25 atenção; 0,25 ou mais crítico.

#### Features categóricas

| Feature | mes_1 p(qui²) | mes_1 JS | mes_2 p(qui²) | mes_2 JS | mes_3_crise p(qui²) | mes_3_crise JS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `situacao_emprego` | 0,9831 | 0,004 | **0,0098** | 0,026 | **7,3e-22** | 0,070 |
| `finalidade_credito` | 0,4162 | 0,016 | **7,3e-19** | 0,068 | **1,0e-88** | 0,142 |

#### Scores e performance do baseline

| Janela | P50 | P90 | P99 | Aprovação | PSI do score | AUC | Δ AUC vs validação | Δ AUC vs controle | Brier | Taxa base |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| mes_1 | 0,030 | 0,145 | 0,433 | 87,6 % | 0,003 | 0,806 | -0,024 | — | 0,050 | 6,1 % |
| mes_2 | 0,034 | 0,201 | 0,514 | 82,4 % | 0,056 | 0,801 | -0,028 | -0,004 | 0,068 | 8,6 % |
| mes_3_crise | 0,047 | 0,282 | 0,590 | 74,7 % | 0,255 | 0,764 | -0,065 | -0,042 | 0,103 | 13,0 % |

### 5.4 Diagnóstico

- **Controle estável.** `mes_1` não tem nenhuma feature com PSI de 0,10 ou mais, nenhum
  drift após Bonferroni e 0 de 14 colunas em drift no Evidently. É a prova de que o
  detector não dá falso alarme.
- **Gradação.** `mes_2` fica na zona de atenção (`taxa_endividamento`, PSI 0,171) e
  `mes_3_crise` é crítico em duas features: `taxa_endividamento` (0,632) e
  `utilizacao_credito` (0,660). Também mudam `renda_mensal` e `comprometimento_renda`
  (atenção) e as duas categóricas (qui-quadrado significativo).
- **Prior shift.** A inadimplência vai de 6,45 % a 12,96 % e o Evidently reprova o teste do
  alvo em `mes_2` e `mes_3_crise`.
- **Concept drift.** Na faixa de score de 550 a 700, a inadimplência passa de 5,0 % na
  referência para 18,9 % no Mês 3 (3,8 vezes). O modelo, treinado antes, subestima esse risco.
- **Degradação do modelo.** O AUC cai de 0,829 (validação) para 0,764 no Mês 3, 0,065 a
  menos, e 0,042 abaixo do controle (0,806). O Brier dobra (0,050 → 0,103), a aprovação cai
  de 87,6 % para 74,7 % e o PSI do score vai a 0,255. P50, P90 e P99 do score sobem a cada
  janela: o modelo pontua clientes mais arriscados e acerta menos.
- **Nativo contra Evidently.** Concordam nas duas features críticas, no alvo e no score.
  Só os testes nativos apontam `tempo_emprego_meses`, `valor_parcela` e `situacao_emprego`
  (deslocamentos pequenos que KS e qui-quadrado detectam, abaixo do corte de 0,1 do
  Evidently). O Evidently informa "Dataset Drift is NOT detected" em `mes_3_crise` porque
  6 de 14 colunas (42,9 %) ficam abaixo do corte padrão de 50 % do dataset; o veredito por
  coluna é o que importa.
- **Ressalva.** `renda_mensal` em `mes_2` tem KS p = 8,4e-18, mas PSI de 0,029 (estável):
  o critério "PSI ≥ 0,10 na renda de `mes_2`" do spec não é atendido, e preferimos
  registrar isso a inflar o cenário (ADR 004 e registro da etapa).

### 5.5 Evidências

- [Relatório do Evidently AI](reports/evidently_drift_report.html) do Mês 3 (crise) e os
  de [`mes_1`](reports/evidently/mes_1_drift_report.html) e
  [`mes_2`](reports/evidently/mes_2_drift_report.html); sumário em
  [`evidently_drift_summary.json`](reports/evidently_drift_summary.json).
- [Página didática da Etapa 2](docs/releases/etapa-2-drift.html): linha do tempo das janelas,
  calculadora de PSI, mapa de calor, curva de concept drift e degradação do modelo.
- [Registro de verificação](docs/evidencias/etapa2/registro.md): hashes, critérios do spec e
  comparação entre os testes nativos e o Evidently.

![Relatório do Evidently AI em mes_3_crise: 6 de 14 colunas com drift](docs/evidencias/etapa2/evidently_mes3_crise.jpg)

## Próximas seções (releases seguintes)

6. Etapa 3 — observabilidade · 7. Governança e LGPD · 8. Análise causal da degradação ·
9. Runbook · 10. Roteiro STAR · 11. Decisões, estrutura e rastreabilidade ·
12. Agradecimentos e transparência.

## Decisões de arquitetura

- [ADR 001 — Dataset sintético com processo gerador explícito](docs/adr/001-dataset-sintetico.md)
- [ADR 002 — Contrato Pandera com circuit breaker em dois níveis](docs/adr/002-pandera-circuit-breaker.md)
- [ADR 003 — Modelo baseline com HistGradientBoosting](docs/adr/003-modelo-baseline-hgb.md)
- [ADR 004 — Cenários de drift: crise de juros em três janelas](docs/adr/004-cenarios-de-drift.md)

## Licença

Distribuído sob a licença MIT. Consulte [LICENSE](LICENSE).
