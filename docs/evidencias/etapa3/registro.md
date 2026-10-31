# Registro de verificação — Etapa 3 (`v0.3.0-rc`)

Evidências coletadas durante a Etapa 3. Cada bloco traz o comando executado e a saída
relevante, para que qualquer pessoa repita a verificação. O ambiente é o mesmo das
etapas anteriores (`uv.lock` não mudou): Python 3.12, pandas 2.3.3, scikit-learn 1.9.1,
Evidently 0.7.23 e MLflow 3.18.0. Os artefatos versionados foram gerados com
`SOURCE_DATE_EPOCH` na data do commit que os guarda.

## 1. Logger JSON e redação de PII (T32)

Cada linha de `logs/pipeline_events.jsonl` valida em `PipelineEvent` (oito campos
obrigatórios). O filtro mascara identificadores **antes** do arquivo e do stderr:

```console
$ uv run pytest tests/unit/test_logger.py -q
17 passed
$ uv run credit-sentinel scan-pii logs reports docs
nenhum dado pessoal encontrado em: logs, reports, docs
```

O teste de integração injeta um identificador de cliente fictício na mensagem de um hook que falha e confirma
que o log traz `mlflow indisponível para ***`.

## 2. Data SLOs e alertas (T33 e T34)

| Critério do `SPEC-observability` | Resultado | Atende |
| :--- | :--- | :---: |
| `SLO2.features` dentro do top-3 do modelo | `score_bureau`, `qtd_parcelas_atraso_12m`, `utilizacao_credito` (teste compara com `feature_importance.json`) | sim |
| Casos de borda do consumo (0,70 e 1,0) | `0,70` → `WARN`, `1,0` → `BREACHED`; objetivo `>=` nunca negativo | sim |
| `BLOCKED` → P1; PSI ≥ 0,25 → P2; `WARN` → P3 | testes parametrizados em `test_slo_monitor.py` | sim |
| Cada alerta aponta para uma âncora existente do runbook | teste lê `docs/runbook_incidentes.md` | sim |

## 3. Pipeline ponta a ponta (T35, T36 e T40)

```console
$ SOURCE_DATE_EPOCH=1793400600 uv run credit-sentinel run-pipeline --window all
janela bad_batch: BLOCKED · alertas P1 · saldo de orçamento -1250% · 0.5 s
  [P1] Lote bad_batch bloqueado pelo contrato: violação estrutural (R01, R02): lote bloqueado
janela mes_1: SUCCESS · alertas P4 · saldo de orçamento +93% · 0.8 s
  [P4] Lote mes_1 dentro dos objetivos (93% de saldo no error budget)
janela mes_2: SUCCESS · alertas P3 · saldo de orçamento +40% · 0.7 s
  [P3] SLO3_consistencia_ks: observado 0.8 (objetivo <= 0.34), 235% do error budget
janela mes_3_crise: SUCCESS · alertas P2, P3 · saldo de orçamento -17% · 0.8 s
  [P2] SLO2_estabilidade_psi: observado 0.6602 (objetivo <= 0.25), 264% do error budget
  [P3] SLO3_consistencia_ks: observado 0.8 (objetivo <= 0.34), 235% do error budget
  [P3] SLO4_performance: observado 0.764 (objetivo >= 0.75), 82% do error budget
4 janela(s): 3 com sucesso, 1 bloqueada(s)
$ uv run credit-sentinel run-pipeline --window bad_batch --quick --no-mlflow --strict; echo $?
2
```

| Critério do `SPEC-observability` | Resultado | Atende |
| :--- | :--- | :---: |
| Log com `SUCCESS` (`mes_1`), `BLOCKED` (`bad_batch`) e alertas de drift (`mes_3_crise`) | 46 eventos; amostra de 40 em `pipeline_events_sample.jsonl` | sim |
| `bad_batch` → P1; `mes_1` → só P4; `mes_2` → ≥ 1 P3; `mes_3_crise` → P2 com ação de retreino | idem ao quadro acima | sim |
| `remaining_budget` negativo em `mes_3_crise` | − 0,166 | sim |
| Painel autocontido com status, gauges, feed, latência, mapa de PSI e links do Evidently | `reports/health_dashboard.html` (52 KB), conferido no Chrome (tema escuro, filtro de alertas, destaque de janela) | sim |
| Pipeline < 120 s por janela e `--quick` < 30 s | ≤ 0,8 s por janela; quatro janelas em modo rápido ≈ 3 s | sim |
| Hook falho não derruba o pipeline | `test_failing_hook_marks_tracking_failed_and_the_pipeline_keeps_going` | sim |

Os relatórios do Evidently foram regravados pela execução e diferiam dos da Etapa 2 só
nos carimbos dos UUIDs e em ruído de 1e-15 nos floats; foram mantidas as versões da
Etapa 2 para não duplicar ~14 MB no histórico.

## 4. MLflow: tracking (T37 e T41)

```console
$ make mlflow     # http://localhost:5001
```

| Print | O que mostra |
| :--- | :--- |
| `mlflow_runs_monitoring.jpg` | experimento `credit-sentinel/monitoring`: 1 run pai e 4 runs aninhados, com `psi_*`, `remaining_budget` e `roc_auc` |
| `mlflow_metricas_mes3_crise.jpg` | 54 métricas do run `mes_3_crise` (`psi_utilizacao_credito` 0,660, `ks_p_*`...) e as tags `decision`, `git_sha`, `release` |
| `mlflow_evidently_artefato.jpg` | o relatório do Evidently como artefato do run (6 de 14 colunas com drift) |

Os carimbos de data da UI são os reais da execução (09/11/2026): o MLflow grava o
relógio de parede e não aceita `SOURCE_DATE_EPOCH`.

## 5. Retreino e guardrails (T38)

```console
$ SOURCE_DATE_EPOCH=1793233200 uv run credit-sentinel retrain-challenger
challenger challenger-1.0.0 salvo em models/challenger
  G1_auc_vs_champion: aprovado · challenger 0.7883 contra champion 0.7605 (+0.0279; exigido +0.02)
  G2_brier: aprovado · challenger 0.0957 contra champion 0.0995
  G3_contrato_de_entrada: aprovado · assinatura idêntica
  G4_fairness_aprovacao: aprovado · pior razão: regiao=Nordeste com 0.934 (faixa 0.80 a 1.25)
  G5_latencia: aprovado · 5000 linhas em 0.011 s (limite 5 s)
holdout temporal: mes_3_crise (2539 linhas)
challenger challenger-1.0.0 aprovado pelos guardrails
```

O `models/challenger/` regenerado saiu byte a byte igual ao versionado; só o
`gates_report.json` ganhou as versões do registry.

## 6. Model Registry e promoção (T39 e T41)

| Print | Estado |
| :--- | :--- |
| `mlflow_registry_antes.jpg` | v1 `@champion`, v2 `@challenger` |
| `mlflow_registry_depois.jpg` | v2 `@champion`; v1 com a tag `retired_reason=drift_mes_3_crise` |
| `mlflow_registry_v1_aposentada.jpg` | página da versão 1 com a tag |

```console
$ uv run credit-sentinel promote-challenger
promoção concluída: credit-scoring-model v2 agora é @champion
  credit-scoring-model v1: sem alias · retired_reason=drift_mes_3_crise
  credit-scoring-model v2: @champion
```

A promoção foi executada por decisão registrada no §12 do spec (2026-11-08); com gates
reprovados o comando termina com exit 3 sem mudar nenhum alias (teste de integração).

## 7. Página didática da Etapa 3 (T42)

`docs/releases/etapa-3-observabilidade.html` é gerada por `credit-sentinel build-explainers`
a partir de `reports/slo_audit_report.json`, `reports/gates_report.json`,
`config/slo_config.yaml` e da amostra de log; nenhum número é digitado no template.

| Critério do `SPEC-release-explainers` | Resultado | Atende |
| :--- | :--- | :---: |
| Até 2,5 MB | 387 KB (inclui três prints em base64) | sim |
| Pelo menos três componentes interativos | replay do pipeline, gauges por janela, feed de alertas filtrável, registry antes e depois, mascaramento de PII ao vivo | sim |
| Números iguais aos relatórios | `test_etapa3_audit_and_gates_are_the_versioned_reports` e `test_etapa3_numbers_come_from_the_audit` | sim |
| No máximo três prints, até 1.200 px, JPEG q80 | `mlflow_runs_monitoring`, `mlflow_evidently_artefato` e `mlflow_registry_depois` | sim |
| Sem dados pessoais no HTML | `scan-pii docs` limpo; o texto do mascaramento é montado no navegador a partir de fragmentos | sim |
| Conferida no navegador | Chrome via HTTP local: sem erros de console, tema claro e escuro, replay de `bad_batch` até `BLOCKED`, filtro de alertas, alternância do registry, 360 px sem rolagem horizontal | sim |

Capturas: `pagina_etapa3_desktop.jpg`, `pagina_etapa3_mobile.jpg` e `pagina_indice.jpg`.

## 8. Verificação da suíte

| Comando | Resultado |
| :--- | :--- |
| `uv run ruff check .` | All checks passed! |
| `uv run ruff format --check .` | 115 arquivos já formatados |
| `uv run pytest --cov=credit_sentinel` | 461 testes aprovados, cobertura de 98 % |
| `uv run credit-sentinel scan-pii logs reports docs README.md` | nenhum dado pessoal encontrado |

## 9. Desvios do spec

1. **Protocolo do retreino.** O spec treinava o challenger sem dados da crise; medido, o
   ganho ficou em +0,015 de AUC (gate G1 exige +0,02). Por decisão do responsável
   (2026-10-28), a primeira quinzena de `mes_3_crise` entra no treino e a segunda é o
   holdout (ADR 005, seção 5).
2. **Features do SLO2.** O exemplo do spec citava `taxa_endividamento` e
   `comprometimento_renda`; o spec também pedia que a lista viesse do modelo, e o top-3
   medido é `score_bureau`, `qtd_parcelas_atraso_12m` e `utilizacao_credito`.
3. **`promote-challenger`.** Subcomando adicional: `retrain-challenger --promote` retreina
   e registra uma versão nova antes de promover; o novo comando promove a versão já
   registrada, que é o que permite o "antes e depois" do registry.
4. **`evaluate_slos`.** Recebe um `SloObservation` em vez de seis argumentos soltos (limite
   de cinco argumentos do Ruff do projeto).
5. **Prints em JPEG.** Os prints estão em JPEG (qualidade 80, ≤ 1.200 px, cada um abaixo
   de 60 KB) em vez de PNG, como na Etapa 2.
