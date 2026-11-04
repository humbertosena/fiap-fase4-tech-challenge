# Runbook de incidentes do `credit-sentinel`

Guia de resposta aos alertas do pipeline de monitoramento. Cada alerta gerado por
`credit-sentinel run-pipeline` aponta para uma seção deste documento pelo campo
`runbook_anchor` (regras em `config/slo_config.yaml`, decisões no ADR 005).

| Severidade | Quando dispara | Prazo de resposta sugerido | Seção |
|---|---|---|---|
| **P1** | o contrato bloqueou o lote (circuit breaker aberto) | imediato | [P1](#p1-circuit-breaker) |
| **P2** | SLO2 (PSI) ou SLO4 (ROC-AUC) estourado: drift crítico | no mesmo dia | [P2](#p2-drift-critico) |
| **P3** | algum SLO com 70 % ou mais do error budget, ou SLO3 estourado | até a próxima janela | [P3](#p3-atencao) |
| **P4** | lote dentro dos objetivos | nenhuma | [P4](#p4-informativo) |

Onde olhar primeiro, em qualquer incidente:

- **Eventos do pipeline:** `logs/pipeline_events.jsonl` (uma linha JSON por evento,
  com PII mascarada); filtre por `run_id` e por `status` diferente de `SUCCESS`.
- **Painel de saúde:** `reports/health_dashboard.html` (status por janela, gauges de
  error budget, mapa de PSI e feed de alertas).
- **Auditoria:** `reports/slo_audit_report.json` (todos os SLOs, consumos e alertas).
- **MLflow:** `make mlflow` e abrir o experimento `credit-sentinel/monitoring`.

<a id="p1-circuit-breaker"></a>

## P1 · Circuit breaker aberto (lote bloqueado)

O contrato de dados rejeitou o lote inteiro. Nenhum score foi calculado, e isso é o
comportamento correto: um lote inválido não pode chegar ao modelo.

1. **Detecção.** Evento `contract_validation` com `status: BLOCKED` no log e alerta P1
   com a mensagem do motivo. Com `--strict`, o `run-pipeline` termina com exit code 2.
2. **Triagem.** Abra `reports/validation/<lote>_report.json` e responda:
   - A violação é **estrutural** (R01 schema e tipos, R02 chave primária) ou o lote
     passou de 1 % de linhas rejeitadas (regras R03 a R12)?
   - Quais colunas e quantas linhas? Há coluna inesperada (por exemplo, um campo com
     identificador direto) ou tipo trocado?
   - O problema é do **produtor** (mudança de schema, exportação quebrada) ou do
     **contrato** (regra desatualizada)?
3. **Mitigação.** Não force o lote e **nunca imputar valores para "fazer passar"**.
   Mantenha o último lote válido como referência operacional, avise o produtor dos
   dados e, se houve coluna de identificador direto, trate como incidente de
   privacidade (seção 7 do `README.md`).
4. **Resolução.** O produtor corrige e reenvia. Rode
   `uv run credit-sentinel validate --input <lote.parquet>` até obter `SUCCESS` e então
   reprocesse com `uv run credit-sentinel run-pipeline --window <janela>`. Feche o
   incidente quando o lote seguinte passar pelo contrato.
5. **Pós-mortem.** Registre linha do tempo, número de registros afetados, causa-raiz e
   ação preventiva. Se a causa for uma regra ausente, abra um ADR e inclua a regra no
   contrato (`config/data_contract.yaml`) com teste.

<a id="p2-drift-critico"></a>

## P2 · Drift crítico (SLO2 ou SLO4 estourado)

O mundo mudou o bastante para o modelo perder qualidade: PSI de 0,25 ou mais nas
features que mais pesam, ou ROC-AUC abaixo de 0,75.

1. **Detecção.** Alerta P2 com o SLO e o valor observado; no painel, o gauge do SLO
   passa de 100 % e o mapa de PSI acende a feature crítica.
2. **Triagem.**
   - Abra o relatório do Evidently da janela (`reports/evidently/<janela>_drift_report.html`)
     e a tabela `reports/drift_table.md` para ver **quais** features mudaram.
   - Descarte primeiro um problema de dados: o lote passou pelo contrato (não houve P1)
     e a janela de controle (`mes_1`) está estável?
   - Distinga o tipo: PSI alto com AUC estável é *covariate shift*; AUC em queda com
     taxa de inadimplência subindo aponta *prior* e *concept drift*. Compare a faixa de
     score de 550 a 700 com a referência. A seção 8.3 do `README.md` mostra como separar
     os dois tipos por intervenção.
3. **Mitigação.** Dispare o retreino com
   `uv run credit-sentinel retrain-challenger`. Enquanto isso, mantenha o champion em
   produção, reforce a revisão manual na faixa de score mais afetada e acompanhe a
   inadimplência observada.
4. **Resolução.** Se os cinco guardrails aprovarem o challenger (AUC, Brier, contrato,
   fairness e latência), promova com `uv run credit-sentinel promote-challenger` (ou
   `retrain-challenger --promote`, que retreina e promove numa única chamada). Antes de promover, rode o
   challenger em *shadow mode* (seção 8.6 do `README.md`). Confira no
   MLflow Model Registry que o alias `champion` foi para a nova versão e que a anterior
   recebeu a tag `retired_reason`. Se algum gate reprovar, o comando termina com exit 3 e
   **nenhum alias muda**. Reprocese a janela e confirme SLO2 em `OK` na janela seguinte.
5. **Pós-mortem.** Registre a causa econômica (veja o ADR 004 como exemplo), o tempo
   entre a detecção e a promoção, o que o challenger mudou e se algum limite do
   `slo_config.yaml` precisa de revisão. Alterar limites exige decisão registrada em ADR.

<a id="p3-atencao"></a>

## P3 · Atenção (error budget alto ou SLO3 estourado)

Nada quebrou, mas a tolerância está acabando ou duas ou mais das cinco features
principais divergem estatisticamente da referência.

1. **Detecção.** Alerta P3 citando o SLO com 70 % ou mais do orçamento consumido, ou o
   SLO3 (KS) estourado.
2. **Triagem.** Veja qual SLO está em `WARN` ou `BREACHED` e a tendência entre as
   janelas. Com milhares de linhas o KS acusa diferenças minúsculas: compare o p-valor
   com a **magnitude** (PSI e Wasserstein normalizado) antes de concluir que há risco.
3. **Mitigação.** Nenhuma ação corretiva imediata. Aumente a atenção na próxima janela
   e deixe o retreino do challenger preparado.
4. **Resolução.** Encerre se a janela seguinte voltar abaixo de 70 % do orçamento.
   Se piorar e algum SLO crítico estourar, o alerta sobe para P2 e vale o runbook P2.
5. **Pós-mortem.** Só é necessário se escalar. Anote o que foi observado e, se for um
   falso positivo recorrente, proponha ajuste do limite via ADR.

<a id="p4-informativo"></a>

## P4 · Informativo

O lote cumpriu todos os objetivos, com saldo positivo de error budget. Não há ação;
o alerta existe para provar que o lote foi avaliado e quanto sobrou de orçamento.

## Onde ver o que aconteceu no MLflow

`make mlflow` abre a UI em <http://localhost:5001>. Os prints da Etapa 3 mostram o
caminho percorrido num incidente P2, em `docs/evidencias/etapa3/`:

1. **Runs do pipeline** (`mlflow_runs_monitoring.jpg`): o run pai e um run por janela,
   com PSI, saldo do orçamento e AUC lado a lado.
2. **Métricas de uma janela** (`mlflow_metricas_mes3_crise.jpg`): `psi_*`, `ks_p_*`,
   `budget_*`, latências e as tags `decision`, `git_sha` e `release`.
3. **Relatório do Evidently como artefato** (`mlflow_evidently_artefato.jpg`): o mesmo
   HTML de `reports/evidently/`, anexado ao run da janela.
4. **Registry antes e depois** (`mlflow_registry_antes.jpg` e `mlflow_registry_depois.jpg`):
   v1 `@champion` e v2 `@challenger`; depois da promoção, v2 `@champion` e a v1 com a
   tag `retired_reason=drift_mes_3_crise` (`mlflow_registry_v1_aposentada.jpg`).
