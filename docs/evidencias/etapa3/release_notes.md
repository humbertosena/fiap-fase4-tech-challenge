# v0.3.0-rc — Observabilidade, Data SLOs e MLflow

**Etapa 3 do Tech Challenge Fase 4 · critério da rubrica: Observabilidade e Logs (20 %).**

Esta release dá à operação o que ela precisa para agir quando o drift aparece: logs que
contam a história sem expor dados pessoais, metas objetivas (Data SLOs) com *error
budget*, alertas com ação recomendada e um registro claro de qual modelo está em
produção, com o challenger promovido só depois de cinco guardrails.

## Destaques

- **Logs JSON estruturados** com oito campos obrigatórios e um filtro que mascara
  identificador de cliente, CPF e e-mail antes do arquivo e do stderr.
- **Cinco Data SLOs** (integridade, estabilidade PSI, consistência KS, performance
  ROC-AUC e latência) com consumo de orçamento normalizado e estados `OK`, `WARN` e
  `BREACHED` (ADR 005).
- **Alertas P1 a P4 com runbook:** o lote corrompido gera P1; `mes_1`, só o P4
  informativo; `mes_2`, P3; `mes_3_crise`, P2 com ação de retreino e saldo de orçamento
  de − 17 %.
- **Pipeline instrumentado** (`run-pipeline`): seis estágios por janela; o lote
  corrompido termina `BLOCKED` e inferência e drift viram `SKIPPED`. As quatro janelas
  rodam em cerca de 7 s, com o Evidently.
- **MLflow:** um run pai por execução e um run por janela, com métricas de drift
  (`psi_*`, `ks_p_*`), saldo do orçamento e os relatórios como artefatos. Uma falha no
  MLflow vira `FAILED` no estágio `tracking` e o pipeline segue.
- **Champion × challenger:** retreino com janela deslizante; AUC de 0,7605 para 0,7883
  (+ 0,0279) no holdout temporal da crise, e os cinco guardrails aprovam (AUC, Brier,
  contrato, fairness e latência). A promoção move o alias `champion` e marca a versão
  antiga com `retired_reason`.
- **Painel de saúde e página didática:** `reports/health_dashboard.html` (autocontido) e
  `docs/releases/etapa-3-observabilidade.html`, com replay do pipeline, gauges de error
  budget, feed de alertas, registry antes e depois e o mascaramento de PII ao vivo.

## Como verificar

```bash
uv sync --all-groups
make data train simulate      # referência, baseline e janelas (Etapas 1 e 2)
make pipeline                 # logs, SLOs, alertas, painel de saúde e runs no MLflow
uv run credit-sentinel retrain-challenger
uv run credit-sentinel promote-challenger
make mlflow                   # UI em http://localhost:5001
uv run pytest                 # suíte completa
uv run credit-sentinel scan-pii reports logs docs
make explainers               # regera as páginas didáticas
```

## Artefatos

`reports/slo_audit_report.json`, `reports/health_dashboard.html`,
`reports/gates_report.json`, `models/challenger/*`, `config/slo_config.yaml`,
`docs/runbook_incidentes.md`, `docs/adr/005-observabilidade-slos-mlflow.md`,
`docs/evidencias/etapa3/` (prints do MLflow UI, amostra de log e registro) e a página
`docs/releases/etapa-3-observabilidade.html`.

## Limitações conhecidas

- **Protocolo do retreino ajustado:** treinado como o spec descreve (sem dados da crise),
  o challenger ganhava só + 0,015 de AUC, abaixo dos + 0,02 do guardrail. A primeira
  quinzena de `mes_3_crise` entrou no treino e a segunda é o holdout (ADR 005, seção 5);
  em produção real o rótulo chega com atraso de meses.
- **Limites calibrados para o cenário simulado:** os SLOs separam bem as quatro janelas
  deste projeto; em dados reais precisariam de histórico para serem recalibrados.
- **O saldo médio mistura SLOs de naturezas diferentes:** é uma simplificação didática; o
  estado por SLO e os alertas continuam sendo a fonte de decisão.
- **A latência é medida:** `slo_audit_report.json` e `gates_report.json` não são idênticos
  byte a byte entre execuções, embora o estado de cada SLO e de cada gate seja estável.
- **Carimbos do MLflow UI são reais** (09/11/2026): ele grava o relógio de parede.
- A governança LGPD, a auditoria de fairness completa e a análise causal chegam na
  Etapa 4 (`v1.0.0-ga`).
