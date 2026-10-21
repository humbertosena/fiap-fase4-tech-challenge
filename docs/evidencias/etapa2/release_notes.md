# v0.2.0-beta — Simulação temporal e detecção de drift

**Etapa 2 do Tech Challenge Fase 4 · critério da rubrica: Detecção de Drift (25 %).**

Esta release mostra a degradação acontecendo: três janelas mensais de uma crise de juros,
testes estatísticos próprios e o Evidently AI apontando o que mudou, e a queda de
performance do baseline medida contra a validação e contra um controle estável.

## Destaques

- **Três janelas de produção** em `data/production/` (5.000 clientes cada, bit a bit
  reproduzíveis): `mes_1` (controle), `mes_2` (inflação) e `mes_3_crise` (choque de juros
  com covariate, prior e concept drift). A inadimplência vai de 6,45 % a 12,96 %.
- **Testes nativos de drift:** KS com correção de Bonferroni, PSI, qui-quadrado,
  Jensen-Shannon e Wasserstein, com 100 % de cobertura de testes.
- **Controle estável, crise crítica:** `mes_1` sem nenhuma feature com PSI de 0,10 ou
  mais; `mes_3_crise` com `taxa_endividamento` (0,632) e `utilizacao_credito` (0,660) em
  nível crítico e o PSI do score predito em 0,255.
- **Evidently AI:** relatório HTML por janela, drift do alvo e do score e qualidade da
  classificação; em `mes_3_crise` marca 6 de 14 colunas, incluindo as duas críticas.
- **Degradação medida:** AUC de 0,829 (validação) para 0,764 (− 0,065; − 0,042 contra o
  controle), Brier de 0,050 para 0,103 e aprovação de 87,6 % para 74,7 %.
- **Concept drift visível:** na faixa de score de 550 a 700, a inadimplência passa de
  5,0 % para 18,9 % (3,8 vezes).
- **Detecção protegida pelo contrato:** `detect-drift` valida cada janela no
  `inference_schema`; um lote bloqueado encerra a execução com exit 2.
- **Página didática interativa:** `docs/releases/etapa-2-drift.html`, com linha do tempo
  das janelas, calculadora de PSI, mapa de calor, curva de concept drift e degradação.

## Como verificar

```bash
uv sync --all-groups
make data train            # referência e baseline (Etapa 1)
make simulate drift        # janelas de produção e relatórios de drift
uv run pytest              # suíte completa
uv run credit-sentinel scan-pii reports docs
make explainers            # regera as páginas didáticas
```

## Artefatos

`data/production/*.parquet`, `reports/evidently_drift_report.html`,
`reports/evidently_drift_summary.json`, `reports/drift_table.md`,
`reports/evidently/<janela>_drift_report.{html,json}`, `reports/predictions/*.parquet`,
`docs/adr/004-cenarios-de-drift.md` e `docs/evidencias/etapa2/registro.md`.

## Limitações conhecidas

- **Um critério do spec não é atendido:** `renda_mensal` em `mes_2` tem KS com p = 8,4e-18,
  mas PSI de 0,029 (o spec pedia 0,10 ou mais). Os fatores de renda do spec foram mantidos;
  o caso serve de exemplo para ler p-valor e magnitude juntos.
- **Cenários calibrados:** os valores iniciais do spec levavam `mes_3_crise` a 21,6 % de
  inadimplência; `gamma_concept` (1,5), a utilização (+ 0,20) e o intercepto residual
  (− 0,64) foram ajustados (ADR 004). A taxa e o AUC variam com a semente.
- O AUC de validação (0,829) vem de 2.400 linhas e é um pouco otimista; o controle
  `mes_1` (0,806) é a comparação mais justa, e ambas aparecem nos relatórios.
- Cada relatório do Evidently pesa cerca de 4,7 MB (front-end embutido); o Git guarda a
  cópia nomeada como o mesmo blob.
- SLOs, alertas e rastreamento no MLflow chegam na Etapa 3 (`v0.3.0-rc`).
