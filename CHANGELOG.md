# Changelog

Todas as mudanças relevantes deste projeto são documentadas neste arquivo.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o projeto adota o [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Unreleased]

## [1.0.0-ga] - em preparação

Etapa 4 — governança LGPD, análise causal e entrega final. A data entra na publicação da release.

### Adicionado

- Auditoria de fairness por subgrupos (`fairness-audit`, `make audit`): gênero, região e faixa etária em
  champion e challenger, em dois conjuntos de dados (12 relatórios), com razão de impacto desigual (regra dos
  4/5), TPR, FPR e diferença de oportunidade em `reports/fairness_audit.{json,md}`. `--strict` termina com
  exit 3 se algum grupo sair dos 4/5. O gate G4 do retreino passou a usar a mesma implementação.
- Ablação causal por intervenção (`causal-ablation`, `make causal`): regera a janela de crise com um choque por
  vez, nas mesmas sementes, e mede o AUC do champion (`reports/causal_ablation.{json,md}`).
- README com as seções 7 (LGPD: inventário, PETs, base legal, retenção e mitigação de viés), 8 (análise causal e
  prescrição), 9 (runbook), 10 (roteiro), 11 (decisões, estrutura, rastreabilidade e limitações) e 12
  (agradecimentos e transparência).
- ADR 006 (privacidade e governança) e runbook refinado com ligação para as seções de governança.
- Roteiro do vídeo STAR cronometrado em 5:00 (`docs/video_star_roteiro.md`) e teste que valida o cronograma.
- Página didática da Etapa 4 (fluxo LGPD, inventário, fairness por grupo, grafo causal, intervenções, retenção
  e roteiro com cronômetro) e índice com as quatro releases.
- Registro de reprodução do zero em `docs/evidencias/etapa4/registro.md` e teste das seções do README.

### Alterado

- `make reproduce` passou a incluir `audit` e `causal`.
- Contraste das páginas didáticas (tokens `--accent`, `--link` e `--good-text`), verificado com o axe-core.

## [0.3.0-rc] - 2026-10-31

Etapa 3 — observabilidade, Data SLOs e MLflow.

### Adicionado

- Logger JSON Lines (`credit_sentinel.observability.logger`) com oito campos
  obrigatórios, `StageTimer` e filtro que mascara `CLI-…`, CPF e e-mail antes de
  qualquer saída.
- Cinco Data SLOs com *error budget* normalizado em `config/slo_config.yaml`, monitor
  (`slo_monitor`) e alertas P1 a P4 com âncora no `docs/runbook_incidentes.md` (ADR 005).
- Orquestrador e subcomando `run-pipeline` (`--window`, `--quick`, `--strict`,
  `--no-mlflow`): seis estágios instrumentados por janela; o lote corrompido termina
  `BLOCKED` e inferência e drift viram `SKIPPED`.
- `reports/slo_audit_report.json` e painel de saúde autocontido
  `reports/health_dashboard.html`; amostra de 40 eventos de log em
  `docs/evidencias/etapa3/`.
- Hook do MLflow (run pai e um run aninhado por janela, com métricas de drift, saldo de
  orçamento e relatórios como artefatos); uma falha no MLflow não derruba o pipeline.
- Retreino do challenger com janela deslizante (`retrain-challenger`), cinco guardrails
  de promoção (`reports/gates_report.json`) e auditoria mínima de fairness.
- Model Registry com os aliases `champion` e `challenger`; `promote-challenger` e
  `retrain-challenger --promote` exigem os guardrails aprovados e marcam a versão
  aposentada com `retired_reason`.
- Job `smoke` na CI: pipeline em modo rápido, lote corrompido com `--strict` (exit 2) e
  varredura de PII em `reports`, `docs` e `logs`.
- Runbook de incidentes (P1, P2 e P3) e prints do MLflow UI como evidência.
- Página didática da Etapa 3 (replay do pipeline, gauges de error budget, feed de alertas,
  champion × challenger e mascaramento de PII ao vivo) e atualização do índice.

### Alterado

- `validate_batch` e o treino carimbam o tempo pelo relógio do projeto, que respeita
  `SOURCE_DATE_EPOCH`.
- `train` agora coloca o baseline no registry com o alias `champion`.

## [0.2.0-beta] - 2026-10-21

Etapa 2 — simulação temporal e detecção de drift.

### Adicionado

- Motor de janelas temporais (`credit_sentinel.drift.simulator`) e subcomando `simulate`:
  `mes_1` (controle), `mes_2` (inflação) e `mes_3_crise` (choque de juros), de 5.000
  linhas cada, em `data/production/` (ADR 004).
- Testes estatísticos nativos: KS com Bonferroni, PSI, qui-quadrado, Jensen-Shannon e
  Wasserstein normalizado, com 100 % de cobertura.
- Predições por janela com `id_cliente_hash` e medição da degradação (AUC, F1, Brier,
  percentis e taxa de aprovação) contra a validação e contra o controle.
- Integração com o Evidently AI 0.7 e subcomando `detect-drift`, que valida cada janela
  no contrato antes de analisar e grava `reports/evidently_drift_summary.json`,
  `reports/drift_table.md`, os relatórios HTML por janela e os scores pseudonimizados.
- Relógio com suporte a `SOURCE_DATE_EPOCH`, também aplicado aos UUIDs v7 do HTML.
- Página didática da Etapa 2 com cinco componentes interativos e atualização do índice.

### Alterado

- O sanitizador de PII ignora o e-mail do autor da biblioteca `deep-eql` no aviso de
  licença embutido nos relatórios do Evidently.
- `pytest` ignora três avisos de depreciação das dependências do Evidently.


## [0.1.0-alpha] - 2026-10-11

Etapa 1 — validação de dados e contratos.

### Adicionado

- Estrutura do pacote `credit_sentinel` com `uv`, `hatchling`, Ruff estrito, pytest com
  markers e workflow de CI (`ci_quality_gates.yml`).
- Configuração centralizada (`Settings`, prefixo `CS_`), sementes determinísticas e CLI
  `credit-sentinel` com subcomandos por módulo.
- Dockerfile multi-stage com usuário não-root e `docker-compose.yml` do MLflow (porta 5001).
- Gerador sintético de carteira de crédito com processo gerador explícito (ADR 001),
  `data/reference/credit_ref.parquet` (12.000 linhas) e dicionário de dados.
- Lote corrompido `data/corrupted/bad_batch.parquet` com dez anomalias propositais.
- Contrato de dados Pandera com 12 regras ICRA, exportado para `config/data_contract.yaml`.
- Circuit breaker em dois níveis com quarentena e inventário completo de violações (ADR 002).
- Sanitizador de privacidade: HMAC-SHA-256 com salt, deny-list, mascaramento e `scan-pii`.
- Modelo baseline `HistGradientBoostingClassifier` com métricas, limiares de F1 e de
  custo, importância por permutação, model card e registro no MLflow (ADR 003).
- Notebook de EDA com diagnóstico ICRA.
- Página didática da Etapa 1 e índice das releases em `docs/` (GitHub Pages).

### Corrigido

- Amostras do relatório de validação quando a única falha da linha é no `id_cliente`.

[Unreleased]: https://github.com/humbertosena/fiap-fase4-tech-challenge/compare/v1.0.0-ga...HEAD
[1.0.0-ga]: https://github.com/humbertosena/fiap-fase4-tech-challenge/compare/v0.3.0-rc...v1.0.0-ga
[0.3.0-rc]: https://github.com/humbertosena/fiap-fase4-tech-challenge/compare/v0.2.0-beta...v0.3.0-rc
[0.2.0-beta]: https://github.com/humbertosena/fiap-fase4-tech-challenge/compare/v0.1.0-alpha...v0.2.0-beta
[0.1.0-alpha]: https://github.com/humbertosena/fiap-fase4-tech-challenge/releases/tag/v0.1.0-alpha
