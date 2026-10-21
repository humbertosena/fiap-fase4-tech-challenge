# Changelog

Todas as mudanças relevantes deste projeto são documentadas neste arquivo.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o projeto adota o [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Unreleased]

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

[Unreleased]: https://github.com/humbertosena/fiap-fase4-tech-challenge/compare/v0.2.0-beta...HEAD
[0.2.0-beta]: https://github.com/humbertosena/fiap-fase4-tech-challenge/compare/v0.1.0-alpha...v0.2.0-beta
[0.1.0-alpha]: https://github.com/humbertosena/fiap-fase4-tech-challenge/releases/tag/v0.1.0-alpha
