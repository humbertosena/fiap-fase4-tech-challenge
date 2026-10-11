# Changelog

Todas as mudanças relevantes deste projeto são documentadas neste arquivo.

O formato segue o [Keep a Changelog](https://keepachangelog.com/pt-BR/1.1.0/)
e o projeto adota o [Versionamento Semântico](https://semver.org/lang/pt-BR/).

## [Unreleased]

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

[Unreleased]: https://github.com/humbertosena/fiap-fase4-tech-challenge/compare/v0.1.0-alpha...HEAD
[0.1.0-alpha]: https://github.com/humbertosena/fiap-fase4-tech-challenge/releases/tag/v0.1.0-alpha
