# v0.1.0-alpha — Contratos de dados e modelo baseline

**Etapa 1 do Tech Challenge Fase 4 · critério da rubrica: Validação de Dados (25 %).**

Esta release garante que dados ruins não cheguem ao modelo: um dataset de referência
sintético e reprodutível, um contrato de dados Pandera com 12 regras e circuit breaker,
e o modelo baseline que será monitorado nas próximas etapas.

## Destaques

- **Contrato de dados com 12 regras ICRA** (`config/data_contract.yaml`): schema estrito,
  chave primária, completude, domínios, razoabilidade e três regras cruzadas.
- **Circuit breaker em dois níveis:** o lote corrompido (600 linhas e dez anomalias,
  uma delas a coluna `cpf` fictícia) é **bloqueado** com exit code 2 e inventário completo das
  violações; erros pontuais de linha (até 1 %) vão para a quarentena.
- **Privacidade por padrão:** relatórios só com `id_cliente_hash` (HMAC-SHA-256) e
  varredura `scan-pii` limpa em `reports/` e `docs/`.
- **Baseline `HistGradientBoostingClassifier`:** ROC-AUC 0,829 · Brier 0,052 · KS 0,541,
  registrado no MLflow como `credit-scoring-model` com assinatura de 14 colunas.
- **Página didática interativa:** `docs/releases/etapa-1-contratos.html`, com validador
  de proposta, simulação do circuit breaker, distribuições e limiar de decisão.

## Como verificar

```bash
uv sync --all-groups
make data validate train   # validate: referência exit 0, lote corrompido exit 2
uv run pytest              # suíte completa
uv run credit-sentinel scan-pii reports docs
```

## Artefatos

`data/reference/credit_ref.parquet` (SHA-256 `cb41f7d3…4224ec4`),
`data/corrupted/bad_batch.parquet`, `reports/validation/*.json`, `models/baseline/*`,
`docs/model_card.md`, ADRs 001 a 003 e o notebook `notebooks/01_eda_and_data_quality.ipynb`.

## Limitações conhecidas

- Dados sintéticos: as métricas avaliam a camada de sustentação, não uma carteira real.
- O SLO de integridade e os alertas que consomem o relatório de validação chegam na
  Etapa 3 (`v0.3.0-rc`).
