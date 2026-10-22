# Atalhos finos sobre a CLI `credit-sentinel` (ver README).
.DEFAULT_GOAL := help
UV ?= uv
RUN := $(UV) run
MLFLOW_PORT ?= 5001

.PHONY: help setup lint format test cov mlflow data validate train simulate drift pipeline audit explainers reproduce

help: ## lista os alvos disponíveis
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "} {printf "  %-12s %s\n", $$1, $$2}'

setup: ## instala runtime + dev + notebook e valida o ambiente
	$(UV) sync --all-groups
	$(RUN) python scripts/validate_env.py

lint: ## ruff check + format --check
	$(RUN) ruff check .
	$(RUN) ruff format --check .

format: ## aplica correções automáticas do ruff
	$(RUN) ruff check --fix .
	$(RUN) ruff format .

test: ## suíte completa de testes
	$(RUN) pytest

cov: ## testes com cobertura mínima de 85 %
	$(RUN) pytest --cov=credit_sentinel --cov-report=term-missing --cov-fail-under=85

mlflow: ## MLflow UI local na porta 5001
	$(RUN) mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port $(MLFLOW_PORT)

data: ## gera referência e lote corrompido (Etapa 1; bit a bit idênticos)
	$(RUN) credit-sentinel generate-data --force

validate: ## valida a referência (exit 0) e o lote corrompido (exit 2 esperado)
	$(RUN) credit-sentinel validate --input data/reference/credit_ref.parquet --schema reference
	-$(RUN) credit-sentinel validate --input data/corrupted/bad_batch.parquet

train: ## valida a referência, treina o baseline e registra no MLflow (Etapa 1)
	$(RUN) credit-sentinel train

simulate: ## materializa as janelas de produção (Etapa 2; bit a bit idênticas)
	$(RUN) credit-sentinel simulate --force

drift: ## testes de drift, scores e relatórios Evidently (Etapa 2)
	$(RUN) credit-sentinel detect-drift --window all

pipeline: ## pipeline instrumentado com SLOs (Etapa 3)
	@echo "disponível na Etapa 3"

audit: ## auditoria de fairness (Etapa 4)
	@echo "disponível na Etapa 4"

explainers: ## páginas HTML didáticas das releases (docs/)
	$(RUN) credit-sentinel build-explainers

reproduce: data validate train simulate drift pipeline audit explainers ## regenera todos os artefatos
