# Atalhos finos sobre a CLI `credit-sentinel` (ver README).
.DEFAULT_GOAL := help
UV ?= uv
RUN := $(UV) run
MLFLOW_PORT ?= 5001

.PHONY: help setup lint format test cov mlflow data train simulate drift pipeline audit explainers reproduce

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

data: ## gera referência e lote corrompido (Etapa 1)
	@echo "disponível a partir do subcomando generate-data"

train: ## treina o baseline (Etapa 1)
	@echo "disponível a partir do subcomando train"

simulate: ## materializa as janelas de produção (Etapa 2)
	@echo "disponível na Etapa 2"

drift: ## detecção de drift e relatórios Evidently (Etapa 2)
	@echo "disponível na Etapa 2"

pipeline: ## pipeline instrumentado com SLOs (Etapa 3)
	@echo "disponível na Etapa 3"

audit: ## auditoria de fairness (Etapa 4)
	@echo "disponível na Etapa 4"

explainers: ## páginas HTML didáticas das releases
	@echo "disponível a partir do subcomando build-explainers"

reproduce: data train simulate drift pipeline audit explainers ## regenera todos os artefatos
