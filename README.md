# Credit Sentinel — Tech Challenge Fase 4 (FIAP MLET)

![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Ruff](https://img.shields.io/badge/lint-ruff-261230)

Camada de **sustentação e confiabilidade** de um modelo de *credit scoring* de uma
fintech fictícia: contratos de dados que bloqueiam lotes inválidos, simulação e
detecção de *drift*, observabilidade operacional e governança sob a LGPD.

> Projeto em construção. As seções abaixo são preenchidas a cada release.

## Estado atual

| Etapa | Tema | Release | Status |
| :--- | :--- | :--- | :--- |
| 1 | Validação de dados e contratos | `v0.1.0-alpha` | em andamento |
| 2 | Simulação e detecção de drift | `v0.2.0-beta` | planejada |
| 3 | Observabilidade de pipelines e modelos | `v0.3.0-rc` | planejada |
| 4 | Governança (LGPD) e vídeo STAR | `v1.0.0-ga` | planejada |

## Início rápido

```bash
uv sync --all-groups
uv run ruff check . && uv run ruff format --check .
uv run pytest
```

## Estrutura

```text
src/credit_sentinel/   # pacote Python (CLI `credit-sentinel`)
tests/                 # unit, integration, schema, smoke, performance
```

## Licença

Distribuído sob a licença MIT. Consulte [LICENSE](LICENSE).
