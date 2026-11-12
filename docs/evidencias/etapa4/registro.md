# Registro de verificação: página didática da Etapa 4

Página: [`docs/releases/etapa-4-governanca.html`](../../releases/etapa-4-governanca.html), gerada por
`credit-sentinel build-explainers` a partir de `reports/fairness_audit.json`,
`reports/causal_ablation.json`, `reports/evidently_drift_summary.json`,
`reports/slo_audit_report.json`, `config/simulation_scenarios.yaml`, dos modelos e de
`docs/video_star_roteiro.md`. O texto jurídico e a política de retenção estão em
`src/credit_sentinel/explainers/static/etapa4_content.yaml`.

## Como foi verificado

O Chrome da automação não abre `file://`, e subir um servidor local só para isso foi descartado.
A página foi aberta pelo Chrome 156 em modo headless, direto do disco, com scripts de teste injetados
em **cópias** da página (nada disso vai para o HTML publicado).

| Verificação | Como | Resultado |
| :--- | :--- | :--- |
| Interação dos componentes | 17 asserções em uma cópia da página: passo do fluxo LGPD, filtros do inventário, seletores de fairness, nós e janelas do grafo, slider de retenção, cronômetro e tabela das intervenções | 17 de 17 em 1280 px, 768 px e 390 px |
| Largura de celular de verdade | a cópia dentro de um `iframe` de 390 px (a janela headless não desce de cerca de 500 px) | sem rolagem horizontal; `scrollWidth` igual a 390 |
| Acessibilidade | axe-core 4.9.1 (regras WCAG 2 A e AA, WCAG 2.1 A e AA e boas práticas), nos temas claro e escuro | 0 violações nos dois temas |
| Funções puras do JavaScript | asserções em Node sobre `blockAt`, `retentionState`, `ratioPos`, `clock`, `useKind` e `daysLabel` | todas aprovadas |
| Tamanho | `ls -l` da página | 111 KB (limite de 2,5 MB) |
| Números iguais aos artefatos | `tests/integration/test_explainer_numbers.py` e `tests/smoke/test_build_explainers.py` | verdes |
| Dados pessoais | `credit-sentinel scan-pii docs` | nenhum achado |

**Sobre o Lighthouse.** O critério do spec fala em Lighthouse de acessibilidade com nota de 90 ou mais. O
Lighthouse não foi executado (a ferramenta não está instalada e não foi baixada). A categoria de
acessibilidade dele é calculada com o axe-core, o mesmo motor usado aqui, que encontrou 0 violações; isso
sustenta a nota, mas **não é a nota**. Quem quiser a pontuação oficial pode abrir a página no Chrome e rodar
o Lighthouse pelo DevTools.

## O que a verificação de acessibilidade corrigiu

O primeiro passe do axe-core achou três tipos de problema, todos corrigidos antes do commit:

- **Contraste (`color-contrast`).** O azul de destaque (`#2a78d6`) dava 4,2:1 sobre o fundo claro e o texto
  branco sobre ele dava 4,4:1, abaixo dos 4,5:1 exigidos; no tema escuro, o verde dos selos também. Os tokens
  compartilhados em `explainer.css` mudaram (`--accent`, novo `--link`, `--good-text`), então as páginas das
  Etapas 1 a 3 foram regeneradas com as mesmas cores melhores. Nenhum texto ou dado mudou nelas, só o
  link para a Etapa 4 na navegação e o hash do commit no rodapé.
- **`nested-interactive`.** O grafo SVG tinha `role="img"` e nós focáveis dentro; virou `role="group"`.
- **`heading-order`.** Os cartões da retenção usavam `h4` logo depois de um `h2`; viraram `h3`.

## Prints

- [`pagina_etapa4_fairness.jpg`](pagina_etapa4_fairness.jpg): fairness por grupo (holdout da crise, região, champion).
- [`pagina_etapa4_grafo_causal.jpg`](pagina_etapa4_grafo_causal.jpg): grafo causal com o nó "Concept drift" selecionado.
- [`pagina_etapa4_mobile.jpg`](pagina_etapa4_mobile.jpg): a página em 390 px.

## Componentes interativos (7, o mínimo é 3)

1. Fluxo LGPD em 8 passos, com artefato, comando de verificação e a marca "implementado" ou "política".
2. Inventário das 18 colunas com filtro por uso no modelo.
3. Fairness por grupo: seletores de conjunto, modelo e atributo, com a faixa dos 4/5.
4. Grafo causal em SVG, com seleção de nó e de janela e evidências numéricas.
5. Intervenções: gráfico de barras (Chart.js, com tabela equivalente se o Chart.js faltar).
6. Linha do tempo da retenção: slider de dias e atalhos.
7. Roteiro STAR com cronômetro de 5 minutos.

## Reprodução do zero

Feita em um clone limpo (`git clone` do repositório local, commit `66db984`, em um diretório novo), seguindo o
guia da seção 3 do README: `cp .env.example .env`, `uv sync --all-groups` e depois cada alvo do
`make reproduce`, cronometrado em separado. MacBook com Apple Silicon e cache de pacotes do `uv` aquecido; a
instalação sem cache depende da rede e não foi medida.

| Passo | Tempo | Saída |
| :--- | ---: | :--- |
| `uv sync --all-groups` | 3 s | ambiente criado |
| `make data` | 43 s (4 s na 2ª execução) | 12.000 linhas, taxa base 6,45 %, SHA-256 igual ao do README |
| `make validate` | 7 s | referência aprovada; lote corrompido `BLOCKED` (exit 2, ignorado pelo `make`) |
| `make train` | 11 s | baseline treinado e registrado |
| `make simulate` | 3 s | `mes_1`, `mes_2` e `mes_3_crise` |
| `make drift` | 5 s | testes de drift e relatórios do Evidently |
| `make pipeline` | 6 s | logs, SLOs, alertas, painel de saúde e runs no MLflow |
| `make audit` | 3 s | 12 relatórios de fairness |
| `make causal` | 4 s | 11 cenários, 10 sementes |
| `make explainers` | 3 s | quatro páginas e o índice |
| **Soma** | **cerca de 85 s** | todos com exit 0 |

No clone, `uv run pytest` passou com **530 testes em 92 s**.

**O que mudou em relação aos arquivos versionados.** `data/`, `models/*/model.joblib` e `models/challenger/` ficam
idênticos byte a byte (o único campo que muda em `models/baseline/metrics.json` é `trained_at`). Mudam os
relatórios que carregam carimbo de tempo, `run_id` ou latência medida (validação, drift, SLOs, painel de saúde,
fairness e ablação) e as predições, porque o guia manda copiar `.env.example`, cujo salt é diferente do salt de
desenvolvimento com que os artefatos versionados foram gerados. Os números que importam (AUC, PSI, razões de
impacto) são os mesmos.

**O que a reprodução apontou.** `make lint` falhou no clone: o `ruff format --check .` também formata os trechos
Python dos arquivos Markdown, e dois trechos do README (seções 7.2 e 7.5) tinham comentários alinhados com
espaços extras. A falha apareceria na CI do GitHub. Foi corrigida neste commit, e agora `ruff check .` e
`ruff format --check .` passam no repositório inteiro.

**Ubuntu.** A execução em Ubuntu acontece na CI do GitHub depois do push e ainda não foi observada.

## Revisão de conformidade com a rubrica

Cada critério da seção 9 do spec foi conferido com um comando ou um artefato. O que depende do push, do vídeo
ou da tag fica marcado como pendente, sem ser dado como feito.

| Critério | Peso | O que foi verificado | Resultado | Estado |
| :--- | ---: | :--- | :--- | :---: |
| Validação de dados | 25 % | `credit-sentinel validate --input data/corrupted/bad_batch.parquet`; `uv run pytest -m schema`; `config/data_contract.yaml` | exit 2 e `BLOCKED`; 8 regras violadas (R01, R02, R03, R04, R07, R08, R10 e R11); 21 testes de schema passam; contrato com 12 regras | ✅ |
| Detecção de drift | 25 % | `reports/evidently_drift_summary.json` e `reports/drift_table.md` | 6 de 14 colunas com drift em `mes_3_crise` (mais o alvo e o score); tabela com p(KS) e PSI por feature e janela; `data/reference` separada de `data/production` | ✅ |
| Observabilidade e logs | 20 % | amostra de log, `slo_audit_report.json`, painel de saúde e prints do MLflow | 4 janelas (1 `BLOCKED` e 3 `SUCCESS`); 40 eventos com os 8 campos; painel gerado; 6 prints em `docs/evidencias/etapa3/` | ✅ |
| Governança (LGPD) | 15 % | README seções 7 a 9, `fairness_audit.json` e o sanitizador | 12 relatórios com `passed: true`; sanitizador coberto por `tests/unit/test_privacy_sanitizer.py` | ✅ |
| Vídeo STAR | 15 % | `docs/video_star_roteiro.md` e o link no README | roteiro de 5:00 com teste de cronograma verde; vídeo publicado no YouTube (<https://youtu.be/IYjAByEJ048>, título "FIAP Fase 4 Tech Challenge Pitch", confirmado pelo oEmbed do YouTube em 11/11/2026); link em destaque no topo do README, no commit `docs(readme): adiciona link do video star` | ✅ |
| `git clone` + `uv sync` + `make reproduce` | | seção "Reprodução do zero" acima | exit 0 em todos os passos, cerca de 85 s no macOS; no Ubuntu, a CI do GitHub roda `uv sync --locked`, ruff, pytest e o smoke do pipeline (execução 43126795080 na branch `etapa/4-governanca`, 11/11/2026, três jobs verdes), mas não o `make reproduce` inteiro | ✅ macOS, ✅ Ubuntu (parcial) |
| CI verde na `main` | | `ruff check .`, `ruff format --check .`, `pytest --cov-fail-under=85`, smoke do pipeline e `scan-pii` | tudo verde localmente (541 testes, 98,11 % de cobertura) e na CI da branch `etapa/4-governanca` (execução 43126795080: lint, testes e smoke com `scan-pii`); a execução na `main` é conferida depois do merge | ✅ branch |
| 4 tags e 4 GitHub Releases | | `git tag -l` e `gh release list` | 3 tags e 3 releases publicadas (`v0.1.0-alpha`, `v0.2.0-beta`, `v0.3.0-rc`); a GA espera o vídeo | ⏳ |
| `docs/index.html` e 4 páginas no Pages | | `docs/` e o Pages | índice e as 4 páginas versionados; as Etapas 1 a 3 já estão no ar, a 4 sobe com o push | ⏳ |
| Sem temporários, segredos ou PII no histórico | | `git log --all --name-only`, `git log --all --format=%B` e `scan-pii` | 244 caminhos distintos, nenhum `.env`, `mlruns/`, `logs/`, `SPEC*`, `tasks/`, `.plan/` ou `.claude/`; maior arquivo de 4,6 MB (relatório do Evidently); 77 commits de um só autor e nenhum trailer de coautoria; `scan-pii` limpo em README, CHANGELOG, `docs`, `reports`, `config`, `logs` e `notebooks` | ✅ |
| Agradecimentos e transparência sobre o assistente de IA | | README, seção 12 | presente, coberta por `tests/smoke/test_readme_sections.py` | ✅ |
| Tag `v1.0.0-ga` depois do commit do vídeo | | | pendente | ⏳ |

**Limpeza.** Foram removidos os arquivos `.DS_Store` e um CSV solto da ablação; `logs/` e `mlruns/` ficam locais e
fora do git (o estado do MLflow serve à demonstração do vídeo). A versão do pacote passou de `0.1.0` para `1.0.0`
e a tag `release` gravada nos runs do MLflow, de `v0.3.0-rc` para `v1.0.0-ga`, com um teste que impede os dois de
ficarem para trás da última release.
