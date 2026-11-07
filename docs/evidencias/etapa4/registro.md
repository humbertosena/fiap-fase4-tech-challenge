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
