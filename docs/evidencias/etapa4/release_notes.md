# Etapa 4 — Governança LGPD, análise causal e entrega final

Esta release (`v1.0.0-ga`) fecha o Tech Challenge da Fase 4: entram a governança de dados sob a LGPD, a auditoria de
fairness por subgrupos, a análise causal da degradação, o runbook, o roteiro do vídeo STAR e o guia de reprodução.
O link do vídeo é adicionado pelo autor, em um commit próprio, antes da tag. Não entram um serviço de inferência nem
as rotinas de produção de retenção, que ficam descritas como política.

Página didática: <https://humbertosena.github.io/fiap-fase4-tech-challenge/releases/etapa-4-governanca.html>

## Entregue

- **Auditoria de fairness** — `credit-sentinel fairness-audit`: gênero, região e faixa etária em champion e challenger,
  em dois conjuntos de dados (12 relatórios), com razão de impacto desigual (regra dos 4/5), TPR, FPR e diferença de
  oportunidade. Todos os grupos ficam dentro de 4/5; a pior razão é 0,894 (Nordeste, champion, crise) e o challenger a
  leva a 0,934. O gate G4 do retreino usa a mesma implementação.
- **Governança LGPD no README** — inventário e minimização das 18 colunas, PETs (HMAC-SHA-256 com salt, mascaramento,
  deny-list e `scan-pii`), base legal (art. 7º, X, V e IX), direitos do titular (art. 18), revisão de decisão
  automatizada (art. 20) e plano de retenção por tipo de dado; ADR 006.
- **Análise causal** — grafo causal, evidências por janela e **intervenções no simulador** (`credit-sentinel
  causal-ablation`): o concept drift sozinho derruba o AUC em 0,036, e o covariate shift sozinho, em 0,010 ± 0,012
  (ruído). Na faixa de score de 550 a 700, o champion aprova 120 inadimplentes onde esperava 44; o challenger, 104
  onde esperava 62. A seção termina com a política de mitigação.
- **Runbook e roteiro** — runbook P1 a P3 refinado com a ligação para a análise causal e roteiro STAR de 5:00, com
  telas, artefatos, comandos pré-executados e plano B.
- **Página didática da Etapa 4** — fluxo LGPD em 8 passos, inventário, fairness por grupo, grafo causal em SVG,
  intervenções, linha do tempo de retenção e roteiro com cronômetro (7 componentes interativos); índice com as quatro
  releases.
- **Rastreabilidade** — matriz com a rubrica, limitações e agradecimentos (com a menção ao apoio do assistente de IA)
  no README.

## Verificações executadas

| Comando | Resultado |
| :--- | :--- |
| `uv run pytest --cov=credit_sentinel --cov-fail-under=85` | 539 testes passando, cobertura de 98,1 % |
| `uv run ruff check .` e `uv run ruff format --check .` | limpos no repositório inteiro |
| `git clone` + `uv sync --all-groups` + `make reproduce` em diretório limpo | todos os passos com exit 0, cerca de 85 s |
| `uv run credit-sentinel fairness-audit` | 12 relatórios, todos dentro de 4/5 |
| `uv run credit-sentinel causal-ablation` | 11 cenários × 10 sementes em 4 s |
| `uv run credit-sentinel scan-pii README.md docs reports logs` | nenhum dado pessoal |
| axe-core 4.9.1 na página da Etapa 4 (tema claro e escuro) | 0 violações |

Evidências completas em `docs/evidencias/etapa4/registro.md`.

## Decisões de escopo

- A ablação causal virou um comando com testes e artefato versionado, para que os números da seção 8 do README e da
  página sejam reproduzíveis; no spec original eram só texto.
- O conjunto da crise na auditoria é o holdout (dias 16 a 30), porque o challenger treinou nos dias 1 a 15.
- O art. 20 da LGPD dá direito de pedir **revisão**; a revisão por analista humano é política interna do projeto.
- A acessibilidade foi medida com o axe-core (o motor que o Lighthouse usa), não com o Lighthouse.

## Limites conhecidos

- Os dados são sintéticos: a análise causal mostra o que o modelo faz diante das hipóteses do gerador, não como a
  economia real se comporta.
- Retenção, controle de acesso, expurgo, cofre de segredos, revisão humana e *shadow mode* são política, não código.
- Sem `CS_PII_SALT`, o código usa um salt de desenvolvimento em vez de abortar.
- O retreino é disparado pelo operador; o alerta P2 é automático.
- A execução em Ubuntu só é observada na CI do GitHub, depois do push.
