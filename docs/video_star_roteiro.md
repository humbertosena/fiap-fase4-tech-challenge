# Roteiro do vídeo STAR (5 minutos) — Credit Sentinel, Tech Challenge Fase 4

> **Tema:** sustentação de um modelo de *credit scoring* diante de uma crise de juros:
> contrato de dados, detecção de drift, observabilidade com SLOs, LGPD, fairness e análise causal.
> **Duração:** 5:00 no máximo (limite da rubrica). O cronograma fecha em 5:00 e as falas foram
> escritas para ocupar de 87 a 123 palavras por minuto, o que deixa folga para as telas.

O método STAR organiza o vídeo em quatro blocos: **Situation** (o problema), **Task** (os
requisitos), **Action** (o que foi construído, com demonstração) e **Result** (o que se mediu
e o que se aprendeu). Cada bloco abaixo tem a tela a mostrar, o artefato que a sustenta, o
comando (quando há) e a fala sugerida.

## Cronograma

| Bloco | Início | Fim | Duração | Tela | Artefato |
| :--- | :---: | :---: | ---: | :--- | :--- |
| S · Situation | 0:00 | 0:45 | 45 s | Abertura: README no GitHub e índice das releases | `README.md`, `docs/index.html` |
| T · Task | 0:45 | 1:20 | 35 s | Índice das releases: as quatro etapas | `docs/index.html` |
| A1 · Action: barreira de dados | 1:20 | 1:55 | 35 s | Terminal com o lote corrompido; depois o relatório JSON | `reports/validation/bad_batch_report.json` |
| A2 · Action: drift | 1:55 | 2:30 | 35 s | Página didática da Etapa 2: mapa de calor do PSI | `docs/releases/etapa-2-drift.html` |
| A3 · Action: observabilidade | 2:30 | 3:05 | 35 s | Painel de saúde; em seguida a interface do MLflow | `reports/health_dashboard.html`, `docs/evidencias/etapa3/mlflow_registry_depois.jpg` |
| A4 · Action: governança | 3:05 | 3:40 | 35 s | README: seção 7 (inventário, base legal, fairness) e seção 8 (DAG e intervenções) | `README.md`, `reports/fairness_audit.md` |
| R1 · Result: a degradação medida | 3:40 | 4:05 | 25 s | Painel de saúde e tabela de SLOs | `reports/slo_audit_report.json`, `reports/health_dashboard.html` |
| R2 · Result: challenger promovido | 4:05 | 4:30 | 25 s | Página da Etapa 3 (champion × challenger) e registry depois da promoção | `docs/releases/etapa-3-observabilidade.html`, `reports/gates_report.json` |
| R3 · Result: lições aprendidas | 4:30 | 4:45 | 15 s | README: tabela de intervenções (seção 8.3) | `README.md` |
| E · Encerramento | 4:45 | 5:00 | 15 s | Página inicial do repositório no GitHub | `README.md` |

Total: 300 s (5:00). Se um ensaio passar de 4:50, use os cortes da seção
"Se o tempo apertar".

## Antes de gravar

### Ambiente e comandos pré-executados

Nada que demore deve acontecer diante da câmera. Prepare tudo antes, de preferência em um
clone separado do repositório, e confira o `git status` ao final (veja o aviso abaixo).

| Passo | Comando | Para quê |
| :---: | :--- | :--- |
| 1 | `make setup` | instalar as dependências e validar o ambiente |
| 2 | `make reproduce` | regenerar dados, treino, janelas, drift, pipeline, auditoria e páginas |
| 3 | `uv run credit-sentinel retrain-challenger` | criar o challenger e o relatório dos cinco guardrails |
| 4 | `uv run credit-sentinel promote-challenger` | mover o alias `champion` para a versão 2 no MLflow |
| 5 | `make mlflow` (em segundo plano) | deixar a interface do registry aberta na porta 5001 |
| 6 | `uv run credit-sentinel scan-pii README.md docs reports logs` | garantir que nenhum dado pessoal apareça nas telas |

Só um comando roda **ao vivo**, no bloco A1: `uv run credit-sentinel validate --input
data/corrupted/bad_batch.parquet`. Ele leva cerca de 2 s e regrava
`reports/validation/bad_batch_report.json` (só o carimbo de tempo muda). Depois da gravação,
rode `git status` e descarte as alterações nos artefatos versionados com `git restore`.

### Telas, na ordem

1. **README no GitHub**, topo da página: <https://github.com/humbertosena/fiap-fase4-tech-challenge> (bloco S).
2. **Índice das releases** no GitHub Pages: <https://humbertosena.github.io/fiap-fase4-tech-challenge/> (bloco T).
3. **Terminal** na raiz do repositório, fonte grande (bloco A1).
4. **Página da Etapa 2**: <https://humbertosena.github.io/fiap-fase4-tech-challenge/releases/etapa-2-drift.html> (bloco A2).
5. **Painel de saúde**, aberto do disco (`reports/health_dashboard.html`, bloco A3 e R1).
6. **MLflow**, <http://localhost:5001>: experimento `credit-sentinel/monitoring` e Model Registry (bloco A3).
7. **README**, seções 7.1, 7.5, 8.1 e 8.3 (blocos A4 e R3).
8. **Página da Etapa 3**, parte champion × challenger (bloco R2).
9. **README no GitHub** de novo, para o encerramento (bloco E).

### Checklist de gravação

- Resolução de 1920 × 1080, fonte do terminal em 18 pt ou mais, zoom do navegador em 125 %.
- Notificações desligadas; abas já abertas na ordem acima; sem `.env` nem chaves na tela.
- Ensaiar duas vezes com cronômetro e anotar o tempo de cada bloco.
- Microfone testado; falar mais devagar nas cifras (PSI, AUC, razão de impacto).
- Exportar em MP4, publicar como **não listado** (YouTube ou Drive) e testar o link em uma janela anônima.

## Roteiro por bloco

### S · Situation (0:00–0:45, 45 s)

- **Tela:** Abertura: README no GitHub e índice das releases.
- **Artefato:** `README.md`, `docs/index.html`.
- **Comando:** nenhum (tudo já aberto).
- **Fala** (91 palavras, 121 por minuto):

> Olá, eu sou o Humberto, e este é o Credit Sentinel, meu projeto da Fase 4 do Tech Challenge. Imagine uma fintech com um modelo de score de crédito em produção. Com inflação e Selic altas, os clientes mudam, mas o modelo continua respondendo normalmente, sem erro nenhum, e decide pior. É a degradação silenciosa. O custo é financeiro, porque aprova quem vai deixar de pagar, e regulatório, porque a LGPD exige explicar a decisão e não discriminar. No meu cenário simulado, a inadimplência dobra, de seis para treze por cento.

### T · Task (0:45–1:20, 35 s)

- **Tela:** Índice das releases: as quatro etapas.
- **Artefato:** `docs/index.html`.
- **Comando:** nenhum.
- **Fala** (64 palavras, 110 por minuto):

> A tarefa tinha quatro requisitos. Primeiro, barrar dado ruim antes que ele chegue ao modelo. Segundo, detectar o drift com estatística, e não por achismo. Terceiro, observar o sistema com SLOs, orçamento de erro e alertas que digam o que fazer. E quarto, cumprir a governança: LGPD, auditoria de fairness e uma análise causal que explique por que o modelo piorou e como corrigir.

### A1 · Action: barreira de dados (1:20–1:55, 35 s)

- **Tela:** Terminal com o lote corrompido; depois o relatório JSON.
- **Artefato:** `reports/validation/bad_batch_report.json`.
- **Comando:** `uv run credit-sentinel validate --input data/corrupted/bad_batch.parquet` (ao vivo, 2 s; mostre `echo $?`, que deve imprimir 2).
- **Fala** (63 palavras, 108 por minuto):

> Começo pela barreira. Este lote traz uma coluna de CPF que não existe no contrato. Rodo o validate e o circuit breaker bloqueia: saída dois, violação estrutural nas regras R01 e R02, com vinte e sete por cento das linhas rejeitadas. Nada disso chega ao modelo. São doze regras em Pandera, e minimizar dados também é LGPD: o CPF é barrado por construção.

### A2 · Action: drift (1:55–2:30, 35 s)

- **Tela:** Página didática da Etapa 2: mapa de calor do PSI.
- **Artefato:** `docs/releases/etapa-2-drift.html`.
- **Comando:** nenhum (página aberta no navegador; passe o mouse sobre a linha de `mes_3_crise`).
- **Fala** (72 palavras, 123 por minuto):

> Agora a crise. Simulei três meses: um controle estável e dois com choque de juros e de renda. Nesta página da Etapa 2, o mapa de calor mostra o PSI de cada feature. Em setembro, o endividamento chega a zero vírgula sessenta e três e a utilização a zero vírgula sessenta e seis, os dois acima do limite crítico de zero vírgula vinte e cinco. O teste KS e o Evidently concordam.

### A3 · Action: observabilidade (2:30–3:05, 35 s)

- **Tela:** Painel de saúde; em seguida a interface do MLflow.
- **Artefato:** `reports/health_dashboard.html`, `docs/evidencias/etapa3/mlflow_registry_depois.jpg`.
- **Comando:** `make mlflow` já rodando em segundo plano (porta 5001); a imagem do registry é o plano B.
- **Fala** (51 palavras, 87 por minuto):

> Para agir, preciso observar. O pipeline grava logs em JSON com dado pessoal mascarado e mede cinco SLOs com orçamento de erro. Este painel mostra setembro com o orçamento estourado e alerta P2. No MLflow, cada janela é um run, e o registry guarda o champion e o challenger como aliases.

### A4 · Action: governança (3:05–3:40, 35 s)

- **Tela:** README: seção 7 (inventário, base legal, fairness) e seção 8 (DAG e intervenções).
- **Artefato:** `README.md`, `reports/fairness_audit.md`.
- **Comando:** nenhum (role o README até as seções 7.1, 7.5, 8.1 e 8.3).
- **Fala** (64 palavras, 110 por minuto):

> Na governança, o README traz o inventário de dados, a base legal, que é a proteção ao crédito, e o plano de retenção. A auditoria de fairness testa gênero, região e faixa etária: todos dentro da regra dos quatro quintos, com o Nordeste como ponto de atenção. E a análise causal mostra, por intervenção, que o concept drift é o que derruba o AUC.

### R1 · Result: a degradação medida (3:40–4:05, 25 s)

- **Tela:** Painel de saúde e tabela de SLOs.
- **Artefato:** `reports/slo_audit_report.json`, `reports/health_dashboard.html`.
- **Comando:** nenhum.
- **Fala** (45 palavras, 108 por minuto):

> O resultado: o AUC cai de zero vírgula oitenta e três para zero vírgula setenta e seis, a aprovação despenca e o orçamento de setembro estoura. O alerta P2 vem do PSI, mas o dano real é um concept drift que o PSI não enxerga.

### R2 · Result: challenger promovido (4:05–4:30, 25 s)

- **Tela:** Página da Etapa 3 (champion × challenger) e registry depois da promoção.
- **Artefato:** `docs/releases/etapa-3-observabilidade.html`, `reports/gates_report.json`.
- **Comando:** nenhum (o retreino e a promoção foram pré-executados).
- **Fala** (51 palavras, 122 por minuto):

> A resposta foi retreinar com janela deslizante. O challenger sobe o AUC de zero vírgula setenta e seis para zero vírgula setenta e nove, passa nos cinco guardrails e vira champion no registry. A equidade melhora: o Nordeste vai de zero vírgula oitenta e nove para zero vírgula noventa e três.

### R3 · Result: lições aprendidas (4:30–4:45, 15 s)

- **Tela:** README: tabela de intervenções (seção 8.3).
- **Artefato:** `README.md`.
- **Comando:** nenhum.
- **Fala** (26 palavras, 104 por minuto):

> Três lições. O PSI avisa, mas não explica. O limiar ajusta, mas é o retreino que corrige. E fairness é monitoramento contínuo, não um relatório único.

### E · Encerramento (4:45–5:00, 15 s)

- **Tela:** Página inicial do repositório no GitHub.
- **Artefato:** `README.md`.
- **Comando:** nenhum.
- **Fala** (25 palavras, 100 por minuto):

> O código, as páginas didáticas de cada release e os relatórios estão no GitHub. O apoio do assistente de IA está declarado no README. Obrigado!

## Se o tempo apertar

Em ordem de corte, do que menos prejudica a rubrica para o que mais prejudica:

1. **A3:** mostrar só o painel de saúde e usar a imagem `mlflow_registry_depois.jpg` no lugar
   da interface ao vivo (economiza de 10 a 15 s).
2. **A4:** cortar a frase sobre retenção e falar só de base legal e fairness (5 s).
3. **T:** reduzir os quatro requisitos a duas frases (10 s).
4. **R3:** dizer só a primeira lição.

## Plano B

| Imprevisto | O que fazer |
| :--- | :--- |
| MLflow não abre (porta 5001 ocupada) | usar `docker compose up -d mlflow` ou mostrar as imagens de `docs/evidencias/etapa3/` |
| GitHub Pages fora do ar | abrir `docs/index.html` e `docs/releases/*.html` direto do disco |
| `validate` demora ou falha ao vivo | mostrar o `reports/validation/bad_batch_report.json` e dizer que o comando termina com saída 2 |
| O ensaio passa de 5:00 | aplicar os cortes acima, sem acelerar a fala |

## Mapa para a rubrica

O vídeo vale 15 % da nota e pede "clareza na demonstração técnica da degradação do modelo e
explicação do impacto, em até 5 minutos".

| O que a rubrica pede | Onde aparece |
| :--- | :--- |
| Risco da degradação silenciosa | S |
| Requisitos técnicos (dado ruim, drift, compliance) | T |
| Ferramentas de validação e observabilidade | A1 (Pandera), A2 (PSI, KS, Evidently), A3 (SLOs, MLflow) |
| Simulação do drift e plano de governança | A2 e A4 |
| Relatório apontando o drift, alertas e lições aprendidas | R1, R2 e R3 |

## Depois de gravar

1. Publicar o vídeo e copiar o link.
2. Substituir o marcador do topo do `README.md` pelo link, em um commit
   `docs(readme): adiciona link do video star` na branch `etapa/4-governanca`.
3. Só então fazer o merge e a tag `v1.0.0-ga`: a tag precisa conter o link do vídeo.

Os mesmos tempos e blocos estão resumidos na seção 10 do `README.md`.
