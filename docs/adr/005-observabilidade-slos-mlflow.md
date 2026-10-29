# ADR 005 — Observabilidade: Data SLOs, error budget e rastreio no MLflow

| Campo | Valor |
|---|---|
| Status | **Aceito** |
| Data | 2026-10-24 (Etapa 3) |
| Decisores | Humberto Sena Santos |
| Módulos afetados | `observability`, `tracking-registry` |
| Relacionados | ADR 002 (circuit breaker), ADR 003 (modelo baseline), ADR 004 (cenários de drift) |

## 1. Contexto

Até a Etapa 2 o projeto **detecta** o problema: o contrato bloqueia o lote ruim e os
testes de drift mostram que o mundo mudou. Falta o que a operação precisa para agir:
um critério objetivo de "está bom ou ruim", um jeito de medir **quanto** da tolerância já
foi gasta e um lugar onde o histórico das execuções possa ser consultado. A rubrica
chama isso de observabilidade e logs; o mercado, de SRE aplicado a dados.

## 2. Decisão

### 2.1 Cinco Data SLOs

Cada lote (janela) é medido por cinco indicadores (SLIs), todos declarados em
`config/slo_config.yaml`. Mudar um limite é mudar o arquivo, não o código.

| SLO | SLI | Meta | Por que esse limite |
|---|---|---|---|
| `SLO1_integridade` | fração de linhas rejeitadas pelo contrato | ≤ 1 % | É o limite do circuit breaker (ADR 002). |
| `SLO2_estabilidade_psi` | maior PSI entre as 3 features mais importantes | ≤ 0,25 | 0,25 é o PSI "crítico" da indústria; só vale olhar quem move o modelo. |
| `SLO3_consistencia_ks` | fração das 5 numéricas principais com KS significativo (Bonferroni) | ≤ 0,34 | Tolera 1 em 5 (falso positivo) e estoura com 2 ou mais. |
| `SLO4_performance` | ROC-AUC do lote, quando há verdade de campo | ≥ 0,75 | Piso operacional abaixo do AUC de validação (0,829). |
| `SLO5_latencia` | duração do processamento do lote | ≤ 60 s | Folga larga: o objetivo é flagrar travamentos, não micro-otimizar. |

As features do SLO2 saem de `models/baseline/feature_importance.json`
(`score_bureau`, `qtd_parcelas_atraso_12m` e `utilizacao_credito`); um teste falha se a
lista do YAML sair do top-3 do modelo. O exemplo do spec citava `taxa_endividamento` e
`comprometimento_renda`, mas o spec também pedia que a lista viesse do modelo, e o top-3
medido é outro.

**SLO4 usa ROC-AUC, não acurácia.** O plano original pedia "acurácia ≥ 80 %". Com
inadimplência de 6 % um modelo que aprova todo mundo acerta 94 %, então o indicador não
discrimina nada. O ROC-AUC é insensível ao desbalanceamento e já é a métrica de
validação do baseline (ADR 003).

### 2.2 Error budget normalizado

O consumo de cada SLO vira um número comparável entre indicadores com unidades
diferentes:

- objetivo `<=`: `consumo = observado / limite`;
- objetivo `>=`: `consumo = (referência − observado) / (referência − limite)`, com a
  referência igual ao AUC de validação do baseline (0,829). Um AUC igual ao limite
  consome 1,0 do orçamento; um AUC igual ou melhor que a referência, 0,0.

| Consumo | Estado do SLO |
|---|---|
| menor que 0,70 | `OK` |
| de 0,70 até menos de 1,0 | `WARN` |
| 1,0 ou mais | `BREACHED` |

O **saldo do orçamento** do lote é `1 − média dos consumos` dos SLOs avaliados; saldo
negativo significa orçamento estourado. A média simples é didática e vira um *gauge*
direto na página e no painel de saúde.

Decisões de borda, todas cobertas por testes:

- o consumo de um objetivo `>=` nunca fica negativo (AUC melhor que a referência não
  "devolve" orçamento);
- SLOs que dependem de insumos que o lote não produziu (drift e AUC de um lote
  bloqueado) **não entram** na média, em vez de contar como zero;
- o consumo não é truncado: um lote bloqueado com 27 % de linhas rejeitadas consome
  27 vezes o SLO1, e o saldo negativo grande é a leitura correta de "lote inutilizável".

### 2.3 Alertas P1 a P4

As regras ficam no YAML e são avaliadas da mais grave para a mais leve:

| Severidade | Gatilho | Ação |
|---|---|---|
| **P1** | o contrato bloqueou o lote | acionar o runbook, isolar o lote e acionar o produtor dos dados |
| **P2** | `SLO2` ou `SLO4` estourado | drift crítico: disparar o retreino do challenger |
| **P3** | qualquer SLO com consumo ≥ 0,70 | atenção: monitorar a próxima janela |
| **P4** | nenhuma regra anterior | informativo |

Cada SLO gera no máximo um alerta, na maior severidade em que se enquadra (um SLO2
estourado é P2, não P2 mais P3). Um lote bloqueado gera **apenas** o P1: os demais
SLOs nem foram medidos e o estouro do SLO1 é consequência do bloqueio, não um segundo
incidente. Cada alerta aponta para uma âncora do `docs/runbook_incidentes.md`.

### 2.4 Logs, pipeline e rastreio

- **Logs em JSON Lines**, um evento por linha, com oito campos obrigatórios
  (`timestamp`, `run_id`, `stage`, `status`, `records_processed`, `rejected_records`,
  `latency_ms`, `model_version`). Um filtro mascara `CLI-xxxxxxxxxx`, CPF e e-mail antes
  da gravação e do stderr; `logs/` fica fora do git e a evidência versionada é uma
  amostra já mascarada.
- **Orquestrador** executa seis estágios por janela (ingestão, validação do contrato,
  inferência, drift, SLO e tracking). Um lote bloqueado termina `BLOCKED` e os estágios
  seguintes viram `SKIPPED`: o bloqueio é um resultado esperado, não uma falha do
  pipeline (por isso o `exit 0`, e o `--strict` para quem quer `exit 2`).
- **MLflow** guarda o histórico (um run pai por execução e um run aninhado por janela,
  com métricas de drift, consumo de orçamento e os relatórios como artefatos). Um hook
  falho **não derruba** o pipeline: o estágio `tracking` registra `FAILED` e a execução
  segue, porque observar não pode ser o motivo de parar de operar.
- **Painel de saúde em HTML** autocontido, gerado do mesmo `AuditReport`, para quem só
  precisa de uma foto do estado sem abrir o MLflow.

## 3. Consequências

- Os limites são **calibrados para o cenário simulado** (ADR 004): `mes_1` fica só com
  alerta informativo, `mes_2` em atenção e `mes_3_crise` estoura o SLO2. Em produção real
  os limites precisariam de histórico para serem recalibrados.
- A latência é um valor **medido**, então `slo_audit_report.json` não é idêntico bit a
  bit entre execuções; o estado de cada SLO é estável porque a latência fica centenas
  de vezes abaixo do limite.
- O saldo médio mistura SLOs de naturezas diferentes. É uma simplificação assumida: o
  estado por SLO e os alertas continuam sendo a fonte de decisão.
- Mudar `slo_config.yaml` depois de os relatórios da release estarem versionados exige
  regenerar a auditoria (regra "perguntar antes", §8 do spec).

## 4. Alternativas consideradas

| Alternativa | Por que não |
|---|---|
| Acurácia ≥ 80 % no SLO4 | Com 6 % de inadimplência não discrimina nada (seção 2.1). |
| Alertar por limiar fixo de PSI, sem orçamento | Não diz o quanto da tolerância restou nem permite comparar indicadores. |
| Saldo acumulado entre os quatro lotes | O lote bloqueado, com consumo de 27, dominaria a média; cada lote tem o próprio saldo. |
| Truncar o consumo em 2,0 | Esconde a gravidade do lote bloqueado e, no `mes_3_crise`, deixaria o saldo positivo. |
| Um alerta por SLO em todas as severidades | Triplica o ruído: o mesmo estouro apareceria como P2 e P3. |
| Prometheus e Grafana | Exigem serviços no ar; o desafio pede evidência reproduzível com `docker compose` mínimo. |
| Logs em texto livre | Impedem validar campos e varrer PII por regra; o JSON Lines permite as duas coisas. |

## 5. Retreino do challenger e guardrails (revisão de 2026-10-28)

### 5.1 Protocolo

O challenger só existe porque o drift foi detectado, então ele é treinado **depois** da
detecção, com o que já tem verdade de campo:

| Conjunto | Dados |
|---|---|
| Treino (janela deslizante) | últimos 6 meses da referência (jan a jun/2026) + `mes_1` + `mes_2` + dias 1 a 15 de `mes_3_crise` |
| Holdout temporal | dias 16 a 30 de `mes_3_crise` (nunca visto no treino) |

Features, pipeline e hiperparâmetros são os do baseline (ADR 003); o champion e o
challenger são comparados **no mesmo holdout**. O treino interno usa o mesmo split 80/20
do baseline para calcular limiares e importância.

### 5.2 Por que a primeira quinzena da crise entra no treino

O spec original treinava o challenger só com a referência recente, `mes_1` e `mes_2`, e
avaliava em `mes_3_crise` inteiro. Medido nos dados do projeto, esse desenho entrega
cerca de **+0,015 de AUC** sobre o champion, abaixo dos +0,02 exigidos pelo gate G1. A
razão é estrutural: o concept drift (ADR 004) só existe em `mes_3_crise`, e um modelo que
nunca viu a crise não aprende a nova relação entre as variáveis e o risco. Com a primeira
quinzena da crise no treino, o ganho no holdout é de **+0,028**.

A decisão foi tomada pelo responsável do projeto em 2026-10-28, entre quatro opções:

| Opção | Resultado |
|---|---|
| **Primeira quinzena da crise no treino, holdout na segunda** (adotada) | G1 aprova; reflete o que acontece na operação (retreinar com o que já foi rotulado) |
| Spec literal | G1 reprova (+0,015): o guardrail bloquearia o challenger e não haveria promoção na demo |
| Spec literal com grade de hiperparâmetros | Sem garantia de chegar a +0,02 e com risco de ajustar contra o próprio holdout |
| Reduzir o limiar do G1 para +0,01 | Enfraquece o guardrail só para o challenger passar |

### 5.3 Guardrails

| Gate | Regra | Observação |
|---|---|---|
| G1 | AUC do challenger ≥ AUC do champion + 0,02 | no holdout temporal |
| G2 | Brier do challenger ≤ Brier do champion | calibração não pode piorar |
| G3 | assinatura de entrada idêntica (nomes e tipos) | o consumidor não muda nada |
| G4 | razão de impacto na taxa de aprovação de cada grupo em [0,80, 1,25] | `genero`, `regiao` e `faixa_etaria`; o grupo de referência é o maior de cada atributo |
| G5 | inferência de 5.000 linhas em até 5 s | amostra do holdout com reposição |

`genero` e `regiao` continuam **fora** das features: só aparecem aqui, para medir o
resultado da decisão. A auditoria completa (taxas de erro por grupo, causas) é da
Etapa 4; o G4 usa a versão mínima de `governance/fairness_audit.py`.

### 5.4 Consequências

- O `gates_report.json` registra o protocolo (linhas por conjunto e datas do holdout),
  para que o leitor saiba exatamente o que foi comparado.
- O ganho depende do dia de corte: com o corte mais cedo haveria menos dados da crise
  e o G1 poderia reprovar. O corte no dia 15 é uma convenção, não um parâmetro tunado.
- Em produção real o rótulo de inadimplência chega com atraso de meses; a premissa de
  ter a primeira quinzena rotulada é uma simplificação do cenário simulado.
