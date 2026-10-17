# ADR 004 — Cenários de drift: crise de juros em três janelas mensais

| Campo | Valor |
|---|---|
| Status | **Aceito** |
| Data | 2026-10-14 (Etapa 2); revisado em 2026-10-17 (seção 2.3) |
| Decisores | Humberto Sena Santos |
| Módulos afetados | `drift-simulation`, `drift-detection`, `tracking-registry` |
| Relacionados | ADR 001 (dataset sintético), ADR 003 (modelo baseline) |

## 1. Contexto

Sem dados reais de produção, a degradação precisa ser **simulada de forma controlada,
explicável e detectável**. O enunciado pede um dataset de produção com mudança
perceptível e documentada; a rubrica avalia se o drift é detectado e *explicado*. Por
isso as janelas reaproveitam o gerador da referência (ADR 001): cada uma sobrescreve só
os parâmetros que a narrativa econômica move, e o resto do mundo fica parado. Assim,
tudo o que o Evidently e os testes estatísticos encontrarem tem causa conhecida.

## 2. Decisão

### 2.1 Narrativa macroeconômica

Três meses consecutivos, logo após a referência (julho/2025 a junho/2026):

| Janela | Período | Cenário |
|---|---|---|
| `mes_1` | julho/2026 | **Controle.** Economia estável; mesma população, nova amostra (semente 43). |
| `mes_2` | agosto/2026 | **Pressão inflacionária.** Inflação corrói a renda real, os juros começam a subir e as famílias recorrem mais ao crédito. Fica na zona de atenção de propósito, para mostrar a gradação do PSI. |
| `mes_3_crise` | setembro/2026 | **Choque de juros.** A Selic alta encarece a parcela, o desemprego sobe e o endividamento das famílias estoura. O risco muda de natureza, não só de nível. |

Cada janela tem 5.000 clientes, `data_referencia` dentro do mês e semente
`42 + posição` (43, 44, 45), de modo que os lotes são distintos e reproduzíveis bit a
bit. O lote traz o `inadimplente` observado (verdade de campo) para medir a degradação
real; a inferência do pipeline usa o `inference_schema` e ignora a coluna.

### 2.2 Parâmetros e justificativa

| Parâmetro (`ScenarioParams`) | Referência | `mes_1` | `mes_2` | `mes_3_crise` | Tipo | Justificativa |
|---|---:|---:|---:|---:|---|---|
| `income_real_factor` | 1,00 | 1,00 | 0,90 | 0,80 | Covariate | A inflação reduz a renda real da carteira. |
| `debt_logit_shift` | 0 | 0 | +0,35 | +0,70 | Covariate | O endividamento das famílias cresce quando a renda não acompanha os preços. |
| `utilization_shift` | 0 | 0 | +0,06 | +0,14 | Covariate | Mais uso de limite rotativo para fechar o mês. |
| `monthly_interest` (Price) | 0,025 | 0,025 | 0,030 | 0,038 | Covariate (via `valor_parcela`) | A Selic alta encarece o crédito e eleva a parcela. |
| `employment_shares[desempregado]` | 0,07 | 0,07 | 0,09 | 0,12 | Covariate (categórica) | O emprego formal cai; a CLT absorve a diferença. |
| `purpose_shares[consolidacao_dividas]` | 0,20 | 0,20 | 0,26 | 0,34 | Covariate (categórica) | Quem já deve toma crédito para refinanciar a dívida; as demais finalidades diminuem na mesma proporção. |
| `coef_commitment` | 0,8 | 0,8 | 1,0 | 1,6 | Concept | Com a renda apertada, o peso da parcela no risco aumenta. |
| `gamma_concept` (faixa 550–700) | 0 | 0 | 0 | 1,5 | Concept | A classe média "limítrofe" deixa de se comportar como em tempos estáveis; o modelo, treinado antes, subestima esse risco. O logit sobe `0,9 × γ` na faixa. |
| `beta0_adjustment` | 0 | 0 | 0 | −0,57 | Prior (residual) | Ver a seção 2.3. |

`genero` e `regiao` seguem fora do logit: nenhum cenário altera o que é sensível.

### 2.3 Calibração: por que os valores diferem da tabela inicial do spec

A tabela inicial do spec previa `γ = 1,0` e ajustes de intercepto de **+0,30**
(`mes_2`) e **+0,55** (`mes_3_crise`), esperando taxas base de 8–9 % e 12–14 %. Os
valores nunca tinham sido medidos. Ao medir (300 mil linhas por cenário, semente 99),
os choques de covariável e de conceito já elevam a inadimplência bem além do alvo:

| Componente (acumulado) | `mes_2` | `mes_3_crise` |
|---|---:|---:|
| Referência (amostra grande; o arquivo tem 6,45 %) | 6,1 % | 6,1 % |
| + covariáveis | 7,9 % | 10,4 % |
| + `coef_commitment` | 8,0 % | 11,3 % |
| + `gamma_concept` = 1,5 | — | 18,2 % |
| + `beta0_adjustment` (valor adotado) | 8,0 % (0) | 12,4 % (−0,57) |
| Valores iniciais do spec (`γ` 1,0; `β0` +0,55) | ≈ 10 % | **21,6 %** |

Com os valores iniciais, `mes_3_crise` chegava a 21,6 % e `mes_2` passava de 9 %.
Três ajustes resolvem, e só eles foram feitos (as demais covariáveis, que determinam o
que o PSI e os testes de hipótese enxergam, ficaram como no spec):

1. **`gamma_concept` de 1,0 para 1,5.** Com 1,0 o AUC de `mes_3_crise` ficava em
   ≈ 0,77 ± 0,01 (média de 8 sementes), colado no teto de 0,779 do critério
   "AUC pelo menos 0,05 abaixo da validação". Com 1,5 fica em ≈ 0,76. É o único parâmetro que
   derruba o AUC: o `coef_commitment` maior até o eleva.
2. **`utilization_shift` de `mes_3_crise` de 0,12 para 0,14** (revisão de 17/10, depois
   de implementar o PSI). Com 0,12 a única feature com PSI crítico era
   `taxa_endividamento` (0,63): `utilizacao_credito` ficava em 0,247, a um fio do corte
   de 0,25, e `renda_mensal` em 0,16. Com 0,14 o PSI da utilização vai a 0,33 e a crise
   passa a ter duas features críticas, como o spec exige. A renda não foi tocada: tirar
   o PSI dela de 0,16 exigiria um fator de 0,72, uma queda de 30 % na mediana.
3. **`beta0_adjustment`** passa a ser o botão de calibração da taxa base. Em `mes_2`
   não é necessário (0). Em `mes_3_crise` fica em −0,57: o choque de conceito empurra
   a taxa para ≈ 18 %, e o intercepto residual negativo representa o **aperto da
   concessão** (cortes de limite e recusas) que contém a inadimplência dos contratos
   aprovados. A bisseção rodou sobre a própria janela entregue (semente e período
   fixos, taxa monotônica no intercepto) até a taxa realizada atingir o alvo de 13 %,
   com arredondamento em duas casas.

`renda_mensal` em `mes_2` merece uma nota: o fator 0,90 desloca a mediana em 11 %, o KS
rejeita a igualdade com folga (p ≈ 1e-17), mas o PSI fica em 0,03, "estável". É o caso
que justifica olhar p-valor e magnitude juntos, e o relatório da Etapa 2 o mostra assim.

O *prior probability shift* continua presente, só que como **resultado**: a taxa
observada salta de 6,5 % para 13,0 %, e é isso que o Evidently mede em
`ValueDrift(inadimplente)`. O intercepto não é a causa dele.

### 2.4 Resultados nas janelas entregues

| Medida | Referência | `mes_1` | `mes_2` | `mes_3_crise` |
|---|---:|---:|---:|---:|
| Taxa de inadimplência | 6,45 % | 6,06 % | 8,60 % | 13,00 % |
| Inadimplência na faixa 550–700 | 4,98 % | 4,90 % | 6,94 % | 18,92 % (3,8×) |
| Mediana de `renda_mensal` (÷ referência) | 1,00 | 0,996 | 0,892 | 0,777 |
| Média de `taxa_endividamento` (− referência) | 0 | +0,001 | +0,076 | +0,150 |
| Share de `desempregado` | 0,071 | 0,072 | 0,085 | 0,118 |
| AUC do baseline | 0,829 (validação) | 0,806 | 0,801 | 0,765 |

O AUC de validação (0,829) vem de 2.400 linhas e 155 inadimplentes; o desvio-padrão
dessa estimativa é de cerca de 0,025. O controle `mes_1` (0,806) é a comparação mais
justa, e por isso o relatório da Etapa 2 informa a queda contra as duas bases.

## 3. Consequências

- Os três critérios do spec são atendidos nas janelas entregues, com `mes_2` na zona de
  atenção e `mes_3_crise` crítico, e a gradação aparece na tabela de PSI.
- A taxa de 13,0 % e o AUC de 0,765 são **sensíveis à semente**: trocar a semente base
  move a taxa em ±0,5 p.p. e o AUC em ±0,01. Alterar qualquer parâmetro depois de os
  relatórios estarem versionados exige regenerar tudo (regra "perguntar antes", §8 do
  spec).
- O intercepto negativo em `mes_3_crise` precisa ser explicado ao leitor; a tabela de
  decomposição da seção 2.3 existe para isso e alimenta a página didática da Etapa 2.

## 4. Alternativas consideradas

| Alternativa | Por que não |
|---|---|
| Manter `γ = 1,0` e `β0 = +0,55` | Taxa de 21,6 % em `mes_3_crise` e `mes_2` acima de 9 %; fora do alvo do spec. |
| Manter `γ = 1,0` e só zerar o intercepto | Taxa de ≈ 15 % e AUC sem folga para o critério de queda. |
| Manter `utilization_shift` em 0,12 | PSI de `utilizacao_credito` em 0,247: sobra uma única feature crítica na crise. |
| Enfraquecer os choques de covariável | Derruba o PSI abaixo de 0,25 e torna a detecção incerta, que é justamente o que a Etapa 2 precisa demonstrar. |
| Concept drift com inversão de sinal em outras faixas | Exigiria mudar o gerador (ADR 001) sem ganho didático. |
| Aumentar `coef_commitment` para derrubar o AUC | Medido: o AUC sobe de 0,771 para 0,783 com o coeficiente em 3,2. |
