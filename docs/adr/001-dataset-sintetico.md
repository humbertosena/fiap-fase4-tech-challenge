# ADR 001 — Dataset sintético com processo gerador explícito

| Campo | Valor |
|---|---|
| Status | **Aceito** |
| Data | 2026-10-03 (Etapa 1) |
| Decisores | Humberto Sena Santos |
| Módulos afetados | `data-reference`, `drift-simulation`, `governance` |
| Relacionados | ADR 002 (contrato Pandera), ADR 003 (modelo baseline) |

## 1. Contexto

O Tech Challenge da Fase 4 pede um modelo de *credit scoring* em produção que sofre
**degradação silenciosa** por mudanças na economia (inflação, novos perfis de
clientes). O grupo precisa: (a) treinar um classificador base em um **dataset de
referência**; (b) criar um **dataset de produção** que simule a passagem do tempo
alterando a distribuição de pelo menos duas variáveis importantes; (c) detectar e
explicar essa mudança, inclusive com uma **análise causal**.

O enunciado sugere bases públicas (German Credit, Home Credit, Lending Club), com
pelo menos 5.000 amostras. Avaliamos essas opções contra três necessidades:

| Necessidade | Base pública | Gerador sintético |
|---|---|---|
| Drift **controlado e explicável** (covariate, prior e concept drift separáveis) | Drift precisa ser fabricado por reamostragem ou ruído sobre dados cujo mecanismo é desconhecido; a "causa" vira suposição | O mecanismo é conhecido: cada choque altera parâmetros nomeados do processo gerador |
| Análise causal honesta | Grafo causal só pode ser postulado | Grafo causal é o próprio processo gerador; a explicação é verificável |
| Privacidade (LGPD) | Dados reais ou pseudo-reais de terceiros, licenças variadas | Nenhum dado pessoal real; PII fictícia só onde precisamos provar o bloqueio |
| Reprodutibilidade | Download externo, versões e licenças variáveis | `make data` regenera o arquivo idêntico (mesmo SHA-256) com seed 42 |
| Contexto brasileiro (renda em BRL, CLT, regiões) | Ausente (bases alemã/americanas) | Desenhado para o cenário de uma fintech brasileira |

## 2. Decisão

Usar um **dataset sintético determinístico** gerado por um **processo gerador de
dados (DGP) explícito**, implementado em `src/credit_sentinel/data/generator.py` e
parametrizado em `config/simulation_scenarios.yaml`. O mesmo gerador produz a
referência (cenário `reference`) e, na Etapa 2, as janelas de produção (cenários
`mes_1`, `mes_2`, `mes_3_crise`), apenas sobrescrevendo parâmetros.

### 2.1 Resumo do DGP (cenário `reference`)

1. **Risco latente** `z ~ N(0, 1)` por cliente: a "verdade" não observada que move
   bureau, endividamento, utilização e atrasos.
2. **Demografia:** idade normal truncada (média 42, desvio 13, 18–85); gênero e região
   com proporções próximas às brasileiras.
3. **Emprego e renda:** situação de emprego categórica (CLT 52 %, autônomo 22 %, …);
   renda log-normal (mediana R$ 3.800) com fatores por situação e região.
4. **Bureau e endividamento:** `score_bureau = 650 − 120·z + ruído`,
   `taxa_endividamento = sigmoid(−0,6 + 0,7·z + ruído)`, utilização Beta deslocada por
   `z`, atrasos Poisson com média crescente em `z`.
5. **Operação:** valor log-normal, prazo categórico, parcela pela **tabela Price**
   (juros de 2,5 % a.m.) limitada a 60 % da renda (política de concessão).
6. **Alvo:** `inadimplente ~ Bernoulli(sigmoid(logit))`, com o logit combinando score,
   endividamento, utilização, atrasos, desemprego, imóvel, renda e comprometimento da
   renda. O intercepto `β0` é **calibrado por bisseção** para taxa base de **6 %**.
7. **Atributos sensíveis:** `genero` e `regiao` **não entram no logit**. Qualquer
   disparidade observada surge apenas de forma indireta (via renda), o que permite à
   auditoria de fairness da Etapa 4 medir efeitos de proxy sem viés injetado.
8. **Identificadores:** `id_cliente = "CLI-" + 10 hexadecimais` sorteados pelo `rng`;
   são pseudonimizados (HMAC-SHA-256 com salt) antes de qualquer saída.

O dicionário completo está em `docs/data_dictionary.md`, gerado do código.

### 2.2 Lote corrompido

Um segundo artefato, `data/corrupted/bad_batch.parquet` (600 linhas), parte de uma
amostra limpa e injeta, com seed 7, dez tipos de anomalia (idade fora do domínio,
nulos críticos, ids duplicados/nulos, categorias inválidas, violações cruzadas e uma
coluna `cpf` fictícia não prevista no contrato). Ele existe para provar que o contrato
de dados bloqueia a ingestão (ADR 002).

## 3. Consequências

**Positivas**
- O drift da Etapa 2 é **rastreável até a causa**: "renda real caiu 20 %" é um
  parâmetro, não uma interpretação; o DAG causal da Etapa 4 é verificável.
- Nenhum dado pessoal real circula no repositório; o plano LGPD trata o cenário como
  se fosse produção, sem risco de exposição.
- Reprodutibilidade bit a bit: o hash SHA-256 da referência fica registrado em
  `docs/evidencias/etapa1/registro.md`.

**Negativas e mitigações**
- **Legitimidade:** dados sintéticos podem parecer "fáceis". Mitigação: ruído no score
  (σ = 60) e no endividamento (σ = 0,5) mantêm a AUC alvo entre 0,80 e 0,90; desbalanço
  realista (6 %).
- **Generalização:** conclusões sobre o modelo não se transferem para carteiras reais.
  Mitigação: o objeto de avaliação é a **camada de sustentação** (contratos, drift,
  observabilidade, governança), que independe da origem dos dados.
- **Viés do autor:** o DGP reflete hipóteses do autor sobre crédito. Mitigação: todos
  os parâmetros estão versionados e documentados; qualquer leitor pode alterá-los.

## 4. Alternativas consideradas

- **German Credit (UCI):** 1.000 linhas, abaixo do mínimo de 5.000. Descartada.
- **Home Credit / Lending Club:** grandes e realistas, mas o drift teria de ser
  fabricado sobre um mecanismo desconhecido, a análise causal ficaria especulativa e o
  download depende de conta no Kaggle. Descartadas pela perda de controle e de
  reprodutibilidade.
