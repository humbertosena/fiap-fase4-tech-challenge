# Model card — `credit-scoring-model` (baseline)

| Campo | Valor |
| :--- | :--- |
| Versão | `baseline-1.0.0` (Model Registry: `credit-scoring-model`, versão 1) |
| Tipo | Classificação binária — probabilidade de inadimplência (PD) |
| Algoritmo | `HistGradientBoostingClassifier` (scikit-learn) em `Pipeline` com `ColumnTransformer` |
| Dados de treino | `data/reference/credit_ref.parquet` (12.000 propostas sintéticas, 2025-07 a 2026-06) |
| Responsável | Humberto Sena Santos |
| Decisões | ADR 001 (dataset), ADR 002 (contrato), ADR 003 (modelo) |

## 1. Finalidade e uso pretendido

Estimar a probabilidade de um cliente pessoa física ficar inadimplente (atraso > 90
dias) para apoiar a **decisão de concessão de crédito** de uma fintech fictícia. É o
modelo `@champion` cuja degradação silenciosa é estudada nas Etapas 2 e 3.

**Fora do escopo:** decisões totalmente automatizadas sem revisão humana (LGPD art. 20
garante ao titular o direito de revisão), precificação de juros e uso em carteiras reais.

## 2. Dados

- **Referência sintética** gerada por processo gerador explícito (ADR 001), sem dados
  pessoais reais; taxa base de inadimplência de 6,45 %.
- Antes do treino, a referência passa pelo **contrato de dados** (12 regras, ADR 002);
  se o contrato bloquear o lote, o treino é abortado (exit 2).
- Split estratificado 80/20 (`random_state=42`): 9.600 propostas de treino e 2.400 de
  validação.

## 3. Features

14 colunas de entrada, registradas na assinatura MLflow (`models/baseline/signature.json`):

- **Numéricas (12):** `idade`, `renda_mensal`, `score_bureau`, `taxa_endividamento`,
  `utilizacao_credito`, `tempo_emprego_meses`, `qtd_parcelas_atraso_12m`,
  `valor_emprestimo`, `prazo_meses`, `valor_parcela`, `possui_imovel` e a derivada
  `comprometimento_renda = valor_parcela / renda_mensal`.
- **Categóricas (2):** `situacao_emprego` e `finalidade_credito` (`OrdinalEncoder`,
  categorias desconhecidas viram ausentes no HGB).
- **Excluídas por desenho:** `genero` e `regiao` (atributos sensíveis, usados apenas na
  auditoria de fairness), `id_cliente`, `data_referencia` e o alvo. Um teste
  automatizado inspeciona o `ColumnTransformer` para garantir a exclusão.

## 4. Desempenho offline

<!-- metrics:start -->
Versão `baseline-1.0.0` · validação com 2400 propostas (treino: 9600) · taxa base 6,45 %

| Métrica | Valor | Critério |
| :--- | ---: | :--- |
| ROC-AUC | 0,8291 | ≥ 0,80 |
| Brier | 0,0516 | ≤ 0,06 |
| KS | 0,5411 | ≥ 0,45 |
| Average precision | 0,3217 | — |
| F1 / precisão / recall | 0,383 / 0,300 / 0,529 | limiar de F1 |

Matriz de confusão no limiar 0,125: VN=2054, FP=191, FN=73, VP=82.
Limiar de custo (FN custa 5 vezes um FP): 0,124.
Top-3 por importância de permutação (ROC-AUC): `score_bureau`, `qtd_parcelas_atraso_12m`, `utilizacao_credito`.
<!-- metrics:end -->

## 5. Limiares de decisão

- **Operacional (F1):** maximiza o F1 na validação; é o limiar usado nas métricas acima.
- **Custo (5:1):** minimiza `5·FN + 1·FP`, pois aprovar um inadimplente custa cerca de
  cinco vezes mais do que negar um bom pagador. É a base da discussão de recalibração
  da Etapa 4.

## 6. Limitações e riscos

- Treinado em dados sintéticos: as métricas medem a camada de sustentação, não o
  desempenho em uma carteira real.
- Pressupõe estabilidade da relação entre features e inadimplência; a Etapa 2 simula
  exatamente a quebra dessa hipótese (*covariate*, *prior* e *concept drift*).
- Variáveis de renda podem atuar como *proxy* de região; a auditoria de fairness da
  Etapa 4 mede o efeito indireto em `genero` e `regiao`.
- Probabilidades calibradas para a taxa base de 6 %; mudança de prior exige
  monitoramento do Brier e possível recalibração.
