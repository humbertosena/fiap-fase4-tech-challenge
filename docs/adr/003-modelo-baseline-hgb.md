# ADR 003 — Modelo baseline com HistGradientBoosting, sem `class_weight`

| Campo | Valor |
|---|---|
| Status | **Aceito** |
| Data | 2026-10-10 (Etapa 1) |
| Decisores | Humberto Sena Santos |
| Módulos afetados | `baseline-model`, `drift-detection`, `tracking-registry` |
| Relacionados | ADR 001 (dataset), ADR 002 (contrato) |

## 1. Contexto

O enunciado pede um classificador base de risco de crédito em Scikit-Learn ou XGBoost.
O modelo é o objeto monitorado nas etapas seguintes: precisa ser forte o bastante para
que a degradação seja visível (AUC ≥ 0,80), produzir **probabilidades calibradas**
(Brier ≤ 0,06, já que a PD alimenta decisão e custo) e expor importância de features
para o SLO de drift das top-3 variáveis (Etapa 3).

## 2. Decisão

1. **Algoritmo:** `HistGradientBoostingClassifier` do scikit-learn dentro de um
   `Pipeline` (`ColumnTransformer` + HGB), com `max_iter=300`, `learning_rate=0.05`,
   `max_leaf_nodes=31`, `l2_regularization=1.0`, *early stopping* e `random_state=42`.
   Lida nativamente com categóricas e valores ausentes, treina em segundos e dispensa a
   dependência extra do XGBoost.
2. **Sem `class_weight="balanced"`.** Na calibração do dataset (12.000 linhas, seed 42)
   comparamos as duas opções na mesma validação:

   | Configuração | ROC-AUC | Brier |
   | :--- | ---: | ---: |
   | `class_weight="balanced"` | 0,828 | 0,135 |
   | sem `class_weight` | 0,829 | 0,052 |

   O balanceamento não melhora a ordenação (AUC igual) e distorce as probabilidades,
   reprovando o critério de Brier. O desbalanço (6 %) é tratado na **decisão**, não no
   treino: o limiar operacional maximiza o F1 e um segundo limiar minimiza o custo
   `5·FN + 1·FP`.
3. **Importância:** `permutation_importance` medida em ROC-AUC (10 repetições) na
   validação; as três primeiras (`score_bureau`, `qtd_parcelas_atraso_12m` e
   `utilizacao_credito`) alimentam o SLO 2 da Etapa 3.
4. **Assinatura e registro:** assinatura MLflow explícita (14 colunas, saída `double`),
   modelo registrado como `credit-scoring-model` no experimento
   `credit-sentinel/baseline`. O MLflow 3.x serializa modelos sklearn com `skops`, que
   exige declarar tipos confiáveis: liberamos apenas os três tipos do nosso próprio
   pipeline HGB (`TreePredictor`, `functools.partial`, `check_array`), em vez de
   recorrer ao `cloudpickle`, que executaria código arbitrário ao carregar.

## 3. Consequências

**Positivas**
- PD calibrada (Brier ≈ 0,05), o que torna a mudança de prior da Etapa 2 visível também
  como piora de calibração.
- Bundle versionado em `models/baseline/` (modelo, métricas, limiares, importância,
  assinatura) reprodutível com seed 42; o model card é atualizado pelo próprio `train`.

**Negativas e mitigações**
- Recall moderado no limiar de F1 (≈ 0,53): documentado no model card. Nesta
  referência o limiar de custo 5:1 ficou praticamente igual ao de F1 (0,124 × 0,125);
  ambos são versionados em `threshold.json` para a discussão de recalibração da Etapa 4.
- `skops_trusted_types` precisa ser revisado se o pipeline mudar; a lista é constante
  nomeada em `ml/mlflow_baseline.py`.
