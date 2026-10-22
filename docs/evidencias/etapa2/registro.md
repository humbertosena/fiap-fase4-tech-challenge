# Registro de verificação — Etapa 2 (`v0.2.0-beta`)

Evidências coletadas durante a Etapa 2. Cada bloco traz o comando executado e a saída
relevante, para que qualquer pessoa repita a verificação. O ambiente é o mesmo da
Etapa 1 (`uv.lock` não mudou): Python 3.12, pandas 2.3.3, scikit-learn 1.9.1 e
Evidently 0.7.23.

## 1. Janelas de produção (T21 a T23)

Três lotes mensais de 5.000 clientes, gerados pelo mesmo processo gerador da referência
(ADR 001) com os parâmetros de `config/simulation_scenarios.yaml` (ADR 004). A semente de
cada janela é `42 + posição` (43, 44, 45).

```console
$ uv run credit-sentinel simulate
janela mes_1: data/production/mes_1.parquet (5000 linhas)
  taxa base: 6.06%
  sha256: 9324f98110f6251aae7d575ceca1601a27f6ffe11f1b76e9e834f33cf9b0cb6e
janela mes_2: data/production/mes_2.parquet (5000 linhas)
  taxa base: 8.60%
  sha256: 053381c3b1463abd1b5f9d1fa510612146647a9f725a466d47016bee9b0dfff1
janela mes_3_crise: data/production/mes_3_crise.parquet (5000 linhas)
  taxa base: 12.96%
  sha256: 7983db5b90b6a5465d248d2279042cb79fc570b84cc6244645ea060e4ffb7a69
deslocamentos em relação à referência:
  indicador                                   referência       mes_1       mes_2 mes_3_crise
  taxa base                                        6.45%       6.06%       8.60%      12.96%
  renda_mensal (mediana)                        3493.390    3480.970    3115.405    2713.485
  taxa_endividamento (média)                       0.372       0.373       0.448       0.522
  utilizacao_credito (média)                       0.400       0.404       0.460       0.595
  comprometimento_renda (média)                    0.267       0.268       0.299       0.342
  situacao_emprego=desempregado                    0.071       0.072       0.085       0.118
  finalidade_credito=consolidacao_dividas          0.198       0.193       0.264       0.346
$ uv run credit-sentinel simulate                # sem --force: imutabilidade
erro: janelas já existem (mes_1, mes_2, mes_3_crise); use --force
$ uv run credit-sentinel simulate --force        # regeração bit a bit
  sha256: 9324f98110f6251aae7d575ceca1601a27f6ffe11f1b76e9e834f33cf9b0cb6e
  sha256: 053381c3b1463abd1b5f9d1fa510612146647a9f725a466d47016bee9b0dfff1
  sha256: 7983db5b90b6a5465d248d2279042cb79fc570b84cc6244645ea060e4ffb7a69
```

### 1.1 Critérios do `SPEC-drift-simulation` nos arquivos versionados

| Critério | Resultado | Atende |
| :--- | :--- | :---: |
| Três parquets com 5.000 linhas e o schema da referência | 5.000 × 18, mesmos tipos | sim |
| Períodos corretos, depois da referência (até 2026-06-30) | 07/2026, 08/2026 e 09/2026 | sim |
| Nenhum cliente repetido entre referência e janelas | 27.000 ids únicos | sim |
| `mes_1`: médias numéricas dentro de ±3 % | maior desvio: 1,19 % (`utilizacao_credito`) | sim |
| `mes_1`: shares categóricas dentro de ±1,5 p.p. | maior desvio: 1,21 p.p. (`genero`) | sim |
| `mes_1`: taxa base entre 5 % e 7 % | 6,06 % | sim |
| `mes_2`: taxa base entre 8 % e 9 % | 8,60 % | sim |
| `mes_3_crise`: mediana de `renda_mensal` ≤ 0,85× a referência | 0,777× | sim |
| `mes_3_crise`: média de `taxa_endividamento` ≥ referência + 0,10 | + 0,150 | sim |
| `mes_3_crise`: share de `desempregado` ≥ 0,11 | 0,118 | sim |
| `mes_3_crise`: taxa base ≥ 0,12 (esperada 12 % a 14 %) | 12,96 % | sim |
| Concept drift: inadimplência na faixa 550–700 ≥ 2× a da referência | 18,87 % contra 4,98 % (3,8×) | sim |
| Reprodutível bit a bit | hashes idênticos em três gerações | sim |

A tabela inicial do spec foi calibrada antes de versionar os relatórios: com `γ = 1,0` e
`β0 = +0,55`, `mes_3_crise` chegava a 21,6 %. A decomposição por componente e os
valores adotados (`γ = 1,5`; `β0 = −0,64`; utilização `+0,20`) estão no ADR 004, seção
2.3. A utilização subiu de 0,12 para 0,20 em duas revisões, depois de medir o PSI das
features (T24) e o PSI do score (T26): com 0,12 a feature ficava em 0,247, abaixo do corte
crítico de 0,25, e com 0,14 o score ficava em 0,21. As revisões regeneraram só o
`mes_3_crise.parquet`; os hashes de `mes_1` e `mes_2` não mudaram.

### 1.2 Testes

```console
$ uv run pytest tests/unit/test_simulator.py tests/unit/test_generator.py
39 passed
$ uv run pytest tests/smoke/test_simulate_cli.py tests/integration/test_simulated_windows.py
16 passed
$ uv run pytest -m "not performance"
288 passed, 1 deselected
$ uv run ruff check . && uv run ruff format --check .
All checks passed!
```

## 2. Detecção de drift (T24 a T28)

Testes nativos (`statistical_tests.py`: KS, PSI, qui-quadrado, Jensen-Shannon e
Wasserstein), predições do baseline por janela (`predictions.py`) e relatório do Evidently
AI 0.7.23 (`evidently_runner.py`). O comando `detect-drift` valida cada janela no
`inference_schema` antes de analisar: um lote reprovado encerra a execução com exit 2 sem
gravar nada. Os carimbos de tempo dos artefatos foram fixados com `SOURCE_DATE_EPOCH`
alguns minutos antes do commit, inclusive o relógio embutido nos UUIDs v7 do HTML.

```console
$ SOURCE_DATE_EPOCH=1792541400 uv run credit-sentinel detect-drift --window all
janela mes_1 (5000 linhas)
  drift nativo: 0 feature(s): nenhuma
  PSI crítico: nenhuma
  score: P50 0.030 · P90 0.145 · P99 0.433 · PSI 0.003
  performance: AUC 0.8056 (-0.024 vs validação) · taxa base 6.06%
  Evidently: 0.0% das features · reports/evidently/mes_1_drift_report.html
janela mes_2 (5000 linhas)
  drift nativo: 7 feature(s): renda_mensal, taxa_endividamento, utilizacao_credito, valor_parcela, comprometimento_renda, situacao_emprego, finalidade_credito
  PSI crítico: nenhuma
  score: P50 0.034 · P90 0.201 · P99 0.514 · PSI 0.056
  performance: AUC 0.8012 (-0.028 vs validação · -0.004 vs controle) · taxa base 8.60%
  Evidently: 28.6% das features · reports/evidently/mes_2_drift_report.html
janela mes_3_crise (5000 linhas)
  drift nativo: 9 feature(s): renda_mensal, taxa_endividamento, utilizacao_credito, tempo_emprego_meses, valor_emprestimo, valor_parcela, comprometimento_renda, situacao_emprego, finalidade_credito
  PSI crítico: taxa_endividamento, utilizacao_credito
  score: P50 0.047 · P90 0.282 · P99 0.590 · PSI 0.255
  performance: AUC 0.7640 (-0.065 vs validação · -0.042 vs controle) · taxa base 12.96%
  Evidently: 42.9% das features · reports/evidently/mes_3_crise_drift_report.html
relatórios: reports/evidently_drift_summary.json, reports/drift_table.md
$ uv run credit-sentinel scan-pii reports docs
nenhum dado pessoal encontrado em: reports, docs
```

Artefatos versionados: `reports/evidently_drift_summary.json`, `reports/drift_table.md`,
`reports/evidently/<janela>_drift_report.{html,json}`, `reports/predictions/<janela>_scores.parquet`
e `reports/evidently_drift_report.html` (cópia idêntica do relatório de `mes_3_crise`).
Cada HTML do Evidently tem cerca de 4,7 MB, porque embute o front-end da biblioteca; o
Git guarda a cópia nomeada como o mesmo blob.

### 2.1 Critérios do `SPEC-drift-detection`

| Janela | Critério | Medido | Atende |
| :--- | :--- | :--- | :---: |
| `mes_1` | nenhuma feature com PSI ≥ 0,10 | maior PSI: 0,004 | sim |
| `mes_1` | no máximo 1 feature com KS bruto; 0 com Bonferroni | 1 (`taxa_endividamento`, p = 0,0489); 0 | sim |
| `mes_1` | Evidently com no máximo 10 % das colunas em drift | 0,0 % | sim |
| `mes_2` | `taxa_endividamento` com PSI de 0,10 ou mais | 0,171 (atenção) | sim |
| `mes_2` | `renda_mensal` com PSI de 0,10 ou mais | **0,029 (estável)** | **não** |
| `mes_2` | KS com p < 0,05 em ambas | 8,4e-18 e 4,4e-103 | sim |
| `mes_3_crise` | PSI ≥ 0,25 em pelo menos 2 features | `taxa_endividamento` 0,632 e `utilizacao_credito` 0,660 | sim |
| `mes_3_crise` | KS com p < 0,001 nessas features | 3,8e-322 e 4,4e-297 | sim |
| `mes_3_crise` | qui-quadrado significativo nas duas categóricas | 7,3e-22 e 1,0e-88 | sim |
| `mes_3_crise` | PSI do score predito ≥ 0,25 | 0,255 | sim |
| Performance | AUC de `mes_3_crise` ≤ AUC de validação − 0,05 | 0,764 contra 0,829 (− 0,065) | sim |
| Performance | taxa base de `mes_3_crise` ≥ 0,12 | 12,96 % | sim |
| Performance | P90 e P99 do score sobem de `mes_1` a `mes_3_crise` | 0,145 → 0,201 → 0,282 e 0,433 → 0,514 → 0,590 | sim |
| Concordância | interseção entre as features críticas nativas e as do Evidently | `taxa_endividamento` e `utilizacao_credito` nos dois | sim |
| Relatório | HTML abre no navegador e destaca as colunas com drift | 6 de 14 colunas em `mes_3_crise`, 0 de 14 em `mes_1`; console sem mensagens | sim |
| Privacidade | nenhum `id_cliente` em claro em `reports/` | `scan-pii` limpo; scores só com `id_cliente_hash` | sim |

**Critério não atendido: `renda_mensal` em `mes_2`.** O fator de renda do spec (0,90)
desloca a mediana em 11 %, e o KS rejeita a igualdade com folga, mas o PSI fica em 0,029.
Levar o PSI a 0,10 pediria um fator de 0,80 e, em cadeia, 0,72 em `mes_3_crise`, uma
queda de 30 % na mediana da renda. Preferimos manter os fatores do spec e registrar o
caso como exemplo de por que p-valor e magnitude devem ser lidos juntos. A gradação do
PSI aparece em `taxa_endividamento` (0,171 em `mes_2`, 0,632 em `mes_3_crise`).

### 2.2 Nativo contra Evidently

- **Concordam** nas duas features críticas, em `renda_mensal`, `comprometimento_renda`,
  `valor_emprestimo` e `finalidade_credito` (em `mes_3_crise`), no alvo (`inadimplente`,
  teste z) e no score.
- **Só os testes nativos** apontam `tempo_emprego_meses`, `valor_parcela` e
  `situacao_emprego`: o KS e o qui-quadrado enxergam deslocamentos minúsculos com 5.000
  linhas (PSI abaixo de 0,04 e Jensen-Shannon de 0,07), abaixo do corte de 0,1 que o
  Evidently usa em Wasserstein e Jensen-Shannon.
- O Evidently informa "Dataset Drift is NOT detected" em `mes_3_crise` porque a
  participação de colunas com drift (42,9 %) fica abaixo do corte padrão de 50 % do
  dataset; o veredito por coluna é o que importa aqui.
- O Evidently compara contra o **holdout de validação** (2.400 linhas), e não contra as
  12.000 da referência, porque os scores das linhas de treino são otimistas e fariam o
  controle parecer em drift. Os testes nativos de features usam a referência completa.

### 2.3 Testes

```console
$ uv run pytest tests/unit/test_statistical_tests.py --cov=credit_sentinel.drift.statistical_tests
32 passed · cobertura de statistical_tests.py: 100 %
$ uv run pytest tests/unit/test_predictions.py
12 passed · cobertura de predictions.py: 100 %
$ uv run pytest tests/integration/test_evidently_runner.py tests/integration/test_detect_drift_cli.py
14 passed
$ uv run pytest -m "not performance" --cov=credit_sentinel --cov-fail-under=85
288 passed, 1 deselected · cobertura total: 97,48 %
```

## 3. Página didática da Etapa 2 (T29)

`uv run credit-sentinel build-explainers` gera `docs/releases/etapa-2-drift.html` (92 KB, limite de
2,5 MB), regera o índice e a página da Etapa 1 (a navegação passa a linkar a Etapa 2) e copia os
relatórios de `reports/` para `docs/reports/`, onde o GitHub Pages os serve. Todos os números da
página vêm de `reports/evidently_drift_summary.json`, de `data/` e de `config/simulation_scenarios.yaml`;
o teste `test_explainer_numbers.py` confere o payload contra esses arquivos.

Verificação no Chrome (via automação), servindo `docs/` em HTTP local:

| Componente | Verificação | Resultado |
| :--- | :--- | :---: |
| Linha do tempo (botões de janela e seletor de feature) | Mês 2 e `taxa_endividamento`: KS p = 4,4e-103, PSI 0,171 (atenção) | ok |
| Calculadora de PSI (controle de deslocamento) | `taxa_endividamento` a 0 dp: 0,000; a 0,25 dp: 0,078; a 0,5 dp: 0,367 (crítico) | ok |
| Mapa de calor feature × janela | clique em `renda_mensal` no Mês 2 mostra KS, PSI 0,029, Wasserstein e o veredito do Evidently | ok |
| Concept drift (marcadores por janela) | faixa 550 a 700: 5,0 % na referência e 18,9 % no Mês 3 (3,8×) | ok |
| Degradação (seletor de métrica) | AUC: validação 0,829, Mês 1 0,806, Mês 2 0,801, Mês 3 0,764 | ok |
| Tema claro e escuro | alternância persistida; gráficos redesenhados | ok |
| Console do navegador | nenhuma mensagem de erro ou aviso | ok |
| Sem internet (cdnjs bloqueado) | as cinco tabelas equivalentes aparecem; nenhum erro de JavaScript | ok |
| Largura de 360 px | sem rolagem horizontal | ok |

O critério de aceitação "`renda_mensal` e `taxa_endividamento` com PSI ≥ 0,10 em `mes_2`" aparece
na página como **Falhou**, com a explicação; é o mesmo ponto da seção 2.1.

Capturas: `pagina_etapa2_desktop.jpg`, `pagina_etapa2_mobile.jpg` e `pagina_indice.jpg`.
