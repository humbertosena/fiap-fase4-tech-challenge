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
  taxa base: 13.00%
  sha256: a7d3ee755d7f552e0cb1dc4b4a47f676e3e48ee9cec0193fd1bc7f7b28ee62b6
deslocamentos em relação à referência:
  indicador                                   referência       mes_1       mes_2 mes_3_crise
  taxa base                                        6.45%       6.06%       8.60%      13.00%
  renda_mensal (mediana)                        3493.390    3480.970    3115.405    2713.485
  taxa_endividamento (média)                       0.372       0.373       0.448       0.522
  utilizacao_credito (média)                       0.400       0.404       0.460       0.538
  comprometimento_renda (média)                    0.267       0.268       0.299       0.342
  situacao_emprego=desempregado                    0.071       0.072       0.085       0.118
  finalidade_credito=consolidacao_dividas          0.198       0.193       0.264       0.346
$ uv run credit-sentinel simulate                # sem --force: imutabilidade
erro: janelas já existem (mes_1, mes_2, mes_3_crise); use --force
$ uv run credit-sentinel simulate --force        # regeração bit a bit
  sha256: 9324f98110f6251aae7d575ceca1601a27f6ffe11f1b76e9e834f33cf9b0cb6e
  sha256: 053381c3b1463abd1b5f9d1fa510612146647a9f725a466d47016bee9b0dfff1
  sha256: a7d3ee755d7f552e0cb1dc4b4a47f676e3e48ee9cec0193fd1bc7f7b28ee62b6
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
| `mes_3_crise`: taxa base ≥ 0,12 (esperada 12 % a 14 %) | 13,00 % | sim |
| Concept drift: inadimplência na faixa 550–700 ≥ 2× a da referência | 18,92 % contra 4,98 % (3,8×) | sim |
| Reprodutível bit a bit | hashes idênticos em três gerações | sim |

A tabela inicial do spec foi calibrada antes de versionar os relatórios: com `γ = 1,0` e
`β0 = +0,55`, `mes_3_crise` chegava a 21,6 %. A decomposição por componente e os
valores adotados (`γ = 1,5`; `β0 = −0,57`; utilização `+0,14`) estão no ADR 004, seção
2.3. A utilização subiu de 0,12 para 0,14 depois de medir o PSI (T24): com 0,12 a feature
ficava em 0,247, abaixo do corte crítico de 0,25. Essa revisão regenerou só o
`mes_3_crise.parquet`; os hashes de `mes_1` e `mes_2` não mudaram.

### 1.2 Testes

```console
$ uv run pytest tests/unit/test_simulator.py tests/unit/test_generator.py
39 passed
$ uv run pytest tests/smoke/test_simulate_cli.py tests/integration/test_simulated_windows.py
16 passed
$ uv run pytest -m "not performance"
200 passed, 1 deselected
$ uv run ruff check . && uv run ruff format --check .
All checks passed!
```
