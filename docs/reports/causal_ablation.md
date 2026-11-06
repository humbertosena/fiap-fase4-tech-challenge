# Ablação causal por intervenção

Cada linha regera a janela `mes_3_crise` (5000 clientes) com as mesmas 10 sementes (45 a 54), mudando só os choques indicados, e mede o modelo `baseline-1.0.0` (sem retreino). `only` parte da referência e liga os choques; `without` parte da crise e os desliga. A variação de AUC é pareada com o controle da mesma semente (média ± desvio-padrão).

| Cenário | Modo | Parâmetros | Taxa base | AUC | Δ AUC vs. controle |
| --- | --- | --- | ---: | ---: | ---: |
| Controle (sem nenhum choque) | only | — | 5,99 % | 0,795 | — |
| Crise completa | without | — | 12,22 % | 0,757 | − 0,038 ± 0,012 |
| Só covariate shift (6 choques) | only | `income_real_factor`, `debt_logit_shift`, `utilization_shift`, `monthly_interest`, `employment_shares`, `purpose_shares` | 10,69 % | 0,785 | − 0,010 ± 0,012 |
| Só renda + endividamento + utilização | only | `income_real_factor`, `debt_logit_shift`, `utilization_shift` | 9,80 % | 0,781 | − 0,014 ± 0,012 |
| Só concept drift (peso da parcela + faixa 550–700) | only | `coef_commitment`, `gamma_concept` | 10,75 % | 0,759 | − 0,036 ± 0,011 |
| Só a faixa 550–700 | only | `gamma_concept` | 10,32 % | 0,752 | − 0,043 ± 0,011 |
| Só juros + peso da parcela | only | `monthly_interest`, `coef_commitment` | 6,49 % | 0,800 | + 0,005 ± 0,005 |
| Só desemprego | only | `employment_shares` | 6,48 % | 0,799 | + 0,003 ± 0,006 |
| Crise sem concept drift | without | `coef_commitment`, `gamma_concept` | 6,49 % | 0,793 | − 0,002 ± 0,007 |
| Crise sem o risco extra da faixa 550–700 | without | `gamma_concept` | 7,21 % | 0,797 | + 0,001 ± 0,005 |
| Crise sem o aperto da concessão | without | `beta0_adjustment` | 18,89 % | 0,756 | − 0,039 ± 0,015 |
