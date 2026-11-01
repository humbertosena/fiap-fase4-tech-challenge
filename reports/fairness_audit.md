# Auditoria de fairness

Todos os grupos ficaram dentro de 4/5. `genero`, `regiao` e a faixa etária (derivada de `idade`) nunca entram nas features do modelo (ADR 003); aqui servem só para medir se a decisão trata os grupos de forma parecida.

A **razão de impacto** é a taxa de aprovação do grupo dividida pela do maior grupo do atributo (a referência) e deve ficar entre 0,80 e 1,25 (regra dos 4/5). **TPR** é a fração dos inadimplentes que o modelo nega; **FPR**, a dos bons pagadores negados; a **diferença de oportunidade** é o TPR do grupo menos o da referência. O modelo nega quando o score é maior ou igual ao limiar operacional (F1).

Conjuntos de dados:

- `referencia_validacao`: validação do baseline (20 % da referência); o challenger treinou com os últimos seis meses da referência e viu parte destes clientes
- `mes_3_crise_holdout`: dias 16 a 30 de `mes_3_crise` (holdout temporal do retreino); nenhum dos modelos treinou com estes clientes

## Resumo

| Conjunto | Modelo | Atributo | Referência | n | Pior razão | Situação |
| --- | --- | --- | --- | ---: | --- | --- |
| `referencia_validacao` | champion (`baseline-1.0.0`) | `genero` | `F` | 2400 | `M` · 1,010 | dentro de 4/5 |
| `referencia_validacao` | champion (`baseline-1.0.0`) | `regiao` | `Sudeste` | 2400 | `Centro-Oeste` · 0,959 | dentro de 4/5 |
| `referencia_validacao` | champion (`baseline-1.0.0`) | `faixa_etaria` | `30-44` | 2400 | `60+` · 0,949 | dentro de 4/5 |
| `referencia_validacao` | challenger (`challenger-1.0.0`) | `genero` | `F` | 2400 | `M` · 0,993 | dentro de 4/5 |
| `referencia_validacao` | challenger (`challenger-1.0.0`) | `regiao` | `Sudeste` | 2400 | `Sul` · 1,027 | dentro de 4/5 |
| `referencia_validacao` | challenger (`challenger-1.0.0`) | `faixa_etaria` | `30-44` | 2400 | `60+` · 0,979 | dentro de 4/5 |
| `mes_3_crise_holdout` | champion (`baseline-1.0.0`) | `genero` | `F` | 2539 | `M` · 1,001 | dentro de 4/5 |
| `mes_3_crise_holdout` | champion (`baseline-1.0.0`) | `regiao` | `Sudeste` | 2539 | `Nordeste` · 0,894 | dentro de 4/5 |
| `mes_3_crise_holdout` | champion (`baseline-1.0.0`) | `faixa_etaria` | `30-44` | 2539 | `60+` · 0,980 | dentro de 4/5 |
| `mes_3_crise_holdout` | challenger (`challenger-1.0.0`) | `genero` | `F` | 2539 | `M` · 1,015 | dentro de 4/5 |
| `mes_3_crise_holdout` | challenger (`challenger-1.0.0`) | `regiao` | `Sudeste` | 2539 | `Nordeste` · 0,934 | dentro de 4/5 |
| `mes_3_crise_holdout` | challenger (`challenger-1.0.0`) | `faixa_etaria` | `30-44` | 2539 | `18-29` · 0,953 | dentro de 4/5 |

## Detalhe por grupo

### `referencia_validacao` · champion (`baseline-1.0.0`)

| Atributo | Grupo | n | Aprovação | TPR | FPR | Inadimpl. obs. | Razão de impacto | Δ oportunidade | Regra dos 4/5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `genero` | `F` | 1220 | 88,2% | 55,6% | 8,7% | 6,6% | 1,000 | +0,000 | dentro de 4/5 |
| `genero` | `M` | 1180 | 89,1% | 50,0% | 8,3% | 6,3% | 1,010 | -0,056 | dentro de 4/5 |
| `regiao` | `Centro-Oeste` | 194 | 86,1% | 46,7% | 11,2% | 7,7% | 0,959 | -0,016 | dentro de 4/5 |
| `regiao` | `Nordeste` | 631 | 86,5% | 59,0% | 10,5% | 6,2% | 0,964 | +0,107 | dentro de 4/5 |
| `regiao` | `Norte` | 210 | 89,5% | 52,6% | 6,3% | 9,0% | 0,998 | +0,044 | dentro de 4/5 |
| `regiao` | `Sudeste` | 1004 | 89,7% | 48,3% | 7,9% | 5,8% | 1,000 | +0,000 | dentro de 4/5 |
| `regiao` | `Sul` | 361 | 90,0% | 58,3% | 6,5% | 6,6% | 1,003 | +0,101 | dentro de 4/5 |
| `faixa_etaria` | `18-29` | 347 | 88,2% | 39,3% | 9,4% | 8,1% | 0,976 | -0,115 | dentro de 4/5 |
| `faixa_etaria` | `30-44` | 985 | 90,4% | 50,8% | 6,9% | 6,2% | 1,000 | +0,000 | dentro de 4/5 |
| `faixa_etaria` | `45-59` | 850 | 87,5% | 58,0% | 9,6% | 5,9% | 0,969 | +0,072 | dentro de 4/5 |
| `faixa_etaria` | `60+` | 218 | 85,8% | 68,8% | 9,9% | 7,3% | 0,949 | +0,179 | dentro de 4/5 |

### `referencia_validacao` · challenger (`challenger-1.0.0`)

| Atributo | Grupo | n | Aprovação | TPR | FPR | Inadimpl. obs. | Razão de impacto | Δ oportunidade | Regra dos 4/5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `genero` | `F` | 1220 | 92,0% | 50,6% | 5,0% | 6,6% | 1,000 | +0,000 | dentro de 4/5 |
| `genero` | `M` | 1180 | 91,4% | 43,2% | 6,3% | 6,3% | 0,993 | -0,074 | dentro de 4/5 |
| `regiao` | `Centro-Oeste` | 194 | 89,7% | 60,0% | 6,1% | 7,7% | 0,978 | +0,100 | dentro de 4/5 |
| `regiao` | `Nordeste` | 631 | 90,5% | 46,2% | 7,1% | 6,2% | 0,986 | -0,038 | dentro de 4/5 |
| `regiao` | `Norte` | 210 | 92,4% | 42,1% | 4,2% | 9,0% | 1,007 | -0,079 | dentro de 4/5 |
| `regiao` | `Sudeste` | 1004 | 91,7% | 50,0% | 5,7% | 5,8% | 1,000 | +0,000 | dentro de 4/5 |
| `regiao` | `Sul` | 361 | 94,2% | 37,5% | 3,6% | 6,6% | 1,027 | -0,125 | dentro de 4/5 |
| `faixa_etaria` | `18-29` | 347 | 90,5% | 39,3% | 6,9% | 8,1% | 0,981 | -0,083 | dentro de 4/5 |
| `faixa_etaria` | `30-44` | 985 | 92,3% | 47,5% | 5,1% | 6,2% | 1,000 | +0,000 | dentro de 4/5 |
| `faixa_etaria` | `45-59` | 850 | 91,8% | 52,0% | 5,5% | 5,9% | 0,994 | +0,045 | dentro de 4/5 |
| `faixa_etaria` | `60+` | 218 | 90,4% | 43,8% | 6,9% | 7,3% | 0,979 | -0,038 | dentro de 4/5 |

### `mes_3_crise_holdout` · champion (`baseline-1.0.0`)

| Atributo | Grupo | n | Aprovação | TPR | FPR | Inadimpl. obs. | Razão de impacto | Δ oportunidade | Regra dos 4/5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `genero` | `F` | 1276 | 74,5% | 53,5% | 21,2% | 13,3% | 1,000 | +0,000 | dentro de 4/5 |
| `genero` | `M` | 1263 | 74,5% | 52,4% | 22,0% | 11,5% | 1,001 | -0,011 | dentro de 4/5 |
| `regiao` | `Centro-Oeste` | 220 | 72,7% | 52,0% | 24,1% | 11,4% | 0,941 | +0,020 | dentro de 4/5 |
| `regiao` | `Nordeste` | 692 | 69,1% | 61,8% | 26,4% | 12,9% | 0,894 | +0,118 | dentro de 4/5 |
| `regiao` | `Norte` | 223 | 70,9% | 65,4% | 24,4% | 11,7% | 0,917 | +0,154 | dentro de 4/5 |
| `regiao` | `Sudeste` | 1039 | 77,3% | 50,0% | 18,5% | 13,5% | 1,000 | +0,000 | dentro de 4/5 |
| `regiao` | `Sul` | 365 | 80,0% | 34,3% | 18,5% | 9,6% | 1,035 | -0,157 | dentro de 4/5 |
| `faixa_etaria` | `18-29` | 348 | 74,4% | 57,5% | 21,4% | 11,5% | 0,996 | +0,020 | dentro de 4/5 |
| `faixa_etaria` | `30-44` | 1018 | 74,8% | 55,5% | 20,9% | 12,6% | 1,000 | +0,000 | dentro de 4/5 |
| `faixa_etaria` | `45-59` | 930 | 74,5% | 48,7% | 22,1% | 12,6% | 0,997 | -0,068 | dentro de 4/5 |
| `faixa_etaria` | `60+` | 243 | 73,3% | 53,3% | 23,0% | 12,3% | 0,980 | -0,021 | dentro de 4/5 |

### `mes_3_crise_holdout` · challenger (`challenger-1.0.0`)

| Atributo | Grupo | n | Aprovação | TPR | FPR | Inadimpl. obs. | Razão de impacto | Δ oportunidade | Regra dos 4/5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `genero` | `F` | 1276 | 76,3% | 60,6% | 18,0% | 13,3% | 1,000 | +0,000 | dentro de 4/5 |
| `genero` | `M` | 1263 | 77,5% | 52,4% | 18,6% | 11,5% | 1,015 | -0,082 | dentro de 4/5 |
| `regiao` | `Centro-Oeste` | 220 | 76,4% | 56,0% | 19,5% | 11,4% | 0,981 | +0,010 | dentro de 4/5 |
| `regiao` | `Nordeste` | 692 | 72,7% | 62,9% | 22,1% | 12,9% | 0,934 | +0,079 | dentro de 4/5 |
| `regiao` | `Norte` | 223 | 78,0% | 57,7% | 17,3% | 11,7% | 1,002 | +0,027 | dentro de 4/5 |
| `regiao` | `Sudeste` | 1039 | 77,9% | 55,0% | 17,0% | 13,5% | 1,000 | +0,000 | dentro de 4/5 |
| `regiao` | `Sul` | 365 | 81,9% | 48,6% | 14,8% | 9,6% | 1,052 | -0,064 | dentro de 4/5 |
| `faixa_etaria` | `18-29` | 348 | 74,1% | 67,5% | 20,5% | 11,5% | 0,953 | +0,089 | dentro de 4/5 |
| `faixa_etaria` | `30-44` | 1018 | 77,8% | 58,6% | 17,0% | 12,6% | 1,000 | +0,000 | dentro de 4/5 |
| `faixa_etaria` | `45-59` | 930 | 77,3% | 51,3% | 18,6% | 12,6% | 0,994 | -0,073 | dentro de 4/5 |
| `faixa_etaria` | `60+` | 243 | 75,7% | 56,7% | 19,7% | 12,3% | 0,973 | -0,019 | dentro de 4/5 |
