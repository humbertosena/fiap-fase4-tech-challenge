# Dicionário de dados — carteira de crédito sintética

> Gerado por `credit_sentinel.data.dictionary.render_markdown()`; não edite à mão.

| Coluna | Tipo | Domínio | Papel | Descrição |
| :--- | :--- | :--- | :--- | :--- |
| `id_cliente` | string | `^CLI-[0-9A-F]{10}$`, único | identificador direto | Identificador do cliente; pseudonimizado (HMAC-SHA-256 com salt) antes de qualquer saída |
| `data_referencia` | datetime64[ns] | referência: 2025-07-01 a 2026-06-30 | janela temporal | Data da proposta de crédito |
| `idade` | int64 | 18 a 100 | feature | Idade do cliente em anos |
| `renda_mensal` | float64 | > 0 e ≤ 1.000.000 (BRL) | feature | Renda mensal declarada em reais |
| `score_bureau` | int64 | 0 a 1000 | feature | Score de birô de crédito |
| `taxa_endividamento` | float64 | 0 a 1 | feature | Comprometimento da renda com dívidas existentes |
| `utilizacao_credito` | float64 | 0 a 1 | feature | Uso do limite de crédito rotativo disponível |
| `tempo_emprego_meses` | int64 | 0 a 600; 0 se desempregado; ≤ (idade - 16) * 12 | feature | Tempo no emprego atual em meses |
| `qtd_parcelas_atraso_12m` | int64 | 0 a 12 | feature | Parcelas pagas em atraso nos últimos 12 meses |
| `valor_emprestimo` | float64 | > 0 (BRL) | feature | Valor solicitado |
| `prazo_meses` | int64 | {6, 12, 24, 36, 48, 60} | feature | Prazo do empréstimo em meses |
| `valor_parcela` | float64 | > 0 e ≤ renda_mensal (BRL) | feature | Parcela mensal calculada pela tabela Price |
| `situacao_emprego` | string | CLT, autonomo, servidor_publico, aposentado, desempregado | feature | Vínculo de trabalho atual |
| `finalidade_credito` | string | veiculo, reforma, educacao, consolidacao_dividas, saude, outros | feature | Finalidade declarada do crédito |
| `possui_imovel` | bool | verdadeiro/falso | feature | Tem imóvel |
| `genero` | string | F, M | sensível (só auditoria de fairness) | Gênero declarado; nunca usado como feature |
| `regiao` | string | Norte, Nordeste, Centro-Oeste, Sudeste, Sul | sensível (só auditoria de fairness) | Região do país; nunca usada como feature |
| `inadimplente` | int64 | {0, 1} | alvo | 1 se o cliente ficou inadimplente (atraso > 90 dias) |

Features do modelo: 13 colunas (+ derivada `comprometimento_renda`).
Atributos sensíveis (`genero`, `regiao`) ficam fora do modelo e entram apenas na auditoria de fairness.
