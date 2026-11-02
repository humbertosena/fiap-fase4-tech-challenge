# ADR 006 — Privacidade e governança: LGPD por desenho, fairness como controle contínuo

| Campo | Valor |
|---|---|
| Status | **Aceito** |
| Data | 2026-11-02 (Etapa 4) |
| Decisores | Humberto Sena Santos |
| Módulos afetados | `contracts` (sanitizador de PII), `observability` (logger), `governance` (auditoria de fairness) |
| Relacionados | ADR 001 (dataset sintético), ADR 002 (contrato e R01), ADR 005 (logs e G4 do retreino) |

## 1. Contexto

Um modelo de crédito decide sobre pessoas. Isso traz três obrigações que a Etapa 3 não
cobre: tratar o mínimo de dados pessoais, ter uma hipótese legal para cada tratamento e
mostrar que a decisão automatizada não discrimina grupos de forma desproporcional. A
rubrica da Fase 4 pede exatamente isso (plano de proteção de dados, base legal,
retenção e mitigação de viés).

Duas restrições moldam a resposta:

- os dados do projeto são **sintéticos** (ADR 001), então não há titulares reais. As
  políticas descrevem o **alvo de produção**; o que o repositório prova com código e
  testes é dito à parte, em cada decisão;
- o texto cita só artigos da LGPD (Lei 13.709/2018) que o autor conseguiu conferir.
  Exigências do Banco Central entram de forma genérica, sem número de resolução, porque
  uma citação imprecisa é pior do que nenhuma.

## 2. Decisão

### 2.1 Minimização: 13 features, nenhum identificador direto

O modelo recebe 13 colunas de entrada mais a derivada `comprometimento_renda`
(`valor_parcela / renda_mensal`). `id_cliente` serve à unicidade e à rastreabilidade e
nunca é feature. CPF, nome, endereço, telefone e e-mail **não existem** no contrato: a
regra R01 (schema estrito) trata qualquer coluna fora da lista como violação
estrutural, e o lote corrompido do projeto, que traz uma coluna `cpf`, termina
`BLOCKED`. A minimização deixa de ser uma intenção e vira um teste que falha.

### 2.2 Pseudonimização com HMAC-SHA-256 e mascaramento na saída

- `id_cliente` vira `id_cliente_hash = HMAC-SHA-256(salt, id)` em predições e relatórios.
  O espaço de ids (`CLI-` mais 10 dígitos hexadecimais, cerca de 1,1 × 10¹² valores) é
  pequeno o bastante para ser enumerado, então um hash **sem** chave seria reversível por
  dicionário. Com o salt guardado fora do código, quem tem só o hash não volta ao id.
- O salt vem de `CS_PII_SALT` (arquivo `.env`, fora do git; `.env.example` traz só um
  marcador). Em produção ele vive em um cofre de segredos.
- Os logs passam por um filtro que troca `CLI-xxxxxxxxxx`, CPF, e-mail e telefone por
  `***` antes de qualquer saída (ADR 005).
- `credit-sentinel scan-pii` varre relatórios, documentação e logs, e roda na CI. Um
  achado falha o job.

### 2.3 Atributos protegidos só para auditoria

`genero` e `regiao` são coletados para **medir** discriminação e ficam fora do modelo
(decisão do contrato: não são `FEATURE_COLUMNS`). `idade` é feature legítima de risco
de crédito e também é auditada em faixas etárias (18–29, 30–44, 45–59, 60+). Retirar o
atributo do modelo não garante equidade, porque outras variáveis podem carregar o mesmo
sinal (a renda, por exemplo, varia com a região). Por isso a ausência da coluna **não**
substitui a auditoria: ela é o controle que testa o resultado.

### 2.4 Base legal

| Tratamento | Hipótese | Dispositivo |
|---|---|---|
| Análise de risco e decisão de crédito (scoring), monitoramento de drift e auditoria | proteção ao crédito (base principal) | LGPD, art. 7º, X |
| Coleta dos dados da proposta, a pedido do titular | execução de contrato ou procedimentos preliminares | LGPD, art. 7º, V |
| Prevenção à fraude | legítimo interesse, com teste de balanceamento | LGPD, art. 7º, IX e art. 10 |

O art. 7º, X, é a base principal porque o tratamento existe para proteger o crédito
concedido. O inciso V cobre a coleta feita para analisar o pedido do próprio titular. O
legítimo interesse fica restrito à prevenção à fraude, que é a finalidade em que o
balanceamento é defensável (README, seção 7.3); usá-lo como base geral do scoring
enfraqueceria a defesa, porque o titular poderia contestar o balanceamento a cada
decisão.

### 2.5 Retenção, acesso e descarte por tipo de dado

Cada tipo de dado tem prazo-alvo, papel com acesso e forma de descarte (README, seção
7.4). O prazo de 5 anos para registros de operações de crédito é um **alvo**, apoiado
nas exigências de guarda do regulador e no art. 16, I, da LGPD, e deve ser confirmado
com o jurídico antes de produção. Logs operacionais têm prazo curto (90 dias), porque
não precisam durar mais do que o diagnóstico que servem. Esta é a aplicação do princípio
da necessidade (art. 6º, III).

### 2.6 Fairness como controle contínuo, não como relatório avulso

- Métrica: razão de impacto desigual (aprovação do grupo ÷ aprovação do maior grupo do
  atributo), aceitável em [0,80, 1,25], a regra dos 4/5. Complementos: TPR, FPR,
  inadimplência observada e diferença de oportunidade igual.
- A mesma implementação roda em dois lugares: o gate G4 do retreino (bloqueia a
  promoção de um challenger desbalanceado) e o comando `credit-sentinel fairness-audit`
  (relatório completo, 3 atributos × 2 modelos × 2 conjuntos de dados).
- Um grupo fora dos 4/5 **não esconde** o resultado: o relatório o marca, e o README
  registra o achado e a mitigação escolhida.

### 2.7 Decisão automatizada e revisão (art. 20)

O art. 20 dá ao titular o direito de **solicitar a revisão** de decisões tomadas
unicamente por tratamento automatizado e de receber informações claras sobre os
critérios e procedimentos usados. O projeto atende o segundo ponto com a importância
das features por permutação (`models/*/feature_importance.json`) e o model card. A
revisão por **analista humano** é uma política interna adotada aqui como boa prática de
governança; o artigo, na redação em vigor, não a exige literalmente. Razões de recusa
por cliente (os fatores que mais pesaram em cada negativa) ficam como evolução, porque
a importância por permutação é global e não explica uma decisão individual.

## 3. Alternativas consideradas

| Alternativa | Por que não |
|---|---|
| Hash SHA-256 simples do id | Reversível por enumeração do espaço de ids; o salt como chave HMAC elimina o ataque. |
| Incluir `genero` e `regiao` como features para "corrigir" o viés | Usa atributo protegido na decisão, o oposto da minimização; a correção, se necessária, vem de reponderação ou revisão de limiar com justificativa jurídica. |
| Legítimo interesse como base única | Exige balanceamento a cada tratamento e é mais frágil do que a proteção ao crédito (art. 7º, X) para o scoring. |
| Anonimizar tudo e abrir mão dos direitos do titular | Dados realmente anonimizados saem do alcance da LGPD, mas o scoring precisa ligar a decisão a uma pessoa; sem isso não há revisão nem correção. |
| Ferramenta de fairness de terceiros | Mais uma dependência para um cálculo curto, testável com dados à mão; a implementação própria fica em pandas. |

## 4. Consequências

**Positivas**

- A minimização, o bloqueio de CPF e o mascaramento dos logs são verificados por teste
  e pela CI, não só descritos.
- O gate G4 e o relatório completo não divergem, porque usam o mesmo código.
- O titular, o auditor e o analista de revisão têm um caminho documentado.

**Negativas e limitações**

- Nenhuma das políticas de retenção, RBAC e descarte é **automatizada** aqui: o
  repositório guarda arquivos e não tem usuários. Elas valem como projeto para produção.
- `id_cliente` aparece em claro nos parquets versionados, porque os dados são sintéticos
  e a pseudonimização ocorre na saída (predições, relatórios, logs). Em produção, a
  ingestão pseudonimiza antes de gravar.
- A importância por permutação não gera razões de recusa individuais.
- Um relatório de impacto à proteção de dados pessoais (art. 38) e a indicação formal do
  encarregado são passos de produção fora do escopo de um projeto acadêmico.
