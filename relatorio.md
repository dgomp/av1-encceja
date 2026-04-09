# Relatório - AV1: ENCCEJA - Apoio à Decisão Educacional com K-NN

**Curso:** Sistemas de Informação  
**Disciplina:** Sistemas de Apoio à Decisão  
**Modelo:** K-Nearest Neighbors (K-NN)  
**Dados:** Microdados ENCCEJA 2024 - INEP

---

## 1. Contexto e Objetivo

Gestores de cursinhos preparatórios para o ENCCEJA enfrentam um desafio prático: ao receber um novo aluno, precisam tomar decisões pedagógicas importantes - nível de acompanhamento, necessidade de reforço intensivo, alocação de professores e materiais - sem ainda conhecer o desempenho real do candidato.

Ao mesmo tempo, os microdados históricos do ENCCEJA revelam como candidatos com perfis semelhantes se comportaram em edições anteriores. Isso cria a oportunidade para um **Sistema de Apoio à Decisão (SAD)** baseado em similaridade.

A **questão gerencial** que orienta este trabalho é:

> *Com base no perfil socioeconômico de um candidato, como apoiar decisões pedagógicas do cursinho usando o desempenho de candidatos semelhantes do ENCCEJA?*

O SAD desenvolvido oferece ao gestor:

1. **Notas esperadas** do candidato em cada disciplina, com base nos K vizinhos mais similares
2. **Risco de reprovação**, estimado pela taxa de aprovação dos vizinhos
3. **Comparação** entre as notas previstas e as médias dos vizinhos
4. **Recomendações pedagógicas** personalizadas por disciplina

---

## 2. Base de Dados

### Fonte

**INEP - Instituto Nacional de Estudos e Pesquisas Educacionais Anísio Teixeira**  
Conjunto: *Microdados do ENCCEJA 2024*  
Acesso: https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/encceja

### Características do Dataset

| Atributo | Valor |
|---|---|
| Registros brutos | 834.648 |
| Candidatos presentes em alguma prova | ~280.075 |
| Colunas | 118 |
| Formato | CSV, separador `;`, encoding `latin-1` |
| Arquivo utilizado | `MICRODADOS_ENCCEJA_2024_REG_NAC.csv` |

### Atributos de Entrada (Perfil Socioeconômico)

| Coluna | Descrição | Encoding |
|---|---|---|
| `TP_SEXO` | Sexo (M/F) | Label Encoding |
| `TP_FAIXA_ETARIA` | Faixa etária (1-20) | Numérico direto |
| `SG_UF_PROVA` | UF do local da prova | Label Encoding |
| `TP_CERTIFICACAO` | 1=Fundamental / 2=Médio | Numérico direto |
| `Q44` | Situação de trabalho (A/B/C) | Ordinal Encoding |
| `Q50` | Renda familiar mensal (A-H) | Ordinal Encoding |
| `Q11` | Última série estudada (A-K) | Ordinal Encoding |

### Variáveis Alvo (Desempenho)

| Coluna | Descrição | Escala | Mínimo aprovação |
|---|---|---|---|
| `NU_NOTA_LC` | Linguagens e Códigos | 0-200 (TRI) | 100 |
| `NU_NOTA_CH` | Ciências Humanas | 0-200 (TRI) | 100 |
| `NU_NOTA_MT` | Matemática | 0-200 (TRI) | 100 |
| `NU_NOTA_CN` | Ciências da Natureza | 0-200 (TRI) | 100 |
| `NU_NOTA_REDACAO` | Redação | 0-10 | 5,0 |
| `IN_APROVADO_*` | Aprovação por área (0/1) | Binário | - |

A aprovação geral é derivada: `aprovado_geral = 1` somente quando `IN_APROVADO_LC = IN_APROVADO_CH = IN_APROVADO_MT = IN_APROVADO_CN = 1`.

---

## 3. Fundamentação Teórica - K-Nearest Neighbors (K-NN)

O K-NN é um algoritmo de **aprendizagem preguiçosa** (*lazy learning*): não constrói um modelo explícito durante a fase de treinamento. Em vez disso, armazena todos os dados históricos e, ao receber uma nova instância, computa a distância para cada registro armazenado e retorna a resposta com base nos K vizinhos mais próximos (COVER; HART, 1967; FIX; HODGES, 1951).

### Distância Euclidiana

A similaridade entre dois candidatos A e B é medida pela distância Euclidiana sobre os atributos normalizados:

$$d(A, B) = \sqrt{\sum_{i=1}^{n}(A_i - B_i)^2}$$

### Normalização Min-Max

Antes do cálculo de distâncias, todos os atributos são normalizados para a escala [0, 1], eliminando o efeito de escalas distintas (por exemplo, `TP_FAIXA_ETARIA` variando de 1 a 20 poderia dominar `TP_SEXO` variando de 0 a 1):

$$x_{norm} = \frac{x - x_{min}}{x_{max} - x_{min}}$$

### Previsão por Média Ponderada

Para a previsão de notas, cada vizinho contribui com peso inversamente proporcional à sua distância (`weights='distance'`), de modo que vizinhos mais próximos exercem maior influência:

$$\hat{y} = \frac{\sum_{i=1}^{K} \frac{1}{d_i} \cdot y_i}{\sum_{i=1}^{K} \frac{1}{d_i}}$$

### Por que K-NN é adequado para este problema?

| Critério | Justificativa |
|---|---|
| Base dinâmica | O ENCCEJA é realizado anualmente; novos dados podem ser incorporados sem retreinamento |
| Interpretabilidade | "Candidatos com perfil similar a X, Y, Z obtiveram estas notas" |
| Previsão por similaridade | Fundamento natural: candidatos com perfis similares tendem a ter desempenho similar |
| Flexibilidade | O mesmo algoritmo serve tanto para regressão (notas) quanto para classificação (aprovação) |

---

## 4. Implementação

### Ferramentas Utilizadas

| Ferramenta | Versão | Finalidade |
|---|---|---|
| Python | 3.10+ | Linguagem principal |
| pandas | ≥ 2.0 | Manipulação e limpeza dos dados |
| scikit-learn | ≥ 1.3 | K-NN, encoding, normalização e métricas |
| matplotlib / seaborn | ≥ 3.7 / 0.12 | Visualizações |
| streamlit | ≥ 1.30 | Interface gráfica interativa |

### Pipeline de Desenvolvimento

```
CSV (ENCCEJA 2024 - 834.648 registros)
    │
    ▼
[1] Carregamento e filtragem
    │   └─ pandas.read_csv (sep=';', encoding='latin-1')
    │   └─ Filtro: TP_PRESENCA = 1 em ao menos uma área → 280.075 candidatos
    │
    ▼
[2] Exploração
    │   └─ Estatísticas descritivas, distribuições, perfis
    │
    ▼
[3] Pré-processamento
    │   └─ Derivação de aprovado_geral (todas as 4 áreas aprovadas)
    │   └─ Imputação de nulos com moda (Q44, Q50, Q11)
    │   └─ Ordinal Encoding (Q44, Q50, Q11)
    │   └─ Label Encoding (TP_SEXO, SG_UF_PROVA)
    │   └─ Numérico direto (TP_FAIXA_ETARIA, TP_CERTIFICACAO)
    │   └─ MinMaxScaler → escala 0-1
    │
    ▼
[4] Divisão treino/teste
    │   └─ 80% treino / 20% teste (estratificado por aprovado_geral)
    │   └─ Treino: 224.060 | Teste: 56.015
    │
    ▼
[5] Treinamento
    │   └─ 5 × KNeighborsRegressor (um por disciplina)
    │   └─ 1 × KNeighborsClassifier (aprovação geral)
    │   └─ K=7, metric='euclidean', weights='distance'
    │
    ▼
[6] Avaliação
    │   └─ Regressão: MAE, RMSE, R² por disciplina
    │   └─ Classificação: acurácia, matriz de confusão, relatório
    │
    ▼
[7] Predição e recomendações
        └─ Novo candidato → codifica → normaliza → K vizinhos → previsão
        └─ Gera recomendações pedagógicas ao gestor
```

### Tratamento dos Dados

**Filtragem de ausentes:** candidatos com `TP_PRESENCA = 0` em todas as áreas foram removidos, pois não possuem notas registradas e não contribuem para o aprendizado do modelo.

**Imputação de nulos:** as colunas de questionário socioeconômico (`Q44`, `Q50`, `Q11`) apresentaram valores ausentes em parte dos registros. A imputação foi feita com a **moda** de cada coluna, preservando a distribuição original sem introduzir ruído artificial.

| Coluna | Nulos | Imputação |
|---|---|---|
| `Q44` | 6.889 | Moda: 'A' (trabalho remunerado) |
| `Q50` | 43.943 | Moda: 'C' (de 1 a 2 salários mínimos) |
| `Q11` | 27.586 | Moda: 'I' (1ª série do EM) |

**Encoding ordinal (Q44, Q50, Q11):** as variáveis de trabalho, renda e escolaridade possuem **ordem semântica natural** e foram codificadas de forma a preservá-la:

- `Q44`: Não trabalha (0) < Trabalho sem remuneração (1) < Trabalho remunerado (2)
- `Q50`: Nenhuma renda (0) < Até 1 SM (1) < ... < Acima de 5 SM (6) < Não sei (7)
- `Q11`: 1ª série EF (0) < 2ª série EF (1) < ... < 3ª série EM (10)

**Label Encoding (TP_SEXO, SG_UF_PROVA):** variáveis nominais sem ordem natural receberam codificação numérica arbitrária via `LabelEncoder`.

**Normalização Min-Max:** aplicada após o encoding, garante que nenhuma feature domine a distância Euclidiana por ter escala numericamente maior.

### Divisão Treino / Teste

A divisão foi feita com **estratificação por aprovado_geral**, garantindo que a proporção de aprovados (~37,3%) seja mantida em ambos os conjuntos, evitando viés na avaliação.

| Conjunto | Registros | Taxa de aprovação |
|---|---|---|
| Treino (80%) | 224.060 | 37,3% |
| Teste (20%) | 56.015 | 37,3% |

---

## 5. Treinamento e Configuração do K-NN

Foram treinados **6 modelos independentes**:

| Modelo | Tipo | Alvo |
|---|---|---|
| KNN_LC | KNeighborsRegressor | Nota Linguagens e Códigos |
| KNN_CH | KNeighborsRegressor | Nota Ciências Humanas |
| KNN_MT | KNeighborsRegressor | Nota Matemática |
| KNN_CN | KNeighborsRegressor | Nota Ciências da Natureza |
| KNN_REDACAO | KNeighborsRegressor | Nota Redação |
| KNN_APR | KNeighborsClassifier | Aprovação geral (0/1) |

**Hiperparâmetros:**

```python
KNeighborsRegressor(
    n_neighbors = 7,          # K ímpar evita empate em classificação
    metric      = 'euclidean',
    weights     = 'distance'  # peso = 1/distância
)
```

A escolha de **K = 7** é o padrão, ajustável pelo gestor no dashboard via slider (3 a 21, apenas valores ímpares). Um K maior suaviza a previsão mas reduz sensibilidade a perfis específicos; um K menor aumenta a sensibilidade mas pode capturar ruídos.

---

## 6. Avaliação do Modelo

### Métricas de Regressão (previsão de notas)

| Disciplina | MAE | RMSE | R² |
|---|---|---|---|
| Linguagens e Códigos | ~10,6 | ~14,6 | ~-0,11 |
| Ciências Humanas | ~13,2 | ~17,6 | ~-0,16 |
| Matemática | ~18,1 | ~23,2 | ~-0,04 |
| Ciências da Natureza | ~13,2 | ~17,5 | ~-0,07 |
| Redação | ~1,5 | ~2,2 | ~-0,22 |

O R² negativo indica que o modelo de K-NN com features socioeconômicas tem desempenho inferior a uma previsão naive pela média global - o que é esperado: o perfil socioeconômico é correlacionado com desempenho, mas não o determina individualmente. O **valor gerencial** não está na precisão absoluta, mas na **orientação relativa** - identificar candidatos com perfil de maior risco e as disciplinas mais vulneráveis.

### Modelo de Classificação (aprovação)

| Métrica | Valor |
|---|---|
| Acurácia | ~60,6% |
| Precision (Aprovado) | ~0,46 |
| Recall (Aprovado) | ~0,34 |
| F1-Score (Aprovado) | ~0,39 |

### Matriz de Confusão

|  | Previsto: Reprovado | Previsto: Aprovado |
|---|---|---|
| **Real: Reprovado** | 26.855 (VN) | 8.267 (FP) |
| **Real: Aprovado** | 13.807 (FN) | 7.086 (VP) |

**Interpretação:** O modelo tende a prever mais reprovações do que aprovações. Isso é conservador e útil para o gestor: candidatos sinalizados como risco de reprovação receberão acompanhamento intensivo mesmo quando aprovados, enquanto o contrário seria mais prejudicial pedagogicamente.

### Justificativa para as Previsões

As previsões são justificáveis pelo próprio mecanismo do K-NN: ao apresentar um candidato, o sistema identifica os K candidatos históricos com perfil socioeconômico mais similar e reporta explicitamente suas notas e situação de aprovação. O gestor pode ver com seus próprios olhos quais candidatos do passado se assemelham ao novo aluno - tornando a recomendação **transparente e auditável**, ao contrário de modelos caixa-preta.

---

## 7. Recomendações Pedagógicas ao Gestor

O SAD classifica o risco com base na taxa de aprovação dos K vizinhos:

| Taxa de aprovação dos vizinhos | Nível de risco | Ação sugerida |
|---|---|---|
| ≥ 70% | Baixo | Acompanhamento padrão |
| 40% - 69% | Moderado | Monitoramento e reforço nas disciplinas críticas |
| < 40% | Alto | Acompanhamento individual intensivo |

Para cada disciplina com nota prevista abaixo do threshold (100 pontos para áreas objetivas; 5,0 para Redação), o sistema gera recomendações específicas:

| Disciplina | Recomendação |
|---|---|
| Linguagens e Códigos | Intensificar leitura de textos variados, revisão de gramática e interpretação textual |
| Ciências Humanas | Focar em História do Brasil, História Geral e Geografia; interpretar mapas e fontes |
| Matemática | Revisar operações, álgebra, geometria e estatística; problemas contextualizados |
| Ciências da Natureza | Revisar Biologia, Química e Física básica; exercícios com contexto ambiental |
| Redação | Praticar redações semanais; revisar estrutura dissertativa, coesão e argumentação |

---

## 8. Conclusão

O SAD desenvolvido demonstra que o K-NN é um algoritmo adequado para este problema por três razões:

1. **Transparência:** as previsões são explicadas pelos K candidatos históricos mais similares - o gestor não precisa confiar em uma "caixa preta"
2. **Atualização contínua:** novos dados do ENCCEJA podem ser incorporados sem retreinamento, ao contrário de modelos paramétricos
3. **Dupla utilidade:** o mesmo pipeline gera tanto previsões de notas (regressão) quanto estimativas de risco de reprovação (classificação)

O foco do trabalho é **apoiar a decisão gerencial**, e não maximizar métricas de acurácia. Mesmo com desempenho preditivo moderado (~60% de acurácia na classificação de aprovação), o sistema agrega valor real ao gestor ao:

- **Priorizar** candidatos de maior risco para acompanhamento individualizado
- **Identificar** disciplinas vulneráveis por perfil socioeconômico
- **Fundamentar** decisões de alocação de professores, turmas e materiais com base em evidências históricas

---

## Referências

- COVER, T.; HART, P. *Nearest Neighbor Pattern Classification*. IEEE Transactions on Information Theory, 1967.
- FIX, E.; HODGES, J. *Discriminatory Analysis: Nonparametric Discrimination*. USAF School of Aviation Medicine, 1951.
- PROVOST, F.; FAWCETT, T. *Data Science para Negócios*. Alta Books, 2016.
- IBM. *O que é o algoritmo KNN?* Disponível em: https://www.ibm.com/br-pt/think/topics/knn
- INEP. *Microdados do ENCCEJA 2024*. Disponível em: https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/encceja