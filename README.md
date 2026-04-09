# AV1 - ENCCEJA: Apoio à Decisão Educacional com K-NN

**Disciplina:** Sistemas de Apoio à Decisão  
**Curso:** Sistemas de Informação

---

![Visão geral do projeto](projeto.png)

## Apresentação

[![Assistir no YouTube](https://img.shields.io/badge/YouTube-Assistir%20Apresentação-red?logo=youtube)](https://youtu.be/cr882EdOB54)

---

## Sobre o Projeto

Sistema de Apoio à Tomada de Decisão para gestores de cursinhos preparatórios para o **ENCCEJA** (Exame Nacional para Certificação de Competências de Jovens e Adultos).

Com base no **perfil socioeconômico** de um novo candidato (coletado no ato da matrícula) e nos **microdados históricos do ENCCEJA 2024**, o sistema utiliza o algoritmo **K-Nearest Neighbors (K-NN)** para:

- Prever as **notas esperadas** do candidato em cada disciplina
- Estimar o **risco de reprovação**
- Identificar os **K candidatos mais similares** do histórico
- Gerar **recomendações pedagógicas** personalizadas ao gestor

**Questão gerencial:**
> *Com base no perfil socioeconômico de um candidato, como apoiar decisões pedagógicas do cursinho usando o desempenho de candidatos semelhantes do ENCCEJA?*

---

## Modelo

- **Algoritmo:** K-Nearest Neighbors (K-NN) - `KNeighborsRegressor` e `KNeighborsClassifier` (scikit-learn)
- **Estratégia:** Lazy learning (aprendizagem preguiçosa)
- **Distância:** Euclidiana
- **Ponderação:** Vizinhos mais próximos recebem maior peso (`weights='distance'`)
- **Normalização:** Min-Max (equaliza escalas antes do cálculo de distâncias)
- **K padrão:** 7 vizinhos (ímpar para evitar empate; ajustável no dashboard)
- **Split:** 80% treino / 20% teste (estratificado por aprovação)

### Por que K-NN?

| Critério | Justificativa |
|---|---|
| Base de dados dinâmica | O ENCCEJA é realizado anualmente; novos dados podem ser incorporados sem retreinamento |
| Interpretabilidade | "Candidatos com perfil similar a X, Y, Z obtiveram estas notas" |
| Previsão por similaridade | Fundamento natural para o problema: candidatos similares tendem a ter desempenho similar |

### Como funciona o K-NN

```
1. Receber perfil socioeconômico do candidato (sexo, idade, UF, renda, escolaridade...)
2. Normalizar os atributos com Min-Max (0 a 1)
3. Calcular distância Euclidiana para todos os registros históricos
4. Selecionar os K vizinhos mais próximos
5. Calcular a média ponderada das notas dos vizinhos → previsão
6. Analisar a taxa de aprovação dos vizinhos → risco
7. Gerar recomendações pedagógicas
```

**Distância Euclidiana:**
```
d(A, B) = √[ (A₁−B₁)² + (A₂−B₂)² + ... + (Aₙ−Bₙ)² ]
```

**Normalização Min-Max:**
```
x_norm = (x − x_min) / (x_max − x_min)
```

---

## Estrutura do Projeto

```
av1-encceja/
├── data/                          # pasta para o CSV (não versionado)
├── output/                        # imagens geradas pelo script
│   ├── distribuicao_notas.png
│   ├── matriz_confusao.png
│   ├── desempenho_disciplinas.png
│   └── aprovacao_por_perfil.png
├── analise_encceja.py             # script principal documentado
├── analise_encceja.ipynb          # notebook Jupyter (apresentação)
├── dashboard.py                   # interface gráfica interativa (Streamlit)
├── relatorio.md                   # texto detalhando o processo de desenvolvimento
├── requirements.txt
└── README.md
```

---

## Como Executar

### 1. Clonar o repositório

```bash
git clone https://github.com/dgomp/av1-encceja.git
cd av1-encceja
```

### 2. Criar e ativar ambiente virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

### 4. Obter os dados

Acesse o portal do INEP e faça o download dos **Microdados do ENCCEJA 2024**:

- https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/encceja

Descompacte o arquivo e salve o CSV em `data/DADOS/MICRODADOS_ENCCEJA_2024_REG_NAC.csv` (caminho esperado pelo script e pelo dashboard).

> Consulte o **Dicionário de Dados** e o **Leia-me** incluídos no pacote de download para entender a estrutura das variáveis.

### 5. Executar

```bash
# Script Python (análise completa + visualizações)
python analise_encceja.py

# Notebook Jupyter (apresentação interativa)
jupyter notebook analise_encceja.ipynb

# Interface gráfica interativa (recomendado para apresentação)
streamlit run dashboard.py
```

#### Alternativa: Google Colab

Abra o `analise_encceja.ipynb` diretamente no [Google Colab](https://colab.research.google.com/). O notebook detecta automaticamente o ambiente e oferece duas opções:

- **Opção A (upload direto):** exibe botão para selecionar o `MICRODADOS_ENCCEJA_2024_REG_NAC.csv` do seu computador
- **Opção B (Google Drive):** descomente as linhas `drive.mount` na mesma célula e ajuste o caminho do arquivo no Drive

---

## Dados

| Item | Detalhe |
|---|---|
| Fonte | INEP - Instituto Nacional de Estudos e Pesquisas Educacionais |
| Exame | ENCCEJA - Exame Nacional para Certificação de Competências |
| Ano | 2024 |
| Arquivo principal | `data/DADOS/MICRODADOS_ENCCEJA_2024_REG_NAC.csv` |
| Registros brutos | ~834.000 inscritos |
| Candidatos presentes | ~280.000 (usados no modelo) |
| Formato | CSV com separador `;` e encoding `latin-1` |
| Escala das notas objetivas | 0 a 200 (TRI) |
| Escala da redação | 0 a 10 |
| Threshold de aprovação | ≥ 100 pontos por área objetiva |

> O arquivo CSV **não é versionado** no repositório por conta do tamanho. Siga as instruções acima para obtê-lo.
> Download direto: https://download.inep.gov.br/microdados/microdados_encceja_2024.zip

---

## Features Utilizadas

### Atributos de Entrada (Perfil Socioeconômico)

| Coluna real no CSV | Descrição | Valores |
|---|---|---|
| `TP_SEXO` | Sexo | M / F |
| `TP_FAIXA_ETARIA` | Faixa etária | 1-20 (menor de 17 a maior de 70) |
| `SG_UF_PROVA` | UF do local da prova | AC, AL, AM … SP, TO |
| `TP_CERTIFICACAO` | Certificação pretendida | 1=Fund. / 2=Médio |
| `Q44` | Situação de trabalho | A=trabalho remunerado / B=sem remuneração / C=Não |
| `Q50` | Renda familiar mensal | A=nenhuma … G=acima 5 SM / H=não sei |
| `Q11` | Última série estudada | A=1ª EF … K=3ª EM |

### Variáveis Alvo (Notas)

| Disciplina | Coluna no CSV | Escala | Mínimo aprovação |
|---|---|---|---|
| Linguagens e Códigos | `NU_NOTA_LC` | 0-200 (TRI) | 100 |
| Ciências Humanas | `NU_NOTA_CH` | 0-200 (TRI) | 100 |
| Matemática | `NU_NOTA_MT` | 0-200 (TRI) | 100 |
| Ciências da Natureza | `NU_NOTA_CN` | 0-200 (TRI) | 100 |
| Redação | `NU_NOTA_REDACAO` | 0-10 | 5.0 |

> **Aprovação geral:** `IN_APROVADO_LC = IN_APROVADO_CH = IN_APROVADO_MT = IN_APROVADO_CN = 1`

---

## Interface Gráfica (Dashboard)

O dashboard Streamlit permite ao gestor:

1. **Configurar o K** (número de vizinhos) via slider lateral
2. **Inserir o perfil** do novo candidato em um formulário
3. **Visualizar as notas previstas** com comparação aos vizinhos
4. **Ver a previsão de aprovação** com probabilidade estimada
5. **Consultar os K vizinhos mais próximos** com suas notas reais
6. **Receber recomendações pedagógicas** detalhadas por disciplina

---

## Referências

- COVER, T.; HART, P. *Nearest Neighbor Pattern Classification*. IEEE, 1967.
- FIX, E.; HODGES, J. *Discriminatory Analysis: Nonparametric Discrimination*. USAF, 1951.
- PROVOST, F.; FAWCETT, T. *Data Science para Negócios*. Alta Books, 2016.
- IBM. *O que é o algoritmo KNN?* https://www.ibm.com/br-pt/think/topics/knn
- INEP. *Microdados do ENCCEJA 2024*. https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/encceja
