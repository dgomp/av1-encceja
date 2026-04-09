"""
=============================================================================
TRABALHO AV1 - SISTEMAS DE APOIO À DECISÃO
Curso: Sistemas de Informação
=============================================================================

Título: ENCCEJA - Apoio à Decisão Educacional com K-Nearest Neighbors (K-NN)
Modelo: K-Nearest Neighbors (lazy learning / aprendizagem preguiçosa)
Dados:  Microdados ENCCEJA 2024 - INEP
Arquivo: data/DADOS/MICRODADOS_ENCCEJA_2024_REG_NAC.csv

Objetivo:
    Desenvolver um SAD que preveja as notas esperadas de um novo candidato
    ao ENCCEJA com base no perfil socioeconômico de candidatos similares
    dos microdados históricos de 2024.

Questão gerencial:
    Com base no perfil socioeconômico de um candidato, como apoiar decisões
    pedagógicas do cursinho usando o desempenho de candidatos semelhantes
    do ENCCEJA?

Estrutura do dado (arquivo REG_NAC):
    - Notas objetivas (0-200, escala TRI): NU_NOTA_LC, NU_NOTA_CH,
      NU_NOTA_MT, NU_NOTA_CN
    - Redação (0-10):                      NU_NOTA_REDACAO
    - Aprovação por área (0/1):            IN_APROVADO_LC/CH/MT/CN
    - Aprovado geral: todas as quatro áreas com IN_APROVADO = 1
    - Threshold aprovação áreas objetivas: ≥ 100 pontos
    - Features socioeconômicas:
        TP_SEXO        → sexo (M/F)
        TP_FAIXA_ETARIA → faixa etária (1-20)
        SG_UF_PROVA    → UF da prova
        TP_CERTIFICACAO → 1=Fund. / 2=Médio
        Q44             → situação de trabalho (A/B/C)
        Q50             → renda familiar (A-H)
        Q11             → última série estudada (A-K)

Algoritmo K-NN:
    1. Filtra candidatos presentes e com notas registradas
    2. Codifica variáveis categóricas (ordinal/label encoding)
    3. Normaliza com MinMaxScaler (escala 0-1) - essencial para não
       distorcer a distância Euclidiana
    4. Ao receber um novo candidato, calcula:
         d(A,B) = √[Σ(Ai − Bi)²]
    5. Seleciona K vizinhos mais próximos
    6. Prevê notas como média ponderada (peso = 1/distância)
    7. Gera recomendações pedagógicas
=============================================================================
"""

import os
import sys
import warnings

# Força UTF-8 no stdout para evitar erro em terminais Windows (cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix, classification_report
)

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CAMINHOS
# ---------------------------------------------------------------------------
CAMINHO_CSV = os.path.join(
    os.path.dirname(__file__),
    'data', 'DADOS', 'MICRODADOS_ENCCEJA_2024_REG_NAC.csv'
)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'output')

# ---------------------------------------------------------------------------
# COLUNAS REAIS DO ARQUIVO
# ---------------------------------------------------------------------------

# Features socioeconômicas (entrada do K-NN)
FEATURES = ['TP_SEXO', 'TP_FAIXA_ETARIA', 'SG_UF_PROVA',
            'TP_CERTIFICACAO', 'Q44', 'Q50', 'Q11']

# Notas alvo
NOTAS = ['NU_NOTA_LC', 'NU_NOTA_CH', 'NU_NOTA_MT', 'NU_NOTA_CN', 'NU_NOTA_REDACAO']

# Nomes legíveis por coluna de nota
NOMES_NOTAS = {
    'NU_NOTA_LC':      'Linguagens e Códigos',
    'NU_NOTA_CH':      'Ciências Humanas',
    'NU_NOTA_MT':      'Matemática',
    'NU_NOTA_CN':      'Ciências da Natureza',
    'NU_NOTA_REDACAO': 'Redação',
}

# Colunas de aprovação por área
APROVACAO_COLS = ['IN_APROVADO_LC', 'IN_APROVADO_CH', 'IN_APROVADO_MT', 'IN_APROVADO_CN']

# Thresholds de aprovação
THRESHOLD_OBJETIVO = 100   # áreas objetivas (escala 0-200)
THRESHOLD_REDACAO  = 5.0   # redação (escala 0-10)

# Hiperparâmetros K-NN
K_VIZINHOS   = 7
RANDOM_STATE = 42
TEST_SIZE    = 0.20

# ---------------------------------------------------------------------------
# MAPEAMENTOS LEGÍVEIS
# ---------------------------------------------------------------------------

MAPA_SEXO = {'M': 'Masculino', 'F': 'Feminino'}

MAPA_FAIXA_ETARIA = {
    1:  'Menor de 17 anos',  2:  '17 anos',          3:  '18 anos',
    4:  '19 anos',           5:  '20 anos',           6:  '21 anos',
    7:  '22 anos',           8:  '23 anos',           9:  '24 anos',
    10: '25 anos',           11: 'Entre 26 e 30 anos', 12: 'Entre 31 e 35 anos',
    13: 'Entre 36 e 40 anos', 14: 'Entre 41 e 45 anos', 15: 'Entre 46 e 50 anos',
    16: 'Entre 51 e 55 anos', 17: 'Entre 56 e 60 anos', 18: 'Entre 61 e 65 anos',
    19: 'Entre 66 e 70 anos', 20: 'Maior de 70 anos',
}

MAPA_CERTIFICACAO = {1: 'Ensino Fundamental', 2: 'Ensino Médio'}

# Q44 - Você trabalha? (Dicionário ENCCEJA 2024)
MAPA_Q44 = {
    'A': 'Sim, exerço um trabalho remunerado',
    'B': 'Sim, mas trabalho sem remuneração',
    'C': 'Não',
}

# Q50 - Renda mensal familiar (Dicionário ENCCEJA 2024)
MAPA_Q50 = {
    'A': 'Nenhuma renda',
    'B': 'Até 1 salário mínimo',
    'C': 'De 1 a 2 salários mínimos',
    'D': 'De 2 a 3 salários mínimos',
    'E': 'De 3 a 4 salários mínimos',
    'F': 'De 4 a 5 salários mínimos',
    'G': 'Acima de 5 salários mínimos',
    'H': 'Não sei',
}

# Q11 - Em que série você parou de estudar? (Dicionário ENCCEJA 2024)
MAPA_Q11 = {
    'A': '1ª série do ensino fundamental',  'B': '2ª série do ensino fundamental',
    'C': '3ª série do ensino fundamental',  'D': '4ª série do ensino fundamental',
    'E': '5ª série do ensino fundamental',  'F': '6ª série do ensino fundamental',
    'G': '7ª série do ensino fundamental',  'H': '8ª série do ensino fundamental',
    'I': '1ª série do ensino médio',        'J': '2ª série do ensino médio',
    'K': '3ª série do ensino médio',
}

UFS_BRASIL = [
    'AC','AL','AM','AP','BA','CE','DF','ES','GO','MA',
    'MG','MS','MT','PA','PB','PE','PI','PR','RJ','RN',
    'RO','RR','RS','SC','SE','SP','TO'
]

# Ordem para encoding ordinal (Q50 e Q11)
ORDEM_Q50 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
ORDEM_Q11 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']
ORDEM_Q44 = ['C', 'B', 'A']   # Não trabalha < sem remun. < com remun.


# ---------------------------------------------------------------------------
# ETAPA 1: CARREGAMENTO
# ---------------------------------------------------------------------------

def carregar_dados() -> pd.DataFrame:
    """
    Carrega o arquivo MICRODADOS_ENCCEJA_2024_REG_NAC.csv.
    Apenas candidatos presentes em pelo menos uma prova são mantidos
    (TP_PRESENCA = 1 em alguma área), pois candidatos ausentes não possuem
    notas e não contribuem para o aprendizado do modelo.
    """
    print(f"\n{'='*60}")
    print("1. CARREGAMENTO DOS DADOS")
    print(f"{'='*60}")

    if not os.path.exists(CAMINHO_CSV):
        raise FileNotFoundError(f"Arquivo não encontrado: {CAMINHO_CSV}")

    print(f"Lendo: {CAMINHO_CSV}")

    df = pd.read_csv(CAMINHO_CSV, sep=';', encoding='latin-1', low_memory=False)

    print(f"Registros brutos : {len(df):,}")

    # Filtrar presentes em ao menos uma prova
    mascara_presenca = (
        (df['TP_PRESENCA_LC'] == 1) |
        (df['TP_PRESENCA_CH'] == 1) |
        (df['TP_PRESENCA_MT'] == 1) |
        (df['TP_PRESENCA_CN'] == 1)
    )
    df = df[mascara_presenca].copy()
    print(f"Presentes em alguma prova: {len(df):,}")
    print(f"Colunas: {df.shape[1]}")
    return df


# ---------------------------------------------------------------------------
# ETAPA 2: EXPLORAÇÃO
# ---------------------------------------------------------------------------

def explorar_dados(df: pd.DataFrame) -> None:
    """Estatísticas exploratórias do dataset."""
    print(f"\n{'='*60}")
    print("2. EXPLORAÇÃO DOS DADOS")
    print(f"{'='*60}")

    print("\n--- Estatísticas das notas (candidatos presentes) ---")
    print(df[NOTAS].describe().round(2).to_string())

    print(f"\n--- Distribuição por certificação ---")
    cert = df['TP_CERTIFICACAO'].map(MAPA_CERTIFICACAO).value_counts()
    print(cert.to_string())

    print(f"\n--- Distribuição por sexo ---")
    print(df['TP_SEXO'].map(MAPA_SEXO).value_counts().to_string())

    print(f"\n--- Situação de trabalho (Q44) ---")
    print(df['Q44'].map(MAPA_Q44).value_counts().to_string())

    print(f"\n--- Renda familiar (Q50) ---")
    print(df['Q50'].map(MAPA_Q50).value_counts().to_string())


# ---------------------------------------------------------------------------
# ETAPA 3: PRÉ-PROCESSAMENTO
# ---------------------------------------------------------------------------

def preprocessar(df: pd.DataFrame):
    """
    Pré-processa os microdados:
      1. Deriva aprovação geral (todas as 4 áreas aprovadas)
      2. Trata valores ausentes nas features com moda
      3. Aplica encoding ordinal (Q44, Q50, Q11) e label encoding (TP_SEXO, SG_UF_PROVA)
      4. Mantém TP_FAIXA_ETARIA e TP_CERTIFICACAO como numéricos
      5. Aplica MinMaxScaler - equaliza escalas para a distância Euclidiana
      6. Divide em treino/teste estratificado por aprovação

    Retorna todos os artefatos necessários para treino e predição.
    """
    print(f"\n{'='*60}")
    print("3. PRÉ-PROCESSAMENTO")
    print(f"{'='*60}")

    df_clean = df.copy()

    # --- Aprovação geral ---
    # Candidato aprovado = aprovado em todas as 4 áreas objetivas
    cols_apr = [c for c in APROVACAO_COLS if c in df_clean.columns]
    df_clean['aprovado_geral'] = (
        df_clean[cols_apr].apply(lambda row: int(all(row == 1)), axis=1)
    )
    taxa = df_clean['aprovado_geral'].mean() * 100
    print(f"Taxa de aprovação geral: {taxa:.1f}%")
    print(df_clean['aprovado_geral'].value_counts().rename({1: 'Aprovado', 0: 'Reprovado'}).to_string())

    # --- Filtrar registros com ao menos uma nota ---
    notas_validas = [c for c in NOTAS if c in df_clean.columns]
    df_clean = df_clean.dropna(subset=notas_validas, how='all')
    print(f"\nRegistros com pelo menos uma nota: {len(df_clean):,}")

    # --- Tratar features ausentes com moda ---
    X_raw = df_clean[FEATURES].copy()
    for col in FEATURES:
        nulos = X_raw[col].isnull().sum()
        if nulos > 0:
            moda = X_raw[col].mode()[0]
            X_raw[col] = X_raw[col].fillna(moda)
            print(f"  Nulos em {col}: {nulos} → preenchidos com '{moda}'")

    # --- Encoding ---
    # Variáveis ordinais: Q44, Q50, Q11 (a ordem importa)
    # Variáveis categóricas nominais: TP_SEXO, SG_UF_PROVA
    # Variáveis numéricas: TP_FAIXA_ETARIA, TP_CERTIFICACAO

    encoders = {}

    def ord_encode(series, ordem):
        """Mapeia letras para índice ordinal baseado na ordem fornecida."""
        mapa = {v: i for i, v in enumerate(ordem)}
        return series.map(mapa).fillna(len(ordem) - 1).astype(float)

    X_enc = pd.DataFrame(index=X_raw.index)

    # Ordinal
    X_enc['Q44'] = ord_encode(X_raw['Q44'], ORDEM_Q44)
    X_enc['Q50'] = ord_encode(X_raw['Q50'], ORDEM_Q50)
    X_enc['Q11'] = ord_encode(X_raw['Q11'], ORDEM_Q11)
    encoders['Q44_ordem'] = ORDEM_Q44
    encoders['Q50_ordem'] = ORDEM_Q50
    encoders['Q11_ordem'] = ORDEM_Q11

    # Label encoding nominal
    for col in ['TP_SEXO', 'SG_UF_PROVA']:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_raw[col].astype(str))
        encoders[col] = le

    # Numérico direto
    X_enc['TP_FAIXA_ETARIA'] = X_raw['TP_FAIXA_ETARIA'].astype(float)
    X_enc['TP_CERTIFICACAO'] = X_raw['TP_CERTIFICACAO'].astype(float)

    feature_cols = ['TP_SEXO', 'TP_FAIXA_ETARIA', 'SG_UF_PROVA',
                    'TP_CERTIFICACAO', 'Q44', 'Q50', 'Q11']

    print(f"\nFeatures utilizadas ({len(feature_cols)}):")
    for f in feature_cols:
        print(f"  {f:<22} → {X_raw[f].nunique()} valores únicos")

    # --- Variáveis alvo ---
    y_notas    = df_clean[notas_validas].copy()
    y_aprovado = df_clean['aprovado_geral']

    # --- Split treino/teste ---
    X_train, X_test, y_n_train, y_n_test, y_a_train, y_a_test = train_test_split(
        X_enc[feature_cols], y_notas, y_aprovado,
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_aprovado
    )
    print(f"\nTreino: {len(X_train):,} | Teste: {len(X_test):,}")

    # --- MinMaxScaler ---
    # Normaliza 0→1 para que nenhuma feature domine a distância Euclidiana.
    # Ex. sem normalização, TP_FAIXA_ETARIA (1-20) dominaria TP_SEXO (0-1).
    scaler = MinMaxScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)

    print("Normalização Min-Max aplicada (equaliza escalas para distância Euclidiana).")

    return (X_train_sc, X_test_sc,
            y_n_train, y_n_test,
            y_a_train, y_a_test,
            scaler, encoders, feature_cols, notas_validas,
            df_clean, X_enc[feature_cols])


# ---------------------------------------------------------------------------
# ETAPA 4: TREINAMENTO
# ---------------------------------------------------------------------------

def treinar_modelos(X_train_sc, y_n_train, y_a_train, notas_validas: list):
    """
    Treina os modelos K-NN:
      - KNeighborsRegressor (um por disciplina) → prevê nota numérica
      - KNeighborsClassifier                    → prevê aprovação geral

    K-NN é lazy learning: na fase de 'treino' apenas armazena os dados.
    O cálculo de distâncias ocorre integralmente na fase de predição.

    Parâmetros:
        n_neighbors = K_VIZINHOS  (K = 7, ímpar para evitar empate)
        metric      = 'euclidean'
        weights     = 'distance'  (vizinhos mais próximos têm maior peso)
    """
    print(f"\n{'='*60}")
    print("4. TREINAMENTO DO MODELO K-NN")
    print(f"{'='*60}")
    print(f"Tipo        : Lazy learning - armazena dados, aprende na predição")
    print(f"K vizinhos  : {K_VIZINHOS}")
    print(f"Distância   : Euclidiana")
    print(f"Pesos       : Ponderados pela distância (1/d)")

    modelos_notas = {}
    for nota in notas_validas:
        knn = KNeighborsRegressor(
            n_neighbors=K_VIZINHOS,
            metric='euclidean',
            weights='distance'
        )
        # Treina apenas com registros não-nulos para esta disciplina
        mask = y_n_train[nota].notna()
        knn.fit(X_train_sc[mask], y_n_train[nota][mask])
        modelos_notas[nota] = knn

    knn_apr = KNeighborsClassifier(
        n_neighbors=K_VIZINHOS,
        metric='euclidean',
        weights='distance'
    )
    knn_apr.fit(X_train_sc, y_a_train)

    print(f"\n{len(modelos_notas)} modelos de regressão (um por disciplina) + 1 classificador")
    return modelos_notas, knn_apr


# ---------------------------------------------------------------------------
# ETAPA 5: AVALIAÇÃO
# ---------------------------------------------------------------------------

def avaliar_modelos(modelos_notas, knn_apr,
                    X_test_sc, y_n_test, y_a_test,
                    notas_validas: list):
    """
    Métricas de avaliação:
      Regressão  → MAE, RMSE, R² por disciplina
      Classificação → acurácia, matriz de confusão, relatório
    """
    print(f"\n{'='*60}")
    print("5. AVALIAÇÃO DOS MODELOS")
    print(f"{'='*60}")

    print(f"\n{'Disciplina':<30} {'MAE':>8} {'RMSE':>8} {'R²':>8}")
    print('-' * 58)

    resultados = []
    for nota in notas_validas:
        mask = y_n_test[nota].notna()
        y_pred = modelos_notas[nota].predict(X_test_sc[mask])
        y_real = y_n_test[nota][mask]
        mae  = mean_absolute_error(y_real, y_pred)
        rmse = np.sqrt(mean_squared_error(y_real, y_pred))
        r2   = r2_score(y_real, y_pred)
        nome = NOMES_NOTAS.get(nota, nota)
        print(f"  {nome:<28} {mae:>8.2f} {rmse:>8.2f} {r2:>8.4f}")
        resultados.append({'nota': nota, 'nome': nome,
                           'mae': mae, 'rmse': rmse, 'r2': r2})

    y_pred_apr = knn_apr.predict(X_test_sc)
    acc = accuracy_score(y_a_test, y_pred_apr)
    cm  = confusion_matrix(y_a_test, y_pred_apr)

    print(f"\n--- Modelo de Aprovação ---")
    print(f"Acurácia: {acc*100:.2f}%")
    print(classification_report(y_a_test, y_pred_apr,
                                 target_names=['Reprovado', 'Aprovado']))
    tn, fp, fn, tp = cm.ravel()
    print(f"Matriz de Confusão:")
    print(f"  VN (reprovado correto)  : {tn:,}")
    print(f"  FP (reprovado → aprovado): {fp:,}")
    print(f"  FN (aprovado → reprovado): {fn:,}")
    print(f"  VP (aprovado correto)   : {tp:,}")

    return resultados, cm


# ---------------------------------------------------------------------------
# ETAPA 6: PREDIÇÃO DE UM CANDIDATO
# ---------------------------------------------------------------------------

def _codificar_candidato(candidato: dict, encoders: dict,
                          feature_cols: list, scaler: MinMaxScaler) -> np.ndarray:
    """Codifica e normaliza um candidato para uso no K-NN."""
    row = {}
    # Ordinal
    for col, ordem_key in [('Q44', 'Q44_ordem'), ('Q50', 'Q50_ordem'), ('Q11', 'Q11_ordem')]:
        val   = str(candidato.get(col, ''))
        ordem = encoders[ordem_key]
        mapa  = {v: i for i, v in enumerate(ordem)}
        row[col] = float(mapa.get(val, len(ordem) - 1))

    # Label encoding nominal
    for col in ['TP_SEXO', 'SG_UF_PROVA']:
        le  = encoders[col]
        val = str(candidato.get(col, le.classes_[0]))
        if val in le.classes_:
            row[col] = float(le.transform([val])[0])
        else:
            row[col] = float(le.transform([le.classes_[0]])[0])

    # Numérico
    row['TP_FAIXA_ETARIA'] = float(candidato.get('TP_FAIXA_ETARIA', 1))
    row['TP_CERTIFICACAO'] = float(candidato.get('TP_CERTIFICACAO', 2))

    arr = np.array([[row[c] for c in feature_cols]])
    return scaler.transform(arr)


def prever_candidato(candidato: dict, modelos_notas, knn_apr,
                     scaler, encoders, feature_cols, notas_validas,
                     df_clean) -> dict:
    """
    Executa a predição K-NN para um novo candidato.
    Retorna notas previstas, aprovação, probabilidade e dados dos vizinhos.
    """
    arr_sc = _codificar_candidato(candidato, encoders, feature_cols, scaler)

    notas_prev = {
        n: round(float(modelos_notas[n].predict(arr_sc)[0]), 2)
        for n in notas_validas
    }

    apr_pred = int(knn_apr.predict(arr_sc)[0])
    prob_apr  = knn_apr.predict_proba(arr_sc)[0]

    # Identifica os K vizinhos mais próximos (usa modelo de referência)
    knn_ref = list(modelos_notas.values())[0]
    dists, idxs = knn_ref.kneighbors(arr_sc, n_neighbors=K_VIZINHOS)

    vizinhos = df_clean.iloc[idxs[0]].copy()
    vizinhos['distancia_knn'] = dists[0]
    cols_exibir = notas_validas + ['aprovado_geral', 'distancia_knn']
    vizinhos_df = vizinhos[[c for c in cols_exibir if c in vizinhos.columns]]

    return {
        'notas_prev': notas_prev,
        'apr_pred':   apr_pred,
        'prob_apr':   prob_apr,
        'vizinhos':   vizinhos_df,
    }


def gerar_recomendacoes(resultado: dict, notas_validas: list) -> None:
    """Exibe as recomendações pedagógicas com base nos resultados do K-NN."""
    print(f"\n{'='*60}")
    print("6. RECOMENDAÇÕES PEDAGÓGICAS")
    print(f"{'='*60}")

    notas  = resultado['notas_prev']
    vizs   = resultado['vizinhos']
    apr    = resultado['apr_pred']
    prob   = resultado['prob_apr']
    taxa_v = vizs['aprovado_geral'].mean() * 100 if 'aprovado_geral' in vizs else 0

    print(f"\n{'Disciplina':<30} {'Prevista':>10} {'Méd.Vizinhos':>14} {'Status':>10}")
    print('-' * 66)
    for nota in notas_validas:
        val  = notas.get(nota, 0)
        mviz = vizs[nota].mean() if nota in vizs.columns else 0
        thr  = THRESHOLD_REDACAO if nota == 'NU_NOTA_REDACAO' else THRESHOLD_OBJETIVO
        st   = "✓ OK" if val >= thr else "⚠ RISCO"
        nome = NOMES_NOTAS.get(nota, nota)
        print(f"  {nome:<28} {val:>10.1f} {mviz:>14.1f} {st:>10}")

    print(f"\n{'='*60}")
    print(f"Aprovação prevista : {'APROVADO' if apr == 1 else 'RISCO DE REPROVAÇÃO'}")
    print(f"Probabilidade      : {prob[1]*100:.1f}%")
    print(f"Taxa vizinhos aprov: {taxa_v:.1f}%  (de {K_VIZINHOS} vizinhos)")

    print("\n--- Plano de ação ---")
    if taxa_v < 30:
        print("  ⚠ ALTO RISCO: maioria dos vizinhos foi reprovada.")
        print("  → Acompanhamento individual intensivo recomendado.")
    elif taxa_v < 60:
        print("  ⚠ RISCO MODERADO: menos de 60% dos vizinhos aprovados.")
        print("  → Monitoramento regular e reforço nas disciplinas críticas.")
    else:
        print("  ✓ Perfil favorável: maioria dos vizinhos aprovada.")
        print("  → Manter ritmo de estudo; focar em disciplinas abaixo do threshold.")

    criticas_obj = [nota for nota in notas_validas
                    if nota != 'NU_NOTA_REDACAO' and notas.get(nota, 0) < THRESHOLD_OBJETIVO]
    if notas.get('NU_NOTA_REDACAO', 0) < THRESHOLD_REDACAO:
        criticas_obj.append('NU_NOTA_REDACAO')

    if criticas_obj:
        print("\n  Disciplinas abaixo do mínimo:")
        for n in criticas_obj:
            nome = NOMES_NOTAS.get(n, n)
            thr  = THRESHOLD_REDACAO if n == 'NU_NOTA_REDACAO' else THRESHOLD_OBJETIVO
            print(f"    → {nome}: {notas[n]:.1f} (mínimo: {thr})")


# ---------------------------------------------------------------------------
# VISUALIZAÇÕES
# ---------------------------------------------------------------------------

def salvar_distribuicao_notas(df_clean, notas_validas, output_dir):
    n = len(notas_validas)
    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 4))
    if n == 1:
        axes = [axes]
    cores = ['#3498db', '#e74c3c', '#27ae60', '#f39c12', '#9b59b6']

    for ax, nota, cor in zip(axes, notas_validas, cores):
        dados = df_clean[nota].dropna()
        ax.hist(dados, bins=35, color=cor, alpha=0.85, edgecolor='white', linewidth=0.4)
        thr = THRESHOLD_REDACAO if nota == 'NU_NOTA_REDACAO' else THRESHOLD_OBJETIVO
        ax.axvline(thr, color='red', linestyle='--', linewidth=1.5,
                   label=f'Mínimo ({thr})')
        ax.set_title(NOMES_NOTAS.get(nota, nota), fontsize=9, fontweight='bold')
        ax.set_xlabel('Nota', fontsize=8)
        ax.legend(fontsize=7)
        ax.spines[['top', 'right']].set_visible(False)

    plt.suptitle('Distribuição das Notas - ENCCEJA 2024',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    caminho = os.path.join(output_dir, 'distribuicao_notas.png')
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {caminho}")


def salvar_matriz_confusao(cm, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['Reprovado', 'Aprovado'],
                yticklabels=['Reprovado', 'Aprovado'],
                cmap='Blues', ax=ax, linewidths=0.5, linecolor='gray',
                annot_kws={'size': 13})
    ax.set_xlabel('Previsto', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title(f'Matriz de Confusão - Aprovação (K-NN, K={K_VIZINHOS})',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    caminho = os.path.join(output_dir, 'matriz_confusao.png')
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {caminho}")


def salvar_desempenho(resultados, output_dir):
    df_res = pd.DataFrame(resultados)
    fig, ax = plt.subplots(figsize=(9, 5))
    cores = ['#c0392b' if v > 10 else '#27ae60' for v in df_res['mae']]
    bars = ax.bar(df_res['nome'], df_res['mae'],
                  color=cores, edgecolor='white')
    for bar, val in zip(bars, df_res['mae']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{val:.2f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_ylabel('MAE (Erro Absoluto Médio)')
    ax.set_title(f'Erro de Previsão por Disciplina - K-NN (K={K_VIZINHOS})',
                 fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=12)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    caminho = os.path.join(output_dir, 'desempenho_disciplinas.png')
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {caminho}")


def salvar_aprovacao_por_renda(df_clean, output_dir):
    if 'Q50' not in df_clean.columns:
        return
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Por renda
    renda_apr = (df_clean.groupby('Q50')['aprovado_geral']
                 .mean() * 100).rename(index=MAPA_Q50)
    renda_apr = renda_apr[renda_apr.index.notna()]
    cores_r = ['#e74c3c' if v < 50 else '#27ae60' for v in renda_apr.values]
    axes[0].barh(renda_apr.index, renda_apr.values, color=cores_r, edgecolor='white')
    axes[0].axvline(50, color='black', linestyle='--', linewidth=1)
    axes[0].set_xlabel('Taxa de Aprovação (%)')
    axes[0].set_title('Aprovação por Renda Familiar', fontweight='bold')
    axes[0].spines[['top', 'right']].set_visible(False)

    # Por trabalho
    trab_apr = (df_clean.groupby('Q44')['aprovado_geral']
                .mean() * 100).rename(index=MAPA_Q44)
    trab_apr = trab_apr[trab_apr.index.notna()]
    cores_t = ['#e74c3c' if v < 50 else '#27ae60' for v in trab_apr.values]
    axes[1].barh(trab_apr.index, trab_apr.values, color=cores_t, edgecolor='white')
    axes[1].axvline(50, color='black', linestyle='--', linewidth=1)
    axes[1].set_xlabel('Taxa de Aprovação (%)')
    axes[1].set_title('Aprovação por Situação de Trabalho', fontweight='bold')
    axes[1].spines[['top', 'right']].set_visible(False)

    plt.suptitle('Perfil Socioeconômico e Aprovação - ENCCEJA 2024',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    caminho = os.path.join(output_dir, 'aprovacao_por_perfil.png')
    plt.savefig(caminho, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Salvo: {caminho}")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    print("\n" + "=" * 60)
    print("  AV1 - K-NN PARA APOIO À DECISÃO EDUCACIONAL (ENCCEJA 2024)")
    print("  Disciplina: Sistemas de Apoio à Decisão")
    print("=" * 60)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = carregar_dados()
    explorar_dados(df)

    (X_train_sc, X_test_sc,
     y_n_train, y_n_test,
     y_a_train, y_a_test,
     scaler, encoders, feature_cols, notas_validas,
     df_clean, X_enc) = preprocessar(df)

    modelos_notas, knn_apr = treinar_modelos(
        X_train_sc, y_n_train, y_a_train, notas_validas
    )

    resultados, cm = avaliar_modelos(
        modelos_notas, knn_apr,
        X_test_sc, y_n_test, y_a_test, notas_validas
    )

    # --- Exemplo de predição ---
    print(f"\n{'='*60}")
    print("EXEMPLO: CANDIDATO HIPOTÉTICO")
    print(f"{'='*60}")
    candidato_ex = {
        'TP_SEXO':          'F',
        'TP_FAIXA_ETARIA':  11,   # Entre 26 e 30 anos
        'SG_UF_PROVA':      'SP',
        'TP_CERTIFICACAO':  2,    # Ensino Médio
        'Q44':              'A',  # Trabalho remunerado
        'Q50':              'C',  # 1 a 2 salários mínimos
        'Q11':              'I',  # 1ª série do EM
    }
    for k, v in candidato_ex.items():
        print(f"  {k}: {v}")

    resultado = prever_candidato(
        candidato_ex, modelos_notas, knn_apr,
        scaler, encoders, feature_cols, notas_validas, df_clean
    )
    gerar_recomendacoes(resultado, notas_validas)

    # --- Visualizações ---
    print(f"\n{'='*60}")
    print("7. SALVANDO VISUALIZAÇÕES")
    print(f"{'='*60}")
    salvar_distribuicao_notas(df_clean, notas_validas, OUTPUT_DIR)
    salvar_matriz_confusao(cm, OUTPUT_DIR)
    salvar_desempenho(resultados, OUTPUT_DIR)
    salvar_aprovacao_por_renda(df_clean, OUTPUT_DIR)

    print(f"\n{'='*60}")
    print(f"CONCLUÍDO. Imagens em: {OUTPUT_DIR}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
