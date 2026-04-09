"""
=============================================================================
DASHBOARD - AV1: ENCCEJA - Apoio à Decisão Educacional com K-NN
=============================================================================
Interface gráfica interativa para prever notas e apoiar decisões pedagógicas.

Arquivo de dados esperado:
    data/DADOS/MICRODADOS_ENCCEJA_2024_REG_NAC.csv

Execução:
    streamlit run dashboard.py
=============================================================================
"""

import os
import warnings
import pandas as pd
import numpy as np
import streamlit as st
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
# CONFIGURAÇÃO DA PÁGINA
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ENCCEJA 2024 - Sistema de Apoio à Decisão (K-NN)",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CONSTANTES
# ---------------------------------------------------------------------------
CAMINHO_CSV = os.path.join(
    os.path.dirname(__file__),
    'data', 'DADOS', 'MICRODADOS_ENCCEJA_2024_REG_NAC.csv'
)

FEATURES = ['TP_SEXO', 'TP_FAIXA_ETARIA', 'SG_UF_PROVA',
            'TP_CERTIFICACAO', 'Q44', 'Q50', 'Q11']

NOTAS = ['NU_NOTA_LC', 'NU_NOTA_CH', 'NU_NOTA_MT', 'NU_NOTA_CN', 'NU_NOTA_REDACAO']

NOMES_NOTAS = {
    'NU_NOTA_LC':      'Linguagens e Códigos',
    'NU_NOTA_CH':      'Ciências Humanas',
    'NU_NOTA_MT':      'Matemática',
    'NU_NOTA_CN':      'Ciências da Natureza',
    'NU_NOTA_REDACAO': 'Redação',
}

ICONES_NOTAS = {
    'NU_NOTA_LC':      '📚',
    'NU_NOTA_CH':      '🌍',
    'NU_NOTA_MT':      '📐',
    'NU_NOTA_CN':      '🔬',
    'NU_NOTA_REDACAO': '✍️',
}

APROVACAO_COLS = ['IN_APROVADO_LC', 'IN_APROVADO_CH', 'IN_APROVADO_MT', 'IN_APROVADO_CN']

THRESHOLD_OBJETIVO = 100   # escala 0-200
THRESHOLD_REDACAO  = 5.0   # escala 0-10

# Mapeamentos
MAPA_SEXO = {'M': 'Masculino', 'F': 'Feminino'}

MAPA_FAIXA_ETARIA = {
    1:  'Menor de 17 anos',   2:  '17 anos',            3:  '18 anos',
    4:  '19 anos',            5:  '20 anos',            6:  '21 anos',
    7:  '22 anos',            8:  '23 anos',            9:  '24 anos',
    10: '25 anos',            11: 'Entre 26 e 30 anos', 12: 'Entre 31 e 35 anos',
    13: 'Entre 36 e 40 anos', 14: 'Entre 41 e 45 anos', 15: 'Entre 46 e 50 anos',
    16: 'Entre 51 e 55 anos', 17: 'Entre 56 e 60 anos', 18: 'Entre 61 e 65 anos',
    19: 'Entre 66 e 70 anos', 20: 'Maior de 70 anos',
}

MAPA_CERTIFICACAO = {1: 'Ensino Fundamental', 2: 'Ensino Médio'}

MAPA_Q44 = {
    'A': 'Sim, exerço um trabalho remunerado',
    'B': 'Sim, mas trabalho sem remuneração',
    'C': 'Não',
}

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

MAPA_Q11 = {
    'A': '1ª série do ensino fundamental', 'B': '2ª série do ensino fundamental',
    'C': '3ª série do ensino fundamental', 'D': '4ª série do ensino fundamental',
    'E': '5ª série do ensino fundamental', 'F': '6ª série do ensino fundamental',
    'G': '7ª série do ensino fundamental', 'H': '8ª série do ensino fundamental',
    'I': '1ª série do ensino médio',       'J': '2ª série do ensino médio',
    'K': '3ª série do ensino médio',
}

ORDEM_Q44 = ['C', 'B', 'A']
ORDEM_Q50 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
ORDEM_Q11 = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K']

UFS_BRASIL = sorted([
    'AC','AL','AM','AP','BA','CE','DF','ES','GO','MA',
    'MG','MS','MT','PA','PB','PE','PI','PR','RJ','RN',
    'RO','RR','RS','SC','SE','SP','TO'
])

# ---------------------------------------------------------------------------
# CARREGAMENTO E PIPELINE
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner="Carregando microdados do ENCCEJA 2024...")
def carregar_dados() -> pd.DataFrame:
    if not os.path.exists(CAMINHO_CSV):
        st.error(
            f"**Arquivo não encontrado:** `{CAMINHO_CSV}`\n\n"
            "Baixe os microdados do ENCCEJA 2024 em:\n"
            "https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/encceja"
        )
        st.stop()

    df = pd.read_csv(CAMINHO_CSV, sep=';', encoding='latin-1', low_memory=False)

    # Apenas candidatos presentes em ao menos uma prova
    mascara = (
        (df['TP_PRESENCA_LC'] == 1) |
        (df['TP_PRESENCA_CH'] == 1) |
        (df['TP_PRESENCA_MT'] == 1) |
        (df['TP_PRESENCA_CN'] == 1)
    )
    df = df[mascara].copy()

    # Aprovação geral
    cols_apr = [c for c in APROVACAO_COLS if c in df.columns]
    df['aprovado_geral'] = df[cols_apr].apply(
        lambda row: int(all(row == 1)), axis=1
    )
    return df


@st.cache_resource(show_spinner="Treinando modelo K-NN...")
def treinar_pipeline(k_vizinhos: int):
    df = carregar_dados()
    notas_validas   = [c for c in NOTAS if c in df.columns]
    df_clean = df.dropna(subset=notas_validas, how='all').copy()

    # Preencher nulos com moda
    X_raw = df_clean[FEATURES].copy()
    for col in FEATURES:
        if X_raw[col].isnull().any():
            X_raw[col] = X_raw[col].fillna(X_raw[col].mode()[0])

    encoders = {}

    def ord_enc(series, ordem):
        mapa = {v: i for i, v in enumerate(ordem)}
        return series.map(mapa).fillna(len(ordem) - 1).astype(float)

    X_enc = pd.DataFrame(index=X_raw.index)
    X_enc['Q44'] = ord_enc(X_raw['Q44'], ORDEM_Q44)
    X_enc['Q50'] = ord_enc(X_raw['Q50'], ORDEM_Q50)
    X_enc['Q11'] = ord_enc(X_raw['Q11'], ORDEM_Q11)
    encoders['Q44_ordem'] = ORDEM_Q44
    encoders['Q50_ordem'] = ORDEM_Q50
    encoders['Q11_ordem'] = ORDEM_Q11

    for col in ['TP_SEXO', 'SG_UF_PROVA']:
        le = LabelEncoder()
        X_enc[col] = le.fit_transform(X_raw[col].astype(str))
        encoders[col] = le

    X_enc['TP_FAIXA_ETARIA'] = X_raw['TP_FAIXA_ETARIA'].astype(float)
    X_enc['TP_CERTIFICACAO'] = X_raw['TP_CERTIFICACAO'].astype(float)

    feature_cols = ['TP_SEXO', 'TP_FAIXA_ETARIA', 'SG_UF_PROVA',
                    'TP_CERTIFICACAO', 'Q44', 'Q50', 'Q11']

    y_notas    = df_clean[notas_validas]
    y_aprovado = df_clean['aprovado_geral']

    X_tr, X_te, y_n_tr, y_n_te, y_a_tr, y_a_te = train_test_split(
        X_enc[feature_cols], y_notas, y_aprovado,
        test_size=0.20, random_state=42, stratify=y_aprovado
    )

    scaler = MinMaxScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    modelos_notas = {}
    for nota in notas_validas:
        mask = y_n_tr[nota].notna()
        knn = KNeighborsRegressor(
            n_neighbors=k_vizinhos, metric='euclidean', weights='distance'
        )
        knn.fit(X_tr_sc[mask], y_n_tr[nota][mask])
        modelos_notas[nota] = knn

    knn_apr = KNeighborsClassifier(
        n_neighbors=k_vizinhos, metric='euclidean', weights='distance'
    )
    knn_apr.fit(X_tr_sc, y_a_tr)

    metricas = []
    for nota in notas_validas:
        mask = y_n_te[nota].notna()
        y_p  = modelos_notas[nota].predict(X_te_sc[mask])
        y_r  = y_n_te[nota][mask]
        metricas.append({
            'nota': nota, 'nome': NOMES_NOTAS.get(nota, nota),
            'mae':  mean_absolute_error(y_r, y_p),
            'rmse': np.sqrt(mean_squared_error(y_r, y_p)),
            'r2':   r2_score(y_r, y_p),
        })

    y_pred_apr = knn_apr.predict(X_te_sc)
    acc_apr = accuracy_score(y_a_te, y_pred_apr)
    cm      = confusion_matrix(y_a_te, y_pred_apr)

    return {
        'modelos_notas':  modelos_notas,
        'knn_apr':        knn_apr,
        'scaler':         scaler,
        'encoders':       encoders,
        'feature_cols':   feature_cols,
        'notas_validas':  notas_validas,
        'df_clean':       df_clean,
        'metricas':       metricas,
        'acc_apr':        acc_apr,
        'cm':             cm,
        'taxa_aprovacao': df_clean['aprovado_geral'].mean() * 100,
    }


def _cod_candidato(candidato: dict, art: dict) -> np.ndarray:
    """Codifica e normaliza um candidato para uso no K-NN."""
    enc  = art['encoders']
    cols = art['feature_cols']
    row  = {}

    for col, ok in [('Q44', 'Q44_ordem'), ('Q50', 'Q50_ordem'), ('Q11', 'Q11_ordem')]:
        val   = str(candidato.get(col, ''))
        ordem = enc[ok]
        mapa  = {v: i for i, v in enumerate(ordem)}
        row[col] = float(mapa.get(val, len(ordem) - 1))

    for col in ['TP_SEXO', 'SG_UF_PROVA']:
        le  = enc[col]
        val = str(candidato.get(col, le.classes_[0]))
        row[col] = float(le.transform([val])[0] if val in le.classes_
                         else le.transform([le.classes_[0]])[0])

    row['TP_FAIXA_ETARIA'] = float(candidato.get('TP_FAIXA_ETARIA', 1))
    row['TP_CERTIFICACAO'] = float(candidato.get('TP_CERTIFICACAO', 2))

    arr = np.array([[row[c] for c in cols]])
    return art['scaler'].transform(arr)


def prever(candidato: dict, art: dict) -> dict:
    arr_sc = _cod_candidato(candidato, art)
    notas_p = art['notas_validas']

    notas_prev = {
        n: round(float(art['modelos_notas'][n].predict(arr_sc)[0]), 2)
        for n in notas_p
    }
    apr_pred = int(art['knn_apr'].predict(arr_sc)[0])
    prob_apr  = art['knn_apr'].predict_proba(arr_sc)[0]

    knn_ref = list(art['modelos_notas'].values())[0]
    dists, idxs = knn_ref.kneighbors(arr_sc, n_neighbors=k_vizinhos)
    vizinhos = art['df_clean'].iloc[idxs[0]].copy()
    vizinhos['distancia_knn'] = dists[0]
    cols_viz = notas_p + ['aprovado_geral', 'distancia_knn']
    vizinhos_df = vizinhos[[c for c in cols_viz if c in vizinhos.columns]]

    return {
        'notas_prev': notas_prev,
        'apr_pred':   apr_pred,
        'prob_apr':   prob_apr,
        'vizinhos':   vizinhos_df,
    }


# ---------------------------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("⚙️ Configurações")
    k_vizinhos = st.slider(
        "Número de vizinhos (K)",
        min_value=3, max_value=21, value=7, step=2,
        help="K ímpar evita empate na classificação."
    )
    st.divider()
    st.caption("AV1 - Sistemas de Apoio à Decisão")
    st.caption("Sistemas de Informação")
    st.divider()
    st.info(
        "**K-NN passo a passo:**\n\n"
        "1. Recebe perfil do candidato\n"
        "2. Normaliza atributos (Min-Max)\n"
        "3. Calcula distância Euclidiana\n"
        "4. Seleciona os K vizinhos mais próximos\n"
        "5. Prevê nota pela média ponderada (peso = 1/d)\n"
        "6. Gera recomendações pedagógicas"
    )

# ---------------------------------------------------------------------------
# CARREGAR E TREINAR
# ---------------------------------------------------------------------------
art = treinar_pipeline(k_vizinhos)
df_clean = art['df_clean']
notas_p  = art['notas_validas']

# ---------------------------------------------------------------------------
# CABEÇALHO
# ---------------------------------------------------------------------------
st.title("🎓 ENCCEJA 2024 - Sistema de Apoio à Decisão Educacional")
st.markdown(
    "**Disciplina:** Sistemas de Apoio à Decisão &nbsp;|&nbsp; "
    "**Modelo:** K-Nearest Neighbors (K-NN) &nbsp;|&nbsp; "
    "**Dados:** Microdados ENCCEJA 2024 - INEP &nbsp;|&nbsp; "
    "**Candidatos regulares presentes:** "
    f"**{len(df_clean):,}**".replace(",", ".")
)
st.divider()

# ---------------------------------------------------------------------------
# SEÇÃO 1 - VISÃO GERAL
# ---------------------------------------------------------------------------
st.header("1. Visão Geral dos Dados")

total     = len(df_clean)
aprovados = int(df_clean['aprovado_geral'].sum())
taxa_apr  = aprovados / total * 100

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Candidatos Presentes", f"{total:,}".replace(",", "."))
c2.metric("Aprovados (todas as áreas)", f"{aprovados:,}".replace(",", "."), f"{taxa_apr:.1f}%")
c3.metric("Reprovados", f"{total-aprovados:,}".replace(",", "."), f"{100-taxa_apr:.1f}%")

if 'NU_NOTA_MT' in df_clean.columns:
    c4.metric("Média MT (0-200)", f"{df_clean['NU_NOTA_MT'].mean():.1f}")
if 'NU_NOTA_REDACAO' in df_clean.columns:
    c5.metric("Média Redação (0-10)", f"{df_clean['NU_NOTA_REDACAO'].mean():.2f}")

st.divider()

col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Distribuição das Notas por Disciplina")
    notas_obj = [n for n in notas_p if n != 'NU_NOTA_REDACAO']
    if notas_obj:
        fig, ax = plt.subplots(figsize=(6, 4))
        cores = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']
        for nota, cor in zip(notas_obj, cores):
            dados = df_clean[nota].dropna()
            ax.hist(dados, bins=30, alpha=0.65,
                    label=NOMES_NOTAS.get(nota, nota), color=cor, density=True)
        ax.axvline(THRESHOLD_OBJETIVO, color='red', linestyle='--',
                   linewidth=1.5, label=f'Mínimo ({THRESHOLD_OBJETIVO})')
        ax.set_xlabel('Nota (0 a 200)', fontsize=10)
        ax.set_ylabel('Densidade', fontsize=10)
        ax.set_title('Distribuição - Áreas Objetivas', fontweight='bold')
        ax.legend(fontsize=7, loc='upper left')
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

with col_b:
    st.subheader("Distribuição da Redação (0-10)")
    if 'NU_NOTA_REDACAO' in df_clean.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        dados_red = df_clean['NU_NOTA_REDACAO'].dropna()
        ax.hist(dados_red, bins=25, color='#9b59b6', alpha=0.85, edgecolor='white')
        ax.axvline(THRESHOLD_REDACAO, color='red', linestyle='--',
                   linewidth=1.5, label=f'Mínimo ({THRESHOLD_REDACAO})')
        ax.set_xlabel('Nota da Redação (0 a 10)', fontsize=10)
        ax.set_ylabel('Frequência', fontsize=10)
        ax.set_title('Distribuição - Redação', fontweight='bold')
        ax.legend(fontsize=9)
        ax.spines[['top', 'right']].set_visible(False)
        st.pyplot(fig)
        plt.close()

# ---------------------------------------------------------------------------
# SEÇÃO 2 - DESEMPENHO DO MODELO
# ---------------------------------------------------------------------------
st.divider()
st.header("2. Desempenho do Modelo K-NN")

acc_apr = art['acc_apr']
cm      = art['cm']
metricas = art['metricas']

cm1, cm2, cm3 = st.columns(3)
cm1.metric("Acurácia (Aprovação)", f"{acc_apr*100:.2f}%")
cm2.metric("K Vizinhos", k_vizinhos)
cm3.metric("Normalização", "Min-Max")

col_r1, col_r2 = st.columns(2)

with col_r1:
    st.subheader("Erro de Previsão por Disciplina (MAE)")
    df_met = pd.DataFrame(metricas)
    fig, ax = plt.subplots(figsize=(6, 4))
    cores_met = ['#e74c3c' if v > 15 else '#27ae60' for v in df_met['mae']]
    bars = ax.barh(df_met['nome'][::-1], df_met['mae'][::-1],
                   color=cores_met[::-1], edgecolor='white')
    for bar, val in zip(bars, df_met['mae'][::-1]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}', va='center', fontsize=9, fontweight='bold')
    ax.set_xlabel('MAE (Erro Absoluto Médio)')
    ax.set_title(f'MAE por Disciplina - K={k_vizinhos}', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    st.pyplot(fig)
    plt.close()

with col_r2:
    st.subheader("Matriz de Confusão - Aprovação")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=['Reprovado', 'Aprovado'],
                yticklabels=['Reprovado', 'Aprovado'],
                cmap='Blues', ax=ax, linewidths=0.5, linecolor='gray',
                annot_kws={'size': 13})
    ax.set_xlabel('Previsto', fontsize=11)
    ax.set_ylabel('Real', fontsize=11)
    ax.set_title('Matriz de Confusão', fontweight='bold')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with st.expander("Ver métricas completas por disciplina"):
    df_show = pd.DataFrame(metricas)[['nome', 'mae', 'rmse', 'r2']].copy()
    df_show.columns = ['Disciplina', 'MAE', 'RMSE', 'R²']
    st.dataframe(df_show.round(3), use_container_width=True)

# ---------------------------------------------------------------------------
# SEÇÃO 3 - ANÁLISE EXPLORATÓRIA
# ---------------------------------------------------------------------------
st.divider()
st.header("3. Análise Exploratória")

col_e1, col_e2 = st.columns(2)

with col_e1:
    st.subheader("Aprovação por Faixa Etária")
    faixa_apr = (df_clean.groupby('TP_FAIXA_ETARIA')['aprovado_geral']
                 .mean() * 100).rename(index=MAPA_FAIXA_ETARIA)
    fig, ax = plt.subplots(figsize=(6, 5))
    cores_f = ['#e74c3c' if v < 50 else '#27ae60' for v in faixa_apr.values]
    ax.barh(faixa_apr.index, faixa_apr.values, color=cores_f, edgecolor='white')
    ax.axvline(50, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Taxa de Aprovação (%)')
    ax.set_title('Aprovação por Faixa Etária', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_e2:
    st.subheader("Aprovação por Renda Familiar (Q50)")
    renda_apr = (df_clean.groupby('Q50')['aprovado_geral']
                 .mean() * 100).rename(index=MAPA_Q50)
    renda_apr = renda_apr[renda_apr.index.notna()]
    fig, ax = plt.subplots(figsize=(6, 5))
    cores_r = ['#e74c3c' if v < 50 else '#27ae60' for v in renda_apr.values]
    ax.barh(renda_apr.index, renda_apr.values, color=cores_r, edgecolor='white')
    ax.axvline(50, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Taxa de Aprovação (%)')
    ax.set_title('Aprovação por Renda Familiar', fontweight='bold')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Aprovação por trabalho e certificação
col_e3, col_e4 = st.columns(2)

with col_e3:
    st.subheader("Aprovação por Situação de Trabalho (Q44)")
    trab_apr = (df_clean.groupby('Q44')['aprovado_geral']
                .mean() * 100).rename(index=MAPA_Q44)
    fig, ax = plt.subplots(figsize=(6, 3))
    cores_t = ['#e74c3c' if v < 50 else '#27ae60' for v in trab_apr.values]
    ax.barh(trab_apr.index, trab_apr.values, color=cores_t, edgecolor='white')
    ax.axvline(50, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Taxa de Aprovação (%)')
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with col_e4:
    st.subheader("Aprovação por Tipo de Certificação")
    cert_apr = (df_clean.groupby('TP_CERTIFICACAO')['aprovado_geral']
                .mean() * 100).rename(index=MAPA_CERTIFICACAO)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(cert_apr.index.astype(str), cert_apr.values,
           color=['#3498db', '#e74c3c'], edgecolor='white', width=0.4)
    for i, v in enumerate(cert_apr.values):
        ax.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')
    ax.set_ylabel('Taxa de Aprovação (%)')
    ax.set_ylim(0, 100)
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ---------------------------------------------------------------------------
# SEÇÃO 4 - ANÁLISE DE NOVO CANDIDATO
# ---------------------------------------------------------------------------
st.divider()
st.header("4. Análise de Novo Candidato")
st.markdown(
    "Preencha o perfil socioeconômico do candidato. O sistema identificará "
    f"os **{k_vizinhos} candidatos mais similares** dos microdados do ENCCEJA 2024 "
    "e preverá as notas esperadas."
)

with st.form("form_candidato"):
    col_f1, col_f2, col_f3 = st.columns(3)

    with col_f1:
        sexo = st.selectbox("Sexo", ['M', 'F'],
                            format_func=lambda x: MAPA_SEXO[x])
        faixa = st.selectbox(
            "Faixa Etária",
            options=list(MAPA_FAIXA_ETARIA.keys()),
            index=10,
            format_func=lambda x: MAPA_FAIXA_ETARIA[x]
        )
        uf = st.selectbox("UF (local da prova)", UFS_BRASIL,
                          index=UFS_BRASIL.index('SP'))

    with col_f2:
        cert = st.selectbox(
            "Certificação Pretendida",
            options=[1, 2],
            format_func=lambda x: MAPA_CERTIFICACAO[x]
        )
        q44 = st.selectbox(
            "Situação de Trabalho",
            options=list(MAPA_Q44.keys()),
            format_func=lambda x: MAPA_Q44[x]
        )

    with col_f3:
        q50 = st.selectbox(
            "Renda Familiar Mensal",
            options=list(MAPA_Q50.keys()),
            index=1,
            format_func=lambda x: MAPA_Q50[x]
        )
        q11 = st.selectbox(
            "Última Série Estudada",
            options=list(MAPA_Q11.keys()),
            index=8,
            format_func=lambda x: MAPA_Q11[x]
        )

    analisar = st.form_submit_button("🔍 Analisar Candidato", use_container_width=True)

# ---------------------------------------------------------------------------
# RESULTADO
# ---------------------------------------------------------------------------
if analisar:
    candidato = {
        'TP_SEXO':          sexo,
        'TP_FAIXA_ETARIA':  faixa,
        'SG_UF_PROVA':      uf,
        'TP_CERTIFICACAO':  cert,
        'Q44':              q44,
        'Q50':              q50,
        'Q11':              q11,
    }

    with st.spinner("Calculando distâncias e identificando vizinhos..."):
        res = prever(candidato, art)

    notas_prev = res['notas_prev']
    apr_pred   = res['apr_pred']
    prob_apr   = res['prob_apr']
    vizinhos   = res['vizinhos']
    taxa_viz   = vizinhos['aprovado_geral'].mean() * 100 if 'aprovado_geral' in vizinhos.columns else 0

    st.divider()
    st.subheader("Resultado da Análise K-NN")

    if apr_pred == 1:
        st.success(
            f"✅ **PROVÁVEL APROVAÇÃO** &nbsp;|&nbsp; "
            f"Probabilidade: **{prob_apr[1]*100:.1f}%** &nbsp;|&nbsp; "
            f"**{taxa_viz:.0f}%** dos {k_vizinhos} vizinhos foram aprovados"
        )
    else:
        st.error(
            f"⚠️ **RISCO DE REPROVAÇÃO** &nbsp;|&nbsp; "
            f"Probabilidade de aprovação: **{prob_apr[1]*100:.1f}%** &nbsp;|&nbsp; "
            f"Apenas **{taxa_viz:.0f}%** dos {k_vizinhos} vizinhos foram aprovados"
        )

    # Notas previstas com métricas
    st.subheader("Notas Previstas")
    cols_notas = st.columns(len(notas_p))
    for col_n, nota in zip(cols_notas, notas_p):
        val      = notas_prev.get(nota, 0)
        mviz     = vizinhos[nota].mean() if nota in vizinhos.columns else 0
        delta    = val - mviz
        thr      = THRESHOLD_REDACAO if nota == 'NU_NOTA_REDACAO' else THRESHOLD_OBJETIVO
        escala   = "(0-10)" if nota == 'NU_NOTA_REDACAO' else "(0-200)"
        cor_d    = "normal" if val >= thr else "inverse"
        col_n.metric(
            label=f"{ICONES_NOTAS.get(nota,'')} {NOMES_NOTAS.get(nota,nota)}\n{escala}",
            value=f"{val:.1f}",
            delta=f"{delta:+.1f} vs vizinhos",
            delta_color=cor_d,
            help=f"Mínimo para aprovação: {thr}"
        )

    # Gráficos de comparação
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.subheader(f"Candidato vs {k_vizinhos} Vizinhos")

        # Separar notas objetivas da redação para escala dupla
        notas_obj = [n for n in notas_p if n != 'NU_NOTA_REDACAO']
        notas_red = [n for n in notas_p if n == 'NU_NOTA_REDACAO']

        fig, ax1 = plt.subplots(figsize=(7, 4.5))

        nomes_obj  = [NOMES_NOTAS.get(n, n) for n in notas_obj]
        vals_c_obj = [notas_prev.get(n, 0) for n in notas_obj]
        vals_v_obj = [vizinhos[n].mean() for n in notas_obj if n in vizinhos.columns]

        x = np.arange(len(notas_obj))
        ax1.bar(x - 0.2, vals_c_obj, 0.35, label='Candidato (previsto)',
                color='#3498db', edgecolor='white')
        ax1.bar(x + 0.2, vals_v_obj, 0.35, label='Média dos vizinhos',
                color='#e67e22', alpha=0.85, edgecolor='white')
        ax1.axhline(THRESHOLD_OBJETIVO, color='red', linestyle='--',
                    linewidth=1.5, label=f'Mínimo obj. ({THRESHOLD_OBJETIVO})')
        ax1.set_xticks(x)
        ax1.set_xticklabels(nomes_obj, rotation=12, ha='right', fontsize=9)
        ax1.set_ylabel('Nota (0-200)')
        ax1.set_title('Previsão vs Vizinhos - Áreas Objetivas', fontweight='bold')
        ax1.set_ylim(0, 220)
        ax1.legend(fontsize=8)
        ax1.spines[['top', 'right']].set_visible(False)

        if notas_red:
            nota_r   = notas_red[0]
            val_c_r  = notas_prev.get(nota_r, 0)
            val_v_r  = vizinhos[nota_r].mean() if nota_r in vizinhos.columns else 0
            ax2 = ax1.twinx()
            ax2.scatter(['Redação (candidato)'], [val_c_r], color='#3498db',
                        s=100, zorder=5, marker='D')
            ax2.scatter(['Redação (vizinhos)'], [val_v_r], color='#e67e22',
                        s=100, zorder=5, marker='D')
            ax2.set_ylabel('Nota Redação (0-10)', color='#9b59b6')
            ax2.set_ylim(0, 11)
            ax2.tick_params(axis='y', labelcolor='#9b59b6')

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_g2:
        st.subheader("Distribuição dos Vizinhos por Área")
        fig, axes = plt.subplots(1, len(notas_obj), figsize=(2.8 * len(notas_obj), 4))
        if len(notas_obj) == 1:
            axes = [axes]
        cores_box = ['#3498db', '#e74c3c', '#27ae60', '#f39c12']

        for ax, nota, cor in zip(axes, notas_obj, cores_box):
            if nota in vizinhos.columns:
                dados = vizinhos[nota].dropna()
                ax.boxplot(dados, patch_artist=True,
                           boxprops=dict(facecolor=cor, alpha=0.7),
                           medianprops=dict(color='white', linewidth=2))
                val = notas_prev.get(nota, 0)
                ax.scatter([1], [val], color='#2c3e50', s=90, zorder=5)
                ax.axhline(THRESHOLD_OBJETIVO, color='red', linestyle='--',
                           linewidth=1, alpha=0.7)
                ax.set_title(NOMES_NOTAS.get(nota, nota)[:14],
                             fontsize=8, fontweight='bold')
                ax.set_ylim(0, 220)
                ax.set_xticks([])
                ax.spines[['top', 'right']].set_visible(False)

        plt.suptitle('Box: vizinhos | Ponto: candidato', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    # Vizinhos detalhados
    with st.expander(f"📋 Ver os {k_vizinhos} vizinhos identificados"):
        viz_show = vizinhos.copy()
        for nota in notas_p:
            if nota in viz_show.columns:
                viz_show[nota] = viz_show[nota].round(1)
        viz_show['distancia_knn'] = viz_show['distancia_knn'].round(4)
        viz_show['aprovado_geral'] = viz_show['aprovado_geral'].map(
            {1: '✅ Aprovado', 0: '❌ Reprovado'}
        )
        rename_map = {n: f"{ICONES_NOTAS.get(n,'')} {NOMES_NOTAS.get(n,n)}" for n in notas_p}
        rename_map['aprovado_geral'] = 'Situação'
        rename_map['distancia_knn']  = 'Distância (K-NN)'
        viz_show = viz_show.rename(columns=rename_map)
        st.dataframe(viz_show.reset_index(drop=True), use_container_width=True)

    # ---------------------------------------------------------------------------
    # RECOMENDAÇÕES PEDAGÓGICAS
    # ---------------------------------------------------------------------------
    st.divider()
    st.subheader("📋 Recomendações Pedagógicas ao Gestor")

    # Disciplinas críticas (abaixo do threshold)
    criticas = []
    for nota in notas_p:
        val = notas_prev.get(nota, 0)
        thr = THRESHOLD_REDACAO if nota == 'NU_NOTA_REDACAO' else THRESHOLD_OBJETIVO
        if val < thr:
            criticas.append((nota, val, thr))

    col_rec1, col_rec2, col_rec3 = st.columns(3)

    if taxa_viz >= 70:
        nivel = "Padrão"
        col_rec1.success(f"**Risco:** Baixo\n\n{taxa_viz:.0f}% dos vizinhos aprovados")
    elif taxa_viz >= 40:
        nivel = "Intensificado"
        col_rec1.warning(f"**Risco:** Moderado\n\n{taxa_viz:.0f}% dos vizinhos aprovados")
    else:
        nivel = "Intensivo"
        col_rec1.error(f"**Risco:** Alto\n\nApenas {taxa_viz:.0f}% dos vizinhos aprovados")

    col_rec2.info(f"**Acompanhamento sugerido:** {nivel}")
    if criticas:
        col_rec3.warning(f"**Disciplinas abaixo da meta:** {len(criticas)} de {len(notas_p)}")
    else:
        col_rec3.success("**Todas as disciplinas** acima do mínimo ✓")

    # Plano por disciplina crítica
    recomendacoes_disc = {
        'NU_NOTA_LC':      "Intensificar leitura de textos variados. Revisar gramática, interpretação textual e produção de texto.",
        'NU_NOTA_CH':      "Focar em História do Brasil, História Geral e Geografia. Praticar leitura e interpretação de mapas e fontes históricas.",
        'NU_NOTA_MT':      "Revisar operações básicas, álgebra, geometria e estatística. Resolver problemas contextualizados do cotidiano.",
        'NU_NOTA_CN':      "Revisar Biologia, Química e Física básica. Praticar exercícios com contexto ambiental e de saúde.",
        'NU_NOTA_REDACAO': "Praticar redações semanais. Revisar estrutura dissertativa, coesão e coerência. Trabalhar argumentação.",
    }

    if criticas:
        st.subheader("Plano de Reforço")
        cols_d = st.columns(min(len(criticas), 3))
        for i, (nota, val, thr) in enumerate(criticas):
            with cols_d[i % 3]:
                with st.container(border=True):
                    nome  = NOMES_NOTAS.get(nota, nota)
                    icone = ICONES_NOTAS.get(nota, '📖')
                    escala = "(0-10)" if nota == 'NU_NOTA_REDACAO' else "(0-200)"
                    st.markdown(f"**{icone} {nome}**")
                    st.markdown(f"Nota prevista: **{val:.1f}** {escala}")
                    st.markdown(f"Mínimo: **{thr}** | Déficit: **{thr - val:.1f} pts**")
                    st.caption(recomendacoes_disc.get(nota, "Revisar conteúdos da disciplina."))

    # Resumo final
    st.subheader("Resumo para o Gestor")
    with st.container(border=True):
        mviz_geral = vizinhos[notas_p].mean().mean() if notas_p else 0
        mprev_geral = np.mean(list(notas_prev.values()))

        st.markdown(f"""
**Perfil do Candidato:**
| Campo | Valor |
|---|---|
| Sexo | {MAPA_SEXO.get(sexo, sexo)} |
| Faixa etária | {MAPA_FAIXA_ETARIA.get(faixa, faixa)} |
| UF da prova | {uf} |
| Certificação | {MAPA_CERTIFICACAO.get(cert, cert)} |
| Trabalho | {MAPA_Q44.get(q44, q44)} |
| Renda familiar | {MAPA_Q50.get(q50, q50)} |
| Última série | {MAPA_Q11.get(q11, q11)} |

**Análise K-NN (K = {k_vizinhos} vizinhos mais próximos):**
- Taxa de aprovação entre vizinhos: **{taxa_viz:.1f}%**
- Previsão de aprovação: **{"APROVADO ✅" if apr_pred == 1 else "RISCO DE REPROVAÇÃO ⚠️"}** (prob. {prob_apr[1]*100:.1f}%)
- Nível de acompanhamento recomendado: **{nivel}**
- Disciplinas abaixo do mínimo: **{len(criticas)} de {len(notas_p)}**
{"- Disciplinas críticas: **" + ", ".join([NOMES_NOTAS.get(n,'') for n,_,_ in criticas]) + "**" if criticas else "- ✅ Todas as disciplinas acima do mínimo"}
        """)

# ---------------------------------------------------------------------------
# RODAPÉ
# ---------------------------------------------------------------------------
st.divider()
st.caption(
    "Fonte: INEP - Microdados ENCCEJA 2024 &nbsp;|&nbsp; "
    "Modelo: K-Nearest Neighbors (scikit-learn) &nbsp;|&nbsp; "
    "Normalização: Min-Max &nbsp;|&nbsp; "
    "Distância: Euclidiana &nbsp;|&nbsp; "
    "Pesos: 1/distância"
)
