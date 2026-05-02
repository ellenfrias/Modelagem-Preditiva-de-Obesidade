import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings('ignore')

# ─── Configuração da página ───────────────────────────────────────────────────
st.set_page_config(
    page_title="Preditor de Obesidade",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── CSS customizado ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        background-color: #2563eb;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton>button:hover { background-color: #1d4ed8; }
    .result-card {
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        margin: 1rem 0;
        font-size: 1.1rem;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    }
    .section-title {
        font-size: 1.1rem;
        font-weight: 600;
        color: #374151;
        margin-bottom: 0.5rem;
    }
    div[data-testid="stSidebar"] { background-color: #1e293b; }
    div[data-testid="stSidebar"] * { color: white !important; }
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] .stNumberInput label { color: #cbd5e1 !important; }
</style>
""", unsafe_allow_html=True)

# ─── Carregar modelo ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    pipeline = joblib.load("models/modelo_obesidade.pkl")
    with open("models/metadata.json") as f:
        meta = json.load(f)
    return pipeline, meta

pipeline, meta = load_model()

# ─── Configurações de classes ─────────────────────────────────────────────────
CLASS_CONFIG = {
    "Insufficient_Weight": {
        "label": "Abaixo do Peso",
        "emoji": "⚠️",
        "color": "#3b82f6",
        "bg": "#eff6ff",
        "imc": "< 18.5",
        "recomendacao": "Consulte um nutricionista. Ganho de peso saudável pode ser necessário."
    },
    "Normal_Weight": {
        "label": "Peso Normal",
        "emoji": "✅",
        "color": "#16a34a",
        "bg": "#f0fdf4",
        "imc": "18.5 – 24.9",
        "recomendacao": "Parabéns! Mantenha seus hábitos saudáveis de alimentação e atividade física."
    },
    "Overweight_Level_I": {
        "label": "Sobrepeso Nível I",
        "emoji": "🟡",
        "color": "#ca8a04",
        "bg": "#fefce8",
        "imc": "25.0 – 27.4",
        "recomendacao": "Atenção! Considere aumentar a frequência de exercícios e reduzir alimentos calóricos."
    },
    "Overweight_Level_II": {
        "label": "Sobrepeso Nível II",
        "emoji": "🟠",
        "color": "#ea580c",
        "bg": "#fff7ed",
        "imc": "27.5 – 29.9",
        "recomendacao": "Recomenda-se acompanhamento médico e nutricional para mudança de hábitos."
    },
    "Obesity_Type_I": {
        "label": "Obesidade Tipo I",
        "emoji": "🔴",
        "color": "#dc2626",
        "bg": "#fef2f2",
        "imc": "30.0 – 34.9",
        "recomendacao": "Encaminhar para acompanhamento médico multidisciplinar com urgência."
    },
    "Obesity_Type_II": {
        "label": "Obesidade Tipo II",
        "emoji": "🔴",
        "color": "#b91c1c",
        "bg": "#fef2f2",
        "imc": "35.0 – 39.9",
        "recomendacao": "Risco de complicações metabólicas. Acompanhamento médico intensivo necessário."
    },
    "Obesity_Type_III": {
        "label": "Obesidade Tipo III (Mórbida)",
        "emoji": "🚨",
        "color": "#7f1d1d",
        "bg": "#fef2f2",
        "imc": "≥ 40.0",
        "recomendacao": "Obesidade grave. Avaliação para intervenção cirúrgica pode ser indicada."
    }
}

ORDER = [
    "Insufficient_Weight", "Normal_Weight",
    "Overweight_Level_I", "Overweight_Level_II",
    "Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"
]

# ─── Sidebar — Formulário de entrada ─────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Dados do Paciente")
    st.markdown("---")

    st.markdown("### 👤 Dados Pessoais")
    gender = st.selectbox("Gênero", ["Female", "Male"],
                          format_func=lambda x: "Feminino" if x == "Female" else "Masculino")
    age    = st.slider("Idade (anos)", 14, 70, 25)
    height = st.slider("Altura (m)", 1.45, 1.98, 1.70, step=0.01, format="%.2f m")
    weight = st.slider("Peso (kg)", 39.0, 173.0, 75.0, step=0.5, format="%.1f kg")

    st.markdown("---")
    st.markdown("### 🍽️ Hábitos Alimentares")

    family_history = st.selectbox(
        "Histórico familiar de obesidade?",
        ["yes", "no"], format_func=lambda x: "Sim" if x == "yes" else "Não"
    )
    favc = st.selectbox(
        "Consome alimentos calóricos com frequência?",
        ["yes", "no"], format_func=lambda x: "Sim" if x == "yes" else "Não"
    )
    fcvc = st.select_slider(
        "Frequência de consumo de vegetais",
        options=[1, 2, 3],
        value=2,
        format_func=lambda x: {1: "Raramente", 2: "Às vezes", 3: "Sempre"}[x]
    )
    ncp = st.select_slider(
        "Número de refeições principais/dia",
        options=[1, 2, 3, 4],
        value=3,
        format_func=lambda x: {1: "1 refeição", 2: "2 refeições", 3: "3 refeições", 4: "4+"}[x]
    )
    caec = st.selectbox(
        "Come entre as refeições?",
        ["no", "Sometimes", "Frequently", "Always"],
        index=1,
        format_func=lambda x: {"no": "Não", "Sometimes": "Às vezes",
                                "Frequently": "Frequentemente", "Always": "Sempre"}[x]
    )
    ch2o = st.select_slider(
        "Consumo diário de água",
        options=[1, 2, 3],
        value=2,
        format_func=lambda x: {1: "< 1 L/dia", 2: "1 – 2 L/dia", 3: "> 2 L/dia"}[x]
    )

    st.markdown("---")
    st.markdown("### 🏃 Estilo de Vida")

    faf = st.select_slider(
        "Frequência de atividade física/semana",
        options=[0, 1, 2, 3],
        value=1,
        format_func=lambda x: {0: "Sedentário", 1: "1–2×/sem",
                                2: "3–4×/sem", 3: "5+×/sem"}[x]
    )
    tue = st.select_slider(
        "Tempo com eletrônicos/dia",
        options=[0, 1, 2],
        value=1,
        format_func=lambda x: {0: "0–2 h", 1: "3–5 h", 2: "> 5 h"}[x]
    )
    smoke = st.selectbox(
        "Fumante?",
        ["no", "yes"], format_func=lambda x: "Não" if x == "no" else "Sim"
    )
    scc = st.selectbox(
        "Monitora calorias ingeridas?",
        ["no", "yes"], format_func=lambda x: "Não" if x == "no" else "Sim"
    )
    calc = st.selectbox(
        "Frequência de consumo de álcool",
        ["no", "Sometimes", "Frequently", "Always"],
        format_func=lambda x: {"no": "Não bebe", "Sometimes": "Às vezes",
                                "Frequently": "Frequentemente", "Always": "Sempre"}[x]
    )
    mtrans = st.selectbox(
        "Meio de transporte habitual",
        ["Public_Transportation", "Automobile", "Walking", "Bike", "Motorbike"],
        format_func=lambda x: {
            "Public_Transportation": "Transporte público",
            "Automobile": "Carro",
            "Walking": "A pé",
            "Bike": "Bicicleta",
            "Motorbike": "Moto"
        }[x]
    )

    st.markdown("---")
    predict_btn = st.button("🔍 Realizar Diagnóstico")

# ─── Área principal ───────────────────────────────────────────────────────────
st.markdown("# 🏥 Sistema Preditivo de Obesidade")
st.markdown("**Ferramenta de apoio à decisão clínica** — Modelo de Machine Learning para triagem de risco de obesidade")
st.markdown("---")

# Métricas do modelo no topo
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="metric-card"><h3>🎯 Acurácia</h3><h2 style="color:#16a34a">'
                f'{meta["accuracy"]:.1%}</h2><p>no conjunto de teste</p></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="metric-card"><h3>📊 F1-Macro</h3><h2 style="color:#2563eb">'
                f'{meta["f1_macro"]:.1%}</h2><p>média entre classes</p></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="metric-card"><h3>🗂️ Classes</h3><h2 style="color:#7c3aed">7</h2>'
                '<p>níveis de obesidade</p></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="metric-card"><h3>🌲 Modelo</h3><h2 style="color:#ea580c" style="font-size:1rem">RF</h2>'
                '<p>Random Forest</p></div>', unsafe_allow_html=True)

st.markdown("---")

if not predict_btn:
    st.info("👈 **Preencha os dados do paciente na barra lateral e clique em 'Realizar Diagnóstico'.**")

    st.markdown("### ℹ️ Como usar este sistema")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Dados necessários:**
        - Dados pessoais: gênero, idade, altura e peso
        - Hábitos alimentares: consumo calórico, vegetais, refeições, água
        - Estilo de vida: atividade física, eletrônicos, transporte
        - Histórico: familiar, tabagismo, álcool
        """)
    with col2:
        st.markdown("""
        **O sistema retorna:**
        - Classificação do nível de obesidade
        - Probabilidade por classe
        - IMC calculado
        - Recomendação clínica
        - Fatores de risco do paciente
        """)

    st.markdown("---")
    st.markdown("> ⚠️ **Aviso:** Este sistema é uma ferramenta de **apoio à decisão**. "
                "O diagnóstico definitivo deve sempre ser realizado por um profissional de saúde.")

else:
    # ── Montar dataframe de entrada ─────────────────────────────────────────
    input_data = pd.DataFrame([{
        "Gender": gender, "Age": age, "Height": height, "Weight": weight,
        "family_history": family_history, "FAVC": favc,
        "FCVC": fcvc, "NCP": ncp, "CAEC": caec, "SMOKE": smoke,
        "CH2O": ch2o, "SCC": scc, "FAF": faf, "TUE": tue,
        "CALC": calc, "MTRANS": mtrans
    }])

    prediction   = pipeline.predict(input_data)[0]
    probabilities = pipeline.predict_proba(input_data)[0]
    bmi = weight / (height ** 2)
    cfg = CLASS_CONFIG[prediction]

    # ── Layout de resultado ─────────────────────────────────────────────────
    col_res, col_prob = st.columns([1, 1])

    with col_res:
        st.markdown("### 📋 Resultado do Diagnóstico")
        st.markdown(f"""
        <div class="result-card" style="background:{cfg['bg']}; border: 2px solid {cfg['color']};">
            <div style="font-size:2.5rem">{cfg['emoji']}</div>
            <div style="font-size:1.6rem; font-weight:700; color:{cfg['color']}; margin: 0.5rem 0">
                {cfg['label']}
            </div>
            <div style="color:#6b7280; font-size:0.95rem">Faixa de IMC correspondente: {cfg['imc']}</div>
        </div>
        """, unsafe_allow_html=True)

        # IMC calculado
        st.markdown(f"""
        <div style="background:white; border-radius:10px; padding:1rem; margin-top:0.5rem;
                    box-shadow:0 1px 4px rgba(0,0,0,0.08); text-align:center">
            <b>IMC Calculado</b><br>
            <span style="font-size:2rem; font-weight:700; color:{cfg['color']}">{bmi:.1f}</span>
            <span style="color:#6b7280"> kg/m²</span>
        </div>
        """, unsafe_allow_html=True)

        # Recomendação
        st.markdown(f"""
        <div style="background:#f1f5f9; border-left: 4px solid {cfg['color']};
                    border-radius:0 8px 8px 0; padding:1rem; margin-top:0.8rem">
            <b>💊 Recomendação Clínica</b><br>
            <span style="color:#374151">{cfg['recomendacao']}</span>
        </div>
        """, unsafe_allow_html=True)

        # Confiança
        pred_idx  = list(pipeline.classes_).index(prediction)
        confidence = probabilities[pred_idx]
        st.markdown(f"""
        <div style="background:white; border-radius:10px; padding:1rem; margin-top:0.8rem;
                    box-shadow:0 1px 4px rgba(0,0,0,0.08); text-align:center">
            <b>🎯 Confiança do Modelo</b><br>
            <span style="font-size:1.8rem; font-weight:700; color:{cfg['color']}">{confidence:.1%}</span>
        </div>
        """, unsafe_allow_html=True)

    with col_prob:
        st.markdown("### 📊 Probabilidade por Classe")

        # Gráfico de probabilidades
        classes_labels = [CLASS_CONFIG[c]["label"] for c in pipeline.classes_]
        colors_bar     = [CLASS_CONFIG[c]["color"] for c in pipeline.classes_]
        alphas         = [1.0 if c == prediction else 0.4 for c in pipeline.classes_]

        fig, ax = plt.subplots(figsize=(7, 4))
        bars = ax.barh(classes_labels, probabilities, color=colors_bar, alpha=0.85)
        for bar, prob, alpha in zip(bars, probabilities, alphas):
            bar.set_alpha(alpha)
            if prob > 0.01:
                ax.text(prob + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{prob:.1%}', va='center', fontsize=9)
        ax.set_xlabel("Probabilidade")
        ax.set_xlim(0, 1.1)
        ax.set_title("Distribuição de probabilidade por classe", fontsize=11)
        ax.axvline(0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Ranking
        prob_df = pd.DataFrame({
            "Classe": classes_labels,
            "Probabilidade": [f"{p:.1%}" for p in probabilities]
        }).sort_values("Probabilidade", ascending=False).reset_index(drop=True)
        prob_df.index = prob_df.index + 1
        st.dataframe(prob_df, use_container_width=True, height=260)

    # ── Fatores de risco do paciente ────────────────────────────────────────
    st.markdown("---")
    st.markdown("### ⚠️ Análise de Fatores de Risco do Paciente")

    risk_col1, risk_col2, risk_col3 = st.columns(3)

    risks = []
    protections = []

    if family_history == "yes":
        risks.append("🔴 Histórico familiar de obesidade presente")
    else:
        protections.append("🟢 Sem histórico familiar de obesidade")

    if favc == "yes":
        risks.append("🔴 Consome alimentos calóricos com frequência")
    else:
        protections.append("🟢 Não consome alimentos calóricos frequentemente")

    if faf == 0:
        risks.append("🔴 Sedentário — nenhuma atividade física")
    elif faf == 1:
        risks.append("🟡 Atividade física insuficiente (1–2×/sem)")
    else:
        protections.append(f"🟢 Pratica atividade física {['3–4×', '5+×'][faf-2]}/semana")

    if caec in ["Frequently", "Always"]:
        risks.append("🔴 Come entre as refeições com frequência")
    elif caec == "Sometimes":
        risks.append("🟡 Come entre as refeições às vezes")

    if ch2o == 1:
        risks.append("🟡 Baixo consumo de água (< 1 L/dia)")
    elif ch2o == 3:
        protections.append("🟢 Boa hidratação (> 2 L/dia)")

    if smoke == "yes":
        risks.append("🔴 Fumante")

    if calc in ["Frequently", "Always"]:
        risks.append("🔴 Consome álcool com frequência")

    if mtrans in ["Walking", "Bike"]:
        protections.append("🟢 Locomove-se de forma ativa (a pé/bicicleta)")
    elif mtrans == "Automobile":
        risks.append("🟡 Usa automóvel (hábito sedentário)")

    if scc == "yes":
        protections.append("🟢 Monitora calorias ingeridas")
    else:
        risks.append("🟡 Não monitora calorias")

    if fcvc == 3:
        protections.append("🟢 Consome vegetais frequentemente")
    elif fcvc == 1:
        risks.append("🟡 Baixo consumo de vegetais")

    with risk_col1:
        st.markdown("**Fatores de Risco**")
        if risks:
            for r in risks:
                st.markdown(r)
        else:
            st.markdown("✅ Nenhum fator de risco identificado")

    with risk_col2:
        st.markdown("**Fatores Protetores**")
        if protections:
            for p in protections:
                st.markdown(p)
        else:
            st.markdown("⚠️ Nenhum fator protetor identificado")

    with risk_col3:
        st.markdown("**Resumo**")
        total = len(risks) + len(protections)
        risk_pct = len(risks) / total if total > 0 else 0
        st.metric("Fatores de risco", len(risks))
        st.metric("Fatores protetores", len(protections))

        if risk_pct > 0.6:
            st.error("Alto risco comportamental")
        elif risk_pct > 0.3:
            st.warning("Risco comportamental moderado")
        else:
            st.success("Perfil comportamental favorável")

    # ── Dados inseridos ─────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📄 Ver dados completos inseridos"):
        labels = {
            "Gender": ("Gênero", lambda x: "Feminino" if x == "Female" else "Masculino"),
            "Age": ("Idade", lambda x: f"{x} anos"),
            "Height": ("Altura", lambda x: f"{x:.2f} m"),
            "Weight": ("Peso", lambda x: f"{x:.1f} kg"),
            "family_history": ("Histórico familiar", lambda x: "Sim" if x == "yes" else "Não"),
            "FAVC": ("Consome alimentos calóricos", lambda x: "Sim" if x == "yes" else "Não"),
            "FCVC": ("Consumo de vegetais", lambda x: {1: "Raramente", 2: "Às vezes", 3: "Sempre"}[x]),
            "NCP": ("Refeições/dia", lambda x: str(x)),
            "CAEC": ("Come entre refeições", lambda x: {"no": "Não", "Sometimes": "Às vezes",
                                                         "Frequently": "Frequentemente", "Always": "Sempre"}[x]),
            "SMOKE": ("Fuma", lambda x: "Sim" if x == "yes" else "Não"),
            "CH2O": ("Água/dia", lambda x: {1: "< 1 L", 2: "1–2 L", 3: "> 2 L"}[x]),
            "SCC": ("Monitora calorias", lambda x: "Sim" if x == "yes" else "Não"),
            "FAF": ("Atividade física", lambda x: {0: "Nenhuma", 1: "1–2×/sem",
                                                    2: "3–4×/sem", 3: "5+×/sem"}[x]),
            "TUE": ("Tempo c/ eletrônicos", lambda x: {0: "0–2 h", 1: "3–5 h", 2: "> 5 h"}[x]),
            "CALC": ("Álcool", lambda x: {"no": "Não bebe", "Sometimes": "Às vezes",
                                           "Frequently": "Frequentemente", "Always": "Sempre"}[x]),
            "MTRANS": ("Transporte", lambda x: {
                "Public_Transportation": "Transporte público", "Automobile": "Carro",
                "Walking": "A pé", "Bike": "Bicicleta", "Motorbike": "Moto"}[x]),
        }
        rows = [{"Campo": v[0], "Valor": v[1](input_data[k].iloc[0])} for k, v in labels.items()]
        rows.append({"Campo": "IMC Calculado", "Valor": f"{bmi:.1f} kg/m²"})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.caption("⚕️ Sistema desenvolvido para fins acadêmicos (Tech Challenge FIAP — Fase 4). "
               "Não substitui avaliação clínica profissional.")
