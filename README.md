# 🏥 Sistema Preditivo de Obesidade — Streamlit App

Aplicação de Machine Learning para triagem de risco de obesidade, desenvolvida como
entregável do Tech Challenge Fase 4 — FIAP Postech Data Analytics.

---

## 📁 Estrutura de Arquivos

```
app_streamlit/
├── app.py                  ← Código principal da aplicação
├── requirements.txt        ← Dependências Python
├── README.md               ← Este arquivo
└── models/
    ├── modelo_obesidade.pkl   ← Modelo treinado (Random Forest)
    └── metadata.json          ← Métricas e metadados do modelo
```

---

## 🚀 Como Rodar Localmente

1. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Execute o app:**
   ```bash
   streamlit run app.py
   ```

3. Acesse no navegador: `http://localhost:8501`

---

## ☁️ Como Fazer Deploy no Streamlit Cloud (gratuito)

1. **Suba o projeto para o GitHub:**
   - Crie um repositório no GitHub
   - Faça upload de todos os arquivos desta pasta (incluindo a pasta `models/`)

2. **Acesse:** https://share.streamlit.io

3. **Clique em "New app"** e configure:
   - **Repository:** seu repositório GitHub
   - **Branch:** main
   - **Main file path:** `app.py`

4. **Clique em "Deploy"** — o Streamlit Cloud instala as dependências automaticamente

5. Você receberá um link público do tipo:
   `https://seu-usuario-nome-do-repo-app-xxxxx.streamlit.app`

---

## 📊 Sobre o Modelo

| Métrica       | Valor   |
|---------------|---------|
| Algoritmo     | Random Forest Classifier |
| Acurácia      | ~88.5%  |
| F1 Macro      | ~88.2%  |
| Classes       | 7 níveis de obesidade |
| Validação     | StratifiedKFold 5-fold |

**Features utilizadas:** Gender, Age, Height, Weight, family_history, FAVC,
FCVC, NCP, CAEC, SMOKE, CH2O, SCC, FAF, TUE, CALC, MTRANS

> ⚠️ BMI não foi incluído como feature para evitar vazamento de informação (data leakage).

---

## ⚕️ Aviso

Este sistema é uma ferramenta de **apoio à decisão clínica** e não substitui
avaliação médica profissional.
