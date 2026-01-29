import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os

# --- Configuração da Página ---
st.set_page_config(page_title="Retail Intelligence App", layout="wide", page_icon="🛒")

# --- 1. Carregamento de Assets (Cache para Performance) ---
@st.cache_resource
def load_assets():
    # Detecta onde o app.py está
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # AJUSTE: Seus modelos estão dentro de 'outputs/models' ou apenas 'outputs'?
    # Vou colocar 'outputs/models' que é o padrão que vimos antes
    model_path = os.path.join(current_dir, "outputs", "models", "supermarket_rf_model.pkl")
    columns_path = os.path.join(current_dir, "outputs", "models", "model_columns.pkl")
    
    # Dados de BI
    data_path = os.path.join(current_dir, "data", "processed", "bi_ready_data.csv")
    
    try:
        model = joblib.load(model_path)
        columns = joblib.load(columns_path)
        data_bi = pd.read_csv(data_path)
        return model, columns, data_bi
    except FileNotFoundError as e:
        st.error(f"Arquivo não encontrado no caminho esperado!")
        st.code(f"Tentamos buscar em: {e.filename}")
        st.info("Dica: Verifique se os arquivos .pkl estão dentro da pasta 'outputs/models'.")
        return None, None, None
    

model, model_columns, df_bi = load_assets()

# --- Interface Principal ---
st.title("🛒 Supermarket Sales Predictor & Insights")
st.markdown("Esta aplicação utiliza **Machine Learning** para prever faturamento e **Clusterização** para visualizar segmentos de clientes.")

# --- Sidebar: Parâmetros de Simulação ---
st.sidebar.header("🛠️ Parâmetros de Simulação")

def get_user_inputs():
    # Inputs Numéricos
    unit_price = st.sidebar.slider("Preço Unitário ($)", 10.0, 100.0, 50.0)
    quantity = st.sidebar.slider("Quantidade", 1, 10, 5)
    hour = st.sidebar.selectbox("Hora do Dia", list(range(10, 21)))
    
    # Inputs Categóricos (Baseados no seu Dataset Original)
    branch = st.sidebar.selectbox("Filial (Branch)", ["A", "B", "C"])
    customer_type = st.sidebar.selectbox("Tipo de Cliente", ["Member", "Normal"])
    product_line = st.sidebar.selectbox("Linha de Produto", [
        "Health and beauty", "Electronic accessories", "Home and lifestyle", 
        "Sports and travel", "Food and beverages", "Fashion accessories"
    ])
    
    data = {
        'Unit price': unit_price,
        'Quantity': quantity,
        'Hour': hour,
        'Branch': branch,
        'Customer type': customer_type,
        'Product line': product_line
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_inputs()

# --- Layout Principal ---
if model is not None:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🔮 Previsão de Faturamento")
        
        # Preprocessamento para o Modelo (One-Hot Encoding)
        # Criamos dummies para o input atual
        input_processed = pd.get_dummies(input_df)
        
        # Alinhamos com as colunas que o modelo espera (model_columns.pkl)
        # Se faltar coluna, preenche com 0. Se sobrar, remove.
        input_final = input_processed.reindex(columns=model_columns, fill_value=0)
        
        try:
            prediction = model.predict(input_final)
            st.metric(label="Receita Total Estimada", value=f"${prediction[0]:,.2f}")
            
            if prediction[0] > 300:
                st.success("🔥 Transação de alto valor detectada!")
            else:
                st.info("✅ Valor de transação dentro da média.")
        except Exception as e:
            st.error(f"Erro na predição: {e}")

    with col2:
        st.subheader("📊 Decomposição de Impostos")
        # Cálculo simples de simulação baseado no input
        raw_total = input_df['Unit price'][0] * input_df['Quantity'][0]
        tax_simulated = raw_total * 0.05
        
        fig_pie = px.pie(
            values=[raw_total, tax_simulated], 
            names=['Receita Líquida', 'Imposto (5%)'], 
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_pie, use_container_width=True)

# --- Seção de BI e Clusters ---
st.divider()
st.subheader("👥 Visão Estratégica: Segmentação de Clientes")

if df_bi is not None:
    # Gráfico de Clusters dinâmico
    tab1, tab2 = st.tabs(["Visualização de Clusters", "Exploração de Dados"])
    
    with tab1:
        # Verifica se a coluna 'cluster' existe (gerada no Notebook 2)
        cluster_col = 'cluster' if 'cluster' in df_bi.columns else df_bi.columns[-1]
        
        fig_scatter = px.scatter(
            df_bi, 
            x="Sales", 
            y="Rating", 
            color=cluster_col,
            title="Segmentação por Faturamento vs Avaliação",
            labels={"Sales": "Faturamento ($)", "Rating": "Nota de Avaliação"},
            template="plotly_white"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab2:
        st.dataframe(df_bi, use_container_width=True)
else:
    st.warning("Aguardando geração do arquivo 'bi_ready_data.csv' para exibir os insights.")

# Rodapé
st.sidebar.markdown("---")
st.sidebar.caption(f"Logado como: **Ricson Ramos**")