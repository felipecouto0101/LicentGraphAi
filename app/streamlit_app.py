"""
Interface Streamlit para LicitGraphAi

Frontend para upload de PDFs e visualização de resultados.
"""

import streamlit as st
import requests
import json
from pathlib import Path

# Configuração da página
st.set_page_config(
    page_title="LicitGraphAi - Analisador de Editais",
    page_icon="📄",
    layout="wide"
)

# URL da API FastAPI
API_URL = "http://127.0.0.1:8000"


def main():
    """Função principal do Streamlit."""
    
    # Header
    st.title("📄 LicitGraphAi - Analisador de Editais")
    st.markdown("Sistema inteligente para análise automática de editais de licitação pública")
    
    # Sidebar
    st.sidebar.title("Configurações")
    api_url = st.sidebar.text_input("URL da API", value=API_URL)
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Análise", "Status", "Sobre"])
    
    with tab1:
        st.header("📤 Upload e Análise de Edital")
        
        # Upload do PDF
        uploaded_file = st.file_uploader(
            "Selecione o arquivo PDF do edital",
            type="pdf",
            help="Selecione um arquivo PDF contendo o edital da licitação"
        )
        
        # Perfil da empresa (opcional)
        st.subheader("Perfil da Empresa (Opcional)")
        company_name = st.text_input("Nome da Empresa")
        company_cnpj = st.text_input("CNPJ")
        
        company_profile = None
        if company_name or company_cnpj:
            company_profile = {
                "name": company_name,
                "cnpj": company_cnpj
            }
        
        # Botão de análise
        if uploaded_file is not None:
            st.info(f"Arquivo selecionado: {uploaded_file.name}")
            
            if st.button("🚀 Iniciar Análise", type="primary"):
                with st.spinner("Processando edital... Isso pode levar alguns minutos."):
                    try:
                        # Enviar arquivo para análise
                        files = {"file": uploaded_file}
                        data = {}
                        if company_profile:
                            data["company_profile"] = json.dumps(company_profile)
                        
                        response = requests.post(
                            f"{api_url}/analyze/upload",
                            files=files,
                            data=data
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            st.success("✅ Análise concluída com sucesso!")
                            display_results(result)
                        else:
                            st.error(f"❌ Erro na análise: {response.text}")
                    
                    except requests.exceptions.ConnectionError:
                        st.error("❌ Não foi possível conectar à API. Verifique se o servidor está rodando.")
                    except Exception as e:
                        st.error(f"❌ Erro: {str(e)}")
        else:
            st.info("👆 Selecione um arquivo PDF para começar")
    
    with tab2:
        st.header("📊 Status do Sistema")
        
        try:
            response = requests.get(f"{api_url}/status")
            if response.status_code == 200:
                status = response.json()
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Status", status["status"])
                col2.metric("Versão", status["version"])
                col3.metric("Orquestração", status["orchestration"])
                
                st.subheader("Nós Disponíveis")
                for node in status["nodes"]:
                    st.success(f"✅ {node}")
            else:
                st.error("❌ Não foi possível obter o status do sistema")
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Não foi possível conectar à API. Verifique se o servidor está rodando.")
        except Exception as e:
            st.error(f"❌ Erro: {str(e)}")
    
    with tab3:
        st.header("ℹ️ Sobre o LicitGraphAi")
        
        st.markdown("""
        ### Sistema de Análise de Editais
        
        O LicitGraphAi é um sistema inteligente para análise automática de editais de licitação pública (Nova Lei 14.133).
        
        ### Funcionalidades
        
        - 📄 **Leitura e Fragmentação**: Extrai texto do PDF e fragmenta em partes menores
        - 🔍 **Embeddings**: Gera representações vetoriais do texto
        - 🤖 **Análise com IA**: Analisa requisitos usando inteligência artificial
        - 📋 **Checklist**: Gera checklist de documentos necessários
        
        ### Stack Tecnológica
        
        - Python 3.11+
        - LangChain + LangGraph
        - Groq API + OpenAI GPT-OSS-120b
        - sentence-transformers
        - ChromaDB
        - FastAPI + Streamlit
        
        ### Como Usar
        
        1. Certifique-se de que a API FastAPI está rodando:
           ```bash
           python -m uvicorn app.api.main:app --reload --host 127.0.0.1 --port 8000
           ```
        
        2. Execute o Streamlit:
           ```bash
           streamlit run app/streamlit_app.py
           ```
        
        3. Faça upload do PDF do edital
        4. Clique em "Iniciar Análise"
        5. Aguarde o processamento
        6. Visualize os resultados
        """)


def display_results(result):
    """Exibe os resultados da análise."""
    
    st.header("📊 Resultados da Análise")
    
    # Tabs para diferentes resultados
    result_tab1, result_tab2, result_tab3, result_tab4 = st.tabs([
        "Resumo",
        "Análise",
        "Checklist",
        "JSON Completo"
    ])
    
    with result_tab1:
        st.subheader("📈 Resumo")
        
        if "result" in result:
            result_data = result["result"]
            
            col1, col2, col3 = st.columns(3)
            
            chunks_count = result_data.get("chunks_count", 0)
            col1.metric("Chunks Gerados", chunks_count)
            
            if result_data.get("embeddings"):
                embeddings_count = result_data["embeddings"].get("total_embeddings", 0)
                col2.metric("Embeddings", embeddings_count)
            
            if result_data.get("checklist"):
                docs_count = result_data["checklist"].get("resumo", {}).get("total_documentos", 0)
                col3.metric("Documentos", docs_count)
    
    with result_tab2:
        st.subheader("🔍 Análise de Requisitos")
        
        if "result" in result and result["result"].get("analysis"):
            analysis = result["result"]["analysis"]
            st.json(analysis)
        else:
            st.info("Análise não disponível")
    
    with result_tab3:
        st.subheader("📋 Checklist de Documentos")
        
        if "result" in result and result["result"].get("checklist"):
            checklist = result["result"]["checklist"]
            
            if checklist.get("resumo"):
                st.write("**Resumo:**")
                st.json(checklist["resumo"])
            
            if checklist.get("checklist"):
                st.write("**Categorias:**")
                for category, data in checklist["checklist"].items():
                    with st.expander(f"📁 {category.upper()}"):
                        st.json(data)
        else:
            st.info("Checklist não disponível")
    
    with result_tab4:
        st.subheader("📄 JSON Completo")
        st.json(result)


if __name__ == "__main__":
    main()
