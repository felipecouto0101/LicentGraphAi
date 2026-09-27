"""
LangGraph Workflow para LicitGraphAi

Orquestra todos os nós do sistema usando LangGraph com estado compartilhado.
"""

from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
import logging
import os
import uuid

# Importar nós existentes
from .node_1_reader_chunker import Node1ReaderChunker
from .node_3_analyzer import Node3RequirementAnalyzer
from .node_4_document_generator import Node4DocumentGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def is_mock_mode() -> bool:
    """O modo de demonstração só é ativado explicitamente."""
    return os.getenv("NODE3_MOCK_MODE", "false").strip().lower() == "true"


def validate_analysis_configuration() -> None:
    """Impede que uma análise real seja trocada silenciosamente por dados simulados."""
    if is_mock_mode():
        return
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key or key == "your_groq_api_key_here":
        raise ValueError(
            "GROQ_API_KEY não configurada. Defina uma chave válida para análise real "
            "ou ative NODE3_MOCK_MODE=true para demonstração."
        )


class LicitGraphState(TypedDict):
    """
    Estado compartilhado entre todos os nós do workflow.
    
    Atributos:
        pdf_path: Caminho para o arquivo PDF do edital
        chunks: Lista de chunks extraídos
        embeddings: Dicionário com embeddings gerados
        analysis: Análise de requisitos do Nó 3
        checklist: Checklist de documentos do Nó 4
        company_profile: Perfil da empresa (opcional)
        error: Erro ocorrido durante o processamento
    """
    pdf_path: str
    chunks: List[Dict]
    embeddings: Optional[Dict]
    analysis: Optional[Dict]
    checklist: Optional[Dict]
    company_profile: Optional[Dict]
    error: Optional[str]


def node_1_reader_chunker(state: LicitGraphState) -> LicitGraphState:
    """
    Nó 1: Leitor e Fragmentador (Wrapper para LangGraph)
    
    Extrai texto do PDF e fragmenta em chunks.
    """
    logger.info("Nó 1: Iniciando leitura e fragmentação do PDF")
    
    try:
        node1 = Node1ReaderChunker()
        result = node1.process_pdf(state["pdf_path"])
        
        state["chunks"] = result["chunks"]
        state["chunks_by_section"] = result.get("chunks_by_section", {})
        state["total_chunks"] = result["total_chunks"]
        
        logger.info(f"Nó 1: {result['total_chunks']} chunks gerados")
        return state
        
    except Exception as e:
        logger.error(f"Nó 1: Erro ao processar PDF: {e}")
        state["error"] = f"Erro no Nó 1: {str(e)}"
        return state


def node_2_embeddings(state: LicitGraphState) -> LicitGraphState:
    """
    Nó 2: Geração de Embeddings (Wrapper para LangGraph)
    
    Gera embeddings dos chunks e armazena no ChromaDB.
    """
    logger.info("Nó 2: Iniciando geração de embeddings")
    
    try:
        # Carrega sentence-transformers e suas dependências somente quando um job
        # chega ao Nó 2; /status não precisa aguardar esse import pesado.
        from .node_2_embeddings import Node2EmbeddingGenerator

        node2 = Node2EmbeddingGenerator()
        # Cada edital usa uma coleção própria: consultas não recuperam trechos
        # de outros documentos nem colidem com IDs de execuções anteriores.
        collection_name = f"licitacoes_{uuid.uuid4().hex}"
        result = node2.process_chunks(
            state["chunks"],
            collection_name=collection_name,
            persist_directory=os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/vector_db")
        )
        
        state["embeddings"] = {
            "total_embeddings": result["total_embeddings"],
            "embedding_model": result["embedding_model"],
            "vector_dimension": result["vector_dimension"],
            "collection_id": result["collection_id"]
        }
        
        logger.info(f"Nó 2: {result['total_embeddings']} embeddings gerados")
        return state
        
    except Exception as e:
        logger.error(f"Nó 2: Erro ao gerar embeddings: {e}")
        state["error"] = f"Erro no Nó 2: {str(e)}"
        return state


def node_3_analyzer(state: LicitGraphState) -> LicitGraphState:
    """
    Nó 3: Analisador de Requisitos (Wrapper para LangGraph)
    
    Analisa requisitos do edital usando IA.
    Usa demonstração somente com NODE3_MOCK_MODE=true.
    """
    logger.info("Nó 3: Iniciando análise de requisitos")
    
    try:
        validate_analysis_configuration()
        mock_mode = is_mock_mode()
        persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/vector_db")
        collection = (state.get("embeddings") or {}).get("collection_id")
        if not collection:
            raise ValueError("Coleção vetorial do edital não disponível")

        if mock_mode:
            logger.info("Nó 3: Modo de demonstração ativado explicitamente")
        else:
            logger.info("Nó 3: Usando API Groq real com RAG")

        node3 = Node3RequirementAnalyzer(
            mock_mode=mock_mode,
            persist_directory=persist_dir,
            collection_name=collection,
        )
        result = node3.process_from_node2(state["chunks"])
        
        state["analysis"] = result
        
        logger.info("Nó 3: Análise concluída")
        return state
        
    except Exception as e:
        logger.error(f"Nó 3: Erro ao analisar requisitos: {e}")
        state["error"] = f"Erro no Nó 3: {str(e)}"
        return state


def node_4_document_generator(state: LicitGraphState) -> LicitGraphState:
    """
    Nó 4: Gerador de Checklist (Wrapper para LangGraph)
    
    Gera checklist de documentos a partir da análise.
    Usa demonstração somente com NODE3_MOCK_MODE=true.
    """
    if state.get("error"):
        logger.info("Nó 4: Ignorado porque uma etapa anterior falhou")
        return state

    logger.info("Nó 4: Iniciando geração de checklist")
    
    try:
        validate_analysis_configuration()
        mock_mode = is_mock_mode()

        node4 = Node4DocumentGenerator(mock_mode=mock_mode)
        result = node4.process_from_node3(state["analysis"])
        
        state["checklist"] = result
        
        logger.info("Nó 4: Checklist gerado")
        return state
        
    except Exception as e:
        logger.error(f"Nó 4: Erro ao gerar checklist: {e}")
        state["error"] = f"Erro no Nó 4: {str(e)}"
        return state


def create_licit_graph_workflow() -> StateGraph:
    """
    Cria o grafo de workflow do LicitGraphAi.
    
    Returns:
        StateGraph compilado pronto para execução
    """
    logger.info("Criando workflow LangGraph")
    
    # Cria o grafo de estado
    workflow = StateGraph(LicitGraphState)
    
    # Adiciona os nós
    workflow.add_node("node_1", node_1_reader_chunker)
    workflow.add_node("node_2", node_2_embeddings)
    workflow.add_node("node_3", node_3_analyzer)
    workflow.add_node("node_4", node_4_document_generator)
    
    # Define o ponto de entrada
    workflow.set_entry_point("node_1")
    
    # Define as transições (fluxo linear)
    workflow.add_edge("node_1", "node_2")
    workflow.add_edge("node_2", "node_3")
    workflow.add_edge("node_3", "node_4")
    workflow.add_edge("node_4", END)
    
    # Compila o grafo
    app = workflow.compile()
    
    logger.info("Workflow LangGraph criado e compilado")
    return app


def run_licit_graph_pipeline(pdf_path: str, company_profile: Optional[Dict] = None) -> LicitGraphState:
    """
    Executa o pipeline completo do LicitGraphAi.
    
    Args:
        pdf_path: Caminho para o arquivo PDF do edital
        company_profile: Perfil da empresa (opcional)
        
    Returns:
        Estado final com todos os resultados
    """
    logger.info(f"Iniciando pipeline completo: {pdf_path}")
    
    # Falha antes de processar o PDF se faltar a configuração da análise real.
    validate_analysis_configuration()

    # Cria o workflow
    app = create_licit_graph_workflow()
    
    # Estado inicial
    initial_state: LicitGraphState = {
        "pdf_path": pdf_path,
        "chunks": [],
        "embeddings": None,
        "analysis": None,
        "checklist": None,
        "company_profile": company_profile,
        "error": None
    }
    
    # Executa o workflow
    final_state = app.invoke(initial_state)
    
    logger.info("Pipeline completo finalizado")
    
    if final_state.get("error"):
        logger.error(f"Pipeline finalizado com erro: {final_state['error']}")
    else:
        logger.info("Pipeline finalizado com sucesso")
    
    return final_state
