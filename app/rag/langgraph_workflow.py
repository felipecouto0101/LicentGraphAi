"""
LangGraph Workflow para LicitGraphAi

Orquestra todos os nós do sistema usando LangGraph com estado compartilhado.
"""

from typing import TypedDict, List, Dict, Optional
from langgraph.graph import StateGraph, END
import logging

# Importar nós existentes
from .node_1_reader_chunker import Node1ReaderChunker
from .node_2_embeddings import Node2EmbeddingGenerator
from .node_3_analyzer import Node3RequirementAnalyzer
from .node_4_document_generator import Node4DocumentGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
        node2 = Node2EmbeddingGenerator()
        result = node2.process_chunks(
            state["chunks"],
            persist_directory="./data/vector_db"
        )
        
        state["embeddings"] = {
            "total_embeddings": result["total_embeddings"],
            "embedding_model": result["embedding_model"],
            "vector_dimension": result["vector_dimension"]
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
    """
    logger.info("Nó 3: Iniciando análise de requisitos")
    
    try:
        node3 = Node3RequirementAnalyzer(mock_mode=True)  # Usa mock mode por padrão
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
    """
    logger.info("Nó 4: Iniciando geração de checklist")
    
    try:
        node4 = Node4DocumentGenerator(mock_mode=True)  # Usa mock mode por padrão
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
