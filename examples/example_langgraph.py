"""
Exemplo de uso do LangGraph Workflow

Demonstra como usar o workflow orquestrado com LangGraph.
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.langgraph_workflow import run_licit_graph_pipeline, create_licit_graph_workflow


def example_langgraph_basic():
    """Exemplo básico do LangGraph workflow."""
    print("=" * 60)
    print("LangGraph Workflow - Exemplo Básico")
    print("=" * 60)
    
    # Caminho para um PDF de exemplo (se existir)
    pdf_path = "data/raw/edital_exemplo.pdf"
    
    # Como não temos um PDF real, vamos simular o workflow
    print(f"\nPipeline completo: {pdf_path}")
    print("\nFluxo do workflow:")
    print("  Nó 1: Leitura e fragmentação do PDF")
    print("  Nó 2: Geração de embeddings")
    print("  Nó 3: Análise de requisitos com IA")
    print("  Nó 4: Geração de checklist de documentos")
    
    # Criar o workflow (sem executar)
    app = create_licit_graph_workflow()
    
    print(f"\nWorkflow criado: {app}")
    print(f"Tipo: {type(app)}")
    
    # Visualizar o grafo (se possível)
    try:
        print("\nEstrutura do grafo:")
        print("  node_1 → node_2 → node_3 → node_4 → END")
    except Exception as e:
        print(f"\nErro ao visualizar grafo: {e}")


def example_langgraph_with_state():
    """Exemplo com estado inicial customizado."""
    print("\n" + "=" * 60)
    print("LangGraph Workflow - Exemplo com Estado")
    print("=" * 60)
    
    # Estado inicial customizado
    initial_state = {
        "pdf_path": "data/raw/edital_exemplo.pdf",
        "chunks": [],
        "embeddings": None,
        "analysis": None,
        "checklist": None,
        "company_profile": {
            "name": "Empresa Exemplo Ltda",
            "cnpj": "00.000.000/0001-00"
        },
        "error": None
    }
    
    print(f"\nEstado inicial:")
    print(f"  PDF: {initial_state['pdf_path']}")
    print(f"  Empresa: {initial_state['company_profile']['name']}")
    print(f"  CNPJ: {initial_state['company_profile']['cnpj']}")
    
    print("\nNota: Para executar o pipeline real, você precisa de um PDF válido.")


def example_compare_manual_vs_langgraph():
    """Compara abordagem manual vs LangGraph."""
    print("\n" + "=" * 60)
    print("Comparação: Manual vs LangGraph")
    print("=" * 60)
    
    print("\nAbordagem Manual (antes):")
    print("  result1 = node1.process_pdf('edital.pdf')")
    print("  result2 = node2.process_chunks(result1['chunks'])")
    print("  result3 = node3.analyze_requirements(result2)")
    print("  result4 = node4.generate_checklist(result3)")
    print("  ->")
    print("  Problemas:")
    print("    - Orquestracao manual")
    print("    - Passagem de dados explicita")
    print("    - Dificil adicionar/remover nos")
    
    print("\nAbordagem LangGraph (agora):")
    print("  final_state = run_licit_graph_pipeline('edital.pdf')")
    print("  ->")
    print("  Vantagens:")
    print("    - Orquestracao automatica")
    print("    - Estado compartilhado")
    print("    - Facil escalar e manter")
    print("    - Visualizacao do grafo")
    print("    - Possibilidade de branching e loops")


if __name__ == "__main__":
    example_langgraph_basic()
    example_langgraph_with_state()
    example_compare_manual_vs_langgraph()
    
    print("\n" + "=" * 60)
    print("Exemplos concluídos!")
    print("=" * 60)
