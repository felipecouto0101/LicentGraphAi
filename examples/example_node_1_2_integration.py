"""
Exemplo de Integração Nó 1 + Nó 2

Demonstra o fluxo completo:
Nó 1: Leitura e Fragmentação do PDF
Nó 2: Geração de Embeddings e Armazenamento no ChromaDB
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.node_1_reader_chunker import Node1ReaderChunker
from app.rag.node_2_embeddings import Node2EmbeddingGenerator
import logging
import tempfile
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_text_only_integration():
    """Exemplo de integração usando texto direto (sem PDF)."""
    print("=" * 60)
    print("Integração Nó 1 + Nó 2 - Processamento de Texto")
    print("=" * 60)
    
    # Texto de exemplo simulando um edital
    sample_text = """
    EDITAL DE LICITAÇÃO nº 001/2024
    
    OBJETO: Aquisição de 50 computadores desktop para uso administrativo.
    
    ESPECIFICAÇÕES TÉCNICAS:
    - Processador: Intel Core i5 ou superior
    - Memória RAM: 8GB mínimo
    - Armazenamento: SSD 256GB
    - Monitor: 24 polegadas Full HD
    
    PRAZO DE ENTREGA: 30 dias corridos após a assinatura do contrato.
    
    DOCUMENTAÇÃO EXIGIDA:
    - CNH válida
    - Comprovante de endereço
    - Certidão negativa de débitos
    """
    
    # Nó 1: Chunking
    print("\n[Nó 1] Iniciando chunking do texto...")
    from app.rag.text_chunker import TextChunker
    
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    sections = chunker.chunk_by_sections(sample_text)
    
    # Prepara chunks no formato esperado pelo Nó 2
    chunks = []
    chunk_id = 0
    for section, section_chunks in sections.items():
        for chunk in section_chunks:
            chunks.append({
                "content": chunk,
                "section": section,
                "metadata": {
                    "chunk_id": chunk_id,
                    "total_chunks": len(chunks) + 1  # estimativa
                }
            })
            chunk_id += 1
    
    print(f"[Nó 1] Chunks gerados: {len(chunks)}")
    print(f"[Nó 1] Distribuição por seção:")
    for section, section_chunks in sections.items():
        print(f"  - {section}: {len(section_chunks)} chunks")
    
    # Nó 2: Embeddings
    print("\n[Nó 2] Iniciando geração de embeddings...")
    temp_dir = tempfile.mkdtemp()
    
    try:
        node2 = Node2EmbeddingGenerator()
        result = node2.process_chunks(chunks, persist_directory=temp_dir)
        
        print(f"[Nó 2] Embeddings gerados: {result['total_embeddings']}")
        print(f"[Nó 2] Modelo: {result['embedding_model']}")
        print(f"[Nó 2] Dimensão: {result['vector_dimension']}")
        print(f"[Nó 2] Tempo de processamento: {result['processing_time']:.2f}s")
        print(f"[Nó 2] Velocidade: {result['chunks_per_second']:.2f} chunks/s")
        print(f"[Nó 2] Coleção ChromaDB: {result['collection_id']}")
        
        # Exemplo de busca semântica
        print("\n[Busca Semântica] Testando busca...")
        query = "Qual processador é necessário?"
        search_results = node2.search_similar(
            query=query,
            collection_name=result['collection_id'],
            persist_directory=temp_dir,
            n_results=2
        )
        
        print(f"[Busca] Query: '{query}'")
        print(f"[Busca] Resultados encontrados: {len(search_results['documents'][0])}")
        for i, doc in enumerate(search_results['documents'][0]):
            print(f"  Resultado {i+1}: {doc[:100]}...")
        
        print("\n[OK] Integração Nó 1 + Nó 2 concluída com sucesso!")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def example_with_pdf_simulation():
    """Exemplo simulando processamento de PDF real."""
    print("\n" + "=" * 60)
    print("Integração Nó 1 + Nó 2 - Simulação de PDF")
    print("=" * 60)
    
    # Simula chunks que viriam do Nó 1
    simulated_chunks = [
        {
            "content": "OBJETO: Aquisição de 50 computadores desktop para uso administrativo.",
            "section": "bens",
            "metadata": {"chunk_id": 0, "total_chunks": 5, "page": 1}
        },
        {
            "content": "ESPECIFICAÇÕES TÉCNICAS: Processador Intel Core i5 ou superior, Memória RAM 8GB mínimo.",
            "section": "exigencias_tecnicas",
            "metadata": {"chunk_id": 1, "total_chunks": 5, "page": 2}
        },
        {
            "content": "Armazenamento SSD 256GB, Monitor 24 polegadas Full HD.",
            "section": "exigencias_tecnicas",
            "metadata": {"chunk_id": 2, "total_chunks": 5, "page": 2}
        },
        {
            "content": "PRAZO DE ENTREGA: 30 dias corridos após a assinatura do contrato.",
            "section": "prazos",
            "metadata": {"chunk_id": 3, "total_chunks": 5, "page": 3}
        },
        {
            "content": "DOCUMENTAÇÃO EXIGIDA: CNH válida, Comprovante de endereço, Certidão negativa de débitos.",
            "section": "documentacao",
            "metadata": {"chunk_id": 4, "total_chunks": 5, "page": 4}
        }
    ]
    
    print(f"[Simulação] Chunks do Nó 1: {len(simulated_chunks)}")
    
    # Nó 2: Processa chunks
    temp_dir = tempfile.mkdtemp()
    
    try:
        node2 = Node2EmbeddingGenerator()
        result = node2.process_chunks(simulated_chunks, persist_directory=temp_dir)
        
        print(f"[Nó 2] Processamento concluído:")
        print(f"  - Coleção: {result['collection_id']}")
        print(f"  - Embeddings: {result['total_embeddings']}")
        print(f"  - Tempo total: {result['processing_time']:.2f}s")
        print(f"  - Tempo embeddings: {result['embedding_time']:.2f}s")
        
        # Busca por diferentes termos
        test_queries = [
            "requisitos de hardware",
            "documentos necessários",
            "tempo de entrega"
        ]
        
        print(f"\n[Teste de Busca] Testando {len(test_queries)} queries:")
        for query in test_queries:
            search_results = node2.search_similar(
                query=query,
                collection_name=result['collection_id'],
                persist_directory=temp_dir,
                n_results=1
            )
            if search_results['documents'][0]:
                print(f"  Query: '{query}'")
                print(f"  Resultado: {search_results['documents'][0][0][:80]}...")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def example_performance_comparison():
    """Compara performance com diferentes tamanhos de batch."""
    print("\n" + "=" * 60)
    print("Comparação de Performance - Diferentes Batch Sizes")
    print("=" * 60)
    
    # Gera chunks de teste
    test_chunks = [
        {"content": f"Chunk de teste número {i} com conteúdo genérico para processamento.", "section": "test"}
        for i in range(100)
    ]
    
    batch_sizes = [8, 16, 32, 64]
    temp_dir = tempfile.mkdtemp()
    
    try:
        print(f"Processando {len(test_chunks)} chunks com diferentes batch sizes:")
        
        for batch_size in batch_sizes:
            node2 = Node2EmbeddingGenerator(batch_size=batch_size)
            result = node2.process_chunks(test_chunks, persist_directory=temp_dir, collection_name=f"test_batch_{batch_size}")
            
            print(f"  Batch {batch_size}: {result['embedding_time']:.2f}s ({result['chunks_per_second']:.1f} chunks/s)")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    # Exemplo 1: Integração com texto direto
    example_text_only_integration()
    
    # Exemplo 2: Simulação de PDF
    example_with_pdf_simulation()
    
    # Exemplo 3: Comparação de performance
    # example_performance_comparison()
