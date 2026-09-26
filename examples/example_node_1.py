"""
Exemplo de uso do Nó 1: Leitor e Fragmentador

Este exemplo demonstra como usar o Nó 1 para processar um PDF de edital.
"""

import sys
from pathlib import Path

# Adiciona o diretório raiz ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.rag.node_1_reader_chunker import process_edital_pdf, Node1ReaderChunker
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_basic_usage():
    """Exemplo básico de uso do Nó 1."""
    print("=" * 60)
    print("Exemplo Básico - Nó 1: Leitor e Fragmentador")
    print("=" * 60)
    
    # Caminho para um PDF de edital (substitua pelo seu arquivo)
    pdf_path = "data/raw/exemplo_edital.pdf"
    
    try:
        # Processa o PDF
        result = process_edital_pdf(
            pdf_path=pdf_path,
            chunk_size=1000,
            chunk_overlap=200,
            chunk_by_sections=True
        )
        
        # Exibe resultados
        print(f"\n[OK] PDF processado com sucesso!")
        print(f"Páginas: {result['page_count']}")
        print(f"Caracteres extraídos: {len(result['full_text'])}")
        print(f"Chunks gerados: {result['total_chunks']}")
        print(f"Método de processamento: {result['processing_method']}")
        
        # Exibe resumo por seções
        if result['processing_method'] == 'by_sections':
            print("\nChunks por seção:")
            section_summary = Node1ReaderChunker().get_section_summary(result['chunks_by_section'])
            for section, count in section_summary.items():
                print(f"  - {section}: {count} chunks")
        
        # Exibe exemplo de chunk
        if result['chunks']:
            print(f"\nExemplo do primeiro chunk:")
            print(f"Seção: {result['chunks'][0].get('section', 'N/A')}")
            print(f"Conteúdo (primeiros 200 caracteres):")
            print(f"  {result['chunks'][0]['content'][:200]}...")
            print(f"\nMetadados: {result['chunks'][0]['metadata']}")
        
    except FileNotFoundError:
        print(f"[ERRO] Arquivo não encontrado: {pdf_path}")
        print("Dica: Coloque um PDF de edital em data/raw/ e atualize o caminho")
    except Exception as e:
        print(f"[ERRO] Erro ao processar PDF: {str(e)}")


def example_advanced_usage():
    """Exemplo avançado com mais controle."""
    print("\n" + "=" * 60)
    print("Exemplo Avançado - Controle Detalhado")
    print("=" * 60)
    
    # Inicializa o nó com parâmetros personalizados
    node = Node1ReaderChunker(
        chunk_size=500,        # Chunks menores
        chunk_overlap=100,     # Menos sobreposição
        chunk_by_sections=True # Organiza por seções
    )
    
    pdf_path = "data/raw/exemplo_edital.pdf"
    
    try:
        # Adiciona metadados personalizados
        custom_metadata = {
            "edital_number": "123/2024",
            "orgao": "Prefeitura Municipal",
            "data_publicacao": "2024-01-15"
        }
        
        result = node.process_pdf(pdf_path, metadata=custom_metadata)
        
        if node.validate_output(result):
            print("[OK] Processamento validado com sucesso!")
            
            # Acessa chunks específicos
            print(f"\nAnalise detalhada:")
            for i, chunk in enumerate(result['chunks'][:3]):  # Primeiros 3 chunks
                print(f"\nChunk {i+1}:")
                print(f"  Seção: {chunk.get('section', 'N/A')}")
                print(f"  Tamanho: {chunk['metadata']['char_count']} caracteres")
                print(f"  Palavras: {chunk['metadata']['word_count']}")
                print(f"  Preview: {chunk['content'][:100]}...")
        
    except Exception as e:
        print(f"[ERRO] Erro: {str(e)}")


def example_text_only():
    """Exemplo processando texto diretamente (sem PDF)."""
    print("\n" + "=" * 60)
    print("Exemplo - Processamento de Texto Direto")
    print("=" * 60)
    
    from app.rag.text_chunker import TextChunker
    
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
    
    chunker = TextChunker(chunk_size=200, chunk_overlap=50)
    
    # Chunk por seções
    sections = chunker.chunk_by_sections(sample_text)
    
    print("Chunks organizados por seção:")
    for section, chunks in sections.items():
        print(f"\n{section.upper()}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            print(f"  {i+1}. {chunk[:80]}...")


if __name__ == "__main__":
    # Exemplo 1: Uso básico (requer PDF real)
    # example_basic_usage()
    
    # Exemplo 2: Uso avançado (requer PDF real)
    # example_advanced_usage()
    
    # Exemplo 3: Texto direto (funciona sem PDF)
    example_text_only()
