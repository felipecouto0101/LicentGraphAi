from .pdf_reader import PDFReader
from .text_chunker import TextChunker
from typing import Dict, List, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Node1ReaderChunker:
    """
    Nó 1: Leitor e Fragmentador (RAG com LangChain)
    
    Responsável por:
    - Extrair texto do PDF do edital
    - Fragmentar o texto em partes menores (bens, prazos, exigências técnicas)
    - Preparar os dados para os próximos nós do fluxo
    """
    
    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        chunk_by_sections: bool = True
    ):
        """
        Inicializa o Nó 1.
        
        Args:
            chunk_size: Tamanho dos chunks
            chunk_overlap: Sobreposição entre chunks
            chunk_by_sections: Se True, organiza chunks por seções
        """
        self.pdf_reader = PDFReader()
        self.text_chunker = TextChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.chunk_by_sections = chunk_by_sections
        
        logger.info(f"Nó 1 inicializado: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    
    def process_pdf(self, pdf_path: str, metadata: Optional[Dict] = None) -> Dict:
        """
        Processa um PDF completo: extrai texto e fragmenta.
        
        Args:
            pdf_path: Caminho para o arquivo PDF
            metadata: Metadados adicionais do documento
            
        Returns:
            Dicionário com o texto completo e os chunks
        """
        logger.info(f"Iniciando processamento do PDF: {pdf_path}")
        
        # Valida o PDF
        if not self.pdf_reader.validate_pdf(pdf_path):
            raise ValueError(f"PDF inválido ou corrompido: {pdf_path}")
        
        # Extrai o texto
        logger.info("Extraindo texto do PDF...")
        full_text = self.pdf_reader.load_pdf(pdf_path)
        page_count = self.pdf_reader.get_page_count()
        
        logger.info(f"Texto extraído: {len(full_text)} caracteres, {page_count} páginas")
        
        # Fragmenta o texto
        logger.info("Fragmentando texto...")
        if self.chunk_by_sections:
            chunks_by_section = self.text_chunker.chunk_by_sections(full_text)
            all_chunks = []
            for section, section_chunks in chunks_by_section.items():
                for chunk in section_chunks:
                    all_chunks.append({
                        "content": chunk,
                        "section": section,
                        "metadata": self.text_chunker.get_chunk_metadata(
                            chunk, len(all_chunks), -1  # -1 porque ainda não sabemos o total
                        )
                    })
            
            # Atualiza o total de chunks nos metadados
            total_chunks = len(all_chunks)
            for chunk in all_chunks:
                chunk["metadata"]["total_chunks"] = total_chunks
            
            result = {
                "full_text": full_text,
                "page_count": page_count,
                "chunks": all_chunks,
                "chunks_by_section": chunks_by_section,
                "total_chunks": total_chunks,
                "processing_method": "by_sections",
                "metadata": metadata or {}
            }
        else:
            chunks = self.text_chunker.create_document_chunks(full_text, metadata)
            
            result = {
                "full_text": full_text,
                "page_count": page_count,
                "chunks": chunks,
                "total_chunks": len(chunks),
                "processing_method": "standard",
                "metadata": metadata or {}
            }
        
        logger.info(f"Processamento concluído: {result['total_chunks']} chunks gerados")
        return result
    
    def get_section_summary(self, chunks_by_section: Dict[str, List[str]]) -> Dict[str, int]:
        """
        Retorna um resumo dos chunks por seção.
        
        Args:
            chunks_by_section: Dicionário com chunks por seção
            
        Returns:
            Dicionário com contagem de chunks por seção
        """
        return {
            section: len(chunks)
            for section, chunks in chunks_by_section.items()
        }
    
    def validate_output(self, output: Dict) -> bool:
        """
        Valida a saída do processamento.
        
        Args:
            output: Dicionário de saída do processamento
            
        Returns:
            True se válido, False caso contrário
        """
        required_fields = ["full_text", "chunks", "total_chunks"]
        
        for field in required_fields:
            if field not in output:
                logger.error(f"Campo obrigatório ausente: {field}")
                return False
        
        if not output["full_text"].strip():
            logger.error("Texto vazio extraído do PDF")
            return False
        
        if not output["chunks"]:
            logger.error("Nenhum chunk gerado")
            return False
        
        logger.info("Validação do output concluída com sucesso")
        return True


# Função de conveniência para uso direto
def process_edital_pdf(pdf_path: str, **kwargs) -> Dict:
    """
    Função de conveniência para processar um PDF de edital.
    
    Args:
        pdf_path: Caminho para o PDF
        **kwargs: Argumentos adicionais para Node1ReaderChunker
        
    Returns:
        Dicionário com o resultado do processamento
    """
    node = Node1ReaderChunker(**kwargs)
    result = node.process_pdf(pdf_path)
    
    if not node.validate_output(result):
        raise ValueError("Output do processamento inválido")
    
    return result
