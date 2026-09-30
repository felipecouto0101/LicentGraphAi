import logging

from .pdf_reader import PDFReader
from .text_chunker import TextChunker

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
        chunk_by_sections: bool = True,
        preserve_document_structure: bool = False,
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
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        self.chunk_by_sections = chunk_by_sections
        self.preserve_document_structure = preserve_document_structure

        logger.info(
            f"Nó 1 inicializado: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}"
        )

    def process_pdf(self, pdf_path: str, metadata: dict | None = None) -> dict:
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
        full_text = (self.pdf_reader.load_pdf(pdf_path, preserve_lines=True)
                     if self.preserve_document_structure else self.pdf_reader.load_pdf(pdf_path))
        page_count = self.pdf_reader.get_page_count()

        logger.info(
            f"Texto extraído: {len(full_text)} caracteres, {page_count} páginas"
        )

        # Fragmenta cada página separadamente para preservar a origem de cada
        # trecho. O número é o índice físico do PDF (1-based), não o número impresso.
        logger.info("Fragmentando texto por página...")
        pages = self.pdf_reader.pages_text or [full_text]
        all_chunks = []
        chunks_by_section = {
            "bens": [], "prazos": [], "exigencias_tecnicas": [],
            "documentacao": [], "outros": [],
        }
        if self.preserve_document_structure:
            from .topic_map import split_topic_sections
            for block in split_topic_sections(pages):
                for chunk in self.text_chunker.create_document_chunks(block["content"], {
                    **(metadata or {}), "page": block["page"],
                    "topic_title": block["topic_title"], "subtopic_title": block["subtopic_title"],
                }):
                    chunk["metadata"]["chunk_id"] = len(all_chunks)
                    all_chunks.append(chunk)
        for page_number, page_text in enumerate(pages if not self.preserve_document_structure else [], 1):
            if not page_text.strip():
                continue
            if self.chunk_by_sections:
                page_sections = self.text_chunker.chunk_by_sections(page_text)
                for section, section_chunks in page_sections.items():
                    chunks_by_section[section].extend(section_chunks)
                    for content in section_chunks:
                        all_chunks.append({
                            "content": content,
                            "section": section,
                            "metadata": {
                                **self.text_chunker.get_chunk_metadata(content, len(all_chunks), -1),
                                "page": page_number,
                            },
                        })
            else:
                for chunk in self.text_chunker.create_document_chunks(
                    page_text, {**(metadata or {}), "page": page_number}
                ):
                    chunk["metadata"]["chunk_id"] = len(all_chunks)
                    all_chunks.append(chunk)

        for chunk in all_chunks:
            chunk["metadata"]["total_chunks"] = len(all_chunks)

        result = {
            "full_text": full_text,
            "page_count": page_count,
            "pages_text": list(pages),
            "chunks": all_chunks,
            "total_chunks": len(all_chunks),
            "processing_method": ("document_structure" if self.preserve_document_structure
                                  else "by_sections" if self.chunk_by_sections else "standard"),
            "metadata": metadata or {},
        }
        if self.chunk_by_sections and not self.preserve_document_structure:
            result["chunks_by_section"] = chunks_by_section

        logger.info(f"Processamento concluído: {result['total_chunks']} chunks gerados")
        return result

    def get_section_summary(
        self, chunks_by_section: dict[str, list[str]]
    ) -> dict[str, int]:
        """
        Retorna um resumo dos chunks por seção.

        Args:
            chunks_by_section: Dicionário com chunks por seção

        Returns:
            Dicionário com contagem de chunks por seção
        """
        return {section: len(chunks) for section, chunks in chunks_by_section.items()}

    def validate_output(self, output: dict) -> bool:
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
def process_edital_pdf(pdf_path: str, **kwargs) -> dict:
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
