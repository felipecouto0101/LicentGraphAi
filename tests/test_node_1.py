import pytest
from app.rag.pdf_reader import PDFReader
from app.rag.text_chunker import TextChunker
from app.rag.node_1_reader_chunker import Node1ReaderChunker
from pathlib import Path
import tempfile


class TestPDFReader:
    """Testes para o PDFReader."""
    
    def test_init(self):
        """Testa inicialização do PDFReader."""
        reader = PDFReader()
        assert reader.text == ""
        assert reader.pdf_path is None
    
    def test_load_nonexistent_file(self):
        """Testa erro ao carregar arquivo inexistente."""
        reader = PDFReader()
        with pytest.raises(FileNotFoundError):
            reader.load_pdf("arquivo_inexistente.pdf")
    
    def test_validate_pdf_nonexistent(self):
        """Testa validação de PDF inexistente."""
        reader = PDFReader()
        assert reader.validate_pdf() is False


class TestTextChunker:
    """Testes para o TextChunker."""
    
    def test_init(self):
        """Testa inicialização do TextChunker."""
        chunker = TextChunker()
        assert chunker.chunk_size == 1000
        assert chunker.chunk_overlap == 200
    
    def test_chunk_empty_text(self):
        """Testa chunking de texto vazio."""
        chunker = TextChunker()
        chunks = chunker.chunk_text("")
        assert chunks == []
    
    def test_chunk_short_text(self):
        """Testa chunking de texto curto."""
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        text = "Este é um texto curto para teste."
        chunks = chunker.chunk_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text
    
    def test_chunk_long_text(self):
        """Testa chunking de texto longo."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = " ".join(["palavra"] * 100)
        chunks = chunker.chunk_text(text)
        assert len(chunks) > 1
    
    def test_chunk_by_sections(self):
        """Testa chunking por seções."""
        chunker = TextChunker()
        text = """
        OBJETO: Aquisição de computadores
        PRAZO: 30 dias para entrega
        EXIGÊNCIA TÉCNICA: Processador Intel i5
        DOCUMENTAÇÃO: CNH e RG
        """
        sections = chunker.chunk_by_sections(text)
        assert "bens" in sections
        assert "prazos" in sections
        assert "exigencias_tecnicas" in sections
        assert "documentacao" in sections
    
    def test_get_chunk_metadata(self):
        """Testa geração de metadados do chunk."""
        chunker = TextChunker()
        chunk = "Este é um chunk de teste"
        metadata = chunker.get_chunk_metadata(chunk, 0, 5)
        
        assert metadata["chunk_id"] == 0
        assert metadata["total_chunks"] == 5
        assert metadata["char_count"] == len(chunk)
        assert metadata["word_count"] == 6  # Corrigido: o chunk tem 6 palavras
        assert metadata["line_count"] == 1
    
    def test_create_document_chunks(self):
        """Testa criação de chunks no formato de documentos."""
        chunker = TextChunker(chunk_size=50, chunk_overlap=10)
        text = " ".join(["palavra"] * 100)
        documents = chunker.create_document_chunks(text, {"source": "test"})
        
        assert len(documents) > 1
        assert all("content" in doc for doc in documents)
        assert all("metadata" in doc for doc in documents)
        assert all(doc["metadata"]["source"] == "test" for doc in documents)


class TestNode1ReaderChunker:
    """Testes para o Node1ReaderChunker."""
    
    def test_init(self):
        """Testa inicialização do Nó 1."""
        node = Node1ReaderChunker()
        assert node.pdf_reader is not None
        assert node.text_chunker is not None
        assert node.chunk_by_sections is True
    
    def test_process_invalid_pdf(self):
        """Testa processamento de PDF inválido."""
        node = Node1ReaderChunker()
        with pytest.raises((FileNotFoundError, ValueError)):
            node.process_pdf("arquivo_inexistente.pdf")
    
    def test_validate_output_missing_fields(self):
        """Testa validação de output com campos faltando."""
        node = Node1ReaderChunker()
        invalid_output = {"full_text": "texto"}
        assert node.validate_output(invalid_output) is False
    
    def test_validate_output_empty_text(self):
        """Testa validação de output com texto vazio."""
        node = Node1ReaderChunker()
        invalid_output = {
            "full_text": "",
            "chunks": [],
            "total_chunks": 0
        }
        assert node.validate_output(invalid_output) is False
    
    def test_get_section_summary(self):
        """Testa resumo de seções."""
        node = Node1ReaderChunker()
        chunks_by_section = {
            "bens": ["chunk1", "chunk2"],
            "prazos": ["chunk3"],
            "exigencias_tecnicas": []
        }
        summary = node.get_section_summary(chunks_by_section)
        
        assert summary["bens"] == 2
        assert summary["prazos"] == 1
        assert summary["exigencias_tecnicas"] == 0


def test_process_edital_pdf_convenience():
    """Testa função de conveniência process_edital_pdf."""
    with pytest.raises((FileNotFoundError, ValueError)):
        from app.rag.node_1_reader_chunker import process_edital_pdf
        process_edital_pdf("arquivo_inexistente.pdf")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
