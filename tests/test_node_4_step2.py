"""
Testes para Etapa 2 do Nó 4: Extração de documentos do edital

TDD Approach: Testes escritos antes da implementação
"""

import pytest
from typing import Dict, List


class TestNode4DocumentExtraction:
    """Testes para extração de documentos."""
    
    def test_extract_documents_from_chunk(self):
        """Testa extração de documentos de um chunk."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        chunk = {
            "content": "Apresentar cnpj, certidão de regularidade fiscal e atestado de capacidade técnica",
            "section": "documentacao"
        }
        
        result = node.extract_documents(chunk)
        
        assert "documents" in result
        assert len(result["documents"]) > 0
        assert any("cnpj" in doc for doc in result["documents"])
    
    def test_extract_documents_empty_chunk(self):
        """Testa extração de chunk vazio."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        chunk = {
            "content": "",
            "section": "documentacao"
        }
        
        result = node.extract_documents(chunk)
        
        assert "documents" in result
        assert len(result["documents"]) == 0
    
    def test_extract_documents_multiple_chunks(self):
        """Testa extração de múltiplos chunks."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        chunks = [
            {"content": "Apresentar cnpj e rg", "section": "documentacao"},
            {"content": "Certidão negativa de débitos", "section": "documentacao"}
        ]
        
        result = node.extract_documents_from_chunks(chunks)
        
        assert "documents" in result
        assert len(result["documents"]) >= 2
    
    def test_extract_document_types(self):
        """Testa identificação de tipos de documentos."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        text = "Apresentar cnpj, rg, cpf e certidão de regularidade fiscal"
        
        result = node._extract_document_types(text)
        
        assert "cnpj" in result
        assert "rg" in result
        assert "cpf" in result
    
    def test_extract_document_deadlines(self):
        """Testa extração de prazos de documentos."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        text = "Apresentar documentos até 5 dias úteis antes da licitação"
        
        result = node._extract_deadlines(text)
        
        assert "5 dias" in result or "5" in result
    
    def test_extract_document_requirements(self):
        """Testa extração de requisitos de documentos."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        text = "Documentos devem ser autenticados em cartório e ter firma reconhecida"
        
        result = node._extract_requirements(text)
        
        assert "autenticado" in result or "cartório" in result
    
    def test_validate_document_extraction(self):
        """Testa validação da extração de documentos."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        valid_output = {
            "documents": ["CNPJ", "RG"],
            "deadlines": ["5 dias"],
            "requirements": ["Autenticado"]
        }
        
        assert node.validate_document_extraction(valid_output) is True
        
        invalid_output = {"documents": []}
        assert node.validate_document_extraction(invalid_output) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
