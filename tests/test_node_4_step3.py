"""
Testes para Etapa 3 do Nó 4: Categorização de documentos

TDD Approach: Testes escritos antes da implementação
"""

import pytest


class TestNode4DocumentCategorization:
    """Testes para categorização de documentos."""
    
    def test_categorize_document_habilitacao(self):
        """Testa categorização de documento de habilitação."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        document = "CNPJ"
        
        category = node.categorize_document(document)
        
        assert category == "habilitacao"
    
    def test_categorize_document_tecnica(self):
        """Testa categorização de documento técnico."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        document = "Certidão ISO 9001"
        
        category = node.categorize_document(document)
        
        assert category == "tecnica"
    
    def test_categorize_document_fiscal(self):
        """Testa categorização de documento fiscal."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        document = "Certidão de Regularidade Fiscal"
        
        category = node.categorize_document(document)
        
        assert category == "fiscal"
    
    def test_categorize_document_juridica(self):
        """Testa categorização de documento jurídico."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        document = "Contrato Social"
        
        category = node.categorize_document(document)
        
        assert category == "juridica"
    
    def test_categorize_documents_list(self):
        """Testa categorização de lista de documentos."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        documents = ["CNPJ", "ISO 9001", "Certidão Fiscal", "Contrato Social"]
        
        result = node.categorize_documents(documents)
        
        assert "habilitacao" in result
        assert "tecnica" in result
        assert "fiscal" in result
        assert "juridica" in result
    
    def test_categorize_unknown_document(self):
        """Testa categorização de documento desconhecido."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        document = "Documento Desconhecido"
        
        category = node.categorize_document(document)
        
        assert category == "outros"
    
    def test_validate_categorization(self):
        """Testa validação da categorização."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        valid_output = {
            "habilitacao": ["CNPJ"],
            "tecnica": ["ISO 9001"],
            "fiscal": ["Certidão Fiscal"],
            "juridica": ["Contrato Social"],
            "trabalhista": [],
            "outros": []
        }
        
        assert node.validate_categorization(valid_output) is True
        
        invalid_output = {"habilitacao": []}
        assert node.validate_categorization(invalid_output) is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
