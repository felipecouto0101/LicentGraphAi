"""
Testes para Etapa 4 do Nó 4: Geração de checklist estruturado

TDD Approach: Testes escritos antes da implementação
"""

import pytest
from typing import Dict, List


class TestNode4ChecklistGeneration:
    """Testes para geração de checklist."""
    
    def test_generate_checklist_item(self):
        """Testa geração de item de checklist."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        document = "CNPJ"
        requirements = ["Autenticado", "Cópia"]
        
        item = node.generate_checklist_item(document, requirements)
        
        assert "documento" in item
        assert "obrigatorio" in item
        assert "observacoes" in item
    
    def test_generate_checklist_category(self):
        """Testa geração de checklist por categoria."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        documents = ["CNPJ", "RG"]
        category = "habilitacao"
        
        checklist = node.generate_checklist_category(documents, category)
        
        assert "categoria" in checklist
        assert checklist["categoria"] == category
        assert "itens" in checklist
        assert len(checklist["itens"]) == 2
    
    def test_generate_complete_checklist(self):
        """Testa geração de checklist completo."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        categorized_docs = {
            "habilitacao": ["CNPJ", "RG"],
            "tecnica": ["ISO 9001"],
            "fiscal": ["Certidão Fiscal"],
            "juridica": ["Contrato Social"],
            "outros": []
        }
        
        checklist = node.generate_complete_checklist(categorized_docs)
        
        assert "checklist" in checklist
        assert "resumo" in checklist
        assert len(checklist["checklist"]) == 4  # "outros" não é adicionado pois está vazio
    
    def test_add_deadline_to_checklist(self):
        """Testa adição de prazo ao checklist."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        item = {"documento": "CNPJ", "obrigatorio": True}
        deadline = "5 dias"
        
        updated_item = node.add_deadline_to_item(item, deadline)
        
        assert "prazo" in updated_item
        assert updated_item["prazo"] == deadline
    
    def test_generate_checklist_summary(self):
        """Testa geração de resumo do checklist."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        checklist = {
            "habilitacao": [{"documento": "CNPJ", "obrigatorio": True}],
            "tecnica": [{"documento": "ISO 9001", "obrigatorio": True}]
        }
        
        summary = node.generate_checklist_summary(checklist)
        
        assert "total_documentos" in summary
        assert "obrigatorios" in summary
        assert "opcionais" in summary
    
    def test_validate_checklist(self):
        """Testa validação do checklist."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        valid_checklist = {
            "checklist": {
                "habilitacao": [{"documento": "CNPJ"}]
            },
            "resumo": {
                "total_documentos": 1
            }
        }
        
        assert node.validate_checklist(valid_checklist) is True
        
        invalid_checklist = {"checklist": {}}
        assert node.validate_checklist(invalid_checklist) is False
    
    def test_empty_checklist_generation(self):
        """Testa geração de checklist vazio."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        categorized_docs = {
            "habilitacao": [],
            "tecnica": [],
            "fiscal": [],
            "juridica": [],
            "outros": []
        }
        
        checklist = node.generate_complete_checklist(categorized_docs)
        
        assert "checklist" in checklist
        assert checklist["resumo"]["total_documentos"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
