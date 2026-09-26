"""
Testes para Etapa 5 do Nó 4: Integração com Nó 3

TDD Approach: Testes escritos antes da implementação
"""

import pytest
from typing import Dict, List


class TestNode4IntegrationWithNode3:
    """Testes para integração do Nó 4 com Nó 3."""
    
    def test_process_from_node3(self):
        """Testa processamento a partir da saída do Nó 3."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node4 = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        node3_output = {
            "structured_info": {
                "documentation": ["CNPJ", "RG", "Certidão Fiscal"]
            }
        }
        
        result = node4.process_from_node3(node3_output)
        
        assert "checklist" in result
        assert "resumo" in result
        assert result["resumo"]["total_documentos"] > 0
    
    def test_complete_workflow_node3_to_node4(self):
        """Testa workflow completo Nó 3 → Nó 4."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node4 = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        node3_analysis = {
            "structured_info": {
                "documentation": ["CNPJ", "ISO 9001", "Certidão Fiscal"],
                "deadlines": ["5 dias"],
                "requirements": ["Autenticado"]
            }
        }
        
        result = node4.process_complete_analysis(node3_analysis)
        
        assert "checklist" in result
        assert "categorized" in result
        assert "summary" in result
    
    def test_extract_docs_from_node3_structure(self):
        """Testa extração de documentos da estrutura do Nó 3."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node4 = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        node3_structure = {
            "documentation": ["CNPJ", "RG", "Certidão"],
            "deadlines": ["5 dias"],
            "requirements": ["Autenticado"]
        }
        
        docs = node4._extract_from_node3_structure(node3_structure)
        
        assert "documents" in docs
        assert "deadlines" in docs
        assert "requirements" in docs
    
    def test_merge_with_deadlines(self):
        """Testa mescla de documentos com prazos."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node4 = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        checklist = {
            "checklist": {
                "habilitacao": {
                    "itens": [{"documento": "CNPJ", "prazo": None}]
                }
            },
            "resumo": {"total_documentos": 1}
        }
        
        deadlines = ["5 dias"]
        
        merged = node4.merge_with_deadlines(checklist, deadlines)
        
        # Como o merge é por correspondência de texto, apenas verifica que a estrutura é mantida
        assert "checklist" in merged
    
    def test_validate_integration_output(self):
        """Testa validação da saída da integração."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node4 = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        valid_output = {
            "checklist": {"habilitacao": {"itens": []}},
            "categorized": {},
            "summary": {"total_documentos": 0}
        }
        
        assert node4.validate_integration_output(valid_output) is True
        
        invalid_output = {"checklist": {}}
        assert node4.validate_integration_output(invalid_output) is False
    
    def test_empty_node3_output(self):
        """Testa processamento com saída vazia do Nó 3."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node4 = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        node3_output = {
            "structured_info": {
                "documentation": []
            }
        }
        
        result = node4.process_from_node3(node3_output)
        
        assert result["resumo"]["total_documentos"] == 0
    
    def test_generate_final_checklist_report(self):
        """Testa geração de relatório final do checklist."""
        from app.rag.node_4_document_generator import Node4DocumentGenerator
        
        node4 = Node4DocumentGenerator(api_key="test_key", mock_mode=True)
        
        checklist = {
            "checklist": {
                "habilitacao": {"itens": [{"documento": "CNPJ"}]}
            },
            "resumo": {"total_documentos": 1}
        }
        
        report = node4.generate_final_report(checklist)
        
        assert "checklist" in report
        assert "recommendations" in report
        assert "priority_actions" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
